//===- CopyCanonicalizationPass.cpp - erase redundant copies   ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass canonicalizes memory copy operations by eliminating intermediate
// allocations that serve as passthroughs between source and destination(s).
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace hexagon;

#define DEBUG_TYPE "copy-canonicalization"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define GEN_PASS_DEF_COPYCANONICALIZATION
#include "hexagon/Transforms/Passes.h.inc"

namespace {

static bool hasSingleStaticElement(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.hasStaticShape() && type.getNumElements() == 1;
}

static Value stripMemRefCasts(Value value) {
  while (auto castOp = value.getDefiningOp<memref::CastOp>())
    value = castOp.getSource();
  return value;
}

static bool isAllocaBacked(Value value) {
  for (;;) {
    if (value.getDefiningOp<memref::AllocaOp>())
      return true;

    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      return false;

    if (auto castOp = dyn_cast<memref::CastOp>(defOp)) {
      value = castOp.getSource();
      continue;
    }
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(defOp)) {
      value = subviewOp.getSource();
      continue;
    }
    if (auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(defOp)) {
      value = reinterpretOp.getSource();
      continue;
    }

    return false;
  }
}

static FailureOr<uint64_t> getStaticCopySizeBytes(MemRefType sourceType,
                                                  MemRefType targetType) {
  if (sourceType.getRank() != targetType.getRank() ||
      sourceType.getElementType() != targetType.getElementType())
    return failure();

  uint64_t elements = 1;
  for (auto dim : llvm::seq<int64_t>(0, sourceType.getRank())) {
    int64_t sourceDim = sourceType.getDimSize(dim);
    int64_t targetDim = targetType.getDimSize(dim);
    if (!ShapedType::isDynamic(sourceDim) &&
        !ShapedType::isDynamic(targetDim) && sourceDim != targetDim)
      return failure();

    int64_t staticDim = ShapedType::isDynamic(sourceDim) ? targetDim : sourceDim;
    if (ShapedType::isDynamic(staticDim))
      return failure();

    elements *= static_cast<uint64_t>(staticDim);
  }

  unsigned elementBitWidth = sourceType.getElementTypeBitWidth();
  if (!elementBitWidth)
    return failure();

  uint64_t elementBytes = llvm::divideCeil(elementBitWidth, 8u);
  return elements * elementBytes;
}

static SmallVector<Value> createCopyBounds(Location loc, PatternRewriter &rewriter,
                                           Value source, Value target) {
  auto sourceType = cast<MemRefType>(source.getType());
  auto targetType = cast<MemRefType>(target.getType());
  SmallVector<Value> bounds;
  bounds.reserve(sourceType.getRank());
  for (int64_t dim = 0; dim < sourceType.getRank(); ++dim) {
    int64_t sourceDim = sourceType.getDimSize(dim);
    int64_t targetDim = targetType.getDimSize(dim);
    if (ShapedType::isDynamic(sourceDim)) {
      bounds.push_back(rewriter.create<memref::DimOp>(loc, source, dim));
      continue;
    }
    if (ShapedType::isDynamic(targetDim)) {
      bounds.push_back(rewriter.create<memref::DimOp>(loc, target, dim));
      continue;
    }
    bounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, sourceDim));
  }
  return bounds;
}

struct EraseSelfCopyPattern final : public OpRewritePattern<memref::CopyOp> {
  EraseSelfCopyPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    if (stripMemRefCasts(op.getSource()) != stripMemRefCasts(op.getTarget()))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

struct SingleElementCopyToLoadStorePattern final
    : public OpRewritePattern<memref::CopyOp> {
  SingleElementCopyToLoadStorePattern(MLIRContext *ctx)
      : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasSingleStaticElement(op.getSource()) ||
        !hasSingleStaticElement(op.getTarget()))
      return failure();

    auto sourceType = cast<MemRefType>(op.getSource().getType());
    auto targetType = cast<MemRefType>(op.getTarget().getType());
    if (sourceType.getRank() != targetType.getRank())
      return failure();

    Location loc = op.getLoc();
    SmallVector<Value> sourceIndices(sourceType.getRank());
    for (Value &index : sourceIndices)
      index = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    SmallVector<Value> targetIndices(targetType.getRank());
    for (Value &index : targetIndices)
      index = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    Value loaded =
        rewriter.create<memref::LoadOp>(loc, op.getSource(), sourceIndices);
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, loaded, op.getTarget(), targetIndices);
    return success();
  }
};

struct SmallAllocaCopyToLoopsPattern final
    : public OpRewritePattern<memref::CopyOp> {
  SmallAllocaCopyToLoopsPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    if (!isAllocaBacked(op.getSource()) && !isAllocaBacked(op.getTarget()))
      return failure();

    auto sourceType = dyn_cast<MemRefType>(op.getSource().getType());
    auto targetType = dyn_cast<MemRefType>(op.getTarget().getType());
    if (!sourceType || !targetType || sourceType.getRank() != targetType.getRank())
      return failure();

    FailureOr<uint64_t> copyBytes =
        getStaticCopySizeBytes(sourceType, targetType);
    if (failed(copyBytes) || *copyBytes > 4096)
      return failure();

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> bounds =
        createCopyBounds(loc, rewriter, op.getSource(), op.getTarget());
    SmallVector<Value> indices;
    indices.reserve(sourceType.getRank());

    auto emitCopy = [&](PatternRewriter &nestedRewriter) {
      Value loaded =
          nestedRewriter.create<memref::LoadOp>(loc, op.getSource(), indices);
      nestedRewriter.create<memref::StoreOp>(loc, loaded, op.getTarget(),
                                             indices);
    };

    auto emitLoops = [&](auto &&emitLoops, unsigned dim) -> void {
      if (dim == bounds.size()) {
        emitCopy(rewriter);
        return;
      }

      scf::ForOp forOp = rewriter.create<scf::ForOp>(loc, zero, bounds[dim], one);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());
      indices.push_back(forOp.getInductionVar());
      emitLoops(emitLoops, dim + 1);
      indices.pop_back();
    };

    emitLoops(emitLoops, 0);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to eliminate a single intermediate buffer allocation in copy chains.
///
/// This pattern matches on a copy operation that writes to an
/// intermediate allocation, then replaces all subsequent reading copies
/// to directly use the original source, eliminating the intermediate buffer.
///
/// Example transformation:
///  ```
///   %src = ...
///   %intermediate = memref.alloc() : memref<NxT>
///   memref.copy %src, %intermediate          // Writing copy (matched here)
///   memref.copy %intermediate, %dst1         // Reading copy 1
///   memref.copy %intermediate, %dst2         // Reading copy 2
///  ```
/// Becomes:
///  ```
///   %src = ...
///   memref.copy %src, %dst1
///   memref.copy %src, %dst2
///  ```
/// The pattern handles both single and multiple reader cases.
struct IntermediateBufferEliminationPattern final
    : public OpRewritePattern<memref::CopyOp> {
  IntermediateBufferEliminationPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    Value source = op.getSource();
    Value target = op.getTarget();
    auto intermediateAlloc = target.getDefiningOp<memref::AllocOp>();
    if (!intermediateAlloc)
      return failure();

    // Validates that all users except this op (writing copy) are copies that
    // read from the intermediate buffer
    SmallVector<memref::CopyOp> readingCopies;
    for (Operation *user : intermediateAlloc->getUsers()) {
      // Skip the current writing copy
      if (user == op.getOperation())
        continue;

      auto copyOp = dyn_cast<memref::CopyOp>(user);
      if (!copyOp)
        return failure();

      // Must be a reading copy (reads from intermediate buffer)
      if (copyOp.getSource() != intermediateAlloc)
        return failure();

      if (copyOp->getBlock() != op->getBlock() || !op->isBeforeInBlock(copyOp))
        return failure();

      readingCopies.push_back(copyOp);
    }

    if (readingCopies.empty())
      return failure();

    for (auto readingCopy : readingCopies) {
      DBG("Replacing reading copy: " << readingCopy);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(readingCopy);
      auto newCopy = rewriter.replaceOpWithNewOp<memref::CopyOp>(
          readingCopy, source, readingCopy.getTarget());
      DBG("With new copy: " << newCopy);
    }

    DBG("Removed writing copy: " << op);
    DBG("Removed intermediate buffer: " << intermediateAlloc);
    rewriter.eraseOp(op);
    rewriter.eraseOp(intermediateAlloc);
    return success();
  }
};

/// This pattern explicitly adds copies if the yielded value does not match the
/// init_arg. This simplifies subsequent buffer deallocation analyses to dynamic
/// memref aliasing via deallocation helper. In conjuction with other passes and
/// patterns, this enables removal of redundant alloc and copies.
///   Example transformation:
///   ```
///   scf.yield %A : memref<10xf32>
///   ```
///   Becomes:
///   ```
///   memref.copy %A, %loop_arg : memref<10xf32> to memref<10xf32>
///   scf.yield %loop_arg : memref<10xf32>
///   ```
struct AddCopyBeforeYieldPattern final : public OpRewritePattern<scf::YieldOp> {
  AddCopyBeforeYieldPattern(MLIRContext *ctx)
      : OpRewritePattern<scf::YieldOp>(ctx) {}

  LogicalResult matchAndRewrite(scf::YieldOp yieldOp,
                                PatternRewriter &rewriter) const override {
    auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
    if (!forOp)
      return failure();

    bool changed = false;
    SmallVector<Value> newYieldOperands =
        llvm::to_vector<4>(yieldOp.getOperands());

    for (const auto &operand : llvm::enumerate(yieldOp.getOperands())) {
      size_t index = operand.index();
      Value yieldedValue = operand.value();

      if (!mlir::isa<MemRefType>(yieldedValue.getType()))
        continue;

      BlockArgument iterArg = forOp.getRegionIterArgs()[index];
      if (yieldedValue == iterArg)
        continue;

      // If the yieldedValue is a memref and is NOT the iter_arg.
      // Then, we need to insert a copy to canonicalize the IR.
      rewriter.setInsertionPoint(yieldOp); // Insert copy before the yieldOp
      auto newCopy = rewriter.create<memref::CopyOp>(yieldOp.getLoc(),
                                                     yieldedValue, iterArg);
      DBG("Added memref.copy before scf.yield: " << newCopy);

      // The new operand for the yield will be the iter_arg itself
      newYieldOperands[index] = iterArg;
      changed = true;
    }

    if (changed) {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, newYieldOperands);
      return success();
    }

    return failure();
  }
};

struct CopyCanonicalizationPass
    : public ::impl::CopyCanonicalizationBase<CopyCanonicalizationPass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<EraseSelfCopyPattern>(patterns.getContext());
    patterns.add<SingleElementCopyToLoadStorePattern>(patterns.getContext());
    patterns.add<SmallAllocaCopyToLoopsPattern>(patterns.getContext());
    patterns.add<IntermediateBufferEliminationPattern>(patterns.getContext());
    patterns.add<AddCopyBeforeYieldPattern>(patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // end anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createCopyCanonicalizationPass() {
  return std::make_unique<CopyCanonicalizationPass>();
}
