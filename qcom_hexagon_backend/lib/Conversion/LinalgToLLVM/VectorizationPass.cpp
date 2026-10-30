//===- Vectorization.cpp - Implementation of Vectorization Pass  ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of inner loop to vector form so that vector
// to llvm pass can convert to vectorized llvm-ir.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"

#define DEBUG_TYPE "hexagon-vectorize"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_HEXAGONVECTORIZATION
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

static LogicalResult vectorizeLinalgOp(linalg::LinalgOp op) {
  auto fnName = op->getAttrOfType<StringAttr>("library_call");
  if (fnName) {
    DBG("-> skipping vectorization. Op will be replaced with a library call.");
    return failure();
  }

  IRRewriter rewriter(op.getContext());

  auto numLoops = op.getNumLoops();
  if (numLoops < 1) {
    return failure();
  }
  auto innerLoopDim = op.getStaticLoopRanges()[numLoops - 1];

  auto dataTileSize = computeDataTileSize(op);
  if (!dataTileSize ||
      !perfectlyVectorizable(dataTileSize.value(), innerLoopDim)) {
    DBG("-> skipping vectorization. data tile size and loop mismatch");
    return failure();
  }

  SmallVector<int64_t> vecSizes(numLoops, 1);
  SmallVector<bool> scalableDims(numLoops, false);
  vecSizes[numLoops - 1] = innerLoopDim;

  FailureOr<mlir::linalg::VectorizationResult> vectorResults =
      linalg::vectorize(rewriter, op, vecSizes, scalableDims);
  if (failed(vectorResults))
    return failure();
  // Replace the original op with the vectorized op.
  rewriter.replaceOp(op, vectorResults->replacements);
  return success();
}


// A narrow, integer-only tensor-to-vector loop-carried-state promotion.
// Keep one final tensor write, but do not allocate/copy a tensor for each K
// iteration. Masked, shifted, partial, nonidentity or multiply-used reads are
// deliberately excluded. This changes storage, never accumulation order.
static void promoteIntegerVectorAccumulator(scf::ForOp loop) {
  auto function = loop->getParentOfType<func::FuncOp>();
  if (!function || !function->hasAttr("hexagon.outlined_matmul") ||
      loop.getNumResults() != 1)
    return;
  auto type = dyn_cast<RankedTensorType>(loop.getResult(0).getType());
  if (!type || !type.hasStaticShape() || type.getRank() != 1 ||
      type.getDimSize(0) != 128 || !type.getElementType().isSignlessInteger(32))
    return;
  auto argument = loop.getRegionIterArgs()[0];
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  auto write = yield.getOperand(0).getDefiningOp<vector::TransferWriteOp>();
  if (!write || write->getBlock() != loop.getBody() ||
      !write->hasOneUse() || write.getMask() ||
      !write.getPermutationMap().isIdentity() ||
      write.getVectorType().getShape() != type.getShape() ||
      write.getVectorType().isScalable())
    return;
  vector::TransferReadOp read;
  for (Operation *user : argument.getUsers()) {
    if (user == write.getOperation())
      continue;
    auto candidate = dyn_cast<vector::TransferReadOp>(user);
    if (!candidate || read || candidate->getBlock() != loop.getBody() ||
        candidate.getMask() || !candidate.getPermutationMap().isIdentity() ||
        candidate.getVectorType() != write.getVectorType())
      return;
    read = candidate;
  }
  if (!read)
    return;
  auto allZero = [](ValueRange indices) {
    return llvm::all_of(indices, [](Value index) {
      return matchPattern(index, m_Zero());
    });
  };
  if (!allZero(read.getIndices()) || !allZero(write.getIndices()))
    return;

  OpBuilder builder(loop);
  Location loc = loop.getLoc();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value padding = arith::ConstantIntOp::create(builder, loc, 0, 32);
  IRMapping initialMapping;
  initialMapping.map(argument, loop.getInitArgs()[0]);
  initialMapping.map(read.getPadding(), padding);
  for (Value index : read.getIndices())
    initialMapping.map(index, zero);
  auto initial = cast<vector::TransferReadOp>(
      builder.clone(*read, initialMapping));
  auto promoted = scf::ForOp::create(
      builder, loc, loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
      ValueRange{initial.getResult()},
      [](OpBuilder &, Location, Value, ValueRange) {});
  IRMapping bodyMapping;
  bodyMapping.map(loop.getInductionVar(), promoted.getInductionVar());
  bodyMapping.map(read.getResult(), promoted.getRegionIterArgs()[0]);
  builder.setInsertionPointToStart(promoted.getBody());
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (&operation == read.getOperation() || &operation == write.getOperation())
      continue;
    builder.clone(operation, bodyMapping);
  }
  scf::YieldOp::create(builder, loc,
                      bodyMapping.lookupOrDefault(write.getVector()));
  builder.setInsertionPointAfter(promoted);
  IRMapping finalMapping;
  finalMapping.map(write.getVector(), promoted.getResult(0));
  finalMapping.map(write.getBase(), loop.getInitArgs()[0]);
  for (Value index : write.getIndices())
    finalMapping.map(index, zero);
  auto finalWrite = cast<vector::TransferWriteOp>(
      builder.clone(*write, finalMapping));
  loop.getResult(0).replaceAllUsesWith(finalWrite->getResult(0));
  loop.erase();
}

struct HexagonVectorizationPass
    : public ::impl::HexagonVectorizationBase<HexagonVectorizationPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();
    moduleOp.walk([&](linalg::LinalgOp op) {
      DBG("vectorization candidate: " << op << "\n");
      if (succeeded(vectorizeLinalgOp(op))) {
        DBG(" -> vectorization succeeded.\n");
      } else {
        DBG(" -> vectorization failed.\n");
      }
      return WalkResult::advance();
    });
    // Vectorization has produced complete output-tile transfers. Promote only
    // the explicitly checked outlined INT8 accumulation loops above.
    moduleOp.walk([&](scf::ForOp loop) {
      promoteIntegerVectorAccumulator(loop);
    });
  }
};
} // namespace
std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createHexagonVectorizationPass() {
  return std::make_unique<HexagonVectorizationPass>();
}
