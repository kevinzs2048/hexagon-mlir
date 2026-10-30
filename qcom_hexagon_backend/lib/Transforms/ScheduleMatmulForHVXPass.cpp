//===- ScheduleMatmulForHVXPass.cpp - Scheduling matmul ops          ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file specifies the rules to schedule matmul op and its variants.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
#include <numeric>
#include <vector>

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Transforms/LinalgUtils.h"
#include "hexagon/Transforms/Transforms.h"

#define DEBUG_TYPE "schedule-matmul-for-hvx"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_SCHEDULEMATMULFORHVX
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// Named transpose builders replace the indexing maps, so only fold canonical
// AB matmuls. In newer MLIR, transpose variants share the MatmulOp class.
static bool hasCanonicalMatmulMaps(linalg::LinalgOp op, bool batch) {
  unsigned rank = batch ? 4 : 3;
  auto *ctx = op.getContext();
  auto m = getAffineDimExpr(rank - 3, ctx);
  auto n = getAffineDimExpr(rank - 2, ctx);
  auto k = getAffineDimExpr(rank - 1, ctx);
  SmallVector<AffineExpr> a, b, c;
  if (batch) {
    auto batchDim = getAffineDimExpr(0, ctx);
    a.push_back(batchDim);
    b.push_back(batchDim);
    c.push_back(batchDim);
  }
  a.append({m, k});
  b.append({k, n});
  c.append({m, n});
  SmallVector<AffineMap> expected{AffineMap::get(rank, 0, a, ctx),
                                AffineMap::get(rank, 0, b, ctx),
                                AffineMap::get(rank, 0, c, ctx)};
  return op.getIndexingMapsArray() == expected;
}

static bool isMatrixTranspose(Operation *op, bool batch) {
  auto transpose = dyn_cast_or_null<linalg::TransposeOp>(op);
  if (!transpose || !transpose.hasPureTensorSemantics())
    return false;
  SmallVector<int64_t> expected = batch ? SmallVector<int64_t>{0, 2, 1}
                                       : SmallVector<int64_t>{1, 0};
  return transpose.getPermutation() == ArrayRef<int64_t>(expected);
}

static Attribute getMatmulCast(linalg::LinalgOp op) {
  if (Attribute attr = op->getAttr("cast"))
    return attr;
  return linalg::TypeFnAttr::get(op.getContext(), linalg::TypeFn::cast_signed);
}

// Named Linalg regions may be customized. Changing an operand's indexing map
// does not authorize replacing its arithmetic, casts, or fast-math attributes.
static void replaceMatmulPreservingBody(IRRewriter &rewriter,
                                       linalg::LinalgOp oldOp,
                                       Operation *newOp) {
  Region &body = newOp->getRegion(0);
  rewriter.eraseBlock(&body.front());
  rewriter.inlineRegionBefore(oldOp->getRegion(0), body, body.end());
  rewriter.replaceOp(oldOp, newOp->getResults());
}

struct ScheduleMatmulForHVXPass
    : public ::impl::ScheduleMatmulForHVXBase<ScheduleMatmulForHVXPass> {
  void runOnOperation() override;
  FailureOr<linalg::GenericOp> GeneralizeOp(IRRewriter &rewriter,
                                            linalg::LinalgOp linalgOp);
};

FailureOr<linalg::GenericOp>
ScheduleMatmulForHVXPass::GeneralizeOp(IRRewriter &rewriter,
                                       linalg::LinalgOp linalgOp) {
  rewriter.setInsertionPoint(linalgOp);
  FailureOr<linalg::GenericOp> generalizedOp =
      linalg::generalizeNamedOp(rewriter, linalgOp);
  if (failed(generalizedOp)) {
    linalgOp->emitOpError("failed to generalize linalg named op");
    signalPassFailure();
    return failure(); // Return a failure result
  }
  return generalizedOp;
}

void ScheduleMatmulForHVXPass::runOnOperation() {
  auto funcOp = getOperation();
  IRRewriter rewriter(&getContext());
  // Optimize below Matmul variants
  //   linalg.matmul(a, transpose_b) -> linalg.matmul_transpose_b
  //   linalg.matmul(transpose_a, b) -> linalg.matmul_transpose_a
  //   linalg.batch_matmul(a, transpose_b) -> linalg.batch_matmul_transpose_b
  //   linalg.batch_matmul(transpose_a, b) -> linalg.batch_matmul_transpose_a
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    if (!linalgOp.hasPureTensorSemantics())
      return;
    if (isa_and_nonnull<linalg::MatmulOp>(linalgOp.getOperation()) &&
        hasCanonicalMatmulMaps(linalgOp, false) &&
        !linalgOp->getAttrOfType<StringAttr>("library_call")) {
      auto firstOperandDef = linalgOp.getDpsInputs()[0].getDefiningOp();
      auto secondOperandDef = linalgOp.getDpsInputs()[1].getDefiningOp();
      rewriter.setInsertionPoint(linalgOp);
      if (isMatrixTranspose(secondOperandDef, false)) {
        // Check if the second operand is a transpose op
        // Convert linalg.transpose + linalg.matmul to linalg.matmul_transpose_b
        auto transposeOp = dyn_cast<linalg::TransposeOp>(secondOperandDef);
        auto matmulTransposeBOp = linalg::MatmulTransposeBOp::create(rewriter,
            linalgOp.getLoc(), linalgOp.getOperation()->getResultTypes(),
            ValueRange{linalgOp.getDpsInputs()[0], transposeOp.getOperand(0)},
            linalgOp.getDpsInits(),
            getMatmulCast(linalgOp));
        replaceMatmulPreservingBody(rewriter, linalgOp, matmulTransposeBOp);
        if (transposeOp->use_empty())
          rewriter.eraseOp(transposeOp);
      } else if (isMatrixTranspose(firstOperandDef, false)) {
        // Check if the first operand is a transpose op
        // Convert linalg.transpose + linalg.matmul to linalg.matmul_transpose_a
        auto transposeOp = dyn_cast<linalg::TransposeOp>(firstOperandDef);
        auto matmulTransposeAOp = linalg::MatmulTransposeAOp::create(rewriter,
            linalgOp.getLoc(), linalgOp.getOperation()->getResultTypes(),
            ValueRange{transposeOp.getOperand(0), linalgOp.getDpsInputs()[1]},
            linalgOp.getDpsInits(),
            getMatmulCast(linalgOp));
        replaceMatmulPreservingBody(rewriter, linalgOp, matmulTransposeAOp);
        if (transposeOp->use_empty())
          rewriter.eraseOp(transposeOp);
      }
    } else if (isa_and_nonnull<linalg::BatchMatmulOp>(
                   linalgOp.getOperation()) &&
               hasCanonicalMatmulMaps(linalgOp, true) &&
               !linalgOp->getAttrOfType<StringAttr>("library_call")) {
      auto firstOperandDef = linalgOp.getDpsInputs()[0].getDefiningOp();
      auto secondOperandDef = linalgOp.getDpsInputs()[1].getDefiningOp();
      rewriter.setInsertionPoint(linalgOp);
      if (isMatrixTranspose(secondOperandDef, true)) {
        // Check if the second operand is a transpose op
        // Convert linalg.transpose + linalg.batch_matmul to
        // linalg.batch_matmul_transpose_b op
        auto transposeOp = dyn_cast<linalg::TransposeOp>(secondOperandDef);
        auto matmulTransposeBOp =
            linalg::BatchMatmulTransposeBOp::create(rewriter,
                linalgOp.getLoc(), linalgOp.getOperation()->getResultTypes(),
                ValueRange{linalgOp.getDpsInputs()[0],
                           transposeOp.getOperand(0)},
                linalgOp.getDpsInits(),
                getMatmulCast(linalgOp));
        replaceMatmulPreservingBody(rewriter, linalgOp, matmulTransposeBOp);
        if (transposeOp->use_empty())
          rewriter.eraseOp(transposeOp);
      } else if (isMatrixTranspose(firstOperandDef, true)) {
        // Check if the first operand is a transpose op
        // Convert linalg.transpose + linalg.batch_matmul to
        // linalg.batch_matmul_transpose_a op
        auto transposeOp = dyn_cast<linalg::TransposeOp>(firstOperandDef);
        auto matmulTransposeAOp =
            linalg::BatchMatmulTransposeAOp::create(rewriter,
                linalgOp.getLoc(), linalgOp.getOperation()->getResultTypes(),
                ValueRange{transposeOp.getOperand(0),
                           linalgOp.getDpsInputs()[1]},
                linalgOp.getDpsInits(),
                getMatmulCast(linalgOp));
        replaceMatmulPreservingBody(rewriter, linalgOp, matmulTransposeAOp);
        if (transposeOp->use_empty())
          rewriter.eraseOp(transposeOp);
      }
    }
  });

  SmallVector<linalg::LinalgOp> batchMatmulOps;
  // Collect all linalg::BatchMatmulOps that do not have the
  // "library_call" attribute
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    if (isa_and_nonnull<linalg::BatchMatmulOp>(linalgOp.getOperation()) &&
        !linalgOp->getAttrOfType<StringAttr>("library_call")) {
      batchMatmulOps.push_back(linalgOp);
    }
  });
  for (auto linalgOp : batchMatmulOps) {
    auto permutation = getBatchMatmulPermutation(linalgOp);
    if (permutation.empty()) {
      linalgOp->emitOpError("failed to determine batch matmul permutation");
      return signalPassFailure();
    }
    // Generalization replaces and erases linalgOp; do not inspect it afterward.
    FailureOr<linalg::GenericOp> generalizedOp =
        GeneralizeOp(rewriter, linalgOp);
    if (failed(generalizedOp))
      return;
    rewriter.setInsertionPoint(*generalizedOp);
    // Apply the interchange transformation
    FailureOr<linalg::GenericOp> interchangedOp =
        linalg::interchangeGenericOp(rewriter, *generalizedOp, permutation);
    if (failed(interchangedOp)) {
      generalizedOp->emitOpError(
          "failed to apply interchange to linalg.generic op");
      return signalPassFailure();
    }
  }

  SmallVector<linalg::LinalgOp> matmulOps;
  // Collect all linalg::MatmulOps that do not have the "library_call" attribute
  funcOp.walk([&](linalg::MatmulOp linalgOp) {
    if (isa_and_nonnull<linalg::LinalgOp>(linalgOp.getOperation()) &&
        !linalgOp->getAttrOfType<StringAttr>("library_call")) {
      matmulOps.push_back(linalgOp);
    }
  });
  for (auto linalgOp : matmulOps) {
    auto permutation = getMatmulPermutation(linalgOp);
    if (permutation.empty()) {
      linalgOp->emitOpError("failed to determine matmul permutation");
      return signalPassFailure();
    }
    FailureOr<linalg::GenericOp> generalizedOp =
        GeneralizeOp(rewriter, linalgOp);
    if (failed(generalizedOp))
      return;
    rewriter.setInsertionPoint(*generalizedOp);
    // Apply the interchange transformation
    FailureOr<linalg::GenericOp> interchangedOp =
        linalg::interchangeGenericOp(rewriter, *generalizedOp, permutation);
    if (failed(interchangedOp)) {
      generalizedOp->emitOpError(
          "failed to apply interchange to linalg.generic op");
      return signalPassFailure();
    }
  }
  // M=1 contractions become vecmat before this pass. Keep contiguous weight
  // columns (N) innermost, just as for the corresponding matmul. This
  // changes traversal of independent outputs, not the K accumulation order.
  // Admit integer W8A8 and full output-column blocks of FP32-accumulating
  // BF16/FP32 projections. Leave small/partial floating-point outputs, such as
  // short-cache attention scores, on their existing schedule.
  SmallVector<linalg::LinalgOp> vecmatOps;
  funcOp.walk([&](linalg::LinalgOp op) {
    if (!isa<linalg::VecmatOp>(op.getOperation()) ||
        !op.hasPureTensorSemantics() || op->hasAttr("library_call"))
      return;
    auto inputs = op.getDpsInputs();
    auto output = cast<RankedTensorType>(op.getDpsInits()[0].getType());
    Type lhs = cast<RankedTensorType>(inputs[0].getType()).getElementType();
    Type rhs = cast<RankedTensorType>(inputs[1].getType()).getElementType();
    bool integerDot = lhs.isSignlessInteger(8) && rhs.isSignlessInteger(8) &&
                      output.getElementType().isSignlessInteger(32);
    bool floatDot = lhs == rhs && (lhs.isF32() || lhs.isBF16()) &&
                    output.getElementType().isF32() && output.hasStaticShape() &&
                    output.getDimSize(0) >= 128 &&
                    output.getDimSize(0) % 128 == 0;
    if (!integerDot && !floatDot)
      return;
    auto *ctx = op.getContext();
    auto n = getAffineDimExpr(0, ctx);
    auto k = getAffineDimExpr(1, ctx);
    SmallVector<AffineMap> maps{AffineMap::get(2, 0, {k}, ctx),
                               AffineMap::get(2, 0, {k, n}, ctx),
                               AffineMap::get(2, 0, {n}, ctx)};
    if (op.getIndexingMapsArray() == maps)
      vecmatOps.push_back(op);
  });
  for (auto op : vecmatOps) {
    auto generic = GeneralizeOp(rewriter, op);
    if (failed(generic))
      return;
    rewriter.setInsertionPoint(*generic);
    if (failed(linalg::interchangeGenericOp(rewriter, *generic, {1, 0}))) {
      generic->emitOpError("failed to schedule contiguous vecmat columns");
      return signalPassFailure();
    }
  }
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createScheduleMatmulForHVXPass() {
  return std::make_unique<ScheduleMatmulForHVXPass>();
}
