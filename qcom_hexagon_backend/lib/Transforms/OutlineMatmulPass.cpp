//===- OutlineMatmulPass.cpp - Reuse DSP-local matmul functions ------------===//
// SPDX-License-Identifier: BSD-3-Clause.

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_OUTLINEMATMUL
#include "hexagon/Transforms/Passes.h.inc"

namespace {
// Compare the complete region, including SSA wiring and arithmetic attributes.
// Matching only the matrix shapes would incorrectly merge customized bodies.
bool equivalent(linalg::LinalgOp op, linalg::LinalgOp prototype) {
  if (op->getOperandTypes() != prototype->getOperandTypes() ||
      op->getResultTypes() != prototype->getResultTypes())
    return false;
  IRMapping mapping;
  for (auto [left, right] :
       llvm::zip(op->getOperands(), prototype->getOperands())) {
    if (mapping.contains(left) && mapping.lookup(left) != right)
      return false;
    mapping.map(left, right);
  }
  return OperationEquivalence::isEquivalentTo(
      op, prototype,
      [&](Value left, Value right) {
        return success(mapping.lookupOrNull(left) == right);
      },
      [&](Value left, Value right) { mapping.map(left, right); },
      OperationEquivalence::IgnoreLocations);
}

struct OutlineMatmulPass
    : public ::impl::OutlineMatmulBase<OutlineMatmulPass> {
  explicit OutlineMatmulPass(bool tile = false) { tileForCache = tile; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    LLVM::LLVMDialect, affine::AffineDialect,
                    arith::ArithDialect, scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<linalg::LinalgOp> work;
    module.walk([&](Operation *operation) {
      auto op = dyn_cast<linalg::LinalgOp>(operation);
      if (!op || !isa<linalg::MatmulOp, linalg::VecmatOp>(operation))
        return;
      auto parent = op->getParentOfType<func::FuncOp>();
      if (!parent || parent->getParentOp() != module ||
          parent->hasAttr("hexagon.outlined_matmul") ||
          !op.hasPureTensorSemantics() || op->hasAttr("library_call"))
        return;
      // M=1 matmuls become vecmat before this pass. Reuse the corresponding
      // DSP-local dot products too. Leave short/partial floating-point outputs
      // (including short-cache attention scores) on the existing path.
      if (isa<linalg::VecmatOp>(operation)) {
        auto lhs = cast<RankedTensorType>(op.getDpsInputs()[0].getType());
        auto rhs = cast<RankedTensorType>(op.getDpsInputs()[1].getType());
        auto output = cast<RankedTensorType>(op.getDpsInits()[0].getType());
        auto leftType = lhs.getElementType();
        auto rightType = rhs.getElementType();
        auto outputType = output.getElementType();
        bool integerDot = leftType.isSignlessInteger(8) &&
                          rightType.isSignlessInteger(8) &&
                          outputType.isSignlessInteger(32);
        bool floatDot = leftType == rightType &&
                        (leftType.isF32() || leftType.isBF16()) &&
                        outputType.isF32() && output.hasStaticShape() &&
                        output.getDimSize(0) >= 128 &&
                        output.getDimSize(0) % 128 == 0;
        if (!integerDot && !floatDot)
          return;
      }
      // Named Linalg regions can legally capture surrounding SSA values.
      // Leave those untouched instead of creating an invalid isolated function.
      llvm::SetVector<Value> captures;
      getUsedValuesDefinedAbove(op->getRegions(), captures);
      if (captures.empty())
        work.push_back(op);
    });

    SymbolTable symbols(module);
    SmallVector<std::pair<linalg::LinalgOp, func::FuncOp>> helpers;
    OpBuilder builder(&getContext());
    for (linalg::LinalgOp op : work) {
      func::FuncOp helper;
      for (auto [prototype, candidate] : helpers) {
        if (equivalent(op, prototype)) {
          helper = candidate;
          break;
        }
      }
      if (!helper) {
        helper = func::FuncOp::create(
            op.getLoc(), isa<linalg::VecmatOp>(op.getOperation())
                             ? "__hexagon_vecmat" : "__hexagon_matmul",
            builder.getFunctionType(op->getOperandTypes(), op->getResultTypes()));
        helper.setPrivate();
        // MLIR symbol visibility alone does not imply LLVM internal linkage.
        helper->setAttr("llvm.linkage", LLVM::LinkageAttr::get(
                                          &getContext(), LLVM::Linkage::Internal));
        helper->setAttr("hexagon.outlined_matmul", builder.getUnitAttr());
        // Survives FuncToLLVM; keep LLVM O3 from rebuilding one giant function.
        helper->setAttr("passthrough", builder.getStrArrayAttr({"noinline"}));
        symbols.insert(helper);
        Block *entry = helper.addEntryBlock();
        builder.setInsertionPointToStart(entry);
        IRMapping mapping;
        mapping.map(op->getOperands(), entry->getArguments());
        auto cloned = cast<linalg::LinalgOp>(builder.clone(*op, mapping));
        func::ReturnOp::create(builder, op.getLoc(), cloned->getResults());
        helpers.emplace_back(cloned, helper);
      }
      builder.setInsertionPoint(op);
      auto call = func::CallOp::create(builder, op.getLoc(), helper,
                                       op->getOperands());
      op->replaceAllUsesWith(call.getResults());
      op->erase();
    }

    // Keep the default schedule unchanged. Only tile after helper equivalence
    // matching is complete, because tiling replaces the prototype operation.
    if (!tileForCache)
      return;
    IRRewriter rewriter(&getContext());
    for (auto [matmul, helper] : helpers) {
      // Unit-M matmuls have already become vecmat. Tile the output axis
      // using its two-dimensional iteration space, not matmul's M/N/K sizes.
      bool isVecmat = isa<linalg::VecmatOp>(matmul.getOperation());
      if (!isVecmat && !isa<linalg::MatmulOp>(matmul.getOperation()))
        continue;
      auto lhs = cast<RankedTensorType>(matmul.getDpsInputs()[0].getType());
      auto rhs = cast<RankedTensorType>(matmul.getDpsInputs()[1].getType());
      auto output = cast<RankedTensorType>(matmul.getDpsInits()[0].getType());
      // Narrow admission: exact W8A8 shapes with no partial output vector.
      // In particular this does not change floating-point reduction order.
      if (!lhs.hasStaticShape() || !rhs.hasStaticShape() ||
          !output.hasStaticShape() || !lhs.getElementType().isSignlessInteger(8) ||
          !rhs.getElementType().isSignlessInteger(8) ||
          !output.getElementType().isSignlessInteger(32) ||
          rhs.getDimSize(1) <= 128 || rhs.getDimSize(1) % 128 != 0)
        continue;
      linalg::LinalgTilingOptions tiling;
      if (isVecmat)
        tiling.setTileSizes(ArrayRef<int64_t>{128, 0});
      else
        tiling.setTileSizes({0, 128, 0});
      rewriter.setInsertionPoint(matmul);
      auto tiled = linalg::tileLinalgOp(rewriter, matmul, tiling);
      if (failed(tiled)) {
        matmul.emitError("failed to cache-tile outlined matmul");
        return signalPassFailure();
      }
      rewriter.replaceOp(matmul, tiled->tensorResults);
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::hexagon::createOutlineMatmulPass(bool tileForCache) {
  return std::make_unique<OutlineMatmulPass>(tileForCache);
}
