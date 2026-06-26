//===-- LinalgToTLERawPass.cpp - recognized kernel to hexkl.tle_raw_call --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//
//
// Recognizes a whole hand-written-backed kernel (by its func.func symbol name,
// via a static registry) and replaces its memref-based elementwise linalg.generic
// with a single hexkl.tle_raw_call naming the hand-written core to invoke.
// Gated by env HEXKL_TLE_RAW at pipeline-insertion time.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_LINALGTOTLERAW
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// Registry: kernel func symbol name -> hand-written core symbol to call.
// Adding a new operator = adding a row here (+ its core wrapper + shim gate).
static const llvm::StringMap<StringRef> &tleRegistry() {
  static const llvm::StringMap<StringRef> reg = {
      {"compiler_add", "triton_add_raw_tle"},
      {"compiler_swiglu", "triton_swiglu_raw_tle"},
  };
  return reg;
}

struct RecognizeTLEKernel final : public OpRewritePattern<linalg::GenericOp> {
  RecognizeTLEKernel(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Must be memref-based (post-bufferization), no tensor results.
    if (op.hasPureTensorSemantics())
      return failure();
    // Look up the containing func by name in the registry.
    auto func = op->getParentOfType<FunctionOpInterface>();
    if (!func)
      return failure();
    auto it = tleRegistry().find(func.getName());
    if (it == tleRegistry().end())
      return failure();
    StringRef callee = it->second;

    // operands[0] = out (init), rest = inputs, in DPS order.
    SmallVector<Value> operands;
    operands.push_back(op.getDpsInitOperand(0)->get());
    for (OpOperand *in : op.getDpsInputOperands())
      operands.push_back(in->get());

    rewriter.replaceOpWithNewOp<hexkl::TLERawCallOp>(
        op, TypeRange{}, operands, rewriter.getStringAttr(callee));
    return success();
  }
};

void populateLinalgToTLERawPatterns(RewritePatternSet &patterns) {
  patterns.add<RecognizeTLEKernel>(patterns.getContext());
}

struct LinalgToTLERawPass
    : public ::impl::LinalgToTLERawBase<LinalgToTLERawPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hexkl::HexKLDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateLinalgToTLERawPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createLinalgToTLERawPass() {
  return std::make_unique<LinalgToTLERawPass>();
}
