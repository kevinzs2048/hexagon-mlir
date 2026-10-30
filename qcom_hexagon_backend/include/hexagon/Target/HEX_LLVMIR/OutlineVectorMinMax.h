// SPDX-License-Identifier: BSD-3-Clause
#ifndef HEXAGON_TARGET_HEX_LLVMIR_OUTLINEVECTORMINMAX_H
#define HEXAGON_TARGET_HEX_LLVMIR_OUTLINEVECTORMINMAX_H

#include "hexagon/Target/HEX_LLVMIR/FloatingPointFeatures.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IntrinsicInst.h"
#include <map>

namespace mlir::hexagon {

// Keep the exact LLVM intrinsic (including NaN/signed-zero semantics), but
// avoid replicating its expensive machine lowering at every unrolled use.
// One HVX-width FP32 vector only; helpers never add a host/DSP RPC boundary.
inline unsigned outlineVectorMinMax(llvm::Module &module) {
  llvm::SmallVector<llvm::IntrinsicInst *> operations;
  for (llvm::Function &function : module) {
    if (!hasIEEEV73VectorPolicy(function) ||
        function.hasFnAttribute("hexagon-vector-minmax-helper") ||
        function.hasFnAttribute(llvm::Attribute::StrictFP))
      continue;
    for (llvm::Instruction &instruction : llvm::instructions(function)) {
      auto *op = llvm::dyn_cast<llvm::IntrinsicInst>(&instruction);
      if (!op || (op->getIntrinsicID() != llvm::Intrinsic::vector_reduce_fmaximum &&
                  op->getIntrinsicID() != llvm::Intrinsic::vector_reduce_fminimum) ||
          op->getFastMathFlags().any() || op->hasOperandBundles() ||
          op->isMustTailCall())
        continue;
      auto *type = llvm::dyn_cast<llvm::FixedVectorType>(
          op->getArgOperand(0)->getType());
      if (type && type->getNumElements() == 32 &&
          type->getElementType()->isFloatTy() && op->getType()->isFloatTy())
        operations.push_back(op);
    }
  }
  std::map<std::pair<llvm::Function *, unsigned>, llvm::Function *> helpers;
  for (llvm::IntrinsicInst *op : operations) {
    auto key = std::make_pair(op->getFunction(),
                             static_cast<unsigned>(op->getIntrinsicID()));
    llvm::Function *&helper = helpers[key];
    if (!helper) {
      auto *type = llvm::FunctionType::get(op->getType(),
          {op->getArgOperand(0)->getType()}, false);
      bool maximum = op->getIntrinsicID() == llvm::Intrinsic::vector_reduce_fmaximum;
      helper = llvm::Function::Create(type, llvm::GlobalValue::InternalLinkage,
          maximum ? "__hexagon_vector_fmaximum_v32" : "__hexagon_vector_fminimum_v32",
          module);
      helper->addFnAttr(llvm::Attribute::NoInline);
      helper->addFnAttr("hexagon-vector-minmax-helper");
      for (llvm::StringRef name : {"target-cpu", "target-features", "target-abi",
               "tune-cpu", "denormal-fp-math", "denormal-fp-math-f32",
               "unsafe-fp-math", "no-nans-fp-math", "no-infs-fp-math",
               "no-signed-zeros-fp-math", "approx-func-fp-math",
               "hexagon-no-bit-copy-generation", "hexagon-no-gen-insert"})
        if (auto attr = op->getFunction()->getFnAttribute(name); attr.isValid())
          helper->addFnAttr(attr);
      llvm::IRBuilder<> body(llvm::BasicBlock::Create(
          module.getContext(), "entry", helper));
      body.CreateRet(body.CreateCall(op->getCalledFunction(), {helper->getArg(0)}));
    }
    llvm::IRBuilder<> builder(op);
    builder.SetCurrentDebugLocation(op->getDebugLoc());
    auto *call = builder.CreateCall(helper, {op->getArgOperand(0)});
    op->replaceAllUsesWith(call);
    op->eraseFromParent();
  }
  return operations.size();
}

} // namespace mlir::hexagon
#endif
