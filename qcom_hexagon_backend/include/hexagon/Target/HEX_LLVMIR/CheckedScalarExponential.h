// SPDX-License-Identifier: BSD-3-Clause
#ifndef HEXAGON_TARGET_HEX_LLVMIR_CHECKEDSCALAREXPONENTIAL_H
#define HEXAGON_TARGET_HEX_LLVMIR_CHECKEDSCALAREXPONENTIAL_H

#include "hexagon/Target/HEX_LLVMIR/FloatingPointFeatures.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IntrinsicInst.h"
#include <map>

namespace mlir::hexagon {

// The tested DSP SDK expf mishandles overflow, NaNs and subnormals. Calls that
// the MLIR HVX-routine pass has not replaced would otherwise lower to expf
// directly. Route ordinary FP32 exp intrinsics through the checked runtime
// entry point; vector intrinsics already scalarize to SDK calls on this target.
// Do not rewrite constrained FP or unverified target policies.
inline unsigned useCheckedScalarExponential(llvm::Module &module,
                                            bool outlineVectors = false) {
  llvm::SmallVector<llvm::IntrinsicInst *> operations;
  for (llvm::Function &function : module) {
    if (!hasIEEEV73VectorPolicy(function))
      continue;
    for (llvm::Instruction &instruction : llvm::instructions(function)) {
      auto *op = llvm::dyn_cast<llvm::IntrinsicInst>(&instruction);
      if (!op || op->getIntrinsicID() != llvm::Intrinsic::exp)
        continue;
      auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(op->getType());
      if (op->getType()->isFloatTy() ||
          (vector && vector->getElementType()->isFloatTy()))
        operations.push_back(op);
    }
  }
  if (operations.empty())
    return 0;
  llvm::IRBuilder<> builder(module.getContext());
  auto checked = module.getOrInsertFunction(
      "__hexagon_mlir_expf", builder.getFloatTy(), builder.getFloatTy());
  // Keep large-model machine functions bounded without changing exp math.
  // Reuse only within the same caller/type so distinct target/FP policies are
  // never silently merged. This is a DSP-local call, not an RPC boundary.
  std::map<std::pair<llvm::Function *, llvm::Type *>, llvm::Function *> helpers;
  for (llvm::IntrinsicInst *op : operations) {
    builder.SetInsertPoint(op);
    builder.SetCurrentDebugLocation(op->getDebugLoc());
    llvm::Value *input = op->getArgOperand(0), *result;
    if (auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(op->getType())) {
      if (outlineVectors) {
        auto key = std::make_pair(op->getFunction(), op->getType());
        llvm::Function *&helper = helpers[key];
        if (!helper) {
          auto *type = llvm::FunctionType::get(vector, {vector}, false);
          helper = llvm::Function::Create(type, llvm::GlobalValue::InternalLinkage,
              "__hexagon_mlir_checked_exp_v" + std::to_string(vector->getNumElements()), module);
          helper->addFnAttr(llvm::Attribute::NoInline);
          for (llvm::StringRef name : {"target-cpu", "target-features", "target-abi",
                   "tune-cpu", "denormal-fp-math", "denormal-fp-math-f32"})
            if (auto attr = op->getFunction()->getFnAttribute(name); attr.isValid())
              helper->addFnAttr(attr);
          llvm::IRBuilder<> body(llvm::BasicBlock::Create(module.getContext(), "entry", helper));
          llvm::Value *output = llvm::PoisonValue::get(vector);
          for (unsigned lane = 0; lane < vector->getNumElements(); ++lane) {
            auto *value = body.CreateExtractElement(helper->getArg(0), body.getInt32(lane));
            auto *exponential = body.CreateCall(checked, {value});
            output = body.CreateInsertElement(output, exponential, body.getInt32(lane));
          }
          body.CreateRet(output);
        }
        result = builder.CreateCall(helper, {input});
      } else {
        result = llvm::PoisonValue::get(vector);
        for (unsigned lane = 0; lane < vector->getNumElements(); ++lane) {
          auto *value = builder.CreateExtractElement(input, builder.getInt32(lane));
          auto *exponential = builder.CreateCall(checked, {value});
          result = builder.CreateInsertElement(result, exponential, builder.getInt32(lane));
        }
      }
    } else {
      result = builder.CreateCall(checked, {input});
    }
    op->replaceAllUsesWith(result);
    op->eraseFromParent();
  }
  return operations.size();
}

} // namespace mlir::hexagon
#endif
