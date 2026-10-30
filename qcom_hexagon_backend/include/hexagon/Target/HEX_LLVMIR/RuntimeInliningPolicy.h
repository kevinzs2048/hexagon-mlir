//===- RuntimeInliningPolicy.h - Preserve Hexagon FP-mode boundaries -------===//
// SPDX-License-Identifier: BSD-3-Clause
#ifndef HEXAGON_TARGET_HEX_LLVMIR_RUNTIMEINLININGPOLICY_H
#define HEXAGON_TARGET_HEX_LLVMIR_RUNTIMEINLININGPOLICY_H

#include "hexagon/Target/HEX_LLVMIR/FloatingPointFeatures.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"

namespace mlir::hexagon {

// Some linked HVX math routines require QFloat intrinsics, while ordinary v73
// model functions use IEEE instructions. Inlining the former into the latter
// loses the callee's target features and can make instruction selection fail.
// Protect only these known incompatible call sites, not all uses of a routine.
// Run even when the local inliner is disabled: exported LLVM IR may subsequently
// pass through an external optimization pipeline.
inline unsigned protectHexagonFloatingPointCallBoundaries(llvm::Module &module) {
  unsigned protectedCalls = 0;
  for (llvm::Function &caller : module) {
    if (!hasIEEEV73VectorPolicy(caller))
      continue;
    for (llvm::Instruction &instruction : llvm::instructions(caller)) {
      auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
      if (!call)
        continue;
      const llvm::Function *callee = call->getCalledFunction();
      if (!callee || !callee->hasFnAttribute("target-features"))
        continue;
      llvm::SmallVector<llvm::StringRef> features;
      callee->getFnAttribute("target-features").getValueAsString().split(features, ',');
      bool qfloat = false;
      for (llvm::StringRef feature : features) {
        feature = feature.trim();
        if (feature == "+hvx-qfloat")
          qfloat = true;
        else if (feature == "-hvx-qfloat")
          qfloat = false;
      }
      if (!qfloat)
        continue;
      call->removeFnAttr(llvm::Attribute::AlwaysInline);
      if (!call->getAttributes().hasFnAttr(llvm::Attribute::NoInline)) {
        call->addFnAttr(llvm::Attribute::NoInline);
        ++protectedCalls;
      }
    }
  }
  return protectedCalls;
}

} // namespace mlir::hexagon
#endif
