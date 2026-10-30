// SPDX-License-Identifier: BSD-3-Clause
#include "hexagon/Target/HEX_LLVMIR/RuntimeInliningPolicy.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include <cassert>

int main() {
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  auto module = llvm::parseAssemblyString(R"IR(
define float @routine(float %x) #0 {
  %y = fadd float %x, 1.0
  ret float %y
}
define float @ieee(float %x) #1 {
  %y = call float @routine(float %x) alwaysinline
  ret float %y
}
define float @compatible(float %x) #0 {
  %y = call float @routine(float %x)
  ret float %y
}
define float @other_arch(float %x) #2 {
  %y = call float @routine(float %x)
  ret float %y
}
define float @no_policy(float %x) {
  %y = call float @routine(float %x)
  ret float %y
}
attributes #0 = { alwaysinline "target-features"="+hvxv73,+hvx-length128b,+hvx-qfloat" }
attributes #1 = { "target-features"="+hvxv73,+hvx-length128b,-hvx-qfloat,+hvx-ieee-fp" }
attributes #2 = { "target-features"="+hvxv79,+hvx-length128b,-hvx-qfloat,+hvx-ieee-fp" }
)IR", diagnostic, context);
  assert(module && !llvm::verifyModule(*module, &llvm::errs()));
  assert(mlir::hexagon::protectHexagonFloatingPointCallBoundaries(*module) == 1);
  assert(mlir::hexagon::protectHexagonFloatingPointCallBoundaries(*module) == 0);
  assert(!llvm::verifyModule(*module, &llvm::errs()));
  assert(module->getFunction("routine")->hasFnAttribute(llvm::Attribute::AlwaysInline));
  llvm::legacy::PassManager passes;
  passes.add(llvm::createAlwaysInlinerLegacyPass());
  passes.run(*module);
  assert(!llvm::verifyModule(*module, &llvm::errs()));
  for (auto name : {"ieee", "compatible", "other_arch", "no_policy"}) {
    unsigned calls = 0;
    for (auto &instruction : llvm::instructions(module->getFunction(name)))
      calls += llvm::isa<llvm::CallBase>(instruction);
    assert(calls == (llvm::StringRef(name) == "ieee" ? 1U : 0U));
  }
  module->print(llvm::outs(), nullptr);
}
