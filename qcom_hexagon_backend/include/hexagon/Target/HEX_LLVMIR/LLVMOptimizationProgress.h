//===- LLVMOptimizationProgress.h - Opt-in pass names and elapsed time -----===//
// SPDX-License-Identifier: BSD-3-Clause.
#ifndef HEXAGON_LLVM_OPTIMIZATION_PROGRESS_H
#define HEXAGON_LLVM_OPTIMIZATION_PROGRESS_H

#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace mlir::hexagon {
// Observation only: no IR bodies/constants and no pass filtering. The state is
// owned by callbacks and scoped to one optimizer invocation, not process-global.
inline void registerLLVMOptimizationProgress(
    llvm::PassInstrumentationCallbacks &callbacks) {
  using Clock = std::chrono::steady_clock;
  struct Frame {
    unsigned long long id;
    std::string pass;
    Clock::time_point started;
  };
  struct State {
    unsigned long long next = 0;
    std::vector<Frame> stack;
  };
  auto state = std::make_shared<State>();
  callbacks.registerBeforeNonSkippedPassCallback(
      [state](llvm::StringRef name, llvm::Any) {
        Frame frame{++state->next, name.str(), Clock::now()};
        llvm::errs() << "[hexmlir-llvm] begin " << frame.id << " "
                     << frame.pass << "\n";
        llvm::errs().flush();
        state->stack.push_back(std::move(frame));
      });
  auto finish = [state](llvm::StringRef name, bool invalidated) {
    if (state->stack.empty() || state->stack.back().pass != name) {
      // A diagnostic mismatch must not alter compiler execution.
      llvm::errs() << "[hexmlir-llvm] unmatched completion " << name << "\n";
      return;
    }
    Frame frame = std::move(state->stack.back());
    state->stack.pop_back();
    double seconds = std::chrono::duration<double>(Clock::now() -
                                                   frame.started).count();
    llvm::errs() << "[hexmlir-llvm] end " << frame.id << " " << frame.pass
                 << " seconds=" << seconds
                 << (invalidated ? " invalidated" : "") << "\n";
    llvm::errs().flush();
  };
  callbacks.registerAfterPassCallback(
      [finish](llvm::StringRef name, llvm::Any,
               const llvm::PreservedAnalyses &) { finish(name, false); });
  callbacks.registerAfterPassInvalidatedCallback(
      [finish](llvm::StringRef name, const llvm::PreservedAnalyses &) {
        finish(name, true);
      });
}
} // namespace mlir::hexagon
#endif
