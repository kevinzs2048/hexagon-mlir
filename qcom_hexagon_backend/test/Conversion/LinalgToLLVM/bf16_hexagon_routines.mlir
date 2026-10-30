// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(llvm-intr-to-hexagon-routines)' | FileCheck %s

// The optimized Hexagon math runtime currently has f16 and f32 entry points,
// but no bf16 entry point.  Unsupported, legal types must remain untouched for
// generic lowering instead of aborting the compiler while constructing a
// runtime symbol name.

// CHECK-LABEL: llvm.func @bf16_is_not_rewritten
// CHECK: llvm.intr.exp(%arg0) : (vector<64xbf16>) -> vector<64xbf16>
// CHECK-NOT: llvm.call @_hexagon_runtime_exp
llvm.func @bf16_is_not_rewritten(%arg0: vector<64xbf16>) -> vector<64xbf16> {
  %0 = llvm.intr.exp(%arg0) : (vector<64xbf16>) -> vector<64xbf16>
  llvm.return %0 : vector<64xbf16>
}
