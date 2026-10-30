// RUN: linalg-hexagon-opt %s --linalg-to-llvm='enable-vtcm-tiling=false enable-convert-to-hexagonmem=false' | FileCheck %s
// Inactive vector lanes can retain UB poison after vector/shape lowering.
// Lower it to LLVM poison, not a zero constant, before LLVM IR translation.
// CHECK-NOT: ub.poison
// CHECK: llvm.func @retained_poison
// CHECK: llvm.mlir.poison : i32
// CHECK: llvm.return
func.func @retained_poison() -> i32 {
  %p = ub.poison : i32
  return %p : i32
}
