// RUN: linalg-hexagon-opt %s -linalg-to-llvm='add-fastmath=false enable-vtcm-tiling=false enable-convert-to-hexagonmem=false enable-hexagon-routines=false' | FileCheck %s --check-prefix=STRICT
// RUN: linalg-hexagon-opt %s -linalg-to-llvm='add-fastmath=true enable-vtcm-tiling=false enable-convert-to-hexagonmem=false enable-hexagon-routines=false' | FileCheck %s --check-prefix=FAST

// Disabling fast math must preserve division/sqrt operations, not merely omit
// attributes after replacing them with an approximate inverse.
// STRICT-LABEL: llvm.func @divide(
// STRICT-NOT: 1597463007
// STRICT: llvm.fdiv
// STRICT: llvm.return
// FAST-LABEL: llvm.func @divide(
// FAST-NOT: 1597463007
// FAST: llvm.fdiv
// FAST: llvm.return
func.func @divide(%x: f32, %y: f32) -> f32 {
  %result = arith.divf %x, %y : f32
  return %result : f32
}

// STRICT-LABEL: llvm.func @inverse_sqrt(
// STRICT-NOT: 1597463007
// STRICT: llvm.intr.sqrt
// STRICT: llvm.fdiv
// STRICT: llvm.return
// FAST-LABEL: llvm.func @inverse_sqrt(
// FAST: 1597463007
// FAST-NOT: llvm.fdiv
// FAST: llvm.return
func.func @inverse_sqrt(%x: f32) -> f32 {
  %one = arith.constant 1.0 : f32
  %root = math.sqrt %x : f32
  %result = arith.divf %one, %root : f32
  return %result : f32
}
