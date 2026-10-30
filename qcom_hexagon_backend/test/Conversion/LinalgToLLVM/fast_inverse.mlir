// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hexagon-fast-inverse)' | FileCheck %s

// Do not use rsqrt(y*y) for arbitrary division: it loses sign and range.
// CHECK-LABEL: func.func @test_scalar_division_f32(
// CHECK-NOT: 1597463007
// CHECK: arith.divf
// CHECK: return
func.func @test_scalar_division_f32(%arg0: vector<128xf32>, %arg1: memref<f32>, %arg2: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %scalar = memref.load %arg1[] : memref<f32>
  // Broadcast the scalar to a vector
  %denominator = vector.broadcast %scalar : f32 to vector<128xf32>
  %div = arith.divf %arg0, %denominator fastmath<fast> : vector<128xf32>
  vector.transfer_write %div, %arg2[%c0] : vector<128xf32>, memref<128xf32>
  return
}

// Do not use rsqrt(y*y) for arbitrary division: it loses sign and range.
// CHECK-LABEL: func.func @test_scalar_division_f16(
// CHECK-NOT: 1597463007
// CHECK: arith.divf
// CHECK: return
func.func @test_scalar_division_f16(%arg0: vector<128xf16>, %arg1: memref<f16>, %arg2: memref<128xf16>) {
  %c0 = arith.constant 0 : index
  %scalar = memref.load %arg1[] : memref<f16>
  // Broadcast the scalar to a vector
  %denominator = vector.broadcast %scalar : f16 to vector<128xf16>
  %div = arith.divf %arg0, %denominator : vector<128xf16>
  vector.transfer_write %div, %arg2[%c0] : vector<128xf16>, memref<128xf16>
  return
}

// Do not use rsqrt(y*y) for arbitrary division: it loses sign and range.
// CHECK-LABEL: func.func @test_fast_inverse_division_f32(
// CHECK-NOT: 1597463007
// CHECK: arith.divf
// CHECK: return
func.func @test_fast_inverse_division_f32(%arg0: vector<256xf32>, %arg1: vector<256xf32>, %arg2: memref<256xf32>) {
  %c0 = arith.constant 0 : index
  %div = arith.divf %arg0, %arg1 fastmath<fast> : vector<256xf32>
  vector.transfer_write %div, %arg2[%c0] : vector<256xf32>, memref<256xf32>
  return
}

// This test ensures that the fp32 division of a constant 1.0f by sqrt(a)
// is transformed directly into the code that implements
// Quake III's Fast Inverse Square Root algorithm
// CHECK-LABEL: func.func @test_one_div_sqrt_f32(
// CHECK-SAME:  %[[A:.*]]: f32)
// For computing the first approximation
// CHECK: %[[I0:.*]] = arith.bitcast %[[A]] : f32 to i32
// CHECK: %[[I1:.*]] = arith.shrui %[[I0]], %c1_i32 : i32
// CHECK: %[[I2:.*]] = arith.subi %c1597463007_i32, %[[I1]] : i32
// CHECK: %[[Y0:.*]] = arith.bitcast %[[I2]] : i32 to f32
// For Newton's first iteration
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: arith.mulf
// For Newton's second iteration
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: arith.mulf
// Important: ensuring that there is no divf produced
// CHECK-NOT: arith.divf
// Important: ensuring that there is no llvm.intr.sqrt produced
// CHECK-NOT: llvm.intr.sqrt
func.func @test_one_div_sqrt_f32(%arg0: f32) -> f32 {
  %cst = arith.constant 1.000000e+00 : f32
  %1 = llvm.intr.sqrt(%arg0) {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
  %2 = arith.divf %cst, %1 fastmath<fast> : f32
  return %2 : f32
}


// This test ensures that the fp32 division of a variable x by sqrt(a)
// is transformed into x multiplied by the code that implements
// Quake III's Fast Inverse Square Root algorithm
// CHECK-LABEL: func.func @test_var_div_sqrt_f32(
// CHECK-SAME:  %[[X:.*]]: f32,
// CHECK-SAME:  %[[A:.*]]: f32)
// For computing the first approximation of rsqrt(a)
// CHECK: %[[I0:.*]] = arith.bitcast %[[A]] : f32 to i32
// CHECK: %[[I1:.*]] = arith.shrui %[[I0]], %c1_i32 : i32
// CHECK: %[[I2:.*]] = arith.subi %c1597463007_i32, %[[I1]] : i32
// CHECK: %[[Y0:.*]] = arith.bitcast %[[I2]] : i32 to f32
// For Newton's first iteration
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: arith.mulf
// For Newton's second iteration
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: arith.mulf
// Finally, x is multiplied by rsqrt(a)
// CHECK: arith.mulf %[[X]],
// Important: ensuring that there is no divf produced
// CHECK-NOT: arith.divf
// Important: ensuring that there is no llvm.intr.sqrt produced
// CHECK-NOT: llvm.intr.sqrt
func.func @test_var_div_sqrt_f32(%arg0: f32, %arg1: f32) -> f32 {
  // %arg0 = x
  // %arg1 = a
  %1 = llvm.intr.sqrt(%arg1) {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
  %2 = arith.divf %arg0, %1 fastmath<fast> : f32
  return %2 : f32
}

// CHECK-LABEL: func.func @strict_inverse_sqrt(
// CHECK-NOT: 1597463007
// CHECK: math.sqrt
// CHECK: arith.divf
// CHECK: return
func.func @strict_inverse_sqrt(%x: f32) -> f32 {
  %one = arith.constant 1.0 : f32
  %root = math.sqrt %x : f32
  %result = arith.divf %one, %root : f32
  return %result : f32
}
