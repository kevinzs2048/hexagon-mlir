// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexagon-lower-tm-tensor))' | FileCheck %s
// RUN: linalg-hexagon-opt %s -linalg-to-llvm='add-fastmath=false enable-vtcm-tiling=false enable-convert-to-hexagonmem=false' | FileCheck %s --check-prefix=LLVM

// CHECK-LABEL: func.func @inclusive(
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.if
// CHECK: arith.addi
// CHECK: tensor.insert
// CHECK-NOT: tm_tensor
// CHECK: return
// LLVM-LABEL: llvm.func @inclusive(
// LLVM-NOT: tm_tensor
// LLVM: llvm.return
func.func @inclusive(%input: tensor<2x4xi64>, %out: tensor<2x4xi64>, %acc: tensor<2xi64>) -> (tensor<2x4xi64>, tensor<2xi64>) {
  %result:2 = tm_tensor.scan dimension(1) inclusive(true)
      ins(%input : tensor<2x4xi64>) outs(%out, %acc : tensor<2x4xi64>, tensor<2xi64>) {
    ^bb0(%previous: i64, %current: i64):
      %sum = arith.addi %previous, %current : i64
      tm_tensor.yield %sum : i64
  } -> tensor<2x4xi64>, tensor<2xi64>
  return %result#0, %result#1 : tensor<2x4xi64>, tensor<2xi64>
}

// CHECK-LABEL: func.func @exclusive_subtract(
// CHECK: scf.for
// CHECK: scf.if
// CHECK: arith.subi
// CHECK-NOT: tm_tensor
// CHECK: return
// LLVM-LABEL: llvm.func @exclusive_subtract(
// LLVM-NOT: tm_tensor
// LLVM: llvm.return
func.func @exclusive_subtract(%input: tensor<4xi32>, %out: tensor<4xi32>, %acc: tensor<i32>) -> (tensor<4xi32>, tensor<i32>) {
  %result:2 = tm_tensor.scan dimension(0) inclusive(false)
      ins(%input : tensor<4xi32>) outs(%out, %acc : tensor<4xi32>, tensor<i32>) {
    ^bb0(%previous: i32, %current: i32):
      %difference = arith.subi %previous, %current : i32
      tm_tensor.yield %difference : i32
  } -> tensor<4xi32>, tensor<i32>
  return %result#0, %result#1 : tensor<4xi32>, tensor<i32>
}

// CHECK-LABEL: func.func @dynamic_axis_zero(
// CHECK: tensor.dim
// CHECK: scf.for
// CHECK: scf.for
// CHECK-NOT: tm_tensor
// CHECK: return
// LLVM-LABEL: llvm.func @dynamic_axis_zero(
// LLVM-NOT: tm_tensor
// LLVM: llvm.return
func.func @dynamic_axis_zero(%input: tensor<?x2xf32>, %out: tensor<?x2xf32>, %acc: tensor<2xf32>) -> (tensor<?x2xf32>, tensor<2xf32>) {
  %result:2 = tm_tensor.scan dimension(0) inclusive(true)
      ins(%input : tensor<?x2xf32>) outs(%out, %acc : tensor<?x2xf32>, tensor<2xf32>) {
    ^bb0(%previous: f32, %current: f32):
      %sum = arith.addf %previous, %current : f32
      tm_tensor.yield %sum : f32
  } -> tensor<?x2xf32>, tensor<2xf32>
  return %result#0, %result#1 : tensor<?x2xf32>, tensor<2xf32>
}
