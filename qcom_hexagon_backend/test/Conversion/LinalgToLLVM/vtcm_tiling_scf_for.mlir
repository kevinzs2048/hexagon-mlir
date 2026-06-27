// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(vtcm-tiling),canonicalize,one-shot-bufferize{bufferize-function-boundaries allow-return-allocs-from-loops})' | FileCheck %s

func.func @add_kernel() -> tensor<32xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c2_i32 = arith.constant 2 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<32xf32>
  %fill = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  %result = scf.for %I = %c0_i32 to %c2_i32 step %c1_i32
                          iter_args(%cF = %fill) -> (tensor<32xf32>)  : i32 {
    %genericRes = linalg.generic
             {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
             iterator_types = ["parallel"]} ins(%cF : tensor<32xf32>) outs(%0: tensor<32xf32>) {
    ^bb0(%in_2: f32, %out_3: f32):
      %5 = arith.addf %in_2, %cst : f32
      linalg.yield %5 : f32
    } -> tensor<32xf32>
    scf.yield %genericRes : tensor<32xf32>
  }
  return %result : tensor<32xf32>
}

// CHECK-LABEL: func.func @add_kernel
// CHECK-NOT: memref<32xf32, 1>
// CHECK-NOT: memref.copy {{.*}} : memref<32xf32>{{.*}} to memref<32xf32, 1>
// CHECK-NOT: memref.copy {{.*}} : memref<32xf32, 1> to memref<32xf32>
// CHECK: linalg.generic

// -----

func.func @row_sum_kernel() -> tensor<4xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %input = tensor.empty() : tensor<4x8xf32>
  %out = tensor.empty() : tensor<4xf32>
  %fill_input = linalg.fill ins(%cst : f32) outs(%input : tensor<4x8xf32>) -> tensor<4x8xf32>
  %fill_out = linalg.fill ins(%cst : f32) outs(%out : tensor<4xf32>) -> tensor<4xf32>
  %result = linalg.generic
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
       iterator_types = ["parallel", "reduction"]}
      ins(%fill_input : tensor<4x8xf32>)
      outs(%fill_out : tensor<4xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
  } -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// CHECK-LABEL: func.func @row_sum_kernel
// CHECK: memref.alloc() {{.*}} : memref<4x8xf32, 1>
// CHECK: memref.copy {{.*}} : memref<4x8xf32> to memref<4x8xf32, 1>
