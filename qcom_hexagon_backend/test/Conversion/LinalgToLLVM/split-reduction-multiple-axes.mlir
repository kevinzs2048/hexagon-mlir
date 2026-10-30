// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hexagon-tiling{enable-split-reduction=true})' | FileCheck %s
// Splitting only k must preserve accumulation over h. Reading the original
// partial accumulator through ins loses all but the last h contribution.
#input = affine_map<(h, k) -> (h, k)>
#scalar = affine_map<(h, k) -> ()>

func.func @two_axis_sum(%input: tensor<4x64xf32>, %initial: tensor<f32>) -> tensor<f32> {
  %sum = linalg.generic {indexing_maps = [#input, #scalar],
                         iterator_types = ["reduction", "reduction"]}
      ins(%input : tensor<4x64xf32>) outs(%initial : tensor<f32>) {
  ^bb0(%value: f32, %acc: f32):
    %next = arith.addf %value, %acc : f32
    linalg.yield %next : f32
  } -> tensor<f32>
  return %sum : tensor<f32>
}

// CHECK-LABEL: func.func @two_axis_sum
// CHECK: iterator_types = ["reduction", "parallel"]} ins(%{{[^,]*}} : tensor<4x32xf32>) outs(%{{.*}} : tensor<32xf32>)
// CHECK-NEXT: ^bb0(%[[VALUE:.*]]: f32, %[[ACC:.*]]: f32):
// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[VALUE]], %[[ACC]] : f32
// CHECK-NEXT: linalg.yield %[[SUM]] : f32
// CHECK: linalg.reduce

// Multiple outputs currently fall back to ordinary reduction tiling. Keep
// this control distinct: it does not exercise split-reduction parallelization.
func.func @two_axis_sum_and_max(%input: tensor<4x64xf32>,
                               %sum_initial: tensor<f32>,
                               %max_initial: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %sum, %max = linalg.generic {indexing_maps = [#input, #scalar, #scalar],
                              iterator_types = ["reduction", "reduction"]}
      ins(%input : tensor<4x64xf32>) outs(%sum_initial, %max_initial : tensor<f32>, tensor<f32>) {
  ^bb0(%value: f32, %sum_acc: f32, %max_acc: f32):
    %next_sum = arith.addf %value, %sum_acc : f32
    %next_max = arith.maximumf %value, %max_acc : f32
    linalg.yield %next_sum, %next_max : f32, f32
  } -> (tensor<f32>, tensor<f32>)
  return %sum, %max : tensor<f32>, tensor<f32>
}

// CHECK-LABEL: func.func @two_axis_sum_and_max
// CHECK: iterator_types = ["reduction", "reduction"]} ins(%{{[^,]*}} : tensor<1x32xf32>) outs(%{{.*}}, %{{.*}} : tensor<f32>, tensor<f32>)
// CHECK-NEXT: ^bb0(%[[V:.*]]: f32, %[[S:.*]]: f32, %[[M:.*]]: f32):
// CHECK-NEXT: %[[NS:.*]] = arith.addf %[[V]], %[[S]] : f32
// CHECK-NEXT: %[[NM:.*]] = arith.maximumf %[[V]], %[[M]] : f32
// CHECK-NEXT: linalg.yield %[[NS]], %[[NM]] : f32, f32
