// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics

func.func @invalid_dimension(%a: tensor<4xi32>, %acc: tensor<i32>) {
  // expected-error @+1 {{scan dimension must be within the input rank}}
  %result:2 = tm_tensor.scan dimension(1) inclusive(true) ins(%a : tensor<4xi32>) outs(%a, %acc : tensor<4xi32>, tensor<i32>) {
    ^bb0(%x: i32, %y: i32):
      tm_tensor.yield %x : i32
  } -> tensor<4xi32>, tensor<i32>
  return
}

// -----

func.func @invalid_accumulator(%a: tensor<2x4xi32>, %acc: tensor<4xi32>) {
  // expected-error @+1 {{accumulator must remove the scan dimension from input}}
  %result:2 = tm_tensor.scan dimension(1) inclusive(true) ins(%a : tensor<2x4xi32>) outs(%a, %acc : tensor<2x4xi32>, tensor<4xi32>) {
    ^bb0(%x: i32, %y: i32):
      tm_tensor.yield %x : i32
  } -> tensor<2x4xi32>, tensor<4xi32>
  return
}

// -----

func.func @invalid_yield(%a: tensor<4xi32>, %acc: tensor<i32>) {
  // expected-error @+1 {{scan region must yield one input-element value}}
  %result:2 = tm_tensor.scan dimension(0) inclusive(true) ins(%a : tensor<4xi32>) outs(%a, %acc : tensor<4xi32>, tensor<i32>) {
    ^bb0(%x: i32, %y: i32):
      tm_tensor.yield %x, %y : i32, i32
  } -> tensor<4xi32>, tensor<i32>
  return
}
