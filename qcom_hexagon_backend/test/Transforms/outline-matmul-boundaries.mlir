// RUN: linalg-hexagon-opt %s -hexagon-outline-matmul --mlir-print-op-generic | FileCheck %s

#a = affine_map<(m, n, k) -> (m, k)>
#b = affine_map<(m, n, k) -> (k, n)>
#c = affine_map<(m, n, k) -> (m, n)>

// Complete region equivalence must include scalar arithmetic attributes, not
// just tensor shapes or the names of the arithmetic operations.
// CHECK: sym_name = "different_bodies"
// CHECK: "func.call"
// CHECK: "func.call"
// CHECK: "func.return"
func.func @different_bodies(%a: tensor<2x2xi32>, %b: tensor<2x2xi32>,
                            %c: tensor<2x2xi32>)
    -> (tensor<2x2xi32>, tensor<2x2xi32>) {
  %x = "linalg.matmul"(%a, %b, %c)
      <{indexing_maps = [#a, #b, #c], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%lhs: i32, %rhs: i32, %acc: i32):
      %product = arith.muli %lhs, %rhs : i32
      %sum = arith.addi %acc, %product : i32
      linalg.yield %sum : i32
  }) : (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  %y = "linalg.matmul"(%a, %b, %c)
      <{indexing_maps = [#a, #b, #c], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%lhs: i32, %rhs: i32, %acc: i32):
      %product = arith.muli %lhs, %rhs overflow<nsw> : i32
      %sum = arith.addi %acc, %product : i32
      linalg.yield %sum : i32
  }) : (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %x, %y : tensor<2x2xi32>, tensor<2x2xi32>
}

// Buffer-semantics operations remain untouched.
// CHECK: sym_name = "buffers"
// CHECK-NOT: "func.call"
// CHECK: "linalg.matmul"
// CHECK: "func.return"
func.func @buffers(%a: memref<2x2xi32>, %b: memref<2x2xi32>, %c: memref<2x2xi32>) {
  linalg.matmul ins(%a, %b : memref<2x2xi32>, memref<2x2xi32>) outs(%c : memref<2x2xi32>)
  return
}

// CHECK: sym_name = "__hexagon_matmul"
// CHECK: "arith.muli"
// CHECK: "func.return"
// CHECK: sym_name = "__hexagon_matmul_0"
// CHECK: "arith.muli"
// CHECK: "func.return"
// CHECK-NOT: sym_name
