// RUN: linalg-hexagon-opt %s -hexagon-outline-matmul -hexagon-outline-matmul | FileCheck %s
// RUN: linalg-hexagon-opt %s -linalg-to-llvm='outline-matmul=true add-fastmath=false enable-vtcm-tiling=false enable-convert-to-hexagonmem=false' | FileCheck %s --check-prefix=LLVM

// Existing symbols must not be overwritten. A second pass is a no-op.
func.func private @__hexagon_matmul()

// CHECK-LABEL: func.func @reuse(
// CHECK: call @[[HELPER:__hexagon_matmul_[0-9]+]](
// CHECK: call @[[HELPER]](
// CHECK-NOT: linalg.matmul
// CHECK: return
// LLVM-LABEL: llvm.func @reuse(
// LLVM: llvm.call @[[LLVMHELPER:__hexagon_matmul_[0-9]+]](
// LLVM: llvm.call @[[LLVMHELPER]](
func.func @reuse(%a: tensor<4x32xi8>, %b: tensor<32x64xi8>,
                 %c: tensor<4x64xi32>, %d: tensor<32x64xi8>) -> tensor<4x64xi32> {
  %x = linalg.matmul ins(%a, %b : tensor<4x32xi8>, tensor<32x64xi8>)
                     outs(%c : tensor<4x64xi32>) -> tensor<4x64xi32>
  %y = linalg.matmul ins(%a, %d : tensor<4x32xi8>, tensor<32x64xi8>)
                     outs(%x : tensor<4x64xi32>) -> tensor<4x64xi32>
  return %y : tensor<4x64xi32>
}

// Different attributes must not be merged with the preceding helper.
// CHECK-LABEL: func.func @different_attributes(
// CHECK: call @[[OTHER:__hexagon_matmul_[0-9]+]](
func.func @different_attributes(%a: tensor<4x32xi8>, %b: tensor<32x64xi8>,
                                %c: tensor<4x64xi32>) -> tensor<4x64xi32> {
  %x = linalg.matmul {test.marker = "distinct"}
                     ins(%a, %b : tensor<4x32xi8>, tensor<32x64xi8>)
                     outs(%c : tensor<4x64xi32>) -> tensor<4x64xi32>
  return %x : tensor<4x64xi32>
}

// CHECK: func.func private @[[HELPER]](
// CHECK-SAME: passthrough = ["noinline"]
// CHECK: linalg.matmul
// CHECK: func.func private @[[OTHER]](
// CHECK: test.marker = "distinct"
// CHECK-NOT: func.func
// LLVM: llvm.func internal @[[LLVMHELPER]](
// LLVM-SAME: passthrough = ["noinline"]
