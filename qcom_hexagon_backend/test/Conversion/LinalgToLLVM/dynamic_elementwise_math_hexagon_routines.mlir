// RUN: linalg-hexagon-opt %s -linalg-to-llvm | FileCheck %s

#map = affine_map<(d0) -> (d0)>

module attributes {llvm.target_triple = "hexagon"} {
  func.func @dynamic_exp(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %n = memref.dim %arg0, %c0 : memref<?xf32>
    %t0 = bufferization.to_tensor %arg0 restrict : memref<?xf32> to tensor<?xf32>
    %e = tensor.empty(%n) : tensor<?xf32>
    %r = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel"]}
        ins(%t0 : tensor<?xf32>)
        outs(%e : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %y = math.exp %in : f32
      linalg.yield %y : f32
    } -> tensor<?xf32>
    bufferization.materialize_in_destination %r in restrict writable %arg1
        : (tensor<?xf32>, memref<?xf32>) -> ()
    return
  }
}

// CHECK-LABEL: llvm.func @dynamic_exp
// CHECK-NOT: llvm.func @malloc
// CHECK-NOT: llvm.func @free
// CHECK-NOT: llvm.call @malloc
// CHECK: llvm.call @_hexagon_runtime_exp__vf
// CHECK-NOT: llvm.intr.exp
// CHECK-LABEL: llvm.func @_hexagon_runtime_exp__vf(vector<32xi32>) -> vector<32xi32>
