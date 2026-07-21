// RUN: linalg-hexagon-opt %s -linalg-to-llvm | FileCheck %s

// This test checks that no-reuse elementwise lowering streams directly from
// input DDR buffers into the destination DDR buffer. It must not stage through
// VTCM or heap-backed temporary output buffers.

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
func.func @kernel(%x: memref<1024x256xf32>, %y: memref<1024x256xf32>, %z: memref<1024x256xf32>) {
   %t0 = bufferization.to_tensor %x restrict writable : memref<1024x256xf32> to tensor<1024x256xf32>
   %t1 = bufferization.to_tensor %y restrict writable : memref<1024x256xf32> to tensor<1024x256xf32>
   %t2 = bufferization.to_tensor %z restrict writable : memref<1024x256xf32> to tensor<1024x256xf32>

   %t3 = linalg.generic {
           indexing_maps = [#map, #map, #map],
           iterator_types = ["parallel", "parallel"]}
           ins(%t0, %t1 : tensor<1024x256xf32>, tensor<1024x256xf32>)
           outs(%t2 : tensor<1024x256xf32>) {
   ^bb0(%in: f32, %in_2: f32, %out: f32):

     %4 = arith.subf %in, %in_2 : f32
     linalg.yield %4 : f32
   } -> tensor<1024x256xf32>

   bufferization.materialize_in_destination %t3 in writable %z
      : (tensor<1024x256xf32>, memref<1024x256xf32>) -> ()
   return
 }
}
// CHECK-LABEL: llvm.func @kernel
// CHECK-NOT: llvm.call @malloc
// CHECK-NOT: llvm.call @free
// CHECK-NOT: llvm.intr.memcpy
// CHECK-NOT: hexagon_runtime_copy
// CHECK: llvm.load {{.*}} : !llvm.ptr -> vector<32xf32>
// CHECK: llvm.load {{.*}} : !llvm.ptr -> vector<32xf32>
// CHECK: llvm.fsub {{.*}} : vector<32xf32>
// CHECK: llvm.store {{.*}} : vector<32xf32>, !llvm.ptr
// CHECK: llvm.return
