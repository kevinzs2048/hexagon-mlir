// RUN: linalg-hexagon-opt %s -split-input-file -linalg-to-llvm="enable-multi-threading=true" | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
func.func @no_reuse_direct_elementwise(%x: memref<1024x256xf32>, %y: memref<1024x256xf32>, %z: memref<1024x256xf32>) {
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

// CHECK-LABEL: llvm.func @no_reuse_direct_elementwise
// CHECK-NOT: mlirAsyncRuntime
// CHECK: llvm.return

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#bcast = affine_map<(d0, d1) -> (d0)>
module {
func.func @broadcast_reuse_elementwise(%x: memref<1024xf32>, %y: memref<1024x256xf32>, %z: memref<1024x256xf32>) {
   %t0 = bufferization.to_tensor %x restrict writable : memref<1024xf32> to tensor<1024xf32>
   %t1 = bufferization.to_tensor %y restrict writable : memref<1024x256xf32> to tensor<1024x256xf32>
   %t2 = bufferization.to_tensor %z restrict writable : memref<1024x256xf32> to tensor<1024x256xf32>

   %t3 = linalg.generic {
           indexing_maps = [#bcast, #map, #map],
           iterator_types = ["parallel", "parallel"]}
           ins(%t0, %t1 : tensor<1024xf32>, tensor<1024x256xf32>)
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

// CHECK-LABEL: llvm.func @broadcast_reuse_elementwise
// CHECK: llvm.call @mlirAsyncRuntimeCreateGroup
// CHECK: llvm.call @async_execute_fn
// CHECK: llvm.call @mlirAsyncRuntimeAddTokenToGroup
// CHECK: llvm.call @mlirAsyncRuntimeAwaitAllInGroup
// CHECK: llvm.func @async_execute_fn
// CHECK: llvm.call @mlirAsyncRuntimeCreateToken
// CHECK: llvm.call @mlirAsyncRuntimeExecute
