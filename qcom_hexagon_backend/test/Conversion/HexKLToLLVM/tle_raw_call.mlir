// RUN: HEXKL_TLE_RAW=1 linalg-hexagon-opt %s -hexkl-to-llvm | FileCheck %s

// CHECK-LABEL: func.func @tle_add
// CHECK: llvm.call @triton_add_raw_tle(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
func.func @tle_add(%out: memref<2048xf32>, %a: memref<2048xf32>, %b: memref<2048xf32>) {
  hexkl.tle_raw_call "triton_add_raw_tle"(%out, %a, %b) : memref<2048xf32>, memref<2048xf32>, memref<2048xf32>
  return
}
