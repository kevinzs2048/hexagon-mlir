// RUN: linalg-hexagon-opt %s | linalg-hexagon-opt | FileCheck %s

// CHECK-LABEL: func.func @tle_add
func.func @tle_add(%out: memref<2048xf32>, %a: memref<2048xf32>, %b: memref<2048xf32>) {
  // CHECK: hexkl.tle_raw_call "triton_add_raw_tle"(%{{.*}}, %{{.*}}, %{{.*}}) : memref<2048xf32>, memref<2048xf32>, memref<2048xf32>
  hexkl.tle_raw_call "triton_add_raw_tle"(%out, %a, %b) : memref<2048xf32>, memref<2048xf32>, memref<2048xf32>
  return
}
