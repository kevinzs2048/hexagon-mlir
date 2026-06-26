// RUN: HEXKL_TLE_RAW=1 linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(linalg-to-tle-raw))' | FileCheck %s

#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @compiler_add
func.func @compiler_add(%a: memref<2048xf32>, %b: memref<2048xf32>, %out: memref<2048xf32>) {
  // CHECK: hexkl.tle_raw_call "triton_add_raw_tle"(%{{.*}}, %{{.*}}, %{{.*}})
  // CHECK-NOT: linalg.generic
  linalg.generic {indexing_maps = [#map, #map, #map],
                  iterator_types = ["parallel"]}
                 ins(%a, %b : memref<2048xf32>, memref<2048xf32>)
                 outs(%out : memref<2048xf32>) {
  ^bb0(%ina: f32, %inb: f32, %o: f32):
    %s = arith.addf %ina, %inb : f32
    linalg.yield %s : f32
  }
  return
}
