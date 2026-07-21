// RUN: linalg-hexagon-opt %s -convert-func-to-llvm -hexkl-to-llvm | FileCheck %s

func.func @weight_q8_range(%vtcm: memref<?xi8, 1>, %w: memref<?x?xi8>,
                           %wt_off: i32, %k: i32, %n: i32,
                           %tile_start: i32, %tile_count: i32) {
  hexkl.hvx_layout_weight_q8(%vtcm, %wt_off, %w, %k, %n, range %tile_start, %tile_count)
      : memref<?xi8, 1>, i32, memref<?x?xi8>, i32, i32
  return
}

// CHECK-LABEL: llvm.func @weight_q8_range
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[IS_ZERO:.*]] = llvm.icmp "eq" %[[COUNT:arg[0-9]+]], %[[ZERO]]
// CHECK: %[[END_COUNT:.*]] = llvm.add %[[START:arg[0-9]+]], %[[COUNT]]
// CHECK: %[[END:.*]] = llvm.select %[[IS_ZERO]], {{.*}}, %[[END_COUNT]]
// CHECK: llvm.br ^{{.*}}(%[[START]] : i32)
// CHECK: llvm.icmp "slt" {{.*}}, %[[END]]
