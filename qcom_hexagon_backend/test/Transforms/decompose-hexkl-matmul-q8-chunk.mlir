// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(decompose-hexkl-matmul{use-inline-hmx enable-q8-weight}))' | FileCheck %s

// Q8 inline-HMX N-chunking must cap a single chunk to 8 output-column tiles.
// Wider chunks have shown v73 failures across dynamic Q8 shapes: 32-tile chunks
// corrupt the second half of N=2048, and 16-tile chunks can hang for M>=64 with
// large N. The Q8 path also splits HMX core-dot over at most two M tiles so
// M128+large-N stress shapes do not run one long 4-row-tile HMX sequence.

func.func @test_q8_chunk_caps_to_32(%arg0: memref<?x?xf16>,
                                    %arg1: memref<?x?xf16>,
                                    %output: memref<?x?xf32>,
                                    %wq8: memref<?x?xi8>) {
    hexkl.matmul ins(%arg0, %arg1 : memref<?x?xf16>, memref<?x?xf16>)
                 outs(%output : memref<?x?xf32>)
    return
}

// CHECK-LABEL: func.func @test_q8_chunk_caps_to_32
// CHECK:       %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[AVAIL:.*]] = arith.maxsi
// CHECK:       %[[CHUNK_RAW:.*]] = arith.divui %[[AVAIL]],
// CHECK:       %[[CHUNK_CAP:.*]] = arith.minui %[[CHUNK_RAW]], %[[C8]] : index
// CHECK:       %[[CHUNK:.*]] = arith.minui %[[CHUNK_CAP]],
// CHECK:       %[[MCHUNK:.*]] = arith.minui {{.*}}%[[C2]]
// CHECK:       scf.for {{.*}} step %[[CHUNK]]
// CHECK:       scf.for {{.*}} step %[[MCHUNK]]
