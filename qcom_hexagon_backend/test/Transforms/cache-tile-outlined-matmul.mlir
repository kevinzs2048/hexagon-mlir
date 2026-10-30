// RUN: linalg-hexagon-opt %s -hexagon-outline-matmul | FileCheck %s --check-prefix=DEFAULT
// RUN: linalg-hexagon-opt %s -hexagon-outline-matmul='cache-tile=true' -hexagon-outline-matmul='cache-tile=true' -canonicalize | FileCheck %s --check-prefix=TILED
// RUN: linalg-hexagon-opt %s -linalg-to-llvm='outline-matmul=true tile-outlined-matmul=true add-fastmath=false enable-vtcm-tiling=false enable-convert-to-hexagonmem=false' | FileCheck %s --check-prefix=LLVM
// RUN: not linalg-hexagon-opt %s -linalg-to-llvm='tile-outlined-matmul=true' 2>&1 | FileCheck %s --check-prefix=ERROR

// DEFAULT-NOT: scf.for
// DEFAULT: linalg.matmul
// DEFAULT-NOT: scf.for
// DEFAULT: linalg.matmul
// DEFAULT-NOT: scf.for
// DEFAULT: linalg.matmul
// DEFAULT-NOT: scf.for
// ERROR: tile-outlined-matmul requires outline-matmul

func.func @integer(%a: tensor<4x32xi8>, %b: tensor<32x256xi8>,
                   %c: tensor<4x256xi32>) -> tensor<4x256xi32> {
  %x = linalg.matmul ins(%a, %b : tensor<4x32xi8>, tensor<32x256xi8>)
                     outs(%c : tensor<4x256xi32>) -> tensor<4x256xi32>
  return %x : tensor<4x256xi32>
}

// Neither floating-point arithmetic nor non-vector-multiple shapes are tiled.
func.func @floating(%a: tensor<4x32xf32>, %b: tensor<32x256xf32>,
                    %c: tensor<4x256xf32>) -> tensor<4x256xf32> {
  %x = linalg.matmul ins(%a, %b : tensor<4x32xf32>, tensor<32x256xf32>)
                     outs(%c : tensor<4x256xf32>) -> tensor<4x256xf32>
  return %x : tensor<4x256xf32>
}

func.func @tail(%a: tensor<4x32xi8>, %b: tensor<32x255xi8>,
                %c: tensor<4x255xi32>) -> tensor<4x255xi32> {
  %x = linalg.matmul ins(%a, %b : tensor<4x32xi8>, tensor<32x255xi8>)
                     outs(%c : tensor<4x255xi32>) -> tensor<4x255xi32>
  return %x : tensor<4x255xi32>
}

// TILED-LABEL: func.func private @__hexagon_matmul(
// TILED: %[[STEP:.*]] = arith.constant 128 : index
// TILED: scf.for {{.*}} step %[[STEP]] iter_args
// TILED-NOT: scf.for
// TILED: tensor.extract_slice
// TILED: linalg.matmul
// TILED: tensor.insert_slice
// TILED: scf.yield
// TILED-NOT: scf.for
// TILED-LABEL: func.func private @__hexagon_matmul_0(
// TILED-NOT: scf.for
// TILED: linalg.matmul
// TILED-LABEL: func.func private @__hexagon_matmul_1(
// TILED-NOT: scf.for
// TILED: linalg.matmul
// TILED-NOT: scf.for
// LLVM: llvm.func internal @__hexagon_matmul(
// LLVM-SAME: passthrough = ["noinline"]
