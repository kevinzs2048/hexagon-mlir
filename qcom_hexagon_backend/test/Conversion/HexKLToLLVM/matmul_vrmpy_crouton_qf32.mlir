// RUN: env HEXKL_VRMPY_CROUTON=1 linalg-hexagon-opt %s -linalg-to-llvm="enable-hexkl=true enable-inline-hmx=true enable-q8-weight=true" | FileCheck %s

// Keep crouton scaling/accumulation on Hexagon qf32 intrinsics. Generic
// vector<32xf32> arithmetic crashes LLVM-19 Hexagon llc in the widening vector
// float pass for this lowering.
func.func @compiler_vrmpy_crouton(%A: memref<?x?xf16>, %B: memref<?x?xf16>,
                                  %out: memref<?x?xf32>, %Aq8: memref<?x?xi8>,
                                  %Wq8: memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dynM = memref.dim %A, %c0 : memref<?x?xf16>
  %dynN = memref.dim %B, %c1 : memref<?x?xf16>
  %At = bufferization.to_tensor %A restrict : memref<?x?xf16> to tensor<?x?xf16>
  %Bt = bufferization.to_tensor %B restrict : memref<?x?xf16> to tensor<?x?xf16>
  %e  = tensor.empty(%dynM, %dynN) : tensor<?x?xf32>
  %r  = linalg.matmul ins(%At, %Bt : tensor<?x?xf16>, tensor<?x?xf16>)
                      outs(%e : tensor<?x?xf32>) -> tensor<?x?xf32>
  bufferization.materialize_in_destination %r in restrict writable %out
      : (tensor<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK-LABEL: llvm.func @compiler_vrmpy_crouton
  // CHECK: llvm.call @llvm.hexagon.V6.vrmpybv.acc.128B
  // CHECK: llvm.call @llvm.hexagon.V6.vconv.sf.w.128B
  // CHECK: llvm.call @llvm.hexagon.V6.vmpy.qf32.hf.128B
  // CHECK: llvm.call @llvm.hexagon.V6.vmpy.qf32.sf.128B
  // CHECK: llvm.call @llvm.hexagon.V6.vadd.qf32.mix.128B
  // CHECK-NOT: llvm.sitofp {{.*}} : vector<32xi32> to vector<32xf32>
  // CHECK-NOT: llvm.fpext {{.*}} : vector<32xf16> to vector<32xf32>
  // CHECK-NOT: llvm.fmul {{.*}} : vector<32xf32>
  // CHECK-NOT: llvm.fadd {{.*}} : vector<32xf32>
  return
}
