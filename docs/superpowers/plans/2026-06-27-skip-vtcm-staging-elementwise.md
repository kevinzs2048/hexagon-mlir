# Skip VTCM Staging For Elementwise Ops Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `vtcm-tiling` leave conservative no-reuse elementwise `linalg.generic` ops on DDR instead of staging them through VTCM.

**Architecture:** Add a small predicate in VTCM tiling option selection that recognizes pure parallel elementwise `linalg.generic` operations. When no explicit tile sizes are provided, return `failure()` for those ops so the existing pass walk skips VTCM staging and the rest of the lowering pipeline handles them directly.

**Tech Stack:** C++ MLIR pass code, `linalg-hexagon-opt`, MLIR FileCheck tests, Ninja build.

---

## File Structure

- Modify: `qcom_hexagon_backend/lib/Conversion/LinalgToLLVM/VTCMTilingOptions.cpp`
  - Owns VTCM tile-size and tiling-option decisions.
  - Add `isNoReuseElementwiseGeneric` helper in the anonymous namespace.
  - Gate `getVTCMTilingOptions` before tile-size calculation when `userProvidedTileSizes` is empty.
- Modify: `qcom_hexagon_backend/test/Conversion/LinalgToLLVM/vtcm_tiling_scf_for.mlir`
  - Convert the existing add test to expect no VTCM staging.
  - Add a reduction test that still expects VTCM staging.

## Task 1: Add RED IR Tests

**Files:**
- Modify: `qcom_hexagon_backend/test/Conversion/LinalgToLLVM/vtcm_tiling_scf_for.mlir`

- [ ] **Step 1: Replace the test file with add skip and reduction keep cases**

Use this full file content:

```mlir
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(vtcm-tiling),canonicalize,one-shot-bufferize{bufferize-function-boundaries allow-return-allocs-from-loops})' | FileCheck %s

func.func @add_kernel() -> tensor<32xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c2_i32 = arith.constant 2 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<32xf32>
  %fill = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  %result = scf.for %I = %c0_i32 to %c2_i32 step %c1_i32
                          iter_args(%cF = %fill) -> (tensor<32xf32>)  : i32 {
    %genericRes = linalg.generic
             {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
             iterator_types = ["parallel"]} ins(%cF : tensor<32xf32>) outs(%0: tensor<32xf32>) {
    ^bb0(%in_2: f32, %out_3: f32):
      %5 = arith.addf %in_2, %cst : f32
      linalg.yield %5 : f32
    } -> tensor<32xf32>
    scf.yield %genericRes : tensor<32xf32>
  }
  return %result : tensor<32xf32>
}

// CHECK-LABEL: func.func @add_kernel
// CHECK-NOT: memref<32xf32, 1>
// CHECK-NOT: memref.copy {{.*}} : memref<32xf32>{{.*}} to memref<32xf32, 1>
// CHECK-NOT: memref.copy {{.*}} : memref<32xf32, 1> to memref<32xf32>
// CHECK: linalg.generic

// -----

func.func @row_sum_kernel() -> tensor<4xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %input = tensor.empty() : tensor<4x8xf32>
  %out = tensor.empty() : tensor<4xf32>
  %fill_input = linalg.fill ins(%cst : f32) outs(%input : tensor<4x8xf32>) -> tensor<4x8xf32>
  %fill_out = linalg.fill ins(%cst : f32) outs(%out : tensor<4xf32>) -> tensor<4xf32>
  %result = linalg.generic
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
       iterator_types = ["parallel", "reduction"]}
      ins(%fill_input : tensor<4x8xf32>)
      outs(%fill_out : tensor<4xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
  } -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// CHECK-LABEL: func.func @row_sum_kernel
// CHECK: memref.alloc() {{.*}} : memref<4x8xf32, 1>
// CHECK: memref.copy {{.*}} : memref<4x8xf32> to memref<4x8xf32, 1>
```

- [ ] **Step 2: Run the test and verify RED**

Set up the environment and use either the PATH tool or the build-tree binary:

```bash
source /home/kevin/hexagon-codex/env.sh
export LLVM_SYSPATH=/home/kevin/hexagon/LLVM_DIR/llvm-project/build
OPT=${OPT:-$(command -v linalg-hexagon-opt || true)}
if [ -z "$OPT" ]; then
  OPT=/home/kevin/hexagon-codex/hexagon-mlir/triton/build/cmake.linux-x86_64-cpython-3.11/third_party/qcom_hexagon_backend/bin/linalg-hexagon-opt
fi
FILECHECK=${FILECHECK:-$(command -v FileCheck || true)}
if [ -z "$FILECHECK" ]; then
  FILECHECK=/home/kevin/hexagon/LLVM_DIR/llvm-project/build/bin/FileCheck
fi
"$OPT" qcom_hexagon_backend/test/Conversion/LinalgToLLVM/vtcm_tiling_scf_for.mlir \
  -split-input-file \
  -pass-pipeline='builtin.module(func.func(vtcm-tiling),canonicalize,one-shot-bufferize{bufferize-function-boundaries allow-return-allocs-from-loops})' \
  | "$FILECHECK" qcom_hexagon_backend/test/Conversion/LinalgToLLVM/vtcm_tiling_scf_for.mlir
```

Expected result before implementation: FAIL because `@add_kernel` still contains `memref<32xf32, 1>` or DDR/VTCM `memref.copy`.

## Task 2: Implement Conservative Elementwise Skip

**Files:**
- Modify: `qcom_hexagon_backend/lib/Conversion/LinalgToLLVM/VTCMTilingOptions.cpp`

- [ ] **Step 1: Add helper in the anonymous namespace**

Insert this helper after `calculateTileSizeLowerBounds` and before the anonymous namespace closes:

```cpp
bool isNoReuseElementwiseGeneric(linalg::LinalgOp op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op.getOperation());
  if (!genericOp)
    return false;

  if (!genericOp.hasPureTensorSemantics() || !genericOp->getNumResults() ||
      genericOp.getNumLoops() < 1)
    return false;

  for (auto iterType : genericOp.getIteratorTypesArray())
    if (iterType != utils::IteratorType::parallel)
      return false;

  return llvm::all_of(genericOp.getIndexingMapsArray(), [](AffineMap map) {
    return map.isProjectedPermutation();
  });
}
```

- [ ] **Step 2: Gate `getVTCMTilingOptions` before tile-size calculation**

Change the beginning of `getVTCMTilingOptions` to:

```cpp
FailureOr<linalg::LinalgTilingOptions>
getVTCMTilingOptions(linalg::LinalgOp op,
                     std::optional<SmallVector<int64_t>> userProvidedTileSizes,
                     SmallVector<bool> &prefetch) {

  DBG("getting vtcm tiling options for: " << op);
  if (!op.hasPureTensorSemantics() || !op->getNumResults() ||
      op.getNumLoops() < 1) {
    DBG("-> get vtcm tiling options aborted. Constraints not "
        "satisfied.");
    return failure();
  }

  if (!userProvidedTileSizes && isNoReuseElementwiseGeneric(op)) {
    DBG("-> get vtcm tiling options skipped for no-reuse elementwise generic.");
    return failure();
  }

  linalg::LinalgTilingOptions options;
  options.setInterchange(getInterchangeVector(op));
  auto tileSizes = getTileSizes(op, userProvidedTileSizes);
  if (!tileSizes)
    return failure();
  setFullTensors(op, *tileSizes, prefetch);
  options.setTileSizes(*tileSizes);
  return options;
}
```

- [ ] **Step 3: Run the focused test and verify GREEN**

```bash
source /home/kevin/hexagon-codex/env.sh
export LLVM_SYSPATH=/home/kevin/hexagon/LLVM_DIR/llvm-project/build
OPT=${OPT:-$(command -v linalg-hexagon-opt || true)}
if [ -z "$OPT" ]; then
  OPT=/home/kevin/hexagon-codex/hexagon-mlir/triton/build/cmake.linux-x86_64-cpython-3.11/third_party/qcom_hexagon_backend/bin/linalg-hexagon-opt
fi
FILECHECK=${FILECHECK:-$(command -v FileCheck || true)}
if [ -z "$FILECHECK" ]; then
  FILECHECK=/home/kevin/hexagon/LLVM_DIR/llvm-project/build/bin/FileCheck
fi
"$OPT" qcom_hexagon_backend/test/Conversion/LinalgToLLVM/vtcm_tiling_scf_for.mlir \
  -split-input-file \
  -pass-pipeline='builtin.module(func.func(vtcm-tiling),canonicalize,one-shot-bufferize{bufferize-function-boundaries allow-return-allocs-from-loops})' \
  | "$FILECHECK" qcom_hexagon_backend/test/Conversion/LinalgToLLVM/vtcm_tiling_scf_for.mlir
```

Expected result after implementation: PASS.

## Task 3: Build And Broader Verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Build `linalg-hexagon-opt`**

```bash
source /home/kevin/hexagon-codex/env.sh
export LLVM_SYSPATH=/home/kevin/hexagon/LLVM_DIR/llvm-project/build
cd /home/kevin/hexagon-codex/hexagon-mlir/triton/build/cmake.linux-x86_64-cpython-3.11
ninja linalg-hexagon-opt
```

Expected result: ninja exits 0 and rebuilds the qcom backend objects if needed.

- [ ] **Step 2: Re-run focused FileCheck test with build-tree binary**

```bash
cd /home/kevin/hexagon-codex/hexagon-mlir
OPT=/home/kevin/hexagon-codex/hexagon-mlir/triton/build/cmake.linux-x86_64-cpython-3.11/third_party/qcom_hexagon_backend/bin/linalg-hexagon-opt
FILECHECK=/home/kevin/hexagon/LLVM_DIR/llvm-project/build/bin/FileCheck
"$OPT" qcom_hexagon_backend/test/Conversion/LinalgToLLVM/vtcm_tiling_scf_for.mlir \
  -split-input-file \
  -pass-pipeline='builtin.module(func.func(vtcm-tiling),canonicalize,one-shot-bufferize{bufferize-function-boundaries allow-return-allocs-from-loops})' \
  | "$FILECHECK" qcom_hexagon_backend/test/Conversion/LinalgToLLVM/vtcm_tiling_scf_for.mlir
```

Expected result: PASS.

- [ ] **Step 3: Build and relink Triton**

```bash
source /home/kevin/hexagon-codex/env.sh
export LLVM_SYSPATH=/home/kevin/hexagon/LLVM_DIR/llvm-project/build
cd /home/kevin/hexagon-codex/hexagon-mlir/triton/build/cmake.linux-x86_64-cpython-3.11
ninja libtriton.so
bash /home/kevin/hexagon-codex/hmx-spike-compiler/relink_triton.sh
```

Expected result: both commands exit 0.

- [ ] **Step 4: Inspect git diff**

```bash
cd /home/kevin/hexagon-codex/hexagon-mlir
git diff -- qcom_hexagon_backend/lib/Conversion/LinalgToLLVM/VTCMTilingOptions.cpp \
  qcom_hexagon_backend/test/Conversion/LinalgToLLVM/vtcm_tiling_scf_for.mlir
```

Expected result: diff contains only the helper/gate and the focused tests.

- [ ] **Step 5: Commit implementation**

```bash
cd /home/kevin/hexagon-codex/hexagon-mlir
git add qcom_hexagon_backend/lib/Conversion/LinalgToLLVM/VTCMTilingOptions.cpp \
  qcom_hexagon_backend/test/Conversion/LinalgToLLVM/vtcm_tiling_scf_for.mlir \
  docs/superpowers/plans/2026-06-27-skip-vtcm-staging-elementwise.md
git commit -m "hexagon: skip vtcm staging for elementwise generics"
```

Expected result: commit succeeds. Existing unrelated untracked files remain untracked.
