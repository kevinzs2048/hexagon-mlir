# Skip VTCM Staging For No-Reuse Elementwise Ops

Date: 2026-06-27

## Context

The `vtcm-tiling` pass currently stages every eligible tensor `linalg.generic`
through VTCM. For pure elementwise kernels such as add, this creates
`alloc_tensor copy(...) {memory_space = 1}` at tensor level and later lowers to
DDR-to-VTCM and VTCM-to-DDR `memref.copy` operations. These copies have no data
reuse benefit and add fixed overhead to small generated kernels.

The first P0-b step is to make the VTCM tiling decision op-aware for the most
conservative no-reuse case.

## Design

Add a conservative helper near the VTCM tiling option selection that identifies
plain elementwise `linalg.generic` operations and declines VTCM tiling for them.
The pass will return `failure()` from `getVTCMTilingOptions`, causing the
existing walk to leave that operation untouched and allowing the later lowering
pipeline to stream directly from DDR.

An op is eligible to skip VTCM staging only when all of these are true:

- It is a `linalg.generic` with pure tensor semantics, results, and at least one
  loop.
- All iterator types are `parallel`.
- All indexing maps are projected permutations, so there is no reduction,
  windowed/stencil access, or repeated access implied by the map.
- Tile sizes were not explicitly provided by the user. Explicit tile sizes keep
  the current behavior so debugging and forced tiling remain possible.

Non-generic linalg ops, reductions, contractions, convolutions, and non-parallel
generics keep the current VTCM staging path.

## Components

- `lib/Conversion/LinalgToLLVM/VTCMTilingOptions.cpp`
  - Add the no-reuse elementwise predicate.
  - Gate `getVTCMTilingOptions` before tile-size calculation when no explicit
    tile sizes were provided.
- `test/Conversion/LinalgToLLVM/vtcm_tiling_scf_for.mlir`
  - Update the existing add regression to expect no VTCM memory-space-1 allocs
    or DDR/VTCM copies.
  - Add or keep a negative reduction-style case proving VTCM staging still
    happens for reuse/reduction patterns.

## Testing

Use IR-level tests first:

```bash
linalg-hexagon-opt qcom_hexagon_backend/test/Conversion/LinalgToLLVM/vtcm_tiling_scf_for.mlir \
  -pass-pipeline='builtin.module(func.func(vtcm-tiling),canonicalize,one-shot-bufferize{bufferize-function-boundaries allow-return-allocs-from-loops})' \
  | FileCheck qcom_hexagon_backend/test/Conversion/LinalgToLLVM/vtcm_tiling_scf_for.mlir
```

Then rebuild the opt tool and libtriton before on-board testing:

```bash
source /home/kevin/hexagon-codex/env.sh
export LLVM_SYSPATH=/home/kevin/hexagon/LLVM_DIR/llvm-project/build
cd /home/kevin/hexagon-codex/hexagon-mlir/triton/build/cmake.linux-x86_64-cpython-3.11
ninja linalg-hexagon-opt
ninja libtriton.so && bash /home/kevin/hexagon-codex/hmx-spike-compiler/relink_triton.sh
```

## Acceptance

- The elementwise add regression no longer produces VTCM `memref<..., 1>`
  staging or DDR/VTCM `memref.copy` operations.
- A reduction/non-elementwise regression still stages through VTCM.
- The build succeeds for `linalg-hexagon-opt` and `libtriton.so`.
- On-board A/B testing can then measure add with identical deploy inputs except
  for this compiler change.
