# Backend support used by the Qwen application

This compiler fork retains general backend changes based on Qualcomm commit
`197407393c07319b8755d92cfa3d3d3ad22b6ef3`:

- Constant partition accounting and resource ownership fixes.
- Strict FP32 sqrt/div semantics and floating-point code generation controls.
- Captured runtime tensor ABI and returned-buffer ownership.
- Tensor scatter/scan lowering for stateful transformer graphs.
- Reusable matmul outlining, scratch handling and whole-model lowering support.
- DSP libc, memref-copy and scalar math compatibility support.
- Explicit compiler options, including the process-wide LLVM GenInsert policy.

Qwen-specific numerical passes, q1/q2 profiles, attention/layout transformations,
model DSP helpers, diagnostic sources and the complete LLVM compatibility recipe
belong to the separate `qwen3-hexagon-runtime` application. They are no longer
stored in this repository. In that application, see `native/`,
`dependencies/llvm/` and `docs/qwen3/SOURCE_BUILD.md`.

The application loads its profiles through CMake's standard top-level include
mechanism and supplies the compiler checkout explicitly to its numerical tools.
The compiler build has no Qwen-specific CMake hook or dependency on application
source paths. The standalone `stabilize-fp32-sum` policy belongs to the application's
policy optimizers, not `linalg-hexagon-opt`.

The first two correctness fixes are independently reviewable. The remaining
backend integration is a separate support change; this boundary is not a claim
that every backend extension is already accepted or minimal for upstream.

Tests for the retained backend remain in `test/` and `qcom_hexagon_backend/test/`.
The application owns numerical policy, LLVM snapshot and SLEEF preparation tests.
Historical full-model qualification is recorded by the application; moving source
ownership does not by itself constitute a new model compilation or DSP acceptance.
