# Standalone constant-partition regression

This host-only test includes and executes the actual `LowerConstantsSeparatelyPass`. Compact splat metadata describes large constants without allocating their payloads. No DSP or model checkpoint is required.

Use an MLIR build compatible with this Hexagon-MLIR checkout:

```bash
cmake -S test/cpp/constant-partition -B /absolute/partition-build \
  -DMLIR_DIR=/absolute/llvm-build/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /absolute/partition-build -j 2
HEXAGON_CONSTANT_PARTITION_PROBE=/absolute/partition-build/constant-partition-probe \
  python -m pytest -q test/cpp/constant-partition/test_constant_partition.py
```

The test refuses to skip when its probe is absent. Cases include an empty module, exact limits, rollover accounting, individually oversized constants, footprints beyond 32-bit range and deterministic mixed sequences. It checks partition membership, order, splat values and preservation of module attributes.

A dedicated resource round-trip case checks that `dense_resource` handles and payloads remain valid in every produced module. These checks exercise pass behavior. They do not validate the complete compiler build or end-to-end DSP inference on the new public baseline.
