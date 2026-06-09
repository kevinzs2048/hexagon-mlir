//===- HexagonRuntimeShared.h - External resource registration for embedding -===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===---------------------------------------------------------------------===//
//
// Enables embedding compiled hexagon-mlir kernels into a long-running host
// runtime (e.g., ExecuTorch BackendInterface, llama.cpp ggml-hexagon, third-
// party inference engines). The host runtime acquires VTCM and HMX resources
// itself; this API lets it inform hexagon-mlir runtime to skip its own
// resource acquisition and use the externally-provided VTCM instead.
//
// Standalone microbenchmark mode (run_main_on_hexagon execution model) is
// unaffected: if this function is not called, HexagonAPI's lazy init uses the
// existing AcquireResources() path.
//
//===---------------------------------------------------------------------===//

#ifndef HEXAGON_BIN_RUNTIME_INCLUDE_HEXAGONRUNTIMESHARED_H_
#define HEXAGON_BIN_RUNTIME_INCLUDE_HEXAGONRUNTIMESHARED_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  HEXMLIR_REGISTER_OK                    = 0,
  HEXMLIR_REGISTER_ALREADY_REGISTERED    = -1,
  HEXMLIR_REGISTER_TOO_LATE              = -2,
  HEXMLIR_REGISTER_INVALID_ARGS          = -3,
  HEXMLIR_REGISTER_INVALID_ALIGNMENT     = -4,
} hexmlir_register_status_t;

/// Register externally-acquired VTCM with hexagon-mlir runtime.
///
/// After this call returns HEXMLIR_REGISTER_OK, the first invocation of any
/// hexagon-mlir kernel will lazily initialize HexagonAPI in shared mode,
/// using the provided VTCM and skipping its own HAP_compute_res_acquire /
/// HAP_power_set HMX calls.
///
/// Marked with default visibility so a host runtime can resolve it across a
/// DT_NEEDED dynamic link even when the runtime is compiled
/// -fvisibility=hidden.
///
/// @return  HEXMLIR_REGISTER_OK (0) on success, or a negative
///          hexmlir_register_status_t code on failure (state unchanged).
__attribute__((visibility("default")))
int hexmlir_register_external_resources(void *vtcm_ptr, size_t vtcm_size);

#ifdef __cplusplus
}  // extern "C"

// Internal helper used by HexagonAPI's lazy-init ctor to atomically commit the
// phase state machine and read back the registered VTCM region. Not part of the
// public C ABI.
namespace hexmlir_internal {
bool commit_and_get_external_resources(void **out_ptr, size_t *out_size);
}  // namespace hexmlir_internal
#endif

#endif  // HEXAGON_BIN_RUNTIME_INCLUDE_HEXAGONRUNTIMESHARED_H_
