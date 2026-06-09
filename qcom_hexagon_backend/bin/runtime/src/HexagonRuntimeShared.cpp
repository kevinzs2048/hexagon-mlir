//===- HexagonRuntimeShared.cpp - External resource registration ---------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Implements the external-resources registration API. Holds process-wide state
// (one VTCM region pointer/size + a monotonic phase) and exposes one C-callable
// entry point plus one internal helper consumed by HexagonAPI's lazy-init ctor.
//
//===----------------------------------------------------------------------===//

#include "HexagonRuntimeShared.h"

#include <atomic>
#include <cstdint>

enum class Phase : int {
  Unregistered = 0,
  Registered = 1,
  Committed = 2,
};

namespace {
struct ExternalResourceState {
  void *vtcm_ptr = nullptr;
  size_t vtcm_size = 0;
  std::atomic<Phase> phase{Phase::Unregistered};
};

ExternalResourceState g_state;

constexpr size_t kVtcmAlignment = 4096; // 4 KiB page alignment
} // namespace

namespace hexmlir_internal {
// Atomically transition the phase to Committed and, if the prior phase was
// Registered, hand back the registered VTCM region. Returns true only when a
// valid registration was committed (i.e. shared mode should be used).
bool commit_and_get_external_resources(void **out_ptr, size_t *out_size) {
  Phase prev = g_state.phase.exchange(Phase::Committed, std::memory_order_acq_rel);
  if (prev == Phase::Registered) {
    *out_ptr = g_state.vtcm_ptr;
    *out_size = g_state.vtcm_size;
    return true;
  }
  return false;
}
} // namespace hexmlir_internal

extern "C" __attribute__((visibility("default"))) int
hexmlir_register_external_resources(void *vtcm_ptr, size_t vtcm_size) {
  if (vtcm_ptr == nullptr || vtcm_size == 0) {
    return HEXMLIR_REGISTER_INVALID_ARGS;
  }
  if ((reinterpret_cast<uintptr_t>(vtcm_ptr) & (kVtcmAlignment - 1)) != 0) {
    return HEXMLIR_REGISTER_INVALID_ALIGNMENT;
  }

  Phase cur = g_state.phase.load(std::memory_order_acquire);
  if (cur == Phase::Committed)
    return HEXMLIR_REGISTER_TOO_LATE;
  if (cur == Phase::Registered)
    return HEXMLIR_REGISTER_ALREADY_REGISTERED;

  // Non-atomic writes; per contract there is no concurrent register call.
  // Their visibility to the subsequent HexagonAPI ctor is established by the
  // release in the phase store below.
  g_state.vtcm_ptr = vtcm_ptr;
  g_state.vtcm_size = vtcm_size;
  g_state.phase.store(Phase::Registered, std::memory_order_release);
  return HEXMLIR_REGISTER_OK;
}
