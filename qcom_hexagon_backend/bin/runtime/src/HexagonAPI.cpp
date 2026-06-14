//===- HexagonAPI.cpp -                                           ---------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// DataSpace: static allocations for Hexagon
//
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include <cstring>

#include "HexagonAPI.h"
#include "HAP_dcvs.h"  /* turbo DCVS corners: lock DSP core/bus to max clock */

HexagonAPI *HexagonAPI::Global() {
  static auto *inst = new HexagonAPI();
  return inst;
}
// DataSpace: static allocations for Hexagon
void *HexagonAPI::Alloc(size_t nbytes, uint64_t alignment, bool isVtcm) {
  // Allocate a single, contiguous memory region.
  void *base_ptr = bufferManager->AllocateHexagonBuffer(
      nbytes, static_cast<size_t>(alignment), isVtcm);
  return base_ptr;
}

// Lock the DSP core+bus to the max DCVS voltage corner (turbo) and disable
// sleep, mirroring the llama.cpp Hexagon backend. Without this the kernel runs
// at the default DCVS clock, which is a uniform slowdown across the whole
// mega-kernel. DCVS corners are clamped, not dynamically varied. Called from
// initialize_and_acquire_hmx, which AcquireResources() runs for every standalone
// kernel (HMX and HVX alike), so all standalone kernels get turbo. Measured: HMX
// mega-kernel 2.1-2.6x faster; pure-HVX mega-kernel is unaffected (not core-clock
// bound). Shared mode (host-registered resources, e.g. llama.cpp path A) skips
// this path — the host already set turbo itself.
static void set_dsp_turbo() {
  static int turbo_ctx;
  {
    HAP_power_request_t req;
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_apptype;
    req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    HAP_power_set((void *)&turbo_ctx, &req);
  }
  {
    HAP_power_request_t req;
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_DCVS_v3;
    req.dcvs_v3.set_dcvs_enable = TRUE;
    req.dcvs_v3.dcvs_enable = FALSE;
    req.dcvs_v3.set_bus_params = TRUE;
    req.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_core_params = TRUE;
    req.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_sleep_disable = TRUE;
    req.dcvs_v3.sleep_disable = TRUE;
    int rc = HAP_power_set((void *)&turbo_ctx, &req);
    if (rc != AEE_SUCCESS)
      FARF(ERROR, "DCVS_v3 turbo request failed rc=%d", rc);
  }
}

uint32_t HexagonAPI::initialize_and_acquire_hmx() {
  int *power_context = (int *)malloc(sizeof(int));

  set_dsp_turbo();

  HAP_power_request_t request;
  memset(&request, 0, sizeof(HAP_power_request_t));
  request.type = HAP_power_set_HMX;
  request.hmx.power_up = TRUE;

  int retVal = HAP_power_set((void *)power_context, &request);
  if (retVal != AEE_SUCCESS) {
    FARF(ERROR,
         "HMX resource allocation failed at HAP_power_set_HMX request with the "
         "return value %d",
         retVal);
    return retVal;
  }

  // Request HMX using HAP compute resource manager
  compute_res_attr_t compute_res;
  uint32_t contextID = 0;

  HAP_compute_res_attr_init(&compute_res);

  // Request the HMX resource
  HAP_compute_res_attr_set_hmx_param(&compute_res, 1);
  contextID = HAP_compute_res_acquire(&compute_res, 100000); // wait till 100ms

  if (contextID == 0) {
    return AEE_ERESOURCENOTFOUND;
  }

  // Locks the HMX unit to the current thread and prepares the thread to execute
  // HMX instructions
  HAP_compute_res_hmx_lock(contextID);

  return contextID;
}

void *HexagonAPI::Alloc(size_t nallocs, size_t nbytes, uint64_t alignment,
                        bool isVtcm) {
  // Allocate the region(s) needed for Hexagon's indirect-tensor format.
  void *base_ptr = bufferManager->AllocateHexagonBuffer(
      nallocs, nbytes, static_cast<size_t>(alignment), isVtcm);
  return base_ptr;
}

void HexagonAPI::Free(void *ptr) {
  if (bufferManager) {
    bufferManager->FreeHexagonBuffer(ptr);
  } else {
    // Either AcquireResources was never called, or ReleaseResources was called.
    // Since this can occur in the normal course of shutdown, log a message and
    // continue.
    CHECK((false), "Free called outside a session");
  }
}

// TODO: Add support to handle buffer aliases on copying
void HexagonAPI::Copy(void *dst, void *src, size_t nbytes) {
  if (bufferManager) {
    bufferManager->Copy(dst, src, nbytes);
  } else {
    // Either AcquireResources was never called, or ReleaseResources was called.
    // Since this can occur in the normal course of shutdown, log a message and
    // continue.
    CHECK((false), "Free called outside a session");
  }
}
