//===- returned_memref_owner.h ---------------------------------------------===//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_RETURNED_MEMREF_OWNER_H
#define HEXAGON_RETURNED_MEMREF_OWNER_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>

// For public functions processed by MLIR's buffer deallocation pipeline:
// returned memrefs transfer ownership to their caller and may not alias inputs.
// Tensor(desc, ...) is deliberately a borrowed view, so this separate owner
// releases the allocated BASE pointers, once each, after all views are done.
// Not valid for arbitrary Triton kernels, VTCM results or borrowed C interfaces.
template <std::size_t Count> class OwnedReturnedMemrefs {
  void *owned[Count ? Count : 1] = {};
  std::size_t count = 0;
  bool adopted = false;

public:
  OwnedReturnedMemrefs() = default;
  OwnedReturnedMemrefs(const OwnedReturnedMemrefs &) = delete;
  OwnedReturnedMemrefs &operator=(const OwnedReturnedMemrefs &) = delete;

  bool adopt(std::initializer_list<void *> results,
             std::initializer_list<void *> inputs) {
    if (adopted || results.size() != Count) return false;
    // Validate the whole collection BEFORE acquiring any ownership. A bad ABI
    // must not cause partial frees or release a caller-owned input allocation.
    for (void *base : results) {
      if (!base) continue;
      if (reinterpret_cast<std::uintptr_t>(base) == 0xdeadbeefU) return false;
      for (void *input : inputs)
        if (base == input) return false;
    }
    for (void *base : results) {
      if (!base) continue;
      bool duplicate = false;
      for (std::size_t i = 0; i < count; ++i)
        duplicate |= owned[i] == base;
      if (!duplicate) owned[count++] = base;
    }
    adopted = true;
    return true;
  }

  std::size_t unique_buffers() const { return count; }

  ~OwnedReturnedMemrefs() {
    for (std::size_t i = 0; i < count; ++i) std::free(owned[i]);
  }
};
#endif
