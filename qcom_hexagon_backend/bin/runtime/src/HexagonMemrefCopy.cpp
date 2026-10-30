//===- HexagonMemrefCopy.cpp - MLIR memref copy on DSP --------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
// Implements the MLIR unranked memrefCopy ABI without the host runner library.
// Like memref.copy, source and destination have the same shape and element type.
// Overlapping source and destination are not supported.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {
struct UnrankedMemRef {
  int64_t rank;
  void *descriptor;
};
struct RankedPrefix {
  char *allocated;
  char *aligned;
  int64_t offset;
};
struct View {
  RankedPrefix prefix;
  const int64_t *sizes;
  const int64_t *strides;
  explicit View(const UnrankedMemRef &value) {
    std::memcpy(&prefix, value.descriptor, sizeof(prefix));
    sizes = reinterpret_cast<const int64_t *>(
        static_cast<const char *>(value.descriptor) + sizeof(prefix));
    strides = sizes + value.rank;
  }
};
} // namespace

extern "C" __attribute__((visibility("default"), used))
void memrefCopy(int64_t elementSize, UnrankedMemRef *source,
                UnrankedMemRef *destination) {
  if (!source || !destination || source->rank < 0 ||
      source->rank != destination->rank || !source->descriptor ||
      !destination->descriptor || elementSize <= 0 ||
      uint64_t(source->rank) > SIZE_MAX / sizeof(int64_t) ||
      uint64_t(elementSize) > SIZE_MAX)
    std::abort();
  const View src(*source), dst(*destination);
  for (int64_t axis = 0; axis < source->rank; ++axis) {
    if (src.sizes[axis] < 0 || src.sizes[axis] != dst.sizes[axis])
      std::abort();
    if (src.sizes[axis] == 0)
      return;
  }
  if (!src.prefix.aligned || !dst.prefix.aligned)
    std::abort();

  // Coalesce the shared contiguous suffix into one memcpy. This includes the
  // common whole-buffer case; noncontiguous/negative/zero strides still use
  // the general indexed traversal below.
  int64_t outerRank = source->rank;
  int64_t blockElements = 1;
  while (outerRank > 0 &&
         src.strides[outerRank - 1] == blockElements &&
         dst.strides[outerRank - 1] == blockElements) {
    if (src.sizes[outerRank - 1] >
        std::numeric_limits<int64_t>::max() / blockElements)
      std::abort();
    blockElements *= src.sizes[--outerRank];
  }
  if (uint64_t(blockElements) > SIZE_MAX / uint64_t(elementSize))
    std::abort();
  const size_t blockBytes = size_t(blockElements) * size_t(elementSize);
  char *srcData = src.prefix.aligned + src.prefix.offset * elementSize;
  char *dstData = dst.prefix.aligned + dst.prefix.offset * elementSize;
  if (outerRank == 0) {
    std::memcpy(dstData, srcData, blockBytes);
    return;
  }

  auto *indices = static_cast<int64_t *>(std::calloc(outerRank, sizeof(int64_t)));
  if (!indices)
    std::abort();
  int64_t read = 0, write = 0;
  for (;;) {
    std::memcpy(dstData + write * elementSize,
                srcData + read * elementSize, blockBytes);
    int64_t axis = outerRank - 1;
    for (; axis >= 0; --axis) {
      ++indices[axis];
      read += src.strides[axis];
      write += dst.strides[axis];
      if (indices[axis] < src.sizes[axis])
        break;
      indices[axis] = 0;
      read -= src.sizes[axis] * src.strides[axis];
      write -= dst.sizes[axis] * dst.strides[axis];
    }
    if (axis < 0)
      break;
  }
  std::free(indices);
}
