//===- HexagonLibcCompat.cpp - DSP libc compatibility --------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Compatibility entry points required by code emitted by the in-tree LLVM
// backend but absent from some deployed Hexagon DSP libc versions.
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>

#if defined(__GNUC__)
#define HEXAGON_LIBC_COMPAT                                                    \
  __attribute__((visibility("default"), used))
#else
#define HEXAGON_LIBC_COMPAT
#endif

extern "C" HEXAGON_LIBC_COMPAT void *aligned_alloc(std::size_t alignment,
                                                    std::size_t size) {
  void *pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size) != 0)
    return nullptr;
  return pointer;
}

extern "C" HEXAGON_LIBC_COMPAT float roundevenf(float value) {
  union FloatBits {
    float value;
    std::uint32_t bits;
  } input = {value};

  const std::uint32_t magnitudeBits = input.bits & 0x7fffffffU;

  // Values at or above 2^23 are already integral in f32. This also preserves
  // infinities and NaNs unchanged.
  if (magnitudeBits >= 0x4b000000U)
    return value;

  // llvm.roundeven must disregard the active rounding mode and must not
  // raise FP exceptions. A floating-point 2^23 bias violates both rules.
  // Round the significand using integers instead, retaining signed zero.
  const std::uint32_t sign = input.bits & 0x80000000U;
  if (magnitudeBits <= 0x3f000000U) {
    input.bits = sign; // |value| <= 0.5, including subnormals.
  } else if (magnitudeBits < 0x3f800000U) {
    input.bits = sign | 0x3f800000U; // 0.5 < |value| < 1.
  } else {
    // Here the unbiased exponent is in [0, 22], so shift is in [1, 23].
    const unsigned shift = 150U - (magnitudeBits >> 23);
    const std::uint32_t unit = 1U << shift;
    const std::uint32_t fraction = magnitudeBits & (unit - 1U);
    std::uint32_t rounded = magnitudeBits & ~(unit - 1U);
    if (fraction > (unit >> 1) ||
        (fraction == (unit >> 1) && ((rounded >> shift) & 1U)))
      rounded += unit;
    input.bits = sign | rounded;
  }
  return input.value;
}

// Qualcomm clang lowers C/C++ assert() to this QuRT-era entry point. Some
// production DSP images omit it even though the toolchain headers expose
// assert(), so retain the fail-stop behavior without a loader dependency.
extern "C" [[noreturn]] HEXAGON_LIBC_COMPAT void
_Assert(const char * /*expression*/, const char * /*file*/) {
  std::abort();
}

// The tested SDK expf does not provide valid overflow/special-value behavior
// and its exponent reconstruction loses subnormal results. Keep the existing
// SDK approximation in its safe normal range, and repair the outer range.
// This is not a correctly rounded libm implementation or a CPU fallback.
extern "C" HEXAGON_LIBC_COMPAT float __hexagon_mlir_expf(float value) {
  union Bits { float value; std::uint32_t bits; } input = {value};
  const std::uint32_t magnitude = input.bits & 0x7fffffffU;
  if (magnitude > 0x7f800000U) {
    input.bits |= 0x00400000U;
    return input.value;
  }
  if (value >= 88.72283935546875f) {
    input.bits = 0x7f800000U;
    return input.value;
  }
  if (value <= -104.0f)
    return 0.0f;
  if (value >= -80.0f && value <= 80.0f)
    return ::expf(value);

  const int scale = value > 80.0f ? 64 : -64;
  // Only rare outer-range inputs take the wider range-reduction calculation.
  const double delta = scale > 0 ? 44.3614195558364998 : -44.3614195558364998;
  const float reduced = static_cast<float>(static_cast<double>(value) - delta);
  Bits result = {::expf(reduced)};
  const int exponent = int((result.bits >> 23) & 255U) + scale;
  if (exponent >= 255) {
    // Original-input classification above owns true overflow. Approximation
    // error just below that threshold must not construct a NaN or infinity.
    result.bits = 0x7f7fffffU;
  } else if (exponent > 0) {
    result.bits += std::uint32_t(scale) << 23;
  } else {
    const unsigned shift = unsigned(1 - exponent);
    if (shift > 24) {
      result.bits = 0;
    } else {
      const std::uint32_t significand = (result.bits & 0x7fffffU) | 0x800000U;
      const std::uint32_t remainder = significand & ((1U << shift) - 1U);
      const std::uint32_t half = 1U << (shift - 1);
      result.bits = significand >> shift;
      if (remainder > half || (remainder == half && (result.bits & 1U)))
        ++result.bits;
    }
  }
  return result.value;
}

#undef HEXAGON_LIBC_COMPAT
