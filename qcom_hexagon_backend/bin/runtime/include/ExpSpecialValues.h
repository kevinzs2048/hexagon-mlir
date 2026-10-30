// SPDX-License-Identifier: BSD-3-Clause
#ifndef HEXAGON_RUNTIME_EXP_SPECIAL_VALUES_H
#define HEXAGON_RUNTIME_EXP_SPECIAL_VALUES_H

#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"

// Repair the exceptional-value boundary independently of the approximation.
// 0x42b17218 is the first FP32 input whose mathematical exp exceeds the FP32
// overflow rounding threshold. Merely clamping to it can construct an exponent
// of 255 with nonzero fraction bits, i.e. a NaN rather than positive infinity.
// This does not assert correct rounding of ordinary or subnormal results.
static inline HVX_Vector hexagon_exp_f32_special_values(HVX_Vector input,
                                                       HVX_Vector approximate) {
  HVX_Vector magnitude = Q6_V_vand_VV(input, Q6_V_vsplat_R(0x7fffffff));
  HVX_Vector infinity = Q6_V_vsplat_R(0x7f800000);
  HVX_VectorPred overflow =
      Q6_Q_vcmp_gt_VwVw(input, Q6_V_vsplat_R(0x42b17217));
  HVX_VectorPred negative_infinity =
      Q6_Q_vcmp_eq_VwVw(input, Q6_V_vsplat_R(0xff800000));
  HVX_VectorPred nan = Q6_Q_vcmp_gt_VwVw(magnitude, infinity);
  HVX_Vector result = Q6_V_vmux_QVV(overflow, infinity, approximate);
  result = Q6_V_vmux_QVV(negative_infinity, Q6_V_vzero(), result);
  return Q6_V_vmux_QVV(nan, Q6_V_vor_VV(input, Q6_V_vsplat_R(0x00400000)), result);
}

#endif
