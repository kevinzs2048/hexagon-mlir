// SPDX-License-Identifier: BSD-3-Clause
#ifndef HEXAGON_RUNTIME_EXP_RANGE_REDUCTION_H
#define HEXAGON_RUNTIME_EXP_RANGE_REDUCTION_H

#include "ExpSpecialValues.h"

// Scale a positive normal FP32 approximation by 2^scale without flushing
// subnormals or wrapping a biased exponent through the NaN encoding.
// Overflow here saturates; the original-input check selects true infinity.
static inline HVX_Vector hexagon_exp_scale_f32(HVX_Vector value,
                                               HVX_Vector scale) {
  HVX_Vector zero = Q6_V_vzero(), one = Q6_V_vsplat_R(1);
  HVX_Vector exponent = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(value, 23),
                                    Q6_V_vsplat_R(255));
  exponent = Q6_Vw_vadd_VwVw(exponent, scale);
  HVX_Vector normal = Q6_Vw_vadd_VwVw(value, Q6_Vw_vasl_VwR(scale, 23));
  HVX_Vector shift = Q6_Vw_vsub_VwVw(one, exponent);
  shift = Q6_Vw_vmax_VwVw(one, Q6_Vw_vmin_VwVw(shift, Q6_V_vsplat_R(31)));
  HVX_Vector significand = Q6_V_vor_VV(Q6_V_vand_VV(value, Q6_V_vsplat_R(0x7fffff)),
                                      Q6_V_vsplat_R(0x800000));
  HVX_Vector truncated = Q6_Vw_vasr_VwVw(significand, shift);
  HVX_Vector mask = Q6_Vw_vsub_VwVw(Q6_Vw_vasl_VwVw(one, shift), one);
  HVX_Vector remainder = Q6_V_vand_VV(significand, mask);
  HVX_Vector half = Q6_Vw_vasl_VwVw(one, Q6_Vw_vsub_VwVw(shift, one));
  HVX_VectorPred odd = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(truncated, one), one);
  HVX_VectorPred ties = Q6_Q_and_QQ(Q6_Q_vcmp_eq_VwVw(remainder, half), odd);
  HVX_VectorPred round_up = Q6_Q_or_QQ(Q6_Q_vcmp_gt_VwVw(remainder, half), ties);
  HVX_Vector subnormal = Q6_Vw_vadd_VwVw(truncated, Q6_V_vmux_QVV(round_up, one, zero));
  HVX_Vector result = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(one, exponent), subnormal, normal);
  return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(exponent, Q6_V_vsplat_R(254)),
                       Q6_V_vsplat_R(0x7f7fffff), result);
}

// Keep the existing approximation unchanged in [-80,80]. Outside that range,
// evaluate exp(x +/- 64*ln(2)) in the safe normal range, then scale its bits.
// This is an approximation with explicit range handling, not a correctly
// rounded libm replacement. The QFloat range-reduction path is v73-v75 only.
static inline HVX_Vector hexagon_exp_f32_checked(
    HVX_Vector input, HVX_Vector (*approximate)(HVX_Vector)) {
#if __HVX_ARCH__ >= 73 && __HVX_ARCH__ <= 75
  HVX_Vector zero = Q6_V_vzero();
  HVX_Vector magnitude = Q6_V_vand_VV(input, Q6_V_vsplat_R(0x7fffffff));
  HVX_VectorPred nan = Q6_Q_vcmp_gt_VwVw(magnitude, Q6_V_vsplat_R(0x7f800000));
  HVX_Vector safe = Q6_V_vmux_QVV(nan, zero, input);
  HVX_Vector lower = Q6_V_vsplat_R(0xc2d00000); // -104
  HVX_Vector upper = Q6_V_vsplat_R(0x42b20000); // 89
  safe = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(lower, safe), lower, safe);
  safe = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(safe, upper), upper, safe);
  HVX_VectorPred high = Q6_Q_vcmp_gt_VsfVsf(safe, Q6_V_vsplat_R(0x42a00000));
  HVX_VectorPred low = Q6_Q_vcmp_gt_VsfVsf(Q6_V_vsplat_R(0xc2a00000), safe);
  HVX_Vector delta_hi = Q6_V_vmux_QVV(high, Q6_V_vsplat_R(0x42317218),
                                     Q6_V_vsplat_R(0xc2317218));
  HVX_Vector delta_lo = Q6_V_vmux_QVV(high, Q6_V_vsplat_R(0xb402e308),
                                     Q6_V_vsplat_R(0x3402e308));
  HVX_Vector reduced = Q6_Vqf32_vsub_VsfVsf(safe, delta_hi);
  reduced = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vsf(reduced, delta_lo));
  // In particular, do not apply a QFloat add/subtract-zero to ordinary inputs.
  reduced = Q6_V_vmux_QVV(Q6_Q_or_QQ(high, low), reduced, safe);
  HVX_Vector scale = Q6_V_vmux_QVV(high, Q6_V_vsplat_R(64), zero);
  scale = Q6_V_vmux_QVV(low, Q6_V_vsplat_R(-64), scale);
  HVX_Vector result = hexagon_exp_scale_f32(approximate(reduced), scale);
  return hexagon_exp_f32_special_values(input, result);
#else
  return hexagon_exp_f32_special_values(input, approximate(input));
#endif
}

#endif
