//===- IntrinsicsHVX.c   - some useful HVX qfloat routines ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Contains useful HVX qfloat routines.
// All routines should be tagged with HEXAGON_INTRIN or HEXAGON_INTRIN_INLINE
// See llvm/utils/Target/Hexagon/h*_protos.h for available intrinsics
//
//===----------------------------------------------------------------------===//

#include "hexagon_types.h"
#include "internal/hvx_internal.h"
#include "internal/qhmath_hvx_convert.h"
#include "internal/qhmath_hvx_vector.h"

#ifndef HEXAGON_INTRIN_INLINE
#define HEXAGON_INTRIN_INLINE                                                  \
  __attribute__((unused, used, always_inline, visibility("hidden")))
#endif
#ifndef HEXAGON_INTRIN
#define HEXAGON_INTRIN __attribute__((visibility("hidden")))
#endif

#ifndef MAP_MLIR_TO_QHMATH_DOUBLE_ARGS
#define MAP_MLIR_TO_QHMATH_DOUBLE_ARGS(FNAME, SUFFIX)                          \
  HEXAGON_INTRIN_INLINE HVX_Vector _hexagon_runtime_##FNAME##__##SUFFIX(       \
      HVX_Vector vin1, HVX_Vector vin2) {                                      \
    return qhmath_hvx_##FNAME##_##SUFFIX(vin1, vin2);                          \
  }
#endif

#ifndef MAP_MLIR_TO_QHMATH_SINGLE_ARG
#define MAP_MLIR_TO_QHMATH_SINGLE_ARG(FNAME, SUFFIX)                           \
  HEXAGON_INTRIN_INLINE HVX_Vector _hexagon_runtime_##FNAME##__##SUFFIX(       \
      HVX_Vector vin) {                                                        \
    return qhmath_hvx_##FNAME##_##SUFFIX(vin);                                 \
  }
#endif

// The following macro is necessary for internal functions, such as
// floor and ceil, which have different naming conventions than the
// qhmath functions exposed publicly. The core difference is the
// having the SUFFIX appear twice in the function name.
#ifndef MAP_MLIR_TO_QHMATH_INTERNAL_SINGLE_ARG
#define MAP_MLIR_TO_QHMATH_INTERNAL_SINGLE_ARG(FNAME, SUFFIX)                  \
  HEXAGON_INTRIN_INLINE HVX_Vector _hexagon_runtime_##FNAME##__##SUFFIX(       \
      HVX_Vector vin) {                                                        \
    return qhmath_hvx_##SUFFIX##_##FNAME##_##SUFFIX(vin);                      \
  }
#endif

static inline HVX_Vector hexagon_runtime_splat_f32(float v) {
  union {
    float f;
    int i;
  } u = {.f = v};
  return Q6_V_vsplat_R(u.i);
}

#if __HVX_ARCH__ < 79
static inline HVX_Vector hexagon_runtime_add_f32(HVX_Vector a, HVX_Vector b) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}

static inline HVX_Vector hexagon_runtime_sub_f32(HVX_Vector a, HVX_Vector b) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b));
}

static inline HVX_Vector hexagon_runtime_mul_f32(HVX_Vector a, HVX_Vector b) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}
#else
static inline HVX_Vector hexagon_runtime_add_f32(HVX_Vector a, HVX_Vector b) {
  return Q6_Vsf_vadd_VsfVsf(a, b);
}

static inline HVX_Vector hexagon_runtime_sub_f32(HVX_Vector a, HVX_Vector b) {
  return Q6_Vsf_vsub_VsfVsf(a, b);
}

static inline HVX_Vector hexagon_runtime_mul_f32(HVX_Vector a, HVX_Vector b) {
  return Q6_Vsf_vmpy_VsfVsf(a, b);
}
#endif

static inline HVX_Vector hexagon_runtime_floor_f32(HVX_Vector in_vec) {
  HVX_Vector mask_mant_v = Q6_V_vsplat_R(0x7FFFFF);
  HVX_Vector mask_impl_v = Q6_V_vsplat_R(0x800000);
  HVX_Vector const_mnlen_v = Q6_V_vsplat_R(23);
  HVX_Vector const_zero_v = Q6_V_vzero();
  HVX_Vector const_negone_v = Q6_V_vsplat_R(0xbf800000);

  HVX_VectorPred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, in_vec);

  HVX_Vector expval_v = in_vec >> 23;
  expval_v &= 0xFF;
  expval_v -= 127;

  HVX_VectorPred q_negexp = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);
  HVX_VectorPred q_expltmn = Q6_Q_vcmp_gt_VwVw(const_mnlen_v, expval_v);
  HVX_VectorPred q_negexp_pos =
      Q6_Q_vcmp_gtand_QVwVw(q_negexp, in_vec, const_zero_v);
  HVX_VectorPred q_negexp_neg =
      Q6_Q_vcmp_gtand_QVwVw(q_negexp, const_zero_v, in_vec);

  mask_mant_v >>= expval_v;
  HVX_Vector neg_addin_v = mask_impl_v >> expval_v;
  HVX_Vector vout_neg_addin = Q6_Vw_vadd_VwVw(in_vec, neg_addin_v);
  HVX_Vector vout =
      Q6_V_vmux_QVV(q_negative, vout_neg_addin, in_vec);

  HVX_Vector mask_chk_v = Q6_V_vand_VV(in_vec, mask_mant_v);
  HVX_VectorPred q_integral = Q6_Q_vcmp_eq_VwVw(const_zero_v, mask_chk_v);

  HVX_Vector not_mask_v = Q6_V_vnot_V(mask_mant_v);
  HVX_Vector vfrfloor_v = Q6_V_vand_VV(vout, not_mask_v);

  vout = in_vec;
  vout = Q6_V_vmux_QVV(q_expltmn, vfrfloor_v, vout);
  vout = Q6_V_vmux_QVV(q_integral, in_vec, vout);
  vout = Q6_V_vmux_QVV(q_negexp_pos, const_zero_v, vout);
  vout = Q6_V_vmux_QVV(q_negexp_neg, const_negone_v, vout);

  return vout;
}

static inline HVX_Vector hexagon_runtime_truncate_f32(HVX_Vector in_vec) {
  HVX_Vector mask_mant_v = Q6_V_vsplat_R(0x7FFFFF);
  HVX_Vector mask_impl_v = Q6_V_vsplat_R(0x800000);
  HVX_Vector const_zero_v = Q6_V_vzero();

  HVX_VectorPred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, in_vec);
  HVX_Vector expval_v = in_vec >> 23;
  expval_v &= 0xFF;
  expval_v -= 127;

  HVX_VectorPred q_negexp = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);
  HVX_Vector rshift_v = 23 - expval_v;
  HVX_Vector mant_v = in_vec & mask_mant_v;
  HVX_Vector vout = Q6_Vw_vadd_VwVw(mant_v, mask_impl_v);

  vout = Q6_Vw_vasr_VwVw(vout, rshift_v);
  vout = Q6_V_vmux_QVV(q_negexp, const_zero_v, vout);
  HVX_Vector neg_vout = -vout;
  return Q6_V_vmux_QVV(q_negative, neg_vout, vout);
}

static inline HVX_Vector hexagon_runtime_inverse_f32(HVX_Vector v_sf) {
  HVX_Vector inv_approx_sf = Q6_V_vsplat_R(0x7EEEEBB3);
  HVX_Vector two_sf = hexagon_runtime_splat_f32(2.0f);
  HVX_Vector i_sf = Q6_Vw_vsub_VwVw(inv_approx_sf, v_sf);

  HVX_Vector r_qf = Q6_Vqf32_vmpy_VsfVsf(
      i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(
                two_sf, Q6_Vsf_equals_Vqf32(
                            Q6_Vqf32_vmpy_VsfVsf(i_sf, v_sf)))));
  r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
      r_qf, Q6_Vqf32_vsub_VsfVsf(
                two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(
                            Q6_Vsf_equals_Vqf32(r_qf), v_sf))));
  r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
      r_qf, Q6_Vqf32_vsub_VsfVsf(
                two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(
                            Q6_Vsf_equals_Vqf32(r_qf), v_sf))));

  return Q6_Vsf_equals_Vqf32(r_qf);
}

static inline HVX_Vector hexagon_runtime_exp_f32(HVX_Vector in_vec) {
  HVX_Vector zero_v = Q6_V_vzero();
  HVX_Vector log2e = Q6_V_vsplat_R(0x3FB8AA3B);
  HVX_Vector logn2 = Q6_V_vsplat_R(0x3F317218);
  HVX_Vector range_hi = Q6_V_vsplat_R(0x42B17218);
  HVX_Vector range_lo = Q6_V_vsplat_R(0xC2B00000);

  HVX_VectorPred pred_cap_right = Q6_Q_vcmp_gt_VsfVsf(in_vec, range_hi);
  HVX_VectorPred pred_cap_left = Q6_Q_vcmp_gt_VsfVsf(range_lo, in_vec);
  in_vec = Q6_V_vmux_QVV(pred_cap_right, range_hi, in_vec);
  in_vec = Q6_V_vmux_QVV(pred_cap_left, range_lo, in_vec);

  HVX_Vector epsilon_v = Q6_Vqf32_vmpy_VsfVsf(log2e, in_vec);
  epsilon_v = Q6_Vsf_equals_Vqf32(epsilon_v);

  HVX_Vector f_v = hexagon_runtime_floor_f32(epsilon_v);
  HVX_Vector k_v = hexagon_runtime_truncate_f32(f_v);
  HVX_Vector x_qf32_v = Q6_Vqf32_vadd_VsfVsf(in_vec, zero_v);
  epsilon_v = Q6_Vqf32_vmpy_VsfVsf(f_v, logn2);
  x_qf32_v = Q6_Vqf32_vsub_Vqf32Vqf32(x_qf32_v, epsilon_v);
  x_qf32_v = Q6_Vqf32_vadd_Vqf32Vsf(x_qf32_v, zero_v);
  HVX_Vector x_v = Q6_Vsf_equals_Vqf32(x_qf32_v);

  HVX_Vector z_qf32_v = Q6_Vqf32_vmpy_Vqf32Vqf32(x_qf32_v, x_qf32_v);
  z_qf32_v = Q6_Vqf32_vadd_Vqf32Vsf(z_qf32_v, zero_v);

  HVX_Vector y_v = Q6_Vqf32_vmpy_VsfVsf(Q6_V_vsplat_R(0x39506967), x_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(0x3AB743CE));
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);
  y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(0x3C088908));
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);
  y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(0x3D2AA9C1));
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);
  y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(0x3E2AAAAA));
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);
  y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(0x3F000000));
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

  y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, z_qf32_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vqf32(y_v, x_qf32_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(0x3f800000));

  y_v = Q6_Vsf_equals_Vqf32(y_v);
  HVX_Vector y_v_exponent = Q6_Vw_vasl_VwR(y_v, 1);
  y_v_exponent = Q6_Vuw_vlsr_VuwR(y_v_exponent, 24);
  y_v_exponent = Q6_Vw_vadd_VwVw(k_v, y_v_exponent);
  HVX_VectorPred q_negative_exponent = Q6_Q_vcmp_gt_VwVw(zero_v, y_v_exponent);
  y_v = Q6_Vw_vaslacc_VwVwR(y_v, k_v, 23);
  return Q6_V_vmux_QVV(q_negative_exponent, zero_v, y_v);
}

static inline HVX_Vector hexagon_runtime_sigmoid_f32(HVX_Vector v) {
  HVX_Vector one = hexagon_runtime_splat_f32(1.0f);
  HVX_Vector max_exp = hexagon_runtime_splat_f32(87.0f);
  HVX_Vector min_exp = hexagon_runtime_splat_f32(-87.0f);
  HVX_VectorPred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
  HVX_VectorPred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

  HVX_Vector qv = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(0x3FB8AA3B));
  qv = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(qv),
                            Q6_V_vsplat_R(0x3F000000));
  HVX_Vector in_int = hexagon_runtime_truncate_f32(Q6_Vsf_equals_Vqf32(qv));
  HVX_Vector x = Q6_Vqf32_vsub_Vqf32Vsf(qv, Q6_Vsf_equals_Vw(in_int));
  HVX_Vector xx = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

  HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx),
                                       Q6_V_vsplat_R(0x3E8D74BD));
  v1 = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(0x3FB8AA3B));

  HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x),
                                       Q6_V_vsplat_R(0x3D009076));
  v2 = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
  v2 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

  HVX_Vector v3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
  HVX_Vector v3_exponent = Q6_Vw_vasl_VwR(v3, 1);
  v3_exponent = Q6_Vuw_vlsr_VuwR(v3_exponent, 24);
  v3_exponent = Q6_Vw_vadd_VwVw(in_int, v3_exponent);
  v3 = Q6_Vw_vaslacc_VwVwR(v3, in_int, 24);

  HVX_Vector v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
  HVX_Vector v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));
  HVX_Vector res = hexagon_runtime_inverse_f32(v5);
  res = Q6_Vqf32_vmpy_VsfVsf(v3, res);

  HVX_Vector out = Q6_Vsf_equals_Vqf32(res);
  out = Q6_V_vmux_QVV(pred_max, out, one);
  return Q6_V_vmux_QVV(pred_min, out, Q6_V_vzero());
}

HEXAGON_INTRIN_INLINE HVX_Vector _hexagon_runtime_exp__vf(HVX_Vector vin) {
  return hexagon_runtime_exp_f32(vin);
}

HEXAGON_INTRIN_INLINE HVX_Vector _hexagon_runtime_log__vf(HVX_Vector vin) {
  return qhmath_hvx_log_vf(vin);
}

HEXAGON_INTRIN_INLINE HVX_Vector _hexagon_runtime_tanh__vf(HVX_Vector vin) {
  HVX_Vector two = hexagon_runtime_splat_f32(2.0f);
  HVX_Vector one = hexagon_runtime_splat_f32(1.0f);
  HVX_Vector x2 = Q6_Vqf32_vmpy_VsfVsf(vin, two);
  HVX_Vector sig2x = hexagon_runtime_sigmoid_f32(Q6_Vsf_equals_Vqf32(x2));
  HVX_Vector res = Q6_Vqf32_vmpy_VsfVsf(sig2x, two);
  res = Q6_Vqf32_vsub_Vqf32Vsf(res, one);
  return Q6_Vsf_equals_Vqf32(res);
}

HEXAGON_INTRIN_INLINE HVX_Vector _hexagon_runtime_cos__vf(HVX_Vector x) {
  HVX_Vector const_inv_pi = hexagon_runtime_splat_f32(0.3183098861837907f);
  HVX_Vector const_half = hexagon_runtime_splat_f32(0.5f);
  HVX_Vector const_pi = hexagon_runtime_splat_f32(3.141592653589793f);
  HVX_Vector const_one = hexagon_runtime_splat_f32(1.0f);
  HVX_Vector const_neg_one = hexagon_runtime_splat_f32(-1.0f);

  HVX_Vector n_float = hexagon_runtime_floor_f32(
      hexagon_runtime_add_f32(hexagon_runtime_mul_f32(x, const_inv_pi),
                              const_half));
  HVX_Vector y =
      hexagon_runtime_sub_f32(x, hexagon_runtime_mul_f32(n_float, const_pi));
  HVX_Vector half_n = hexagon_runtime_mul_f32(n_float, const_half);
  HVX_Vector floor_half_n = hexagon_runtime_floor_f32(half_n);
  HVX_VectorPred is_odd = Q6_Q_vcmp_gt_VsfVsf(half_n, floor_half_n);
  HVX_Vector sign = Q6_V_vmux_QVV(is_odd, const_neg_one, const_one);

  HVX_Vector z = hexagon_runtime_mul_f32(y, y);
  HVX_Vector c4 = hexagon_runtime_splat_f32(2.3557242013849433e-05f);
  HVX_Vector c3 = hexagon_runtime_splat_f32(-0.0013871428263450528f);
  HVX_Vector c2 = hexagon_runtime_splat_f32(0.041665895266688284f);
  HVX_Vector c1 = hexagon_runtime_splat_f32(-0.4999999360426369f);
  HVX_Vector c0 = hexagon_runtime_splat_f32(0.9999999999071725f);

  HVX_Vector cos_y =
      hexagon_runtime_add_f32(c3, hexagon_runtime_mul_f32(z, c4));
  cos_y = hexagon_runtime_add_f32(c2, hexagon_runtime_mul_f32(z, cos_y));
  cos_y = hexagon_runtime_add_f32(c1, hexagon_runtime_mul_f32(z, cos_y));
  cos_y = hexagon_runtime_add_f32(c0, hexagon_runtime_mul_f32(z, cos_y));

  return hexagon_runtime_mul_f32(cos_y, sign);
}

HEXAGON_INTRIN_INLINE HVX_Vector _hexagon_runtime_sin__vf(HVX_Vector x) {
  HVX_Vector const_inv_pi = hexagon_runtime_splat_f32(0.3183098861837907f);
  HVX_Vector const_half = hexagon_runtime_splat_f32(0.5f);
  HVX_Vector const_pi = hexagon_runtime_splat_f32(3.141592653589793f);
  HVX_Vector const_one = hexagon_runtime_splat_f32(1.0f);
  HVX_Vector const_neg_one = hexagon_runtime_splat_f32(-1.0f);

  HVX_Vector n_float = hexagon_runtime_floor_f32(
      hexagon_runtime_add_f32(hexagon_runtime_mul_f32(x, const_inv_pi),
                              const_half));
  HVX_Vector y =
      hexagon_runtime_sub_f32(x, hexagon_runtime_mul_f32(n_float, const_pi));
  HVX_Vector half_n = hexagon_runtime_mul_f32(n_float, const_half);
  HVX_Vector floor_half_n = hexagon_runtime_floor_f32(half_n);
  HVX_VectorPred is_odd = Q6_Q_vcmp_gt_VsfVsf(half_n, floor_half_n);
  HVX_Vector sign = Q6_V_vmux_QVV(is_odd, const_neg_one, const_one);

  HVX_Vector z = hexagon_runtime_mul_f32(y, y);
  HVX_Vector s4 = hexagon_runtime_splat_f32(2.642186986152672e-06f);
  HVX_Vector s3 = hexagon_runtime_splat_f32(-0.00019825318964070864f);
  HVX_Vector s2 = hexagon_runtime_splat_f32(0.00833326283319605f);
  HVX_Vector s1 = hexagon_runtime_splat_f32(-0.16666666082087775f);
  HVX_Vector s0 = hexagon_runtime_splat_f32(0.999999999915155f);

  HVX_Vector sin_y =
      hexagon_runtime_add_f32(s3, hexagon_runtime_mul_f32(z, s4));
  sin_y = hexagon_runtime_add_f32(s2, hexagon_runtime_mul_f32(z, sin_y));
  sin_y = hexagon_runtime_add_f32(s1, hexagon_runtime_mul_f32(z, sin_y));
  sin_y = hexagon_runtime_add_f32(s0, hexagon_runtime_mul_f32(z, sin_y));
  sin_y = hexagon_runtime_mul_f32(y, sin_y);

  return hexagon_runtime_mul_f32(sin_y, sign);
}

HEXAGON_INTRIN_INLINE HVX_Vector _hexagon_runtime_rsqrt__vf(HVX_Vector vin) {
  HVX_Vector rsqrtconst = Q6_V_vsplat_R(0x5f3759df);
  HVX_Vector onehalf = Q6_V_vsplat_R(0x3f000000);
  HVX_Vector threehalfs = Q6_V_vsplat_R(0x3fc00000);

  HVX_Vector x2 = Q6_Vqf32_vmpy_VsfVsf(vin, onehalf);
  x2 = Q6_Vqf32_vadd_Vqf32Vsf(x2, Q6_V_vzero());

  HVX_Vector y = Q6_Vw_vasr_VwR(vin, 1);
  y = Q6_Vw_vsub_VwVw(rsqrtconst, y);

  HVX_Vector ypower2 = Q6_Vqf32_vmpy_VsfVsf(y, y);
  ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
  HVX_Vector temp = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
  temp =
      Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
  temp = Q6_Vqf32_vmpy_VsfVsf(y, Q6_Vsf_equals_Vqf32(temp));

  y = Q6_Vqf32_vadd_Vqf32Vsf(temp, Q6_V_vzero());
  ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
  ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
  temp = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
  temp =
      Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
  temp = Q6_Vqf32_vmpy_Vqf32Vqf32(y, temp);

  y = Q6_Vqf32_vadd_Vqf32Vsf(temp, Q6_V_vzero());
  ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
  ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
  temp = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
  temp =
      Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
  temp = Q6_Vqf32_vmpy_Vqf32Vqf32(y, temp);

  return Q6_Vsf_equals_Vqf32(temp);
}

HEXAGON_INTRIN_INLINE HVX_Vector _hexagon_runtime_sqrt__vf(HVX_Vector vin) {
  return hexagon_runtime_mul_f32(_hexagon_runtime_rsqrt__vf(vin), vin);
}

//===----------------------------------------------------------------------===//
// Exponential
//===----------------------------------------------------------------------===//

MAP_MLIR_TO_QHMATH_SINGLE_ARG(exp, vhf);

//===----------------------------------------------------------------------===//
// Power
//===----------------------------------------------------------------------===//
MAP_MLIR_TO_QHMATH_DOUBLE_ARGS(pow, vf);
MAP_MLIR_TO_QHMATH_DOUBLE_ARGS(pow, vhf);

//===----------------------------------------------------------------------===//
// Trigonometric Functions
//===----------------------------------------------------------------------===//
// MAP_MLIR_TO_QHMATH_SINGLE_ARG(sin, vhf);
// MAP_MLIR_TO_QHMATH_SINGLE_ARG(cos, vhf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(tan, vf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(tanh, vhf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(acos, vf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(acos, vhf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(asin, vf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(asin, vhf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(atan, vf);
MAP_MLIR_TO_QHMATH_SINGLE_ARG(atan, vhf);

//===----------------------------------------------------------------------===//
// Sqrt
//===----------------------------------------------------------------------===//
MAP_MLIR_TO_QHMATH_SINGLE_ARG(sqrt, vhf);

//===----------------------------------------------------------------------===//
// Rsqrt
//===----------------------------------------------------------------------===//
MAP_MLIR_TO_QHMATH_SINGLE_ARG(rsqrt, vhf);

//===----------------------------------------------------------------------===//
// Floor
//===----------------------------------------------------------------------===//
MAP_MLIR_TO_QHMATH_INTERNAL_SINGLE_ARG(floor, vhf);
MAP_MLIR_TO_QHMATH_INTERNAL_SINGLE_ARG(floor, vsf);

//===----------------------------------------------------------------------===//
// Ceil
//===----------------------------------------------------------------------===//
MAP_MLIR_TO_QHMATH_INTERNAL_SINGLE_ARG(ceil, vhf);
MAP_MLIR_TO_QHMATH_INTERNAL_SINGLE_ARG(ceil, vsf);
