"""Host oracle for the integer-only exponential exceptional-value repair.

The small intrinsic shim emulates integer lanes, not DSP floating arithmetic.
Actual HVX code generation and execution need the separate board tests.
"""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture
def hvx_integer_headers(tmp_path):
    (tmp_path / "hexagon_types.h").write_text("""
#pragma once
#include <array>
#include <cstdint>
using HVX_Vector = std::array<uint32_t,32>;
using HVX_VectorPred = std::array<bool,32>;
""")
    (tmp_path / "hvx_hexagon_protos.h").write_text("""
#pragma once
#include "hexagon_types.h"
#include <algorithm>
inline HVX_Vector Q6_V_vsplat_R(uint32_t x) { HVX_Vector v; v.fill(x); return v; }
inline HVX_Vector Q6_V_vzero() { return {}; }
inline HVX_Vector Q6_V_vand_VV(HVX_Vector a, HVX_Vector b) { for(int i=0;i<32;++i)a[i]&=b[i]; return a; }
inline HVX_Vector Q6_V_vor_VV(HVX_Vector a, HVX_Vector b) { for(int i=0;i<32;++i)a[i]|=b[i]; return a; }
inline HVX_VectorPred Q6_Q_vcmp_gt_VwVw(HVX_Vector a, HVX_Vector b) { HVX_VectorPred p; for(int i=0;i<32;++i)p[i]=int32_t(a[i])>int32_t(b[i]); return p; }
inline HVX_VectorPred Q6_Q_vcmp_eq_VwVw(HVX_Vector a, HVX_Vector b) { HVX_VectorPred p; for(int i=0;i<32;++i)p[i]=a[i]==b[i]; return p; }
inline HVX_Vector Q6_V_vmux_QVV(HVX_VectorPred p, HVX_Vector a, HVX_Vector b) { for(int i=0;i<32;++i)a[i]=p[i]?a[i]:b[i]; return a; }
inline HVX_Vector Q6_Vuw_vlsr_VuwR(HVX_Vector a, int s) { for(auto &v:a)v>>=s; return a; }
inline HVX_Vector Q6_Vw_vasl_VwR(HVX_Vector a, int s) { for(auto &v:a)v<<=s; return a; }
inline HVX_Vector Q6_Vw_vadd_VwVw(HVX_Vector a, HVX_Vector b) { for(int i=0;i<32;++i)a[i]+=b[i]; return a; }
inline HVX_Vector Q6_Vw_vsub_VwVw(HVX_Vector a, HVX_Vector b) { for(int i=0;i<32;++i)a[i]-=b[i]; return a; }
inline HVX_Vector Q6_Vw_vmin_VwVw(HVX_Vector a, HVX_Vector b) { for(int i=0;i<32;++i)a[i]=uint32_t(std::min(int32_t(a[i]),int32_t(b[i]))); return a; }
inline HVX_Vector Q6_Vw_vmax_VwVw(HVX_Vector a, HVX_Vector b) { for(int i=0;i<32;++i)a[i]=uint32_t(std::max(int32_t(a[i]),int32_t(b[i]))); return a; }
inline HVX_Vector Q6_Vw_vasr_VwVw(HVX_Vector a, HVX_Vector b) { for(int i=0;i<32;++i)a[i]=uint32_t(int32_t(a[i])>>b[i]); return a; }
inline HVX_Vector Q6_Vw_vasl_VwVw(HVX_Vector a, HVX_Vector b) { for(int i=0;i<32;++i)a[i]<<=b[i]; return a; }
inline HVX_VectorPred Q6_Q_and_QQ(HVX_VectorPred a, HVX_VectorPred b) { for(int i=0;i<32;++i)a[i]=a[i]&&b[i]; return a; }
inline HVX_VectorPred Q6_Q_or_QQ(HVX_VectorPred a, HVX_VectorPred b) { for(int i=0;i<32;++i)a[i]=a[i]||b[i]; return a; }
""")
    return tmp_path


def compile_and_run(source, tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    root = Path(__file__).resolve().parents[3]
    binary = source.with_suffix("")
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(tmp_path),
                    "-I", str(root / "qcom_hexagon_backend/bin/runtime/include"),
                    str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_exp_exceptional_values_do_not_turn_overflow_into_nan(tmp_path, hvx_integer_headers):
    source = tmp_path / "oracle.cpp"
    source.write_text("""
#include "ExpSpecialValues.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
int main() {
  const uint32_t samples[] = {0, 0x80000000, 0x42b17216, 0x42b17217,
    0x42b17218, 0x42b17219, 0x7f7fffff, 0xff7fffff, 0x7f800000,
    0xff800000, 0x7f800001, 0xff800001, 0x7fc12345, 0xffc12345};
  for (uint32_t bits : samples) {
    float x; std::memcpy(&x, &bits, sizeof(x));
    uint32_t expected = 0x3f812345; // Keep ordinary approximations unchanged.
    if (std::isnan(x)) expected = bits | 0x00400000;
    else if (x == -std::numeric_limits<float>::infinity()) expected = 0;
    else if (std::exp(double(x)) > double(std::numeric_limits<float>::max())) expected = 0x7f800000;
    auto result = hexagon_exp_f32_special_values(Q6_V_vsplat_R(bits), Q6_V_vsplat_R(0x3f812345));
    for (auto lane : result) assert(lane == expected);
  }
}
""")
    compile_and_run(source, tmp_path)


def test_exp_binary_scaling_preserves_subnormal_rounding(tmp_path, hvx_integer_headers):
    source = tmp_path / "scaling.cpp"
    source.write_text("""
#include "ExpRangeReduction.h"
#include <cassert>
#include <cmath>
#include <cstring>
int main() {
  for (uint32_t exponent=1; exponent<255; ++exponent)
    for (uint32_t fraction : {0U,1U,0x3fffffU,0x400000U,0x7ffffeU,0x7fffffU})
      for (int scale : {-256,-149,-128,-127,-126,-125,-64,-24,-23,-1,0,1,64,127,128,256}) {
        uint32_t bits=(exponent<<23)|fraction;
        float value; std::memcpy(&value,&bits,4);
        float expected=float(std::ldexp(double(value),scale));
        uint32_t want; std::memcpy(&want,&expected,4);
        if (std::isinf(expected)) want=0x7f7fffff; // Input classifier owns infinity.
        auto actual=hexagon_exp_scale_f32(Q6_V_vsplat_R(bits),Q6_V_vsplat_R(uint32_t(scale)));
        for(auto lane:actual) assert(lane==want);
      }
}
""")
    compile_and_run(source, tmp_path)
