"""Host semantic checks for the DSP libc shim (not device performance tests)."""
import ctypes
import math
from pathlib import Path
import random
import shutil
import struct
import subprocess

import pytest


@pytest.fixture(scope="module")
def compat(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    root = Path(__file__).resolve().parents[3]
    directory = tmp_path_factory.mktemp("libc-compat")
    wrapper = directory / "wrapper.cpp"
    wrapper.write_text(r'''
#include <cfenv>
#include <cstdint>
extern "C" float roundevenf(float);
extern "C" uint32_t check_roundeven(uint32_t bits, int mode, int *exceptions) {
  const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
  const int previous = fegetround();
  fesetround(modes[mode]);
  feclearexcept(FE_ALL_EXCEPT);
  union { uint32_t bits; float value; } input = {bits}, output;
  output.value = roundevenf(input.value);
  *exceptions = fetestexcept(FE_ALL_EXCEPT);
  fesetround(previous);
  return output.bits;
}
''')
    library = directory / "compat.so"
    subprocess.run([
        compiler, "-std=c++17", "-O2", "-fPIC", "-shared",
        "-frounding-math", "-fno-builtin", "-Wl,-Bsymbolic", str(wrapper),
        str(root / "qcom_hexagon_backend/bin/runtime/src/HexagonLibcCompat.cpp"),
        "-o", str(library),
    ], check=True, capture_output=True, text=True)
    dll = ctypes.CDLL(str(library))
    dll.check_roundeven.argtypes = [ctypes.c_uint32, ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_int)]
    dll.check_roundeven.restype = ctypes.c_uint32
    dll.__hexagon_mlir_expf.argtypes = [ctypes.c_float]
    dll.__hexagon_mlir_expf.restype = ctypes.c_float
    return dll


@pytest.mark.parametrize("value", [-math.inf, -104., -103., -100., -96., -90., -88.5,
                                  -88., -80., -1., -0., 0., 1., 80., 88., 88.5,
                                  88.7, 88.7228317, 88.7228394, 89., math.inf, math.nan])
def test_checked_scalar_exp_range_and_special_values(compat, value):
    value = ctypes.c_float(value).value
    actual = compat.__hexagon_mlir_expf(value)
    if math.isnan(value):
        assert math.isnan(actual)
    elif value >= 88.72283935546875:
        assert actual == math.inf
    else:
        expected = ctypes.c_float(math.exp(value)).value
        assert math.isclose(actual, expected, rel_tol=5e-6, abs_tol=2**-149)


def expected_bits(bits):
    magnitude = bits & 0x7fffffff
    if magnitude >= 0x4b000000:
        return bits  # Integral values, infinities and NaN payloads.
    value = struct.unpack("f", struct.pack("I", bits))[0]
    rounded = round(value)
    if rounded == 0:
        return bits & 0x80000000
    return struct.unpack("I", struct.pack("f", float(rounded)))[0]


@pytest.mark.parametrize("mode", range(4))
def test_roundeven_ties_boundaries_and_rounding_modes(compat, mode):
    # Adjacent representable values around ties, powers of two and subnormals.
    cases = [0, 1, 0x007fffff, 0x00800000, 0x7f800000, 0x7fc01234]
    for value in (0.5, 1.0, 1.5, 2.5, 3.5, 127.5, 128.5,
                  32767.5, 4194303.5, 8388607.5, 8388608.0):
        bits = struct.unpack("I", struct.pack("f", value))[0]
        cases.extend((bits - 1, bits, bits + 1))
    cases += [bits | 0x80000000 for bits in cases]
    random_generator = random.Random(20260910)
    cases += [random_generator.getrandbits(32) for _ in range(4096)]
    for bits in cases:
        exceptions = ctypes.c_int()
        actual = compat.check_roundeven(bits, mode, ctypes.byref(exceptions))
        assert actual == expected_bits(bits), f"mode={mode}, input=0x{bits:08x}"
        assert exceptions.value == 0, f"unexpected FP exception for 0x{bits:08x}"
