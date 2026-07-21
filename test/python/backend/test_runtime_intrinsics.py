import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_fast_f32_transcendentals_do_not_call_qhmath(tmp_path):
    hexagon_tools = Path(os.environ["HEXAGON_TOOLS"])
    hexagon_sdk = Path(os.environ["HEXAGON_SDK_ROOT"])
    hexkl_root = Path(os.environ["HEXKL_ROOT"])
    arch = os.environ["HEXAGON_ARCH_VERSION"]

    source = ROOT / "qcom_hexagon_backend/bin/runtime/IntrinsicsHVX.c"
    output = tmp_path / "IntrinsicsHVX.ll"
    compiler = hexagon_tools / "bin/hexagon-clang"

    cmd = [
        str(compiler),
        f"-I{hexagon_sdk}/incs",
        f"-I{hexagon_sdk}/incs/stddef",
        f"-I{hexagon_sdk}/libs/qhl_hvx/inc",
        f"-I{hexkl_root}/include",
        f"-I{hexagon_sdk}/rtos/qurt/computev{arch}/include/posix",
        f"-I{hexagon_sdk}/rtos/qurt/computev{arch}/include/qurt",
        f"-I{source.parent}/include",
        "-O2",
        "-Wall",
        f"-mv{arch}",
        "-mhvx",
        "-mhvx-qfloat",
        "-S",
        "-emit-llvm",
        "-fPIC",
        "-fvisibility=hidden",
        "-fvisibility-inlines-hidden",
        str(source),
        "-o",
        str(output),
    ]
    env = os.environ.copy()
    env["CPLUS_INCLUDE_PATH"] = ""
    subprocess.run(cmd, check=True, env=env)

    ir = output.read_text()
    assert "1050868099" in ir  # 0x3ea2f983, 1/pi range-reduction constant.
    assert "1078530011" in ir  # 0x40490fdb, pi range-reduction constant.
    assert "935697611" in ir  # llama fast cos polynomial coefficient.
    assert "909201517" in ir  # llama fast sin polynomial coefficient.
    assert "1597463007" in ir  # 0x5f3759df fast-rsqrt seed.
    assert "1023447158" in ir  # llama fast sigmoid/tanh coefficient.
    assert "1049457853" in ir  # llama fast sigmoid/tanh coefficient.
    assert "@c0_coeffs_tanh_vf" not in ir
    assert "llvm.hexagon.V6.vmpy" in ir
