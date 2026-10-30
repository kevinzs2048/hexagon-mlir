"""Opt-in LLVM regression for incompatible QFloat/IEEE call-site inlining."""
import os
from pathlib import Path
import re
import shlex
import subprocess

import pytest


def test_qfloat_calls_keep_their_target_mode_after_inlining(tmp_path):
    config, compiler = os.getenv("QWEN_LLVM_CONFIG"), os.getenv("QWEN_HOST_CXX")
    if not config or not compiler:
        pytest.skip("Select QWEN_LLVM_CONFIG and QWEN_HOST_CXX")
    root = Path(__file__).resolve().parents[3]
    flags = shlex.split(subprocess.check_output(
        [config, "--cxxflags", "--ldflags", "--libs", "core", "asmparser", "ipo", "--system-libs"], text=True))
    binary = tmp_path / "inlining"
    subprocess.run([compiler, "-O2", "-UNDEBUG", "-I" + str(root / "qcom_hexagon_backend/include"),
                    str(Path(__file__).with_suffix(".cpp")), *flags, "-UNDEBUG", "-o", str(binary)],
                   check=True, capture_output=True, text=True, timeout=180)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=30)
    source = tmp_path / "protected.ll"
    source.write_text(result.stdout)
    optimized = subprocess.run([str(Path(config).with_name("opt")), "-passes=default<O3>",
                                "-S", str(source), "-o", "-"],
                               check=True, capture_output=True, text=True, timeout=60).stdout
    caller = re.search(r"define float @ieee\(.*?^\}", optimized, re.M | re.S)
    assert caller and "call float @routine" in caller[0]
    assert "noinline" in optimized
