import dataclasses
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

from triton.backends.qcom_hexagon_backend import compiler
from triton.backends.qcom_hexagon_backend.hexagon_options import HexagonOptions


@contextmanager
def temporary_env(**updates):
    old_values = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def compile_inline_hmx_q8_object(tmp_path, *, crouton=False):
    mlir = """
module {
  func.func @compiler_mm(%A: memref<?x?xf16>, %B: memref<?x?xf16>,
                         %out: memref<?x?xf32>, %Wq8: memref<?x?xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dynM = memref.dim %A, %c0 : memref<?x?xf16>
    %dynN = memref.dim %B, %c1 : memref<?x?xf16>
    %At = bufferization.to_tensor %A restrict : memref<?x?xf16> to tensor<?x?xf16>
    %Bt = bufferization.to_tensor %B restrict : memref<?x?xf16> to tensor<?x?xf16>
    %e  = tensor.empty(%dynM, %dynN) : tensor<?x?xf32>
    %r  = linalg.matmul ins(%At, %Bt : tensor<?x?xf16>, tensor<?x?xf16>)
                        outs(%e : tensor<?x?xf32>) -> tensor<?x?xf32>
    bufferization.materialize_in_destination %r in restrict writable %out
        : (tensor<?x?xf32>, memref<?x?xf32>) -> ()
    return
  }
}
"""
    opts = dataclasses.replace(
        HexagonOptions(),
        enableHexKL=True,
        enableInlineHMX=True,
        enableQ8Weight=True,
        enableBufferization=True,
    )

    env = {"HEXKL_WQ8_CROUTON": "1"} if crouton else {"HEXKL_WQ8_CROUTON": None}
    with temporary_env(**env):
        obj = compiler.ttsharedir_to_obj(mlir, opts, {})

    obj_path = tmp_path / ("compiler_mm_crouton.o" if crouton else "compiler_mm.o")
    obj_path.write_bytes(obj)

    return subprocess.check_output(
        [
            str(Path(os.environ["HEXAGON_TOOLS"]) / "bin/hexagon-llvm-objdump"),
            "-d",
            str(obj_path),
        ],
        text=True,
    )


def assert_inline_hmx_without_tile_calls(dump):
    assert "hexkl_micro_hmx" not in dump
    assert "call hexkl_micro" not in dump
    assert any(token in dump.lower() for token in ("mxmem", "mxclracc", "mxcvt"))


def test_inline_hmx_q8_object_has_hmx_instructions_and_no_tile_calls(tmp_path):
    dump = compile_inline_hmx_q8_object(tmp_path)
    assert_inline_hmx_without_tile_calls(dump)


def test_inline_hmx_crouton_q8_object_has_hmx_instructions_and_no_tile_calls(tmp_path):
    dump = compile_inline_hmx_q8_object(tmp_path, crouton=True)
    assert_inline_hmx_without_tile_calls(dump)
