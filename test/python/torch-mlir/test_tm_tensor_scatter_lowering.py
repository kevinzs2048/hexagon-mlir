"""Verify indexed writes and lower them to ordered loops with the actual pass."""
import os
import subprocess

import pytest


def scatter_source(updates="tensor<3xi32>", indices="tensor<3x2xi64>",
                   original="tensor<3x4xi32>", dimensions=(0, 1),
                   element="i32", body=None, result=None, unique=False):
    result = result or original
    body = body or f"tm_tensor.yield %u : {element}"
    mapping = ", ".join(map(str, dimensions))
    attr = "array<i64" + (": " + mapping if mapping else "") + ">"
    return f"""module {{
      func.func @Scatter(%updates: {updates}, %indices: {indices}, %original: {original}) -> {result} {{
        %r = tm_tensor.scatter {{dimension_map = {attr}}} unique_indices({str(unique).lower()})
          ins(%updates, %indices : {updates}, {indices}) outs(%original : {original}) {{
          ^bb0(%u: {element}, %old: {element}): {body}
        }} -> {result}
        return %r : {result}
      }}
    }}"""


VALID = [
    {},
    dict(dimensions=(1, 0)),
    dict(updates="tensor<3x2xi32>", indices="tensor<3x1xi32>", dimensions=(0,)),
    dict(updates="tensor<3x1x2xi32>", dimensions=(1, 0)),
    dict(updates="tensor<3x1x2xi32>", indices="tensor<3x0xi64>", dimensions=()),
    dict(updates="tensor<0xi32>", indices="tensor<0x2xi64>"),
    dict(body="%s = arith.addi %u, %old : i32 tm_tensor.yield %s : i32"),
    dict(updates="tensor<3xbf16>", original="tensor<3x4xbf16>", element="bf16", unique=True),
    dict(updates="tensor<3xf32>", original="tensor<3x4xf32>", element="f32"),
]


def compile_source(tmp_path, source):
    optimizer = os.environ.get("HEXAGON_OPT")
    if not optimizer:
        pytest.fail("Set HEXAGON_OPT to the built linalg-hexagon-opt")
    path = tmp_path / "scatter.mlir"
    path.write_text(source)
    return subprocess.run([
        optimizer, str(path),
        "--pass-pipeline=builtin.module(func.func(hexagon-lower-tm-tensor))",
        "--verify-each",
    ], capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("case", VALID)
def test_scatter_lowers_to_ordered_loops(tmp_path, case):
    process = compile_source(tmp_path, scatter_source(**case))
    assert process.returncode == 0, process.stdout + process.stderr
    assert "tm_tensor.scatter" not in process.stdout
    assert "scf.for" in process.stdout
    assert "tensor.insert" in process.stdout


@pytest.mark.parametrize("case,message", [
    (dict(indices="tensor<3x2xf32>"), "indices must be rank-two integers"),
    (dict(indices="tensor<3x?xi64>"), "static index depth"),
    (dict(indices="tensor<2x2xi64>"), "same number of rows"),
    (dict(dimensions=(0,)), "dimension map must match index depth"),
    (dict(dimensions=(0, 0)), "distinct valid dimensions"),
    (dict(dimensions=(0, 2)), "distinct valid dimensions"),
    (dict(dimensions=(-1, 0)), "distinct valid dimensions"),
    (dict(updates="tensor<3x2xi32>", indices="tensor<3x1xi64>", dimensions=(1,)), "cover the original rank"),
    (dict(updates="tensor<3x5xi32>", indices="tensor<3x1xi64>", dimensions=(0,)), "slice exceeds"),
    (dict(updates="tensor<3x1x1x1xi32>"), "invalid updates or original rank"),
    (dict(result="tensor<4x3xi32>"), "original/result and scalar update types must match"),
    (dict(body="tm_tensor.yield %u, %old : i32, i32"), "yield one scalar"),
])
def test_invalid_scatter_is_rejected_before_codegen(tmp_path, case, message):
    process = compile_source(tmp_path, scatter_source(**case))
    assert process.returncode != 0
    assert message in process.stderr, process.stderr
