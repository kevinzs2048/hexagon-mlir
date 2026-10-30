"""Structural regression tests for the Hexagon matmul scheduling pass.

Use an explicit isolated optimizer; no board or host-specific input artifacts.
"""
import os
from pathlib import Path
import re
import subprocess

import pytest


def run(text, *extra, schedule=True, optimizer_env="QWEN_MATMUL_SCHEDULE_OPT"):
    configured = os.environ.get(optimizer_env) or os.environ.get("QWEN_MATMUL_SCHEDULE_OPT")
    if not configured:
        pytest.skip("Set QWEN_MATMUL_SCHEDULE_OPT to the candidate optimizer")
    executable = Path(configured).resolve(strict=True)
    flags = ["--pass-pipeline=builtin.module(func.func(schedule-matmul-for-hvx))"] if schedule else []
    result = subprocess.run(
        [str(executable), "-", *flags, "--verify-each", *extra],
        input=text, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def module(batch=False, operand="b", shared=False, permutation=None,
           buffer=False, library=False):
    rank = 3 if batch else 2
    shape = "2x" * rank + "f32"
    ty = f"{'memref' if buffer else 'tensor'}<{shape}>"
    name = "batch_matmul" if batch else "matmul"
    if permutation is None:
        permutation = [0, 2, 1] if batch else [1, 0]
    perm = ", ".join(str(v) for v in permutation)
    producer = f"linalg.transpose ins(%{operand} : {ty}) outs(%tmp : {ty}) permutation = [{perm}]"
    inputs = ("%t", "%b") if operand == "a" else ("%a", "%t")
    attr = ' {library_call = "external_matmul"}' if library else ""
    if buffer:
        inputs = ("%tmp", "%b") if operand == "a" else ("%a", "%tmp")
        body = f"{producer}\n linalg.{name}{attr} ins({inputs[0]}, {inputs[1]} : {ty}, {ty}) outs(%init : {ty})\n return"
        signature = ""
    else:
        results = f"({ty}, {ty})" if shared else ty
        signature = f" -> {results}"
        ret = f"return %result, %t : {ty}, {ty}" if shared else f"return %result : {ty}"
        body = f"%t = {producer}\n %result = linalg.{name}{attr} ins({inputs[0]}, {inputs[1]} : {ty}, {ty}) outs(%init : {ty}) -> {ty}\n {ret}"
    return f"func.func @probe(%a: {ty}, %b: {ty}, %tmp: {ty}, %init: {ty}){signature} {{\n {body}\n}}"


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("operand", ["a", "b"])
def test_matrix_transpose_folds(batch, operand):
    output = run(module(batch=batch, operand=operand))
    assert "linalg.generic" in output
    assert re.search(r"ins\(%arg0, %arg1\s*:", output)
    assert "linalg.transpose " not in output


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("operand", ["a", "b"])
def test_shared_transpose_producer_is_preserved(batch, operand):
    output = run(module(batch=batch, operand=operand, shared=True))
    assert "linalg.transpose " in output
    assert "linalg.generic" in output
    assert re.search(r"ins\(%arg0, %arg1\s*:", output)


@pytest.mark.parametrize("batch,permutation", [(False, [0, 1]),
    (True, [0, 1, 2]), (True, [1, 0, 2]), (True, [2, 1, 0])])
@pytest.mark.parametrize("operand", ["a", "b"])
def test_non_matrix_transpose_is_not_folded(batch, permutation, operand):
    # Equal-sized axes ensure these are valid inputs, not rejected shape errors.
    output = run(module(batch=batch, operand=operand, permutation=permutation))
    assert "linalg.transpose " in output
    assert "matmul_transpose" not in output
    assert "linalg.generic" in output


@pytest.mark.parametrize("batch", [False, True])
def test_library_call_is_untouched(batch):
    output = run(module(batch=batch, library=True))
    assert 'library_call = "external_matmul"' in output
    assert "linalg.transpose " in output
    assert "matmul_transpose" not in output


@pytest.mark.parametrize("batch", [False, True])
def test_buffer_transpose_write_is_not_removed(batch):
    output = run(module(batch=batch, buffer=True))
    assert "linalg.transpose " in output
    assert "matmul_transpose" not in output


@pytest.mark.parametrize("batch", [False, True])
def test_noncanonical_matmul_maps_are_preserved(batch):
    if batch:
        maps = ["(b,m,n,k)->(b,k,m)", "(b,m,n,k)->(b,k,n)", "(b,m,n,k)->(b,m,n)"]
    else:
        maps = ["(m,n,k)->(k,m)", "(m,n,k)->(k,n)", "(m,n,k)->(m,n)"]
    name = "linalg.batch_matmul" if batch else "linalg.matmul"
    attributes = " indexing_maps = [" + ", ".join(f"affine_map<{m}>" for m in maps) + "]"
    text = module(batch=batch, operand="a").replace(name+" ins", name+attributes+" ins")
    output = run(text)
    producer = re.search(r"(%[\w]+) = linalg.transpose ", output)
    assert producer, output
    assert re.search(r"ins\("+re.escape(producer[1])+r", %arg1\s*:", output), output


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("operand", ["a", "b"])
def test_unsigned_input_widening_is_preserved(batch, operand):
    shape = "2x2x2" if batch else "2x2"
    inp, acc = f"tensor<{shape}xi8>", f"tensor<{shape}xi32>"
    perm = "[0, 2, 1]" if batch else "[1, 0]"
    name = "batch_matmul" if batch else "matmul"
    lhs, rhs = ("%t", "%b") if operand == "a" else ("%a", "%t")
    text = f"""func.func @probe(%a: {inp}, %b: {inp}, %tmp: {inp}, %init: {acc}) -> {acc} {{
      %t = linalg.transpose ins(%{operand} : {inp}) outs(%tmp : {inp}) permutation = {perm}
      %r = linalg.{name} {{cast = #linalg.type_fn<cast_unsigned>}} ins({lhs}, {rhs} : {inp}, {inp}) outs(%init : {acc}) -> {acc}
      return %r : {acc}
    }}"""
    output = run(text)
    assert output.count("arith.extui") == 2
    assert "arith.extsi" not in output
    assert "linalg.transpose " not in output


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("operand", ["a", "b"])
def test_arithmetic_attributes_are_not_dropped(batch, operand):
    generic = run(module(batch=batch, operand=operand), "--mlir-print-op-generic", schedule=False)
    assert generic.count('"arith.mulf"') == 1
    custom = generic.replace('#arith.fastmath<none>', '#arith.fastmath<contract>', 1)
    output = run(custom)
    assert "fastmath<contract>" in output
