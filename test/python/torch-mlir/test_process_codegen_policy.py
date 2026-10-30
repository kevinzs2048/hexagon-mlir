"""Exercise the real extension in fresh workers, including LLVM's pass pipeline.

Set HEXAGON_TEST_LIBTRITON to compiler/libtriton.so from a source build.
The q1/q2 specializations require complete model IR and cannot use this fixture.
"""
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
PREAMBLE = r'''
import importlib.util
import os
import sys
from pathlib import Path

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

extension = load("triton._C.libtriton", os.environ["HEXAGON_TEST_LIBTRITON"])
backend = extension.qcom_hexagon_backend
options = load("worker_options", Path(os.environ["HEXAGON_TEST_SOURCE"]) /
               "qcom_hexagon_backend/backend/hexagon_options.py").HexagonOptions()
options = {key: str(value) for key, value in options.__dict__.items()}
context = extension.ir.context()
backend.load_dialects(context)

def module():
    return backend.parse_mlir_module_from_str(
        "module { func.func @identity(%x: f32) -> f32 { return %x : f32 } }",
        context)
'''


def worker(body):
    path = os.environ.get("HEXAGON_TEST_LIBTRITON")
    if not path:
        pytest.skip("Set HEXAGON_TEST_LIBTRITON to the source-built extension")
    result = subprocess.run(
        [sys.executable, "-c", PREAMBLE + body], capture_output=True, text=True,
        timeout=60, env=dict(os.environ, HEXAGON_TEST_SOURCE=str(ROOT),
                             LLVM_IR_LIST_PASSES="true"))
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("enabled", [True, False])
def test_official_option_changes_actual_codegen_pipeline(enabled):
    result = worker(f'''
backend.configure_process_geninsert({enabled})
backend.configure_process_geninsert({enabled})
text = backend.translate_linalg_to_llvmir(module(), options)
assert '"hexagon-no-gen-insert"' not in text
objects = backend.translate_linalg_to_obj(module(), options)
assert objects and bytes(objects[0]).startswith(b"\\x7fELF")
try:
    backend.configure_process_geninsert({enabled})
except ValueError as error:
    assert "before translation" in str(error)
else:
    raise AssertionError("Late configuration was accepted")
''')
    assert ('Hexagon generate "insert" instructions' in result.stderr) == enabled


def test_conflicting_worker_policy_is_rejected():
    worker('''
backend.configure_process_geninsert(False)
try:
    backend.configure_process_geninsert(True)
except ValueError as error:
    assert "fresh process" in str(error)
else:
    raise AssertionError("Conflicting policy was accepted")
''')


def test_translation_before_configuration_is_rejected():
    worker('''
backend.translate_linalg_to_llvmir(module(), options)
try:
    backend.configure_process_geninsert(False)
except ValueError as error:
    assert "before translation" in str(error)
else:
    raise AssertionError("Late first configuration was accepted")
''')


def test_process_and_function_policies_cannot_be_mixed():
    worker('''
backend.configure_process_geninsert(False)
options["enableHexagonGenInsert"] = "False"
try:
    backend.translate_linalg_to_llvmir(module(), options)
except ValueError as error:
    assert "cannot be combined" in str(error)
else:
    raise AssertionError("Mixed policy was accepted")
''')
