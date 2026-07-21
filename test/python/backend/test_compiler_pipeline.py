from triton.backends.qcom_hexagon_backend import compiler
from triton.backends.qcom_hexagon_backend.hexagon_options import HexagonOptions


def test_ttsharedir_to_obj_uses_mlir_pipeline_for_elementwise_kernel(monkeypatch):
    calls = []

    class FakeContext:
        pass

    class FakeIr:
        @staticmethod
        def context():
            return FakeContext()

    class FakeQcomBackend:
        @staticmethod
        def load_dialects(context):
            calls.append(("load_dialects", context))

        @staticmethod
        def parse_mlir_module_from_str(mod, context):
            calls.append(("parse", mod, context))
            return "parsed-module"

        @staticmethod
        def extract_func_name_from_mlir_module(module):
            calls.append(("name", module))
            return "add_f32"

        @staticmethod
        def get_return_list(module, name):
            calls.append(("returns", module, name))
            return []

        @staticmethod
        def translate_linalg_to_obj(module, options_map):
            calls.append(("translate_linalg_to_obj", module, options_map))
            return [b"mlir-object"]

    monkeypatch.setattr(compiler, "ir", FakeIr)
    monkeypatch.setattr(compiler, "qcom_hexagon_backend", FakeQcomBackend)
    monkeypatch.setattr(compiler, "parse_return_types", lambda values: [])
    if hasattr(compiler, "_compile_streaming_elementwise_obj"):
        monkeypatch.setattr(
            compiler,
            "_compile_streaming_elementwise_obj",
            lambda name: b"name-matched-object",
        )

    result = compiler.ttsharedir_to_obj("module {}", HexagonOptions(), {})

    assert result == b"mlir-object"
    assert any(call[0] == "translate_linalg_to_obj" for call in calls)
