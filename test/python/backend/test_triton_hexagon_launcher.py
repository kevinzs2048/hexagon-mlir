from collections import namedtuple

from triton.backends.qcom_hexagon_backend.hexagon_options import HexagonOptions
from triton.backends.qcom_hexagon_backend.triton_hexagon_launcher import (
    TritonHexagonWrapperGenerator,
)


InputProfile = namedtuple(
    "InputProfile",
    ("idx", "input_id", "input_type", "value", "dtype", "rank", "shape"),
)


def _tensor(idx):
    return InputProfile(
        idx=idx,
        input_id=idx,
        input_type="tensor",
        value=None,
        dtype="float",
        rank=1,
        shape=("{131072}", "{1}"),
    )


def test_single_thread_triton_wrapper_uses_unranked_memref_descriptor_abi():
    generator = TritonHexagonWrapperGenerator(
        [_tensor(0), _tensor(1), _tensor(2)],
        iterations=10,
        func_name="add_kernel",
        output_profs=[],
        launch_grid=(1, 1, 1),
        options=HexagonOptions().__dict__,
    )

    wrapper = generator.generate_cpp_wrapper("add_kernel", "/tmp")

    assert (
        "extern \"C\" void add_kernel(int64_t, MemRefDescriptor<float, 1> *, "
        "int64_t, MemRefDescriptor<float, 1> *, "
        "int64_t, MemRefDescriptor<float, 1> *, int, int, int, int, int, int);"
        in wrapper
    )
    assert "add_kernel(1, dt0, 1, dt1, 1, dt2, 1, 1, 1, 0, 0, 0);" in wrapper
    assert "add_kernel(dt0->allocated" not in wrapper
