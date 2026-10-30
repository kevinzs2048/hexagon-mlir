"""Host validation of the DSP runtime's MLIR descriptor ABI and copy layout."""
import ctypes as C
import itertools
from pathlib import Path
import shutil
import subprocess

import pytest


class Unranked(C.Structure):
    _fields_ = [("rank", C.c_int64), ("descriptor", C.c_void_p)]


@pytest.fixture(scope="module", params=["backend", "fallback", "both"])
def memref_copy(tmp_path_factory, request):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    root = Path(__file__).resolve().parents[3]
    library = tmp_path_factory.mktemp("memref-copy") / "copy.so"
    backend = root / "qcom_hexagon_backend/bin/runtime/src/HexagonMemrefCopy.cpp"
    fallback = root / "test/utils/dsp/include/CRunnerUtils.cpp"
    sources = {"backend": [backend], "fallback": [fallback], "both": [fallback, backend]}[request.param]
    subprocess.run([
        compiler, "-std=c++17", "-O2", "-shared", "-fPIC",
        *map(str, sources),
        "-o", str(library),
    ], check=True, capture_output=True, text=True)
    if shutil.which("nm"):
        symbols = subprocess.check_output(["nm", "-D", str(library)], text=True)
        binding = "W" if request.param == "fallback" else "T"
        assert f" {binding} memrefCopy" in symbols
    dll = C.CDLL(str(library))
    dll.memrefCopy.argtypes = [C.c_int64, C.POINTER(Unranked), C.POINTER(Unranked)]
    dll.memrefCopy.restype = None
    return dll.memrefCopy


@pytest.mark.parametrize("element_size", [1, 2, 4, 8, 13])
@pytest.mark.parametrize("shape,src_strides,dst_strides,src_offset,dst_offset", [
    ((), (), (), 2, 3),
    ((8,), (1,), (1,), 2, 3),
    ((2, 3), (3, 1), (3, 1), 0, 0),
    ((2, 3), (1, 2), (3, 1), 1, 2),  # transposed source
    ((2, 3), (3, 1), (1, 2), 1, 2),  # transposed destination
    ((2, 3), (8, 2), (9, 2), 1, 2),  # padded and strided
    ((2, 3), (3, -1), (3, 1), 2, 0),
    ((2, 3), (3, 1), (3, -1), 0, 2),
    ((2, 3), (0, 1), (3, 1), 0, 0),  # broadcast source
    ((2, 1, 2, 3), (12, 6, 3, 1), (15, 6, 3, 1), 0, 1),
    ((2, 0, 3), (3, 3, 1), (3, 3, 1), 0, 0),
])
def test_copy_shapes_offsets_and_strides(memref_copy, element_size, shape,
                                        src_strides, dst_strides, src_offset, dst_offset):
    rank = len(shape)
    class Ranked(C.Structure):
        _fields_ = [("allocated", C.c_void_p), ("aligned", C.c_void_p),
                    ("offset", C.c_int64), ("sizes", C.c_int64 * rank),
                    ("strides", C.c_int64 * rank)]
    nbytes = 128 * element_size
    source = (C.c_ubyte * nbytes)(*[i % 251 for i in range(nbytes)])
    destination = (C.c_ubyte * nbytes)(*[0xa5] * nbytes)
    expected = bytearray(destination)
    for coordinates in itertools.product(*(range(size) for size in shape)):
        read = (src_offset + sum(i * s for i, s in zip(coordinates, src_strides))) * element_size
        write = (dst_offset + sum(i * s for i, s in zip(coordinates, dst_strides))) * element_size
        expected[write:write+element_size] = bytes(source)[read:read+element_size]
    src = Ranked(C.addressof(source), C.addressof(source), src_offset,
                 (C.c_int64 * rank)(*shape), (C.c_int64 * rank)(*src_strides))
    dst = Ranked(C.addressof(destination), C.addressof(destination), dst_offset,
                 (C.c_int64 * rank)(*shape), (C.c_int64 * rank)(*dst_strides))
    memref_copy(element_size, C.byref(Unranked(rank, C.addressof(src))),
                C.byref(Unranked(rank, C.addressof(dst))))
    assert bytes(destination) == expected
