"""A single compiled ranked entry must accept different runtime lengths."""

from pathlib import Path
import resource
import shutil
import struct
import subprocess
from types import SimpleNamespace

import pytest
import torch

from qcom_hexagon_backend.backend import hexagon_dynamic_buffer_io as dynamic


ROOT = Path(__file__).resolve().parents[3]


def contract(max_query=8):
    return dynamic.make_contract(
        ["i64", "tensor<1x?xf32>"], ["tensor<1x?xf32>"],
        variables={"query": (1, max_query)},
        symbolic_dimensions={("input", 1, 1): "query", ("output", 0, 1): "query"},
        scalar_bindings={0: "query"},
    )


def wrapper():
    return SimpleNamespace(
        iterations=1, release_returned_buffers=True, func_name="kernel",
        input_profs=[SimpleNamespace(input_type="scalar", dtype="int64_t"),
                     SimpleNamespace(input_type="tensor", idx=0, rank=2, dtype="float")],
        output_profs=[SimpleNamespace(idx=0, rank=2, dtype="float")],
    )


@pytest.mark.parametrize("query", [1, 3, 8])
def test_one_contract_packs_distinct_runtime_shapes(query):
    values = struct.pack("<" + "f" * query, *range(query))
    request, layout = dynamic.pack_request(contract(), {"query": query}, [values])
    assert request[:8] == dynamic.MAGIC
    assert struct.unpack("<q", request[8:16]) == (query,)
    assert request[16:16 + len(values)] == values
    assert len(request) == 16 + layout["input_bytes"]
    assert layout["outputs"][0]["shape"] == [1, query]


@pytest.mark.parametrize("query", [1, 5, 8, 17, 32])
def test_torch_arguments_infer_query_and_restore_owned_output(query):
    source = torch.arange(query, dtype=torch.float32).reshape(1, query)
    c = contract(32)
    packet, layout, values = dynamic.pack_torch_request(c, [query, source])
    assert values == {"query": query}
    assert packet[16:16 + source.numel() * 4] == source.numpy().tobytes()
    response = bytearray(layout["output_bytes"])
    struct.pack_into("<" + "f" * query, response, 0, *range(query))
    output, = dynamic.unpack_torch_outputs(c, values, response)
    torch.testing.assert_close(output, source)
    response[:] = b"\x00" * len(response)
    torch.testing.assert_close(output, source)  # The caller owns the result.


@pytest.mark.parametrize("arguments, message", [
    ([4, torch.ones(1, 5)], "Inconsistent symbolic dimension"),
    ([9, torch.ones(1, 9)], "outside"),
    ([5, torch.ones(1, 5, dtype=torch.float64)], "compiled dtype"),
    ([5, torch.ones(1, 5, requires_grad=True)], "inference-only"),
    ([5, torch.ones(5, 1)], "compiled contract"),
    ([5, torch.ones(1, 10)[:, ::2]], "contiguous"),
    ([True, torch.ones(1, 1)], "concrete Python integer"),
])
def test_torch_request_rejects_unusable_runtime_shapes(arguments, message):
    with pytest.raises(ValueError, match=message):
        dynamic.pack_torch_request(contract(), arguments)


def test_contract_rejects_unbound_shape_and_out_of_range_request():
    with pytest.raises(ValueError, match="Unbound symbolic"):
        dynamic.make_contract(["tensor<?xf32>"], ["tensor<?xf32>"],
                              variables={"query": (1, 8)},
                              symbolic_dimensions={("input", 0, 0): "query"},
                              scalar_bindings={})
    with pytest.raises(ValueError, match="outside"):
        dynamic.layout_for(contract(), {"query": 9})


def test_signature_rejects_static_relabeling():
    with pytest.raises(ValueError, match="fixed tensor"):
        dynamic.make_contract(["tensor<1x8xf32>"], ["tensor<1x8xf32>"],
                              variables={"query": (1, 8)},
                              symbolic_dimensions={("input", 0, 1): "query"},
                              scalar_bindings={})


def test_qwen_signature_contract_has_same_library_for_q5_and_q17():
    # The actual lifted Qwen signature keeps both query and KV capacity
    # symbolic. This caller varies query while binding capacity to 1024.
    kv = "tensor<1x8x?x128xf32>"
    types = (["i64", "tensor<1x?xi64>", "tensor<?xi64>",
              "tensor<1x?xi64>", "i64", kv, kv,
              "tensor<1x1x?x?xf32>"] + [kv] * 54 + ["i64"])
    outputs = [kv] * 56 + ["tensor<1x1x151936xf32>"]
    assert len(types) == 63
    symbolic = {("input", 1, 1): "query", ("input", 2, 0): "query",
                ("input", 3, 1): "query", ("input", 7, 2): "query",
                ("input", 7, 3): 1024}
    symbolic.update({("input", i, 2): 1024 for i in range(5, 62)
                     if i != 7})
    symbolic.update({("output", i, 2): 1024 for i in range(56)})
    c = dynamic.make_contract(
        types, outputs, variables={"query": (1, 32)},
        symbolic_dimensions=symbolic,
        scalar_bindings={0: "query", 4: 1024, 62: 1},
    )
    q5 = dynamic.layout_for(c, {"query": 5})
    q17 = dynamic.layout_for(c, {"query": 17})
    assert q5["input_bytes"] != q17["input_bytes"]
    assert q5["output_bytes"] == q17["output_bytes"]
    assert c["schema"] == 1 and c["variables"] == {"query": (1, 32)}
    profiles = []
    tensor_index = 0
    for typename in types:
        if typename == "i64":
            profiles.append(SimpleNamespace(input_type="scalar", dtype="int64_t"))
        else:
            profiles.append(SimpleNamespace(input_type="tensor", idx=tensor_index,
                                            rank=typename.count("x"),
                                            dtype="int64_t" if typename.endswith("xi64>") else "float"))
            tensor_index += 1
    output_profiles = [SimpleNamespace(idx=i, rank=typename.count("x"), dtype="float")
                       for i, typename in enumerate(outputs)]
    source = dynamic.generate_buffer_entry(
        SimpleNamespace(iterations=1, release_returned_buffers=True,
                        func_name="qwen_dynamic_model", input_profs=profiles,
                        output_profs=output_profiles), c)
    assert "qwen_dynamic_model(r, query, dt0, dt1, dt2, 1024," in source
    assert "r->r56" in source and "in_offset_59" in source
    assert "query < 1 || query > 32" in source
    assert "query < 17" not in source and "query > 17" not in source


@pytest.fixture(scope="module", params=[False, True], ids=['one-shape-scalar', 'repeated-shape-scalars'])
def executable(tmp_path_factory, request):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("Host C++ compiler unavailable")
    root = tmp_path_factory.mktemp("dynamic-buffer-entry")
    before = r'''
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
enum class MemType {HEAP};
template<class T,int N> struct MemRefDescriptor {
  T *allocated,*aligned; int64_t offset,sizes[N],strides[N];
};
struct Shape {std::initializer_list<int64_t> sizes,strides; int alignment;};
template<class T,int N> class Tensor {
  MemRefDescriptor<T,N> d{};
public:
  Tensor(Shape shape,MemType) {
    int i=0; for(auto n:shape.sizes) d.sizes[i++]=n;
    i=0; for(auto n:shape.strides) d.strides[i++]=n;
    int64_t count=1; for(auto n:shape.sizes) count*=n;
    d.allocated=d.aligned=(T*)std::malloc(sizeof(T)*count);
  }
  auto *toMemRefDesc(){return &d;}
  ~Tensor(){std::free(d.allocated);}
};
struct FuncResult {MemRefDescriptor<float,2> r0;};
static int calls=0;
extern "C" void kernel(void *ret,int64_t query,MemRefDescriptor<float,2>*input) {
  ++calls;auto*r=(FuncResult*)ret;
  r->r0.allocated=r->r0.aligned=(float*)std::malloc(sizeof(float)*query);
  r->r0.sizes[0]=1;r->r0.sizes[1]=query;
  r->r0.strides[0]=query;r->r0.strides[1]=1;
  for(int64_t i=0;i<query;++i)r->r0.aligned[i]=input->aligned[i]*2+1;
}
template<class F> uint64_t benchmark_time_us(int,F f){f();return 123;}
'''
    after = r'''
int main(int argc,char**argv) {
  if(argc!=2)return 10;
  int64_t query=std::atoi(argv[1]);
  unsigned char input[144]={},output[128]={};
  std::memcpy(input,"HXDY\x01\x00\x00\x00",8);
  std::memcpy(input+8,&query,8);
  for(int64_t i=0;i<query && i<32;++i){float v=float(i);std::memcpy(input+16+4*i,&v,4);}
  uint64_t us=0;
  int status=hexagon_forward_dynamic_buffers(input,144,output,128,&us);
  if(query==33)return status==2 && calls==0?0:11;
  if(status!=0 || calls!=1 || us!=123)return 12;
  for(int64_t i=0;i<query;++i){float v;std::memcpy(&v,output+4*i,4);
    if(v!=float(2*i+1))return 13;}
  return 0;
}
'''
    actual_wrapper, actual_contract = wrapper(), contract(32)
    if request.param:
        # AOT can emit several shape arguments for equal tensor dimensions.
        # Exercise the real compiled C argument order, not only source strings.
        before = before.replace('MemRefDescriptor<float,2>*input) {',
            'MemRefDescriptor<float,2>*input,int64_t q2,int64_t capacity,int64_t q3) {\n'
            '  if(q2!=query || q3!=query || capacity!=1024)std::abort();')
        actual_wrapper.input_profs.extend([
            SimpleNamespace(input_type='scalar', dtype='int64_t') for _ in range(3)])
        actual_contract = dynamic.make_contract(
            ['i64', 'tensor<1x?xf32>', 'i64', 'i64', 'i64'], ['tensor<1x?xf32>'],
            variables={'query': (1, 32)},
            symbolic_dimensions={('input', 1, 1): 'query', ('output', 0, 1): 'query'},
            scalar_bindings={0: 'query', 2: 'query', 3: 1024, 4: 'query'})
    source = root / "entry.cpp"
    source.write_text(before + dynamic.generate_buffer_entry(actual_wrapper, actual_contract) + after)
    build = subprocess.run([compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined", "-I",
                            str(ROOT / "test/utils/dsp/include"), str(source),
                            "-o", str(root / "entry")], capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    return root / "entry"


@pytest.mark.parametrize("query", [1, 5, 17, 32, 33])
def test_one_generated_cpp_entry_runs_multiple_lengths(executable, query):
    result = subprocess.run([str(executable), str(query)], capture_output=True,
                            text=True, timeout=10,
                            preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
    assert result.returncode == 0, result.stderr
