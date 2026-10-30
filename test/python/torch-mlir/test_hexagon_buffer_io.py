"""Synthetic host buffer ABI tests; no model or DSP execution."""
import copy
import importlib.util
from pathlib import Path
import resource
import shutil
import subprocess
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('buffer_io_test',ROOT/'qcom_hexagon_backend/backend/hexagon_buffer_io.py')
buffer_io=importlib.util.module_from_spec(spec);spec.loader.exec_module(buffer_io)


def small_layout():
    return buffer_io.make_layout([dict(dtype='float32',shape=[4])],[dict(dtype='float32',shape=[4])])


def test_full_model_layout_sizes_and_roundtrip():
    inputs=[dict(dtype='int64',shape=[1,1]),dict(dtype='int64',shape=[1])]+[dict(dtype='float32',shape=[1,8,32,128])]*56
    outputs=[dict(dtype='float32',shape=[1,1,151936])]+[dict(dtype='float32',shape=[1,8,32,128])]*56
    layout=buffer_io.make_layout(inputs,outputs)
    assert layout['input_bytes']==7340288 and layout['output_bytes']==7947776
    payloads=[bytes([i])*row['bytes'] for i,row in enumerate(layout['inputs'])]
    packed=buffer_io.pack_inputs(layout,payloads)
    assert all(packed[r['offset']:r['offset']+r['bytes']]==p for r,p in zip(layout['inputs'],payloads))
    unpacked=buffer_io.unpack_outputs(layout,bytes(layout['output_bytes']))
    assert [len(p) for p in unpacked]==[r['bytes'] for r in layout['outputs']]


@pytest.mark.parametrize('fault',['overlap','stride','total','dtype','extra','shape'])
def test_noncanonical_layout_rejected(fault):
    layout=small_layout()
    if fault=='overlap':layout['inputs'][0]['offset']=1
    elif fault=='stride':layout['inputs'][0]['strides']=[2]
    elif fault=='total':layout['output_bytes']-=1
    elif fault=='dtype':layout['inputs'][0]['dtype']='BF16'
    elif fault=='extra':layout['extra']=True
    else:layout['inputs'][0]['shape']=[-1]
    with pytest.raises(ValueError):buffer_io.validate_layout(layout)


def test_payload_lengths_are_exact():
    layout=small_layout()
    with pytest.raises(ValueError):buffer_io.pack_inputs(layout,[bytes(15)])
    with pytest.raises(ValueError):buffer_io.pack_inputs(layout,[])
    with pytest.raises(ValueError):buffer_io.unpack_outputs(layout,bytes(127))


def wrapper():
    return SimpleNamespace(iterations=1,release_returned_buffers=True,func_name='kernel',
        input_profs=[SimpleNamespace(input_type='tensor',idx=0,dtype='float',rank=1,value=SimpleNamespace(shape=(4,)))],
        output_profs=[SimpleNamespace(idx=0,dtype='float',rank=1)],
        generate_llvm_function_call=lambda:'kernel(r, dt0);\n')


@pytest.mark.parametrize('fault',['ownership','iterations','input_shape','output_dtype','symbol'])
def test_generator_rejects_wrong_abi(fault):
    instance=wrapper();kw={}
    if fault=='ownership':instance.release_returned_buffers=False
    elif fault=='iterations':instance.iterations=2
    elif fault=='input_shape':instance.input_profs[0].value.shape=(5,)
    elif fault=='output_dtype':instance.output_profs[0].dtype='int64_t'
    else:kw['symbol']='bad;symbol'
    with pytest.raises(ValueError):buffer_io.generate_buffer_entry(instance,small_layout(),**kw)


@pytest.mark.parametrize('mapping',[[],{True:0},{-1:0},{1:0},{0:1},{0:False}])
def test_invalid_state_mapping_rejected(mapping):
    with pytest.raises((ValueError,TypeError)):
        buffer_io.generate_buffer_entry(wrapper(),small_layout(),validate_input_copy=True,input_update_outputs=mapping)


def test_update_requires_diagnostics_and_exact_tensor_shape():
    with pytest.raises(ValueError,match='diagnostics'):
        buffer_io.generate_buffer_entry(wrapper(),small_layout(),input_update_outputs={0:0})
    layout=buffer_io.make_layout([dict(dtype='float32',shape=[4])],[dict(dtype='float32',shape=[2,2])])
    with pytest.raises(ValueError,match='complete tensor ABI'):
        buffer_io.generate_buffer_entry(wrapper(),layout,validate_input_copy=True,input_update_outputs={0:0})


def test_state_update_check_keeps_input_copy_and_output_validation():
    text=buffer_io.generate_buffer_entry(wrapper(),small_layout(),validate_input_copy=True,input_update_outputs={0:0})
    assert 'return 6;' in text and 'return 8;' in text
    assert text.index('return 4;') < text.index('return 8;') < text.index('std::memset(output')
    assert 'input + 0, 16) &&' in text
    assert 'r->r0.aligned), 16)' in text


@pytest.fixture(scope='module',params=[False,True,'state'],ids=['default-copy','checked-copy','declared-update'])
def executable(tmp_path_factory,request):
    root=tmp_path_factory.mktemp('buffer-entry-host');compiler=shutil.which('c++')
    if not compiler:pytest.skip('No host C++ compiler')
    before=r'''
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <cassert>
#include <cstring>
enum class MemType {HEAP};
template<class T,int N> struct MemRefDescriptor {
 T *allocated,*aligned; int64_t offset,sizes[N],strides[N];
};
struct Shape {std::initializer_list<int64_t> sizes,strides; int alignment;};
template<class T,int N> class Tensor {
 MemRefDescriptor<T,N> d{};
public:
 Tensor(Shape shape,MemType) {
   int i=0;for(auto n:shape.sizes)d.sizes[i++]=n;
   i=0;for(auto n:shape.strides)d.strides[i++]=n;
   d.allocated=d.aligned=(T*)std::malloc(sizeof(T)*4);
 }
 auto *toMemRefDesc(){return &d;}
 ~Tensor(){std::free(d.allocated);}
};
struct FuncResult {MemRefDescriptor<float,1> r0;};
static int mode=0,calls=0;
extern "C" void kernel(void *ret,MemRefDescriptor<float,1>*input) {
 ++calls;auto*r=(FuncResult*)ret;
 r->r0=input[0];
 if(mode==2)return; // Bad borrowed return ABI: wrapper must reject, not free input.
 r->r0.allocated=r->r0.aligned=(float*)std::malloc(16);
 for(int i=0;i<4;++i)r->r0.aligned[i]=input->aligned[i]*2+1;
 if(mode==1)r->r0.strides[0]=2;
 if(mode==5)input->aligned[0]=0;
 if(mode==6)for(int i=0;i<4;++i)input->aligned[i]=r->r0.aligned[i];
}
template<class F> uint64_t benchmark_time_us(int,F f){f();return 123;}
'''
    after=r'''
int main(int argc,char**argv) {
 assert(argc==2);mode=std::atoi(argv[1]);
 alignas(128) unsigned char input[128]={},output[128];
 float values[]={1,2,3,4};std::memcpy(input,values,16);
 std::memset(output,0xa5,128);uint64_t us=999;
 if(mode==3) {
   assert(hexagon_forward_buffers(input,127,output,128,&us)==2);
   assert(calls==0 && output[0]==0xa5 && us==999);return 0;
 }
 if(mode==4) {
   assert(hexagon_forward_buffers(input,128,input,128,&us)==2);
   assert(calls==0);return 0;
 }
 int status=hexagon_forward_buffers(input,128,output,128,&us);
 if(mode==5) {assert(status==BAD_UPDATE_STATUS && output[0]==0xa5);return 0;}
 if(mode==6 && REJECT_VALID_UPDATE) {assert(status==7 && output[0]==0xa5);return 0;}
 if(mode==1 || mode==2) {
   assert(status==(mode==1?4:3) && output[0]==0xa5);return 0;
 }
 assert(status==0 && calls==1 && us==123);
 float expected[]={3,5,7,9};assert(!std::memcmp(output,expected,16));
 for(int i=16;i<128;++i)assert(output[i]==0);
 assert(!std::memcmp(input,values,16));return 0;
}
'''
    checked = request.param is not False
    updates = {0: 0} if request.param == 'state' else {}
    after=after.replace('BAD_UPDATE_STATUS','8' if updates else '7').replace(
        'REJECT_VALID_UPDATE','true' if checked and not updates else 'false')
    source=root/'test.cpp';source.write_text(before+buffer_io.generate_buffer_entry(wrapper(),small_layout(),
        validate_input_copy=checked,input_update_outputs=updates)+after)
    result=subprocess.run([compiler,'-std=c++17','-O1','-Wall','-Wextra','-Werror',
        '-fsanitize=address,undefined','-I',str(ROOT/'test/utils/dsp/include'),str(source),'-o',str(root/'test')],
        capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    return root/'test',checked


@pytest.mark.parametrize('mode',[0,1,2,3,4,6])
def test_actual_generated_cpp_with_sanitizers(executable,mode):
    result=subprocess.run([str(executable[0]),str(mode)],capture_output=True,text=True,timeout=10,
        preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_CORE,(0,0)))
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('executable', [True, 'state'], indirect=True)
def test_checked_entry_detects_input_mutation_before_output_copy(executable):
    result=subprocess.run([str(executable[0]),'5'],capture_output=True,text=True,timeout=10,
        preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_CORE,(0,0)))
    assert result.returncode==0,result.stderr
