"""Host ownership regressions using the actual C++ helper and code generator."""
from pathlib import Path
import shutil
import resource
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope='module')
def owner_program(tmp_path_factory):
    out = tmp_path_factory.mktemp('returned-memref-owner')
    compiler = shutil.which('c++')
    if not compiler: pytest.skip('Host C++ compiler unavailable')
    source = out/'test.cpp'
    source.write_text(r'''
#include "returned_memref_owner.h"
#include <cassert>
#include <cstring>
extern "C" void __real_free(void*);
static void *freed[32]; static unsigned frees;
extern "C" void __wrap_free(void *p) {
  for (unsigned i=0;i<frees;++i) assert(freed[i]!=p);
  if(p) freed[frees++]=p;
  __real_free(p);
}
int main(int argc,char **argv) {
  assert(argc==2);
  void *a=std::malloc(32), *b=std::malloc(64), *input=std::malloc(128);
  assert(a && b && input);
  if (!strcmp(argv[1],"unique") || !strcmp(argv[1],"exception")) {
    try {
      OwnedReturnedMemrefs<3> owner;
      assert(owner.adopt({a,b,a},{input}));
      assert(owner.unique_buffers()==2);
      assert(!owner.adopt({a,b,a},{input}));
      if(!strcmp(argv[1],"exception")) throw 7;
    } catch(int error) { assert(error==7); }
    assert(frees==2 && freed[0]==a && freed[1]==b);
    std::free(input);
  } else if (!strcmp(argv[1],"null")) {
    { OwnedReturnedMemrefs<2> owner; assert(owner.adopt({nullptr,a},{input})); }
    assert(frees==1 && freed[0]==a); std::free(b); std::free(input);
  } else {
    {
      OwnedReturnedMemrefs<2> owner;
      if(!strcmp(argv[1],"alias")) assert(!owner.adopt({a,input},{input}));
      else if(!strcmp(argv[1],"poison")) assert(!owner.adopt({a,(void*)0xdeadbeefU},{input}));
      else if(!strcmp(argv[1],"count")) assert(!owner.adopt({a},{input}));
      else assert(false);
      assert(owner.unique_buffers()==0);
    }
    assert(frees==0); std::free(a); std::free(b); std::free(input);
  }
  assert(frees==3);
  { OwnedReturnedMemrefs<0> empty; assert(empty.adopt({},{})); }
  return 0;
}
''')
    # Link-time --wrap interposition must remain visible to the optimizer;
    # the built-in free contract otherwise lets it cache the observer counter.
    run = subprocess.run([compiler,'-std=c++17','-O1','-Wall','-Wextra','-Werror',
        '-fno-builtin-malloc','-fno-builtin-free',
        '-I',str(ROOT/'test/utils/dsp/include'),str(source),'-Wl,--wrap=free','-o',str(out/'test')],
        capture_output=True,text=True)
    assert run.returncode == 0, run.stderr
    return out/'test'


@pytest.mark.parametrize('case', ['unique','exception','null','alias','poison','count'])
def test_real_cpp_ownership(owner_program,case):
    run = subprocess.run([str(owner_program),case],capture_output=True,text=True,timeout=10,
        preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE,(0,0)))
    assert run.returncode == 0, run.stderr
