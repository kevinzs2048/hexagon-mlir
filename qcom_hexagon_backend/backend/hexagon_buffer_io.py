# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit fixed-shape buffer ABI for a torch-mlir DDR callable.

This is transport metadata/code generation, not model arithmetic. The first
version copies RPC inputs into the same aligned Tensor allocations as the file
wrapper, and copies owned results out before freeing them. No zero-copy claim.
"""
import math
import re

DTYPES = {'float32': ('float',4), 'int64': ('int64_t',8)}
ALIGNMENT = 128
LIMIT = 256*1024**2


def make_layout(inputs, outputs):
    def rows(specs):
        if not 1 <= len(specs) <= 256: raise ValueError('Expected a bounded nonempty tensor list')
        result=[];offset=0
        for index,spec in enumerate(specs):
            dtype=spec['dtype'];shape=list(spec['shape'])
            if (dtype not in DTYPES or not 1<=len(shape)<=8
                    or any(type(n) is not int or n<=0 for n in shape)):
                raise ValueError('Buffer ABI requires positive static FP32/int64 ranked tensors')
            length=math.prod(shape)*DTYPES[dtype][1]
            if length>64*1024**2: raise ValueError('Tensor exceeds bounded ABI size')
            offset=(offset+ALIGNMENT-1)//ALIGNMENT*ALIGNMENT
            strides=[math.prod(shape[i+1:]) for i in range(len(shape))]
            result.append(dict(index=index,dtype=dtype,shape=shape,strides=strides,offset=offset,bytes=length))
            offset+=length
        total=(offset+ALIGNMENT-1)//ALIGNMENT*ALIGNMENT
        if total>LIMIT: raise ValueError('Total buffer size exceeds ABI limit')
        return result,total
    incoming,in_bytes=rows(inputs);outgoing,out_bytes=rows(outputs)
    return dict(schema=1,byte_order='little',alignment=ALIGNMENT,inputs=incoming,outputs=outgoing,
                input_bytes=in_bytes,output_bytes=out_bytes)


def validate_layout(layout):
    expected=make_layout(layout['inputs'],layout['outputs'])
    if layout!=expected: raise ValueError('Layout is not canonical or contains overlapping/inconsistent slices')
    return layout


def pack_inputs(layout, payloads):
    validate_layout(layout)
    if len(payloads)!=len(layout['inputs']): raise ValueError('Wrong input tensor count')
    buffer=bytearray(layout['input_bytes'])
    for row,data in zip(layout['inputs'],payloads):
        if len(data)!=row['bytes']: raise ValueError('Wrong input tensor byte count')
        buffer[row['offset']:row['offset']+row['bytes']]=data
    return bytes(buffer)


def unpack_outputs(layout, buffer):
    validate_layout(layout)
    if len(buffer)!=layout['output_bytes']: raise ValueError('Wrong output buffer length')
    return [buffer[r['offset']:r['offset']+r['bytes']] for r in layout['outputs']]


def generate_buffer_entry(wrapper, layout, *, symbol='hexagon_forward_buffers', validate_input_copy=False,
                          input_update_outputs=None):
    """Append this to an owned torch-mlir file wrapper; retain its original main.

The existing FuncResult definition and _mlir_ciface declaration are reused.
All runtime shapes/strides are checked before copying any result to the caller.
"""
    validate_layout(layout)
    if type(validate_input_copy) is not bool: raise TypeError('Input-copy diagnostics require an explicit bool')
    updates = {} if input_update_outputs is None else input_update_outputs
    if not isinstance(updates, dict): raise TypeError('Explicit input/output update mapping required')
    if updates and not validate_input_copy: raise ValueError('State updates require input-copy diagnostics')
    for incoming, outgoing in updates.items():
        if (type(incoming) is not int or type(outgoing) is not int
                or not 0 <= incoming < len(layout['inputs']) or not 0 <= outgoing < len(layout['outputs'])):
            raise ValueError('Invalid state input/output index')
        a, b = layout['inputs'][incoming], layout['outputs'][outgoing]
        if any(a[k] != b[k] for k in ('shape', 'dtype', 'bytes', 'strides')):
            raise ValueError('State update must preserve the complete tensor ABI')
    if (not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',symbol)
            or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',wrapper.func_name)):
        raise ValueError('Invalid C entry symbol')
    if wrapper.iterations!=1 or getattr(wrapper,'release_returned_buffers',False) is not True:
        raise ValueError('Buffer entry requires the explicit one-call owned DDR wrapper')
    if len(wrapper.input_profs)!=len(layout['inputs']) or len(wrapper.output_profs)!=len(layout['outputs']):
        raise ValueError('Wrapper and buffer layout tensor counts differ')
    for profile,row in zip(wrapper.input_profs,layout['inputs']):
        if (profile.input_type!='tensor' or profile.idx!=row['index'] or profile.dtype!=DTYPES[row['dtype']][0]
                or profile.rank!=len(row['shape']) or tuple(profile.value.shape)!=tuple(row['shape'])):
            raise ValueError('Input ABI differs from compiled wrapper')
    for profile,row in zip(wrapper.output_profs,layout['outputs']):
        if profile.idx!=row['index'] or profile.dtype!=DTYPES[row['dtype']][0] or profile.rank!=len(row['shape']):
            raise ValueError('Output ABI differs from compiled wrapper')
    code=['\n#include <cstring>\n#include "returned_memref_owner.h"\n',
          f'extern "C" int {symbol}(const unsigned char *input, int input_len,',
          '    unsigned char *output, int output_len, uint64_t *kernel_us) {',
          f'  if (!input || !output || !kernel_us || input_len != {layout["input_bytes"]} || output_len != {layout["output_bytes"]}) return 2;',
          '  uintptr_t ib = reinterpret_cast<uintptr_t>(input), ob = reinterpret_cast<uintptr_t>(output);',
          '  if ((ib <= ob && ob - ib < static_cast<uintptr_t>(input_len)) ||',
          '      (ob < ib && ib - ob < static_cast<uintptr_t>(output_len))) return 2;',
          '  *kernel_us = 0;', '  try {']
    if validate_input_copy:
        code.insert(0, '''
static __attribute__((noinline)) bool hexagon_input_bytes_equal(
    const volatile unsigned char *a, const volatile unsigned char *b, int n) {
  for (int i=0;i<n;++i) if (a[i]!=b[i]) return false;
  return true;
}
''')
    for row in layout['inputs']:
        index=row['index'];dtype=DTYPES[row['dtype']][0];rank=len(row['shape'])
        sizes='{'+', '.join(map(str,row['shape']))+'}'
        strides='{'+', '.join(map(str,row['strides']))+'}'
        code.extend([f'  Tensor<{dtype}, {rank}> T{index}({{{sizes}, {strides}, 128}}, MemType::HEAP);',
            f'  auto *dt{index} = T{index}.toMemRefDesc();',
            f'  std::memcpy(dt{index}->aligned, input + {row["offset"]}, {row["bytes"]});'])
        if validate_input_copy:
            code.append(f'  if (!hexagon_input_bytes_equal(reinterpret_cast<const unsigned char *>(dt{index}->aligned), input + {row["offset"]}, {row["bytes"]})) return 6;')
    code.extend(['  FuncResult result_storage{}; auto *r = &result_storage;',
                 f'  OwnedReturnedMemrefs<{len(layout["outputs"])}> owner;',
                 '  *kernel_us = benchmark_time_us(1, [&]() {',
                 '    '+wrapper.generate_llvm_function_call().strip(), '  });'])
    results=', '.join(f'r->r{i}.allocated' for i in range(len(layout['outputs'])))
    inputs=', '.join(f'dt{i}->allocated' for i in range(len(layout['inputs'])))
    code.append('  if (!owner.adopt({'+results+'}, {'+inputs+'})) return 3;')
    if validate_input_copy:
        for row in layout['inputs']:
            if row['index'] in updates: continue
            code.append(f'  if (!hexagon_input_bytes_equal(reinterpret_cast<const unsigned char *>(dt{row["index"]}->aligned), input + {row["offset"]}, {row["bytes"]})) return 7;')
    for row in layout['outputs']:
        desc=f'r->r{row["index"]}'
        conditions=[f'!{desc}.allocated',f'!{desc}.aligned',f'{desc}.offset != 0']
        for axis,(size,stride) in enumerate(zip(row['shape'],row['strides'])):
            conditions.extend([f'{desc}.sizes[{axis}] != {size}',f'{desc}.strides[{axis}] != {stride}'])
        code.append('  if ('+' || '.join(conditions)+') return 4;')
    # Only DSP-owned copies corresponding to explicit functional update outputs
    # may change. Never disable pre-copy, immutable-input or output-shape checks.
    # Validate output descriptors above before reading an update output here.
    for incoming, outgoing in sorted(updates.items()):
        row = layout['inputs'][incoming]
        code.append(f'  if (!hexagon_input_bytes_equal(reinterpret_cast<const unsigned char *>(dt{incoming}->aligned), input + {row["offset"]}, {row["bytes"]}) && '
                    f'!hexagon_input_bytes_equal(reinterpret_cast<const unsigned char *>(dt{incoming}->aligned), reinterpret_cast<const unsigned char *>(r->r{outgoing}.aligned), {row["bytes"]})) return 8;')
    code.append('  std::memset(output, 0, output_len);')
    for row in layout['outputs']:
        code.append(f'  std::memcpy(output + {row["offset"]}, r->r{row["index"]}.aligned, {row["bytes"]});')
    code.extend(['  return 0;', '  } catch (...) { return 5; }', '}'])
    return '\n'.join(code)+'\n'
