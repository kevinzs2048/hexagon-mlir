# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Bounded symbolic-shape buffer ABI for a ranked torch-mlir forward.

The shape variables are request data, not values baked into a model wrapper.
This module only provides the transport contract and C entry; it does not
establish that a compiled model is numerically correct for every shape.
"""

import re
import struct
import sys

from .hexagon_buffer_io import (ALIGNMENT, DTYPES, make_layout, pack_inputs,
                                unpack_outputs)


MAGIC = b"HXDY\x01\x00\x00\x00"
_TYPE = re.compile(r"tensor<([0-9?x]+)x(i64|f32)>")


def _tensor_type(value):
    match = _TYPE.fullmatch(value)
    if match is None:
        raise ValueError("Expected a ranked i64/f32 tensor type")
    dims = match.group(1).split("x")
    if not 1 <= len(dims) <= 8:
        raise ValueError("Unsupported tensor rank")
    return [None if part == "?" else int(part) for part in dims], {
        "i64": "int64", "f32": "float32"
    }[match.group(2)]


def make_contract(input_types, output_types, *, variables, symbolic_dimensions,
                  scalar_bindings):
    """Bind every symbolic dimension and scalar to a bounded request variable.

    ``symbolic_dimensions`` keys are ("input"/"output", argument, axis).
    Scalar bindings are either an integer literal or a variable name.
    """
    if not variables or len(variables) > 8 or any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name)
            or type(bounds) not in (tuple, list) or len(bounds) != 2
            or any(type(n) is not int for n in bounds)
            or not 1 <= bounds[0] <= bounds[1] <= 1024
            for name, bounds in variables.items()):
        raise ValueError("One to eight bounded positive shape variables required")
    names = tuple(sorted(variables))
    if (not input_types or not output_types or len(input_types) > 256
            or len(output_types) > 256):
        raise ValueError("Bounded nonempty model signature required")
    used_dimensions = set()

    def rows(types, side):
        result = []
        for index, typename in enumerate(types):
            if typename == "i64" and side == "input":
                if index not in scalar_bindings:
                    raise ValueError("Unbound runtime scalar")
                value = scalar_bindings[index]
                if type(value) is not int and value not in names:
                    raise ValueError("Scalar must be an integer or shape variable")
                result.append(dict(kind="scalar", value=value))
                continue
            shape, dtype = _tensor_type(typename)
            for axis, dim in enumerate(shape):
                key = (side, index, axis)
                if dim is None:
                    bound = symbolic_dimensions.get(key)
                    if bound not in names and (type(bound) is not int or bound <= 0):
                        raise ValueError("Unbound symbolic tensor dimension")
                    shape[axis] = bound
                    used_dimensions.add(key)
                elif dim <= 0 or key in symbolic_dimensions:
                    raise ValueError("Invalid fixed tensor dimension binding")
            result.append(dict(kind="tensor", dtype=dtype, shape=shape))
        return result

    inputs = rows(input_types, "input")
    outputs = rows(output_types, "output")
    if (set(symbolic_dimensions) != used_dimensions
            or set(scalar_bindings) != {i for i, value in enumerate(input_types)
                                        if value == "i64"}):
        raise ValueError("Extraneous symbolic dimension or scalar binding")
    if not any(row["kind"] == "tensor" for row in inputs):
        raise ValueError("At least one runtime tensor required")
    contract = dict(schema=1, variables={name: tuple(variables[name]) for name in names},
                    inputs=inputs, outputs=outputs)
    layout_for(contract, {name: variables[name][1] for name in names})
    return contract


def layout_for(contract, values):
    if (set(values) != set(contract["variables"])
            or any(type(values[name]) is not int
                   or not lo <= values[name] <= hi
                   for name, (lo, hi) in contract["variables"].items())):
        raise ValueError("Runtime dimensions outside the compiled contract")

    def tensor_rows(rows):
        return [dict(dtype=row["dtype"], shape=[values.get(dim, dim)
                                                 for dim in row["shape"]])
                for row in rows if row["kind"] == "tensor"]

    return make_layout(tensor_rows(contract["inputs"]),
                       tensor_rows(contract["outputs"]))


def pack_request(contract, values, payloads):
    layout = layout_for(contract, values)
    names = tuple(contract["variables"])
    return (MAGIC + struct.pack("<" + "q" * len(names),
                                *(values[name] for name in names))
            + pack_inputs(layout, payloads)), layout


def pack_torch_request(contract, arguments):
    """Pack concrete PyTorch forward arguments without choosing a shape variant.

    Tensor sizes and scalar arguments must agree on every symbolic dimension.
    This is a host transport adapter, not a graph compiler or DSP admission.
    """
    import torch

    if sys.byteorder != "little":
        raise ValueError("The buffer ABI requires a little-endian host")
    if len(arguments) != len(contract["inputs"]):
        raise ValueError("Wrong model argument count")
    values = {}
    tensors = []

    def bind_dimension(expected, actual):
        if type(actual) is not int or actual <= 0:
            raise ValueError("Runtime tensor dimensions must be positive integers")
        if type(expected) is int:
            if actual != expected:
                raise ValueError("Runtime tensor shape differs from compiled contract")
        elif expected in values:
            if actual != values[expected]:
                raise ValueError("Inconsistent symbolic dimension")
        else:
            values[expected] = actual

    for row, argument in zip(contract["inputs"], arguments):
        if row["kind"] == "scalar":
            if type(argument) is not int:
                raise ValueError("Runtime scalar must be a concrete Python integer")
            expected = row["value"]
            if type(expected) is int:
                if argument != expected:
                    raise ValueError("Runtime scalar differs from compiled contract")
            elif expected in values:
                if argument != values[expected]:
                    raise ValueError("Runtime scalar disagrees with tensor shape")
            else:
                values[expected] = argument
            continue
        dtype = {"float32": torch.float32, "int64": torch.int64}[row["dtype"]]
        if (not isinstance(argument, torch.Tensor) or argument.dtype != dtype
                or argument.device.type != "cpu" or argument.layout != torch.strided
                or not argument.is_contiguous() or argument.requires_grad):
            raise ValueError("Expected contiguous, inference-only CPU tensor with compiled dtype")
        if argument.ndim != len(row["shape"]):
            raise ValueError("Runtime tensor rank differs from compiled contract")
        for expected, actual in zip(row["shape"], argument.shape):
            bind_dimension(expected, int(actual))
        tensors.append(argument)
    # A shape-specialized graph (notably Dynamo's q=1 graph) can omit the
    # scalar entirely. A fixed protocol field is supplied only when the
    # explicit ABI bounds prove one value, never from an example hint.
    for name, (lo, hi) in contract["variables"].items():
        if name not in values and lo == hi:
            values[name] = lo
    if set(values) != set(contract["variables"]):
        raise ValueError("Cannot infer every runtime shape variable from model inputs")
    # Validate all bounds and byte counts before making large host copies.
    layout_for(contract, values)
    packet, layout = pack_request(contract, values,
                                  [tensor.numpy().tobytes() for tensor in tensors])
    return packet, layout, values


def unpack_torch_outputs(contract, values, response):
    """Restore owned CPU tensors using the exact runtime output dimensions."""
    import torch

    if sys.byteorder != "little":
        raise ValueError("The buffer ABI requires a little-endian host")
    layout = layout_for(contract, values)
    payloads = unpack_outputs(layout, response)
    dtypes = {"float32": torch.float32, "int64": torch.int64}
    return tuple(torch.frombuffer(bytearray(payload), dtype=dtypes[row["dtype"]])
                 .clone().reshape(row["shape"])
                 for row, payload in zip(layout["outputs"], payloads))


def generate_buffer_entry(wrapper, contract, *, symbol="hexagon_forward_dynamic_buffers"):
    """Append a runtime-shaped C entry to an owned, ranked model wrapper."""
    if (not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", symbol)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", wrapper.func_name)):
        raise ValueError("Invalid C entry symbol")
    if wrapper.iterations != 1 or wrapper.release_returned_buffers is not True:
        raise ValueError("Dynamic entry requires a one-call owned DDR wrapper")
    ins, outs = contract["inputs"], contract["outputs"]
    tensor_ins = [row for row in ins if row["kind"] == "tensor"]
    if len(wrapper.input_profs) != len(ins) or len(wrapper.output_profs) != len(outs):
        raise ValueError("Wrapper/model signature cardinality differs")
    tensor_index = 0
    for profile, row in zip(wrapper.input_profs, ins):
        if row["kind"] == "scalar":
            if profile.input_type != "scalar" or profile.dtype != "int64_t":
                raise ValueError("Wrapper scalar ABI differs")
        else:
            if (profile.input_type != "tensor" or profile.idx != tensor_index
                    or profile.rank != len(row["shape"])
                    or profile.dtype != DTYPES[row["dtype"]][0]):
                raise ValueError("Wrapper tensor ABI differs")
            tensor_index += 1
    for index, (profile, row) in enumerate(zip(wrapper.output_profs, outs)):
        if (row["kind"] != "tensor" or profile.idx != index
                or profile.rank != len(row["shape"])
                or profile.dtype != DTYPES[row["dtype"]][0]):
            raise ValueError("Wrapper result ABI differs")
    names = tuple(contract["variables"])
    header = len(MAGIC) + 8 * len(names)
    code = ["\n#include <cstring>", '#include "returned_memref_owner.h"',
            f'extern "C" int {symbol}(const unsigned char *input, int input_len,',
            "    unsigned char *output, int output_len, uint64_t *kernel_us) {",
            f"  if (!input || !output || !kernel_us || input_len < {header} || output_len < 0) return 2;",
            "  *kernel_us = 0;",
            f"  if (std::memcmp(input, \"HXDY\\x01\\x00\\x00\\x00\", {len(MAGIC)}) != 0) return 2;"]
    for i, name in enumerate(names):
        lo, hi = contract["variables"][name]
        code += [f"  int64_t {name}; std::memcpy(&{name}, input + {len(MAGIC) + 8*i}, 8);",
                 f"  if ({name} < {lo} || {name} > {hi}) return 2;"]
    code += ["  uintptr_t ib = reinterpret_cast<uintptr_t>(input), ob = reinterpret_cast<uintptr_t>(output);",
             "  if ((ib <= ob && ob - ib < static_cast<uintptr_t>(input_len)) ||",
             "      (ob < ib && ib - ob < static_cast<uintptr_t>(output_len))) return 2;",
             "  try {", "  size_t in_offset = 0, out_offset = 0;"]

    def expr(dim):
        return str(dim)

    def size_expr(shape, start=0):
        return " * ".join(expr(dim) for dim in shape[start:]) or "1"

    for index, row in enumerate(tensor_ins):
        shape = row["shape"]
        width = DTYPES[row["dtype"]][1]
        code += [f"  in_offset = (in_offset + {ALIGNMENT-1}) & ~size_t({ALIGNMENT-1});",
                 f"  const size_t in_offset_{index} = in_offset;",
                 f"  in_offset += ({size_expr(shape)}) * {width};"]
    code += [f"  if (static_cast<size_t>(input_len) != {header} + ((in_offset + {ALIGNMENT-1}) & ~size_t({ALIGNMENT-1}))) return 2;"]
    for index, row in enumerate(outs):
        shape = row["shape"]
        width = DTYPES[row["dtype"]][1]
        code += [f"  out_offset = (out_offset + {ALIGNMENT-1}) & ~size_t({ALIGNMENT-1});",
                 f"  const size_t out_offset_{index} = out_offset;",
                 f"  out_offset += ({size_expr(shape)}) * {width};"]
    code += [f"  if (static_cast<size_t>(output_len) != ((out_offset + {ALIGNMENT-1}) & ~size_t({ALIGNMENT-1}))) return 2;"]
    for index, row in enumerate(tensor_ins):
        shape = row["shape"]
        dtype = DTYPES[row["dtype"]][0]
        width = DTYPES[row["dtype"]][1]
        sizes = "{" + ", ".join(map(expr, shape)) + "}"
        strides = "{" + ", ".join(size_expr(shape, i+1)
                                    for i in range(len(shape))) + "}"
        code += [f"  Tensor<{dtype}, {len(shape)}> T{index}({{{sizes}, {strides}, {ALIGNMENT}}}, MemType::HEAP);",
                 f"  auto *dt{index} = T{index}.toMemRefDesc();",
                 f"  std::memcpy(dt{index}->aligned, input + {header} + in_offset_{index}, ({size_expr(shape)}) * {width});"]
    code += ["  FuncResult result_storage{}; auto *r = &result_storage;",
             f"  OwnedReturnedMemrefs<{len(outs)}> owner;",
             "  *kernel_us = benchmark_time_us(1, [&]() {"]
    call_args = []
    tensor_index = 0
    for row in ins:
        if row["kind"] == "tensor":
            call_args.append(f"dt{tensor_index}")
            tensor_index += 1
        else:
            call_args.append(expr(row["value"]))
    code += [f"    {wrapper.func_name}(r, {', '.join(call_args)});", "  });"]
    results = ", ".join(f"r->r{i}.allocated" for i in range(len(outs)))
    inputs = ", ".join(f"dt{i}->allocated" for i in range(len(tensor_ins)))
    code.append("  if (!owner.adopt({" + results + "}, {" + inputs + "})) return 3;")
    for index, row in enumerate(outs):
        shape = row["shape"]
        conditions = [f"!r->r{index}.allocated", f"!r->r{index}.aligned",
                      f"r->r{index}.offset != 0"]
        for axis, dim in enumerate(shape):
            conditions += [f"r->r{index}.sizes[{axis}] != {expr(dim)}",
                           f"r->r{index}.strides[{axis}] != ({size_expr(shape, axis+1)})"]
        code.append("  if (" + " || ".join(conditions) + ") return 4;")
    code.append("  std::memset(output, 0, output_len);")
    for index, row in enumerate(outs):
        width = DTYPES[row["dtype"]][1]
        code.append(f"  std::memcpy(output + out_offset_{index}, r->r{index}.aligned, ({size_expr(row['shape'])}) * {width});")
    code += ["  return 0;", "  } catch (...) { return 5; }", "}"]
    return "\n".join(code) + "\n"
