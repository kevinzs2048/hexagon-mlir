# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Derive ranked buffer dimensions from actual AOT metadata and Linalg types.

Shape bounds/scalar specialization remain explicit backend policy. Tensor
indices, axes, ranks and output counts come from the graph, not a model's KV
layout. This deliberately rejects expressions outside the current transport
ABI instead of specializing them to the example input's concrete hint.
"""

from .hexagon_dynamic_buffer_io import make_contract


def scalar_policy_from_tensor_contract(graph, retained_inputs, tensor_policy, *, variables,
                                       allowed_literals):
    """Bind shape scalars by symbolic identity, never by an example's hint.

    The selected ABI states which tensor axes must agree. Distinct captured
    symbols may map to that same runtime variable; pack_torch_request checks
    ALL associated tensors/scalars before dispatch. No graph is rewritten.
    """
    import sympy
    import torch
    placeholders = [node for node in graph.graph.nodes if node.op == 'placeholder']
    indices = list(retained_inputs)
    if indices != sorted(set(indices)) or any(type(i) is not int or not 0 <= i < len(placeholders) for i in indices):
        raise ValueError('Ordered retained graph inputs required')
    inputs = [placeholders[i].meta['val'] for i in indices]
    tensors = [value for value in inputs if isinstance(value, torch.Tensor)]
    if len(tensors) != len(tensor_policy):
        raise ValueError('Captured tensor cardinality differs from selected ABI')
    symbols = {}
    for value, policy in zip(tensors, tensor_policy):
        if (policy['kind'] != 'tensor' or str(value.dtype) != 'torch.'+policy['dtype']
                or len(value.shape) != len(policy['shape'])):
            raise ValueError('Captured tensor rank/dtype differs from selected ABI')
        for dim, expected in zip(value.shape, policy['shape']):
            expression = dim.node.expr if isinstance(dim, torch.SymInt) else sympy.Integer(dim)
            if isinstance(expression, sympy.Symbol):
                if expression in symbols and symbols[expression] != expected:
                    raise ValueError('Conflicting selected ABI policies for one shape symbol')
                symbols[expression] = expected
            elif isinstance(expression, sympy.Integer):
                bounds = (expected, expected) if type(expected) is int else variables[expected]
                if bounds[0] != bounds[1] or int(expression) != bounds[0]:
                    raise ValueError('Captured fixed dimension differs from selected ABI')
            else:
                raise ValueError('Compound shape expression requires explicit lowering support')
    result = {}
    for index, value in enumerate(inputs):
        if isinstance(value, torch.Tensor):
            continue
        if type(value) is int:
            expression = sympy.Integer(value)
        elif isinstance(value, torch.SymInt):
            expression = value.node.expr
        else:
            raise ValueError('Only integer shape scalars can be bound')
        if expression in symbols:
            result[index] = symbols[expression]
        elif isinstance(expression, sympy.Integer) and int(expression) in allowed_literals:
            result[index] = int(expression)
        else:
            raise ValueError('Shape scalar has no declared tensor-axis or literal policy')
    return result


def derive_captured_contract(graph, retained_inputs, signature, *, variables,
                             scalar_bindings):
    import sympy
    import torch
    from torch_mlir import ir

    placeholders = [node for node in graph.graph.nodes if node.op == 'placeholder']
    indices = list(retained_inputs)
    if (any(type(index) is not int for index in indices)
            or indices != sorted(set(indices))
            or any(not 0 <= index < len(placeholders) for index in indices)):
        raise ValueError('Ordered unique retained AOT input indices required')
    outputs = [node for node in graph.graph.nodes if node.op == 'output']
    if len(outputs) != 1 or len(outputs[0].args) != 1:
        raise ValueError('One flat AOT output list required')
    output_nodes = outputs[0].args[0]
    if not isinstance(output_nodes, (tuple, list)) or not all(
            isinstance(node, torch.fx.Node) for node in output_nodes):
        raise ValueError('Only flat tensor-node AOT outputs are supported')
    inputs = [placeholders[index] for index in indices]
    if any('val' not in node.meta for node in inputs+list(output_nodes)):
        raise ValueError('Authoritative AOT value metadata missing')
    with ir.Context():
        function = ir.FunctionType(ir.Type.parse(signature))
        input_types = [str(value) for value in function.inputs]
        output_types = [str(value) for value in function.results]
        if len(inputs) != len(input_types) or len(output_nodes) != len(output_types):
            raise ValueError('Captured and lowered signature arities differ')
        # Use the MLIR type parser, not textual splitting of tensor syntax.
        parsed = {}
        for side, types in (('input', function.inputs), ('output', function.results)):
            for index, value in enumerate(types):
                if side == 'input' and str(value) == 'i64':
                    parsed[side, index] = None
                    continue
                if not isinstance(value, ir.RankedTensorType):
                    raise ValueError('Ranked tensor ABI required')
                tensor = ir.RankedTensorType(value)
                dtype = {'f32': torch.float32, 'i64': torch.int64}.get(str(tensor.element_type))
                if dtype is None:
                    raise ValueError('Only f32/i64 buffer tensor types are supported')
                parsed[side, index] = (dtype, [None if ir.ShapedType.is_dynamic_size(dim)
                                               else int(dim) for dim in tensor.shape])

    replacements = {}
    policy_symbols = {name: sympy.Dummy(name) for name in variables}
    for index, typename in enumerate(input_types):
        if typename != 'i64':
            continue
        if index not in scalar_bindings:
            raise ValueError('Every lowered scalar needs explicit shape policy')
        value = inputs[index].meta['val']
        policy = scalar_bindings[index]
        if type(policy) is not int and (not isinstance(policy, str) or policy not in variables):
            raise ValueError('Unknown shape policy variable')
        if type(value) is int:
            if type(policy) is not int or value != policy:
                raise ValueError('Captured literal differs from scalar specialization')
        elif isinstance(value, torch.SymInt):
            expression = value.node.expr
            if isinstance(expression, sympy.Integer):
                if type(policy) is not int or int(expression) != policy:
                    raise ValueError('Captured symbolic literal differs from scalar specialization')
                continue
            if not isinstance(expression, sympy.Symbol):
                raise ValueError(f'Shape scalar {index} must be a single symbol or literal: {expression}')
            resolved = sympy.Integer(policy) if type(policy) is int else policy_symbols[policy]
            if expression in replacements and replacements[expression] != resolved:
                raise ValueError('Conflicting scalar policies for one captured symbol')
            limits = value.node.shape_env.var_to_range.get(expression)
            if limits is None:
                raise ValueError('Captured symbolic range missing')
            lo, hi = (policy, policy) if type(policy) is int else variables[policy]
            if (type(lo) is not int or type(hi) is not int
                    or not bool(lo >= limits.lower) or not bool(hi <= limits.upper)):
                raise ValueError('Requested shape policy exceeds captured symbolic range')
            replacements[expression] = resolved
        else:
            raise ValueError('Scalar ABI requires integer AOT metadata')

    def resolve_dimension(value):
        if type(value) is int:
            return value
        if not isinstance(value, torch.SymInt):
            raise ValueError('Integer tensor dimension metadata required')
        expression = value.node.expr.xreplace(replacements)
        if isinstance(expression, sympy.Integer):
            return int(expression)
        for name, symbol in policy_symbols.items():
            if expression == symbol:
                return name
        raise ValueError('Unbound or compound tensor dimension is outside the buffer ABI')

    dimensions = {}
    for side, nodes in (('input', inputs), ('output', output_nodes)):
        for index, node in enumerate(nodes):
            lowered = parsed[side, index]
            if lowered is None:
                continue
            dtype, shape = lowered
            value = node.meta['val']
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                    or value.dtype != dtype or value.ndim != len(shape)):
                raise ValueError('AOT tensor rank/dtype differs from lowered signature')
            for axis, (actual, dim) in enumerate(zip(value.shape, shape)):
                resolved = resolve_dimension(actual)
                if dim is None:
                    dimensions[side, index, axis] = resolved
                elif resolved != dim:
                    raise ValueError('Lowered fixed tensor dimension differs from captured shape')
    contract = make_contract(input_types, output_types, variables=variables,
                             symbolic_dimensions=dimensions, scalar_bindings=scalar_bindings)
    referenced = set()
    for row in contract['inputs']:
        referenced.update(value for value in ([row['value']] if row['kind'] == 'scalar' else row['shape'])
                          if isinstance(value, str))
    if any(variables[name][0] != variables[name][1] for name in set(variables)-referenced):
        raise ValueError('Every nonconstant runtime shape variable must be inferable from captured inputs')
    return contract
