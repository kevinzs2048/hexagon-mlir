"""Actual MLIR types + AOT shape metadata, without handwritten tensor axes."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from qcom_hexagon_backend.backend.hexagon_captured_buffer_contract import derive_captured_contract
from qcom_hexagon_backend.backend.hexagon_dynamic_buffer_io import pack_torch_request


def test_independent_and_duplicate_shape_symbols_bind_by_tensor_axes():
    from qcom_hexagon_backend.backend.hexagon_captured_buffer_contract import scalar_policy_from_tensor_contract
    env = ShapeEnv()
    q, p = env.create_unbacked_symint(), env.create_unbacked_symint()
    for value in (q, p):
        env.constrain_symbol_range(value.node.expr, 2, 32)
    with FakeTensorMode(shape_env=env):
        ids, positions = torch.empty(1, q, dtype=torch.int64), torch.empty(p, dtype=torch.int64)
    graph = torch.fx.Graph()
    nodes = []
    for index, value in enumerate((q, ids, p, positions, q, 1)):
        node = graph.placeholder(f'arg{index}')
        node.meta['val'] = value
        nodes.append(node)
    graph.output((nodes[1],))
    model = torch.fx.GraphModule({}, graph)
    tensor_policy = [dict(kind='tensor', dtype='int64', shape=[1, 'query']),
                     dict(kind='tensor', dtype='int64', shape=['query'])]
    policy = scalar_policy_from_tensor_contract(model, range(6), tensor_policy,
        variables={'query': (2, 32)}, allowed_literals={1})
    assert policy == {0: 'query', 2: 'query', 4: 'query', 5: 1}
    contract = derive_captured_contract(model, range(6),
        '(i64, tensor<1x?xi64>, i64, tensor<?xi64>, i64, i64) -> tensor<1x?xi64>',
        variables={'query': (2, 32)}, scalar_bindings=policy)
    pack_torch_request(contract, (6, torch.zeros(1, 6, dtype=torch.int64), 6,
                                 torch.zeros(6, dtype=torch.int64), 6, 1))
    with pytest.raises(ValueError, match='disagrees'):
        pack_torch_request(contract, (6, torch.zeros(1, 6, dtype=torch.int64), 5,
                                     torch.zeros(5, dtype=torch.int64), 6, 1))
    nodes[5].meta['val'] = env.create_unbacked_symint()
    with pytest.raises(ValueError, match='no declared'):
        scalar_policy_from_tensor_contract(model, range(6), tensor_policy,
            variables={'query': (2, 32)}, allowed_literals={1})


SIGNATURE = '(i64, tensor<?x4xf32>) -> tensor<?x4xf32>'


def graph_fixture():
    env = ShapeEnv()
    query = env.create_unbacked_symint()
    env.constrain_symbol_range(query.node.expr, 2, 32)
    with FakeTensorMode(shape_env=env):
        value = torch.empty(query, 4)
    graph = torch.fx.Graph()
    scalar = graph.placeholder('query')
    scalar.meta['val'] = query
    tensor = graph.placeholder('value')
    tensor.meta['val'] = value
    output = graph.call_function(torch.ops.aten.neg.default, (tensor,))
    output.meta['val'] = value
    graph.output((output,))
    return torch.fx.GraphModule({}, graph), query, value


def derive(graph, signature=SIGNATURE, **kwargs):
    return derive_captured_contract(graph, kwargs.pop('retained', [0, 1]), signature,
        variables=kwargs.pop('variables', {'query': (2, 32)}),
        scalar_bindings=kwargs.pop('scalar_bindings', {0: 'query'}), **kwargs)


def test_tensor_axes_derived_and_runtime_length_validated():
    graph, _, _ = graph_fixture()
    before = graph.code
    contract = derive(graph)
    assert contract['inputs'] == [dict(kind='scalar', value='query'),
        dict(kind='tensor', dtype='float32', shape=['query', 4])]
    assert contract['outputs'] == [dict(kind='tensor', dtype='float32', shape=['query', 4])]
    assert graph.code == before
    _, _, values = pack_torch_request(contract, (7, torch.zeros(7, 4)))
    assert values == {'query': 7}
    with pytest.raises(ValueError, match='Inconsistent'):
        pack_torch_request(contract, (7, torch.zeros(6, 4)))


@pytest.mark.parametrize('bounds', [(1, 32), (2, 33)])
def test_cannot_widen_dynamo_guards(bounds):
    graph, _, _ = graph_fixture()
    with pytest.raises(ValueError, match='exceeds captured'):
        derive(graph, variables={'query': bounds})


@pytest.mark.parametrize('signature,match', [
    ('(i64, tensor<?x4xi64>) -> tensor<?x4xf32>', 'rank/dtype'),
    ('(i64, tensor<?x5xf32>) -> tensor<?x4xf32>', 'fixed tensor'),
    ('(i64, tensor<7x4xf32>) -> tensor<?x4xf32>', 'fixed tensor'),
    ('(i64, tensor<?x4xf32>) -> (tensor<?x4xf32>, tensor<?x4xf32>)', 'arities'),
    ('(i64, tensor<?x4xf32>) -> tensor<?x4xf16>', 'f32/i64'),
])
def test_lowered_signature_mismatch_is_rejected(signature, match):
    graph, _, _ = graph_fixture()
    with pytest.raises(ValueError, match=match):
        derive(graph, signature)


def test_fixed_shape_policy_is_explicit_not_taken_from_a_hint():
    graph, _, _ = graph_fixture()
    with pytest.raises(ValueError, match='inferable'):
        derive(graph, scalar_bindings={0: 7})


@pytest.mark.parametrize('retained', [[1, 0], [0, 0], [0, 2]])
def test_bad_retained_indices_fail(retained):
    graph, _, _ = graph_fixture()
    with pytest.raises(ValueError, match='retained'):
        derive(graph, retained=retained)


def test_compound_shape_rejected_not_replaced_with_example_length():
    graph, query, _ = graph_fixture()
    with FakeTensorMode(shape_env=query.node.shape_env):
        larger = torch.empty(query+1, 4)
    list(graph.graph.nodes)[2].meta['val'] = larger
    with pytest.raises(ValueError, match='compound tensor'):
        derive(graph)


def test_variable_query_and_fixed_capacity_are_separate_symbols():
    env = ShapeEnv()
    query, capacity = env.create_unbacked_symint(), env.create_unbacked_symint()
    env.constrain_symbol_range(query.node.expr, 2, 32)
    env.constrain_symbol_range(capacity.node.expr, 2, 1024)
    with FakeTensorMode(shape_env=env):
        ids = torch.empty(1, query, dtype=torch.int64)
        kv = torch.empty(1, 2, capacity, 8)
        mask = torch.empty(1, 1, query, capacity)
        logits = torch.empty(1, query, 16)
    graph = torch.fx.Graph()
    nodes = []
    for index, value in enumerate((query, ids, capacity, kv, mask, 1)):
        node = graph.placeholder(f'arg{index}')
        node.meta['val'] = value
        nodes.append(node)
    output = graph.call_function(torch.ops.aten.neg.default, (nodes[3],))
    output.meta['val'] = logits
    graph.output((nodes[3], output))
    model = torch.fx.GraphModule({}, graph)
    signature = ('(i64, tensor<1x?xi64>, i64, tensor<1x2x?x8xf32>, '
                 'tensor<1x1x?x?xf32>, i64) -> (tensor<1x2x?x8xf32>, tensor<1x?x16xf32>)')
    contract = derive_captured_contract(model, range(6), signature,
        variables={'query': (2, 32)}, scalar_bindings={0: 'query', 2: 1024, 5: 1})
    assert contract['inputs'][3]['shape'] == [1, 2, 1024, 8]
    assert contract['inputs'][4]['shape'] == [1, 1, 'query', 1024]
    assert contract['outputs'][1]['shape'] == [1, 'query', 16]


def test_no_unknown_symbol_is_resolved_by_its_example_hint():
    graph, query, _ = graph_fixture()
    env = query.node.shape_env
    other = env.create_unbacked_symint()
    env.constrain_symbol_range(other.node.expr, 2, 32)
    with FakeTensorMode(shape_env=env):
        unknown = torch.empty(other, 4)
    list(graph.graph.nodes)[2].meta['val'] = unknown
    with pytest.raises(ValueError, match='Unbound'):
        derive(graph)


def test_symint_integer_literal_is_not_an_unbound_dynamic_axis():
    import sympy
    from torch.fx.experimental.sym_node import SymNode
    graph, query, _ = graph_fixture()
    nodes = list(graph.graph.nodes)
    with graph.graph.inserting_before(nodes[1]):
        literal = graph.graph.placeholder('literal')
        literal.meta['val'] = torch.SymInt(SymNode(sympy.Integer(1), query.node.shape_env, int, hint=1))
    graph.recompile()
    contract = derive_captured_contract(graph, [0, 1, 2],
        '(i64, i64, tensor<?x4xf32>) -> tensor<?x4xf32>',
        variables={'query': (2, 32)}, scalar_bindings={0: 'query', 1: 1})
    assert contract['inputs'][1] == dict(kind='scalar', value=1)


def test_static_graph_omitted_scalar_uses_only_singleton_protocol_bound():
    graph = torch.fx.Graph()
    value = graph.placeholder('value')
    value.meta['val'] = torch.empty(1, 4)
    graph.output((value,))
    model = torch.fx.GraphModule({}, graph)
    signature = '(tensor<1x4xf32>) -> tensor<1x4xf32>'
    contract = derive_captured_contract(model, [0], signature,
        variables={'query': (1, 1)}, scalar_bindings={})
    packet, _, dimensions = pack_torch_request(contract, (torch.ones(1, 4),))
    assert dimensions == {'query': 1} and int.from_bytes(packet[8:16], 'little') == 1
    with pytest.raises(ValueError, match='inferable'):
        derive_captured_contract(model, [0], signature,
            variables={'query': (1, 32)}, scalar_bindings={})
