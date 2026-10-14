"""Whole-Program finite state contracts; native execution is received separately."""
import pytest
import pops
from pops.algebra import IndexFamily
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.time._program.detach import detach_compiled_program
from examples.api_v02.bundle_remap_v14 import build_bundle_remap_v14


def test_zero_block_complete_packet_resolves_and_lowers_on_system():
    case = build_bundle_remap_v14()
    plan = pops.resolve(pops.validate(case.case), layout=case.layout)
    assert not plan.blocks
    assert plan.target == "system"
    cpp = emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert 'pops_install_program(pops::System<' in cpp
    assert 'ctx.continuation_publish(' in cpp
    assert 'ctx.commit_many(' not in cpp
    assert 'ctx.continuation_new_domain(' in cpp
    detached = detach_compiled_program(plan.time)
    assert detached._ir_hash() == plan.time._ir_hash()


def test_empty_case_does_not_gain_an_implicit_runtime_or_state():
    with pytest.raises((ValueError, RuntimeError), match='finite-state|finite state|Continuation'):
        pops.validate(pops.Case('empty'))


def test_same_family_is_not_same_instance_even_at_zero_cardinality():
    p = pops.Program('instances')
    family = IndexFamily(capacity=8)
    left = p.range_set(0, family=family)
    right = p.range_set(0, family=family)
    again = p.range_set(2, family=left)
    assert left.attrs['signature']['family'] == right.attrs['signature']['family']
    assert left.id != right.id != again.id
    assert p.same_domain(left, right).op == 'finite_same_domain'
    assert again.inputs[1] is left
    with pytest.raises(ValueError, match='capacity'):
        p.range_set(2, family=left, capacity=9)


def test_scatter_reduce_uses_real_ordered_regions_and_empty_instance_guard():
    p = pops.Program('scatter')
    domain = p.range_set(3, capacity=8, ownership='replicated')
    target = p.range_set(2, capacity=8, ownership='replicated')
    values = p.map(domain, lambda q, i: q.float64(q.index_value(i))+1)
    indices = p.map(domain, lambda q, i: q.index(target, q.index_value(i) % 2))
    result = p.scatter_reduce(values, indices, target, p.float64(0), lambda q,a,b: 10*a+b,
                              order='ascending')
    assert result.op == 'finite_map'
    assert any(v.op == 'finite_same_domain' for v in p._values)
    assert any(v.op == 'structured_fold' for v in result.attrs['body_block'])
    with pytest.raises(ValueError, match='ascending'):
        p.scatter_reduce(values, indices, target, p.float64(0), lambda q,a,b:a+b, order='unordered')
