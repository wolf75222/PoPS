"""Actual Case validate/resolve/codegen for the staged distributed frontend."""
import pytest
import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from examples.api_v02.distributed_program_checks import build_distributed_program


@pytest.mark.parametrize('ranks', [2, 3])
@pytest.mark.parametrize('kind', ['roundtrip', 'user_payload', 'corrupt_data'])
def test_full_distributed_case_validates_resolves_and_emits(kind, ranks):
    spec = build_distributed_program(kind, ranks=ranks)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    cpp = emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert 'ctx.distributed_registry()' in cpp
    assert '::map<pops::kNativeDimension>' in cpp
    assert 'accepted_after_distributed_checks' in cpp
    if kind != 'user_payload':
        assert '::requests(' in cpp and '::import_values(' in cpp
        assert '::csr_offsets(' in cpp and '::import_csr(' in cpp
        assert '::at_local(' in cpp and '7001' in cpp
    else:
        assert '7002' in cpp


def test_unhandled_corrupt_import_is_a_real_case_with_a_guarded_commit():
    spec = build_distributed_program('corrupt_data', ranks=3, recover=False)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    cpp = emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert '::at_local(' in cpp and '7001' in cpp
    assert 'reject_typed_failure<' in cpp


def _terminal_domain_payload_case(kind, *, replicated):
    """Author a genuine failing Result, nested twice in Product and once in Variant."""
    from examples.api_v02.methods_T01_T12 import _storage, _initial
    from pops.time import FixedDt
    from pops.time._program.distributed_arrays import Distributed
    from pops.time.variant_types import Float64, Int64, Result, Variant, type_of
    case, layout, _, block, state, p, q, initial = _storage(
        'terminal_domain_payload', ('u',), (1.,))
    if kind not in ('domain', 'array'):
        raise ValueError('this graph helper supports only Domain and Array payloads')
    if replicated:
        domain = p.range_set(p.int64(2), capacity=2, ownership='replicated')
        leaf = p.map(domain, lambda g, i: g.index_value(i)) if kind == 'array' else domain
    else:
        distributed = Distributed(p)
        domain = distributed.domain((0, 1, 2), capacity=2)
        leaf = domain
        if kind == 'array':
            leaf = p.require_success(distributed.local_map(
                domain, lambda g, i: g.variant(Result(Int64, Int64), 'ok', distributed.gid(i)),
                byte_capacity=4096))
    tag = p.variant(Variant(detail=type_of(leaf)), 'detail', leaf)
    payload = p.product(error=p.product(envelope=tag))
    failure = p.variant(Result(Float64, type_of(payload)), 'err', payload)
    status = p.require_success(failure)
    p.commit(q.next, p.value('guarded_terminal_commit', status*q.n, at=q.next.point))
    p.step_strategy(FixedDt(.01))
    case.program(p)
    _initial(case, block[state], initial)
    return pops.resolve(pops.validate(case), layout=layout)


@pytest.mark.parametrize('kind', ('domain', 'array'))
def test_terminal_failure_retains_nested_distributed_domain_payload(kind):
    plan = _terminal_domain_payload_case(kind, replicated=False)
    cpp=emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert 'ctx.distributed_registry_if_present()' in cpp
    assert 'reject_typed_failure<' in cpp
    assert 'finite_structural::pack<' in cpp
    assert ('ContinuationKind::DistributedArray' if kind=='array' else
            'ContinuationKind::DistributedDomain') in cpp


def test_terminal_failure_guard_retains_local_index_signature_without_global_lowering():
    """The terminal codec accepts an exact local Index type, including nested paths."""
    from pops.time import Program
    from pops.time._program.distributed_arrays import Distributed
    from pops.time.variant_types import Int64, Product, Result, Variant, type_of
    from pops.codegen.program_emit_typed_failure import _require_supported_transport

    p = Program('terminal_local_index_signature')
    distributed = Distributed(p)
    domain = distributed.domain((0, 1, 2), capacity=2)
    index_types = []

    def local_body(g, index):
        index_types.append(type_of(index))
        return g.variant(Result(Int64, Int64), 'ok', distributed.gid(index))

    distributed.local_map(domain, local_body, byte_capacity=4096)
    assert len(index_types) == 1
    signature = Product(error=Product(envelope=Variant(detail=index_types[0]))).signature
    _require_supported_transport(signature)


def test_terminal_failure_keeps_nested_replicated_array_payload():
    plan = _terminal_domain_payload_case('array', replicated=True)
    cpp = emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert 'reject_typed_failure<' in cpp
    assert 'finite_structural::pack<' in cpp
    assert all('ContinuationKind::' + kind in cpp for kind in ('Array', 'Variant', 'Product'))
    assert 'guarded_terminal_commit' in cpp
