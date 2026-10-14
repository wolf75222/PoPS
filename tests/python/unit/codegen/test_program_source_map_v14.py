"""Exact generated-source correspondence; no compilation or execution claim."""
import hashlib
import json

import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_source_map import source_map
from pops.graph_functions import GraphFunction, graph
from pops.time import Program
from pops.time.variant_types import Float64, Product
from typed_program_support import typed_state


def _case(coefficient=2, roundtrip=False):
    @graph(inputs=Product(x=Float64), output=Float64)
    def f(p, a):
        return p.fold(a['x'], 2, lambda b, state, i: b.branch(
            state > 0, lambda c: state*coefficient, lambda c: state-1), max_steps=2)
    if roundtrip:
        f = GraphFunction.from_data(json.loads(json.dumps(f.to_data())))
    p = Program('source_map')
    q = typed_state(p, 'material', state_name='U')
    q.n
    answer = f(p, x=p.float64(1))
    p.commit(q.next, p.value('answer', answer*q.n, at=q.next.point))
    return p, emit_cpp_program(p)


def test_nested_regions_own_disjoint_exact_lines_and_actual_bindings():
    p, cpp = _case()
    data = source_map(cpp)
    assert data['source_sha256'] == hashlib.sha256(cpp.encode()).hexdigest()
    assert data['evidence'] == 'source_correspondence_only'
    assert {'structured_fold', 'branch', 'scalar_op'} <= {row['op'] for row in data['operations']}
    lines = cpp.splitlines()
    owned = set()
    for row in data['operations']:
        for first, last in row['ranges']:
            assert 1 <= first <= last <= len(lines)
            for line in range(first, last+1):
                assert line not in owned
                owned.add(line)
                assert '// pops-source-map@1' not in lines[line-1]
        if row['identifier_binding']:
            assert any(row['identifier_binding'] in lines[line-1]
                       for first,last in row['ranges'] for line in range(first,last+1))
    assert owned
    assert len(owned) < len(lines)
    assert data['inventories'][0]['program_hash'] == p._ir_hash()


def test_from_data_and_repeated_lowering_have_identical_map_and_cpp():
    p, original = _case()
    _, loaded = _case(roundtrip=True)
    assert original == loaded == emit_cpp_program(p)
    assert source_map(original) == source_map(loaded)


def test_changed_equation_changes_source_and_map_hash():
    _, left = _case(2)
    _, right = _case(3)
    assert left != right
    assert source_map(left)['source_sha256'] != source_map(right)['source_sha256']
    assert source_map(left)['map_sha256'] != source_map(right)['map_sha256']


def test_mapping_refuses_truncation_instead_of_widening_a_range():
    _, cpp = _case()
    lines = cpp.splitlines()
    end = next(i for i,line in enumerate(lines) if '// pops-source-map@1 end ' in line)
    del lines[end]
    with pytest.raises(ValueError, match='unbalanced|unterminated|inside an open'):
        source_map('\n'.join(lines))
    plain = source_map('int actual_symbol = 1;\n')
    assert plain['operations'] == [] and plain['complete'] is False


def test_distributed_local_fold_and_aliases_are_mapped_without_fake_ranges():
    from pops.time._program.distributed_arrays import Distributed
    p = Program('local_mapping')
    q = typed_state(p, 'material', state_name='U'); q.n
    d = Distributed(p); domain = d.domain((0, 1, 1, 4), capacity=4)
    values = p.map(domain, lambda b,i: b.fold(d.gid(i), 2,
        lambda c,s,k: s+k, max_steps=2), byte_capacity=65536)
    total = p.reduce(values, p.int64(0), lambda b,s,x:s+x,
                     order='global_gid_ascending', byte_capacity=65536)
    p.commit(q.next, p.value('answer', p.float64(total)*q.n, at=q.next.point))
    data = source_map(emit_cpp_program(p))
    local = [row for row in data['operations'] if row['channel'] == 'distributed_local']
    assert any(row['op'] == 'structured_fold' and row['ranges'] for row in local)
    assert any(row['op'] == 'int64_op' and row['ranges'] for row in local)
    aliases = [row for row in data['operations'] if not row['ranges']]
    assert aliases and all(row['status'] == 'no_direct_statements' for row in aliases)


def test_multiple_realized_occurrences_retain_the_exact_use_metadata():
    from pops.frames import Cartesian2D
    from pops.math import ddt
    from pops.physics import Model
    from pops.realization import Realize, Use
    from pops.codegen.module_lowering import lower_and_validate
    from typed_program_support import state_refs
    m = Model('two_occurrences', frame=Cartesian2D())
    state = m.state('U', components=('x','y'))
    source = m.source('exchange', on=state, value=(-state[0], state[0]))
    balance = m.rate('equation', equation=ddt(state) == source+source)
    p = Program('two_occurrences')
    block,_ = state_refs(p, 'material', model=m, state=state)
    q = p.state(block[state])
    uses = tuple(Use(item, 'cell', 'current_stage', 'inventory') for item in balance.occurrences)
    assert len(uses) == 2
    results = []
    for i,use in enumerate(uses):
        results.append(Realize('same_name', owns=(use,), bindings={'old':q.n},
            body=lambda b,old:b.value('flux', (-old[0],old[0])), outputs={use:()}).evaluate(p)[use])
    p.commit(q.next, p.value('answer', q.n+p.dt*(results[0]+results[1]), at=q.next.point))
    lowered,_ = lower_and_validate(m, facade=m)
    data = source_map(emit_cpp_program(p, model=lowered))
    mapped = {json.dumps(row['realization_use_v02']['use'],sort_keys=True):row
              for row in data['operations'] if 'realization_use_v02' in row and row['ranges']}
    assert set(mapped) == {json.dumps(use.to_data(),sort_keys=True) for use in uses}
    assert len({row['ssa'] for row in mapped.values()}) == 2


def test_compiled_handle_exposes_only_its_retained_program_source():
    from pops.codegen.loader import CompiledProblem
    from pops.codegen._compiled_artifact import CompiledSimulationArtifact
    from types import SimpleNamespace
    _, cpp = _case()
    handle = object.__new__(CompiledProblem)
    handle._generated_cpp = cpp
    assert handle.source_maps['program'] == source_map(cpp)
    artifact = SimpleNamespace(layout_programs=(), program=handle)
    assert CompiledSimulationArtifact.source_maps.fget(artifact) == handle.source_maps
    artifact.layout_programs = (SimpleNamespace(layout_id='mesh', program=handle),)
    assert list(CompiledSimulationArtifact.source_maps.fget(artifact)) == ['layout:mesh']
    handle._generated_cpp = None
    assert handle.source_maps == {}


def test_strict_scientific_realization_through_newton_array_graph_has_effective_ranges():
    import pops
    from examples.api_v02.scientific_spatial_v14 import build_scientific_spatial_v14
    from pops.codegen.program_models import ProgramModelGraph
    spec = build_scientific_spatial_v14(count=3)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    cpp = emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    data = source_map(cpp)
    assert data['complete']
    assert not [row for row in data['unlocated'] if row['status'] == 'not_emitted_or_unmapped']
    assert {'structured_iterate', 'finite_map'} <= {row['op'] for row in data['operations']}
    uses = [row for row in data['operations'] if 'realization_use_v02' in row
            and row['channel'] != 'install']
    assert len(uses) >= 2
    for row in uses:
        assert row['realization_use_v02']['use']['occurrence']
        assert row['realization_use_v02']['checking'] == 'strict'
        assert row['effective_ranges']
        if not row['ranges']:
            assert row['location_basis'] == 'identical_emitter_binding'
            assert row['binding_origins']
    assert any('scientific_graph_v14' in row for row in data['operations'])


def test_artifact_spec_binds_map_digest_to_the_full_emitted_source(monkeypatch):
    from pops.codegen._artifact_identity import program_artifact_spec
    from types import SimpleNamespace
    import pops.identity
    import pops.identity.semantic
    import pops.codegen.toolchain
    _, cpp = _case()
    monkeypatch.setattr(pops.identity, 'artifact_spec_identity', lambda semantic, **kwargs: kwargs)
    monkeypatch.setattr(pops.identity.semantic, 'semantic_identity_of', lambda **kwargs: 'semantic')
    monkeypatch.setattr(pops.codegen.toolchain, '_native_feature_key', lambda: 'fixture-native')
    _, spec = program_artifact_spec(snapshot=object(), model_authority=None,
        program=SimpleNamespace(name='source_map'), program_graph=SimpleNamespace(graph_hash='graph'),
        target='system', abi_key='fixture', compiler='c++', standard='c++20', source=cpp,
        cflags=(), lflags=(), optflags=(), libraries=(), native_dimension=2)
    assert spec['components']['generated_source'].hex() == source_map(cpp)['source_sha256']
    assert spec['components']['program_source_map'] == source_map(cpp)['map_sha256']


def _profile_source_map(identifier):
    import pops
    from examples.api_v02.reception import build
    from pops.codegen._orchestration_compile import build_program_model_graph
    from pops._balance_due_contract import BalanceDueContract
    spec = build(identifier)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    cpp = emit_cpp_program(plan.time, target=plan.target,
        model_graph=build_program_model_graph(plan), field_plans=plan.field_plans,
        balance_due_contract=BalanceDueContract.from_consumer_graph(plan.consumer_graph))
    return cpp, source_map(cpp)


def test_field_operator_bodies_and_same_named_formals_keep_distinct_lambda_regions():
    cpp, data = _profile_source_map('M08')
    assert data['complete'] and not data['unlocated']
    rows = data['operations']
    outputs = [r for r in rows if r['op'] == 'apply_out' and r['channel'] == 'operator_apply']
    assert len(outputs) >= 2
    assert len({tuple(r['lexical_regions']) for r in outputs}) == len(outputs)
    for row in outputs:
        assert row['cpp_binding'] == 'out' and not row['ranges']
        assert row['binding_origins'] and row['effective_ranges']
        for origin in row['binding_origins']:
            producer = rows[origin]
            assert producer['op'] == 'field_problem_apply'
            assert producer['lexical_regions'] == row['lexical_regions']
            assert any('apply_general_field' in cpp.splitlines()[line-1]
                for first,last in producer['ranges'] for line in range(first,last+1))


def test_binding_alias_does_not_borrow_an_identical_name_from_another_lambda():
    from pops.codegen.program_source_map import _mark
    def start(ssa, op):
        return _mark('begin', dict(ssa=ssa, op=op, name=op, inputs=[],
                                  program_hash='fixture', channel='operator_apply'))
    def end(ssa, op, binding=None):
        return _mark('end', dict(ssa=ssa, op=op, binding=binding))
    cpp = '\n'.join([start(1,'lambda'), 'auto first = [] {', start(2,'producer'),
        'double out = 1;', end(2,'producer','out'), '};', end(1,'lambda'),
        start(3,'lambda'), 'auto second = [](double out) {', start(4,'formal'),
        end(4,'formal','out'), '};', end(3,'lambda')])
    data = source_map(cpp)
    alias = next(r for r in data['operations'] if r['ssa'] == 4)
    assert alias['cpp_binding'] == 'out'
    assert alias['binding_origins'] == [] and alias['effective_ranges'] == []


@pytest.mark.parametrize('identifier,channel,opcode', [
    ('T06', 'local_residual', 'pointwise_expression'),
    ('T08', 'post_synchronization', 'post_synchronization'),
])
def test_direct_residual_and_post_sync_routes_are_mapped(identifier, channel, opcode):
    _, data = _profile_source_map(identifier)
    assert data['complete'] and not data['unlocated']
    assert any(r['channel'] == channel and r['op'] == opcode and r['ranges']
               for r in data['operations'])


def test_coalesced_rhs_outputs_share_only_the_exact_native_group_call():
    cpp, data = _profile_source_map('M22')
    assert data['complete'] and not data['unlocated']
    rows = [r for r in data['operations'] if r['channel'] == 'rhs_group_call']
    assert len(rows) == 4
    groups = {}
    for row in rows:
        groups.setdefault(row['shared_emission_group'], []).append(row)
        assert row['op'] == 'rhs'
        assert len(row['ranges']) == 1 and row['ranges'][0][0] == row['ranges'][0][1]
        assert 'ctx.rhs_group(' in cpp.splitlines()[row['ranges'][0][0]-1]
    assert len(groups) == 2
    for members in groups.values():
        assert len(members) == 2
        assert {r['ssa'] for r in members} == set(members[0]['shared_ssa'])
        assert members[0]['ranges'] == members[1]['ranges']


def test_uniform_spatial_residual_diffusion_is_located_at_its_emitted_call():
    cpp, data = _profile_source_map('T02')
    assert data['complete'] and not data['unlocated']
    rows = [r for r in data['operations'] if r['op'] == 'diffusive_rhs']
    assert rows and all(r['ranges'] for r in rows if r['channel'] != 'install')
    assert any('diffusive' in cpp.splitlines()[line-1]
               for r in rows for first,last in r['ranges'] for line in range(first,last+1))


def test_amr_spatial_residual_diffusion_and_formal_have_exact_emitter_regions():
    from pops.codegen.module_lowering import lower_and_validate
    from tests.python.unit.time.test_implicit_diffusion_request import resolve_case
    plan = resolve_case(nonlinear=True)
    cpp = emit_cpp_program(plan.time, model=lower_and_validate(plan.blocks[0].model)[0],
                           target='amr_system')
    data = source_map(cpp)
    assert data['complete'] and not data['unlocated']
    rows = [r for r in data['operations'] if r['op'] == 'diffusive_rhs'
            and r['channel'] != 'install']
    assert rows and all(r['ranges'] for r in rows)
    assert any(r.get('binding_role') == 'lexical_formal' for r in data['operations'])


def test_balanced_removal_of_computing_region_is_incomplete_despite_retained_binding():
    from pops.codegen.program_source_map import _mark
    begin = _mark('begin', dict(ssa=2, op='scalar_op', name='compute', inputs=[],
                               program_hash='fixture', channel='body'))
    end = _mark('end', dict(ssa=2, op='scalar_op', binding='answer'))
    inventory = _mark('inventory', dict(program_hash='fixture', nodes=[
        dict(ssa=2, op='scalar_op', name='compute', inputs=[], cpp_bindings=['answer'])]))
    intact = '\n'.join((begin, 'const double answer = 1 + 2;', end, inventory))
    assert source_map(intact)['complete']
    missing = source_map('const double answer = 1 + 2;\n'+inventory)
    assert not missing['complete']
    assert missing['unlocated'][0]['op'] == 'scalar_op'
    assert missing['unlocated'][0]['cpp_bindings'] == ['answer']
    assert missing['unlocated'][0]['status'] == 'not_emitted_or_unmapped'
