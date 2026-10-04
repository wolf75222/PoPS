"""Shared definitions remain reusable; native component addresses belong to instances."""
from dataclasses import replace
import json
import pytest
from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen._compiler_lowering import require_compiler_lowering
from pops.codegen.provider_instances import emitter_contract, runtime_key, require_instance_contract
from pops.codegen.component_provider_packs import require_emitter_provider_carrier
from tests.python.support.field_publication_instance_case import build, NAMES


@pytest.mark.parametrize('cells,reverse,permuted,different', (
    ((7,3),False,False,False), ((3,7),True,True,False),
    ((11,2),True,False,True), ((2,11),False,True,True)))
def test_actual_publication_and_native_routes_use_each_instance(cells,reverse,permuted,different):
    resolved=build(cells=cells,reverse=reverse,permuted=permuted,different_values=different)
    graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    code=emit_cpp_program(resolved.time,model_graph=graph)
    assert code==emit_cpp_program(resolved.time,model_graph=graph)
    keys=[]
    claims={}
    for block in resolved.blocks:
        emitter=graph.model_for_block(block.name)
        contract=emitter_contract(emitter)
        if block.name==NAMES[3]:
            assert contract is None
            continue
        assert block.declares_auxiliary_providers is True
        assert require_instance_contract(block.resolved_operations,owner_qid=block.instance_owner_qid)==contract
        model_cpp=require_compiler_lowering(emitter).native_loader_source(
            name='Instance_'+block.name,consumer_owner_qid=block.instance_owner_qid,
            declare_auxiliary_providers=block.declares_auxiliary_providers)
        pack=emitter._auxiliary_provider_pack
        for key in pack:
            projected=runtime_key(key,contract)
            assert projected.owner_qid==block.instance_owner_qid
            assert key.owner_qid==contract['definition_owner_qid'] # declaration retained
            keys.append(projected)
            marker='Key{%s, %s, %s, %s}'%tuple(json.dumps(x) for x in projected.to_data().values())
            assert marker in model_cpp # outputs, dependencies and consumer plans
        claim=block.resolved_operations.provider_evidence['program_field_publications'][0]
        claims[block.name]=claim
        assert ('{"%s", "field", "imposed solved field", "published_potential"}'%block.instance_owner_qid) in code
        retained=[row for row in resolved.continuation_transitions.to_data()['objects']
                  if row['kind']=='auxiliary' and row['name']==block.name]
        assert retained and all(row['validity']['owner_qid']==block.instance_owner_qid for row in retained)
        assert all(row['transitions']['rollback']['action']=='preserve' for row in retained)
    assert len(keys)==len(set(keys))==6 # three field outputs plus three dependent gains
    if different:
        assert claims[NAMES[0]]['unknown'] != claims[NAMES[1]]['unknown']
        assert claims[NAMES[0]]['unknown'] == claims[NAMES[2]]['unknown']
    # The explicit coupled evaluation takes its declared first State's provider instance,
    # independently of declaration order or which other block produces output rates.
    first=NAMES[2 if permuted else 0]
    contract=emitter_contract(graph.model_for_block(first))
    gain=next(k for k in graph.model_for_block(first)._auxiliary_provider_pack if k.space_kind=='aux')
    expected=runtime_key(gain,contract)
    assert ('Dependency{Key{%s, "aux", "nonlinear_gain", "nonlinear_gain"}'%json.dumps(expected.owner_qid)) in code


@pytest.mark.parametrize('corruption', ('definition','instance','contract'))
def test_instance_projection_rejects_foreign_authority(corruption):
    resolved=build();block=resolved.blocks[0]
    data=dict(block.resolved_operations.provider_evidence)
    contract=dict(data['native_provider_instance'])
    if corruption=='definition': contract['definition_owner_qid']='model_definition:foreign'
    elif corruption=='instance': contract['instance_owner_qid']=resolved.blocks[1].instance_owner_qid
    else: contract['contract']='unreviewed@99'
    data['native_provider_instance']=contract
    plan=replace(block.resolved_operations,provider_evidence=data)
    module=require_compiler_lowering(block.model).source_module
    with pytest.raises(ValueError): plan.require_provider_packs(module)


def test_formula_carrier_projection_is_attested_and_cannot_be_revoked():
    resolved=build();graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    emitter=graph.model_for_block(NAMES[0]);formula=getattr(emitter,'_m',emitter)
    original=formula._native_provider_instance
    object.__setattr__(formula,'_native_provider_instance',{**original,'instance_owner_qid':'foreign'})
    with pytest.raises(ValueError,match='witness'):require_emitter_provider_carrier(formula)
    object.__setattr__(formula,'_native_provider_instance',original)
    require_emitter_provider_carrier(formula)
    with pytest.raises(ValueError):
        require_compiler_lowering(emitter).native_loader_source(
            name='ForeignOwner',consumer_owner_qid=resolved.blocks[1].instance_owner_qid)


def test_runtime_projection_preserves_foreign_keys_and_local_slot_contracts():
    from pops.model.provider_pack import ComponentKey
    resolved=build();graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    emitter=graph.model_for_block(NAMES[0]);contract=emitter_contract(emitter)
    foreign=ComponentKey('field-owner','field','shared space','potential')
    assert runtime_key(foreign,contract) is foreign
    declared=emitter._auxiliary_provider_pack
    for key in declared:
        entry=declared.declared_entry(key)
        projected=runtime_key(key,contract)
        assert projected.space==key.space and projected.component==key.component
        assert entry.slot is not None and declared.contract(key).centering=='cell'


def test_distinct_published_unknown_consumes_its_exact_beta_instance():
    resolved=build(cells=(11,2),reverse=True,different_values=True,first_beta=True)
    graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    code=emit_cpp_program(resolved.time,model_graph=graph)
    beta=next(b for b in resolved.blocks if b.name==NAMES[1])
    assert ('Dependency{Key{%s, "aux", "nonlinear_gain", "nonlinear_gain"}'%json.dumps(beta.instance_owner_qid)) in code
    alpha=next(b for b in resolved.blocks if b.name==NAMES[0])
    assert beta.resolved_operations.provider_evidence['program_field_publications'][0]['unknown'] != \
        alpha.resolved_operations.provider_evidence['program_field_publications'][0]['unknown']


def test_program_consumer_rejects_unversioned_instance_table():
    from pops.codegen.program_emit_kernels import ProgramProviderPlans
    with pytest.raises(ValueError,match='invalid consumed provider instance contract'):
        ProgramProviderPlans(instance_contracts=({'contract':'unversioned'},))
