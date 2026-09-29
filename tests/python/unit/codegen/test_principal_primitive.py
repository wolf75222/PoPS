"""Joint primitive maps are physical declarations, independent of storage grouping."""
import pops
from pops.frames import Cartesian2D


def test_joint_coordinates_accept_cross_state_recovery_and_explicit_inverse():
    model=pops.Model('joint',frame=Cartesian2D())
    mass=model.species('mass',state=('rho',))
    momentum=model.species('momentum',state=('m',))
    rho,=mass
    m,=momentum
    velocity=model.primitive('velocity',m/rho)
    model.primitive_state(rho,velocity,states=(mass,momentum),conservative=(rho,rho*velocity))
    model.recovery_admissibility(states=(mass,momentum),rho=rho>0)
    assert len(model.module.primitive_coordinates()) == 1


import pytest
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from tests.python.support.principal_primitive_case import primitive_case


@pytest.mark.parametrize("widths",((1,1),(2,3),(1,4)))
@pytest.mark.parametrize("reverse",(False,True))
def test_joint_primitive_map_resolves_full_flux_and_keeps_exact_native_variable_mode(widths,reverse):
    case,layout,*_=primitive_case(widths,reverse=reverse)
    plan=pops.resolve(pops.validate(case),layout=layout)
    graph=ProgramModelGraph.from_resolved_blocks(plan.blocks)
    source=emit_cpp_program(plan.time,model=graph)
    assert ',pops::nd::ReconstructionVariables::Primitive>' in source
    assert 'primitive_domain(result.value)' in source
    assert 'if(!primitive_domain(p))' in source
    assert 'parameter_sets[0]' in source and 'parameter_sets[1]' in source
    assert 'params.get(0)' in source
    assert 'pops_principal_p' not in source


def test_primitive_group_requires_authored_inverse_and_coordinates():
    case,layout,*_=primitive_case(coordinates=False)
    with pytest.raises((ValueError,NotImplementedError),match='primitive_state|joint.*coordinate'):
        pops.resolve(pops.validate(case),layout=layout)


def test_joint_map_hashes_domain_and_refuses_foreign_coordinate_or_parameter():
    model=pops.Model('same_name',frame=Cartesian2D())
    a=model.species('a',state=('rho',)); b=model.species('b',state=('m',))
    rho,=a; m,=b
    p=model.primitive('v',m/rho)
    model.primitive_state(rho,p,states=(a,b),conservative=(rho,rho*p))
    before=model.module.module_hash()
    model.recovery_admissibility(states=(a,b),rho=rho>0)
    assert model.module.module_hash() != before
    data=model.module.manifest().to_dict()
    assert data['schema_version'] == 10
    assert data['expressions']['primitive_coordinates']
    from pops.model._module_manifest import ModuleManifest
    assert ModuleManifest.from_dict(data).to_dict() == data
    other=pops.Model('same_name',frame=Cartesian2D())
    aa=other.species('a',state=('rho',)); bb=other.species('b',state=('m',))
    foreign=other.primitive('v',bb[0]/aa[0])
    with pytest.raises(ValueError,match='owned|foreign|another'):
        other.primitive_state(aa[0],p,states=(aa,bb),conservative=(aa[0],aa[0]*p))
    with pytest.raises(ValueError,match='another|foreign'):
        other.primitive_state(aa[0],foreign,states=(a,bb),conservative=(aa[0],aa[0]*foreign))


def test_joint_map_order_is_independent_of_the_principal_pack_order():
    from pops.codegen.principal_lowering import principal_for_value
    case,layout,*_=primitive_case((2,3),reverse=True,map_reverse=True)
    plan=pops.resolve(pops.validate(case),layout=layout)
    graph=ProgramModelGraph.from_resolved_blocks(plan.blocks)
    source=emit_cpp_program(plan.time,model=graph)
    assert 'ReconstructionVariables::Primitive' in source
    model=graph.model_for_block('block0')
    entry=model._m._principal_groups[0]
    assert tuple(map(len,entry['conversion']['forward'])) == (2,3)
    assert tuple(map(len,entry['conversion']['inverse'])) == (2,3)
    assert tuple(map(len,entry['conversion']['constraints'])) == (1,0)


def test_single_state_coordinate_authoring_keeps_the_existing_native_contract():
    model=pops.Model('single_primitive',frame=Cartesian2D())
    state=model.state('state',components=('rho','m'))
    rho,m=state
    v=model.primitive('v',m/rho)
    model.primitive_state(rho,v,conservative=(rho,rho*v))
    model.recovery_admissibility(rho=rho>0)
    assert model._dsl._m.prim_state == ['rho','v']
    assert model.module.primitive_coordinates() == ()


def test_joint_primitive_sampling_promotes_every_rows_storage_halo():
    case,layout,*_=primitive_case((2,3),sample_offsets=(0,2))
    plan=pops.resolve(pops.validate(case),layout=layout)
    graph=ProgramModelGraph.from_resolved_blocks(plan.blocks)
    for block in plan.blocks:
        numerical = block.numerics
        expected = max(method.ghost_depth for group in numerical.principal_groups
                       for method in group.methods)
        assert numerical.primary_spatial().runtime_spatial().ghost_depth == expected
        native = graph.model_for_block(block.name)._m
        assert native._program_state_ghost_depth == expected
