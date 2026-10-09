"""Actual typed mathematical producer snapshots, independent of case names/formulas."""
from fractions import Fraction
from dataclasses import replace
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import pops
from pops import math
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.fields import FieldProblem,FieldDiscretization,FieldBoundary,bcs
from pops.fields.methods import CellCenteredSecondOrder
from pops.model import Handle,OwnerPath
from pops.mesh import CartesianGrid,PeriodicAxes
from pops.layouts import Uniform
from pops.numerics import Diffusion,DiscretizationPlan
from pops.solvers import CG
from pops.time import Program,FailRun,FixedDt
from pops.codegen import Production
from pops.codegen._orchestration_compile import build_program_model_graph
from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp
from pops.codegen.program_codegen import emit_cpp_program


def composed(*, stages=2, fields=2, suffix='', foreign=False, mutation=False, negative=False):
    frame=CartesianDomain('domain'+suffix,(0.,),(1.,)).frame(Cartesian1D())
    model=pops.Model('free-definition'+suffix,frame=frame)
    U=model.state('material'+suffix,components=('m',))
    names=tuple('property'+str(i)+suffix for i in range(fields))
    quantities=tuple(model.aux(name) for name in names)
    coefficient=1+sum(x**2 for x in quantities)
    flux=model.diffusive_flux('constitutive'+suffix,state=U,value=coefficient*math.grad(U[0]))
    rate=model.rate('balance'+suffix,equation=math.ddt(U)==math.div(flux))
    case=pops.Case('composed'+suffix);block=case.block('receiver'+suffix,model)
    methods=DiscretizationPlan();methods.rates.add(rate,Diffusion(flux=flux));case.numerics(methods,block=block)
    foreign_ref=None;foreign_rate=None
    if foreign:
        other=pops.Model('other'+suffix,frame=frame);V=other.state('other_state'+suffix,components=('v',))
        other_flux=other.diffusive_flux('other_flux'+suffix,state=V,value=Fraction(1,10)*math.grad(V[0]))
        foreign_rate=other.rate('other_rate'+suffix,equation=math.ddt(V)==math.div(other_flux))
        other_block=case.block('other_block'+suffix,other);foreign_ref=other_block[V]
        other_methods=DiscretizationPlan();other_methods.rates.add(foreign_rate,Diffusion(flux=other_flux));case.numerics(other_methods,block=other_block)
    unknowns=tuple(Handle('unknown'+str(i)+suffix,kind='field',owner=OwnerPath.model('equations'+suffix)) for i in range(fields))
    equations=tuple(-math.laplacian(x)+math.Reaction(x,i+1)==(i+1)*U[0]-(V[0] if foreign else 0) for i,x in enumerate(unknowns))
    problem=FieldProblem('registered-system'+suffix,unknowns=unknowns,equations=equations,
        boundaries=tuple(FieldBoundary(x,bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(),bcs.Periodic())) for x in unknowns))
    field=case.field(problem,FieldDiscretization(method=CellCenteredSecondOrder(),boundaries=(),solver=CG(max_iter=100,rel_tol=1e-12,abs_tol=1e-14)))
    p=Program('authored-composition'+suffix);t=p.state(block[U]);other_t=p.state(foreign_ref) if foreign else None
    carrier=block[model.module.field_handle(model.module.field_spaces()['fields'])]
    def snapshot(state,point):
        bindings={block[U]:state}
        if foreign:bindings[foreign_ref]=other_t.n
        solved=field.observe(p.solve(field,values=bindings,at=point).consume(action=FailRun()))
        return solved.publish({(carrier,name):solved[field[x]] for name,x in zip(names,unknowns)},states={block[U]:state})
    point0=p.stage('zero'+suffix,c=0);pub0=snapshot(t.n,point0)
    if mutation:snapshot(t.n,point0)
    r0=rate(t.n,pub0)
    point1=p.stage('one'+suffix,c=1);predictor=p.value('predictor'+suffix,t.n+p.dt*r0,at=point1)
    if stages==1:end=p.value('endpoint'+suffix,t.n+p.dt*r0,at=t.next.point);pub1=None;r1=None
    else:
        pub1=snapshot(predictor,point1);r1=rate(predictor,pub1)
        expression=(t.n+predictor+p.dt*r1)/2 if not negative else (t.n+predictor)/2+p.dt*r0-p.dt*r1/2
        end=p.value('endpoint'+suffix,expression,at=t.next.point)
    p.commit(t.next,end)
    if foreign:p.commit(other_t.next,p.value('other_endpoint'+suffix,other_t.n+p.dt*foreign_rate(other_t.n),at=other_t.next.point))
    p.step_strategy(FixedDt(1e-5));case.program(p)
    layout=Uniform(CartesianGrid(frame=frame,cells=(8,),periodic=PeriodicAxes(frame.axes)))
    resolved=pops.resolve(pops.validate(case),layout=layout,backend=Production())
    graph=build_program_model_graph(resolved)
    return p,graph,resolved,{'pub0':pub0,'pub1':pub1,'r0':r0,'r1':r1,'predictor':predictor,'t':t,'end':end}


def prove(case):
    p,g,_,_=case;return prove_accepted_update_ssp(p,model_authority=g)

@pytest.mark.parametrize('fields,stages',[(1,1),(2,1),(1,2),(2,2)])
def test_joint_fields_and_arbitrary_fe_composition(fields,stages):
    case=composed(fields=fields,stages=stages);p,g,resolved,_=case
    before=p.to_graph().graph_hash;certificate,reason=prove(case)
    assert certificate is not None,reason
    assert p.to_graph().graph_hash==before
    assert len(certificate.contributing_rates)==stages
    code=emit_cpp_program(p,model_graph=g)
    assert code.count('.stage_accepted_exchanges(')==stages
    assert 'combined_transport_diffusion_stability' in code


def test_public_resolution_factory_supplies_field_authority_automatically():
    case=composed();p,g,resolved,_=case
    assert resolved.program_field_plans and not resolved.field_plans
    assert g._resolved_program_field_sources
    certificate,reason=prove(case);assert certificate is not None,reason


def test_rename_keeps_numeric_convex_decomposition():
    first,_=prove(composed());second,reason=prove(composed(suffix='_unrelated'))
    assert second is not None,reason
    assert (first.A,first.b,first.c)==(second.A,second.b,second.c)


def test_foreign_evolving_state_refuses_without_joint_owner_premise():
    certificate,reason=prove(composed(fields=1,stages=1,foreign=True))
    assert certificate is None and 'Field/evaluation' in reason


def test_intervening_publication_refuses_exact_read_token():
    certificate,reason=prove(composed(stages=1,mutation=True))
    assert certificate is None and 'replaced' in reason


def test_negative_accepted_weights_remain_refused():
    certificate,reason=prove(composed(negative=True))
    assert certificate is None and 'negative' in reason

@pytest.mark.parametrize('damage',['clock','point','region','context','component','source_hash','field_plan_hash','load_payload','resolved_plan_payload','apply_binder_role','emitted_pack'])
def test_stage_snapshot_authentication_refusals(damage):
    case=composed();p,g,resolved,v=case;pub=v['pub1']
    if damage=='clock':
        point=pub.point.time if hasattr(pub.point,'time') else pub.point
        object.__setattr__(pub,'point',replace(point,clock=replace(pub.clock,name='changed')))
    if damage=='point':object.__setattr__(pub,'point',v['pub0'].point)
    if damage=='region':object.__setattr__(pub,'region',99)
    if damage=='context':
        from pops.time.field_context import FieldContext
        object.__setattr__(pub,'field_context',FieldContext(pub.field_context.field,((v['t'].n.block,v['t'].n.id),),pub.field_context.outputs))
    if damage=='component':
        attrs=dict(pub.attrs);rows=[dict(row) for row in attrs['bindings']];rows[0]['source_component']=9;attrs['bindings']=tuple(rows);object.__setattr__(pub,'attrs',attrs)
    if damage=='source_hash':
        from pops.identity import make_identity
        attrs=dict(pub.attrs);attrs['field_problem_identity']=make_identity('field-problem',{'changed':True}).token;object.__setattr__(pub,'attrs',attrs)
    if damage=='field_plan_hash':
        from pops.fields._identity import field_identity
        plan=next(iter(resolved.program_field_plans.values()));object.__setattr__(plan,'identity',field_identity('resolved-program-field',{'changed':True}))
    if damage=='load_payload':
        node=next(x for x in p._values if x.op=='field_problem_load');attrs=dict(node.attrs);attrs['expressions']=({'kind':'literal','value':{'kind':'integer','value':'0'}},);object.__setattr__(node,'attrs',attrs)
    if damage=='resolved_plan_payload':
        _,plan,_=g._resolved_provider_sources[v['t'].n.block.local_id]
        op=plan.operations[0];object.__setattr__(plan,'operations',(replace(op,stencil_radius=op.stencil_radius+1),*plan.operations[1:]))
    if damage=='apply_binder_role':
        node=next(v for v in p._values if v.op=='matrix_free_operator')
        apply=next(v for v in node.attrs['apply_block'] if v.op=='field_problem_apply')
        object.__setattr__(apply,'inputs',(apply.inputs[1],apply.inputs[0],apply.inputs[2]))
    if damage=='emitted_pack':
        from pops.codegen.program_emit_kernels import _model_impl
        from pops.model.provider_pack import ProviderPack
        model=_model_impl(g.model_for_block(v['t'].n.block));pack=model._auxiliary_provider_pack.to_data()
        pack['entries'][0]['provider']['producer']='runtime_input'
        object.__setattr__(model,'_auxiliary_provider_pack',ProviderPack.from_data(pack))
    certificate,reason=prove(case)
    assert certificate is None and reason


def test_production_program_driver_carries_authority_without_client_assembly(monkeypatch):
    from pops.codegen import _compile_drivers
    case=composed();_,_,resolved,_=case
    captured={};sentinel=object()
    def observe(**kwargs):
        captured.update(kwargs)
        certificate,reason=prove_accepted_update_ssp(kwargs['time'],model_authority=kwargs['model_graph'])
        assert certificate is not None,reason
        assert kwargs['model_graph']._resolved_program_field_sources
        return sentinel
    # Stop exactly at the ordinary native compiler handoff, after actual source
    # plan preparation. No compiler, Native loading or binary fabrication occurs.
    monkeypatch.setattr(_compile_drivers,'_compile_problem_impl',observe)
    assert _compile_drivers._compile_resolved_problem(resolved) is sentinel
    assert captured['time'] is resolved.time


@pytest.mark.parametrize("damage", ("load_boundary", "coefficient_apply_boundary"))
def test_current_registered_physical_boundary_authenticates_linear_operations(damage):
    case = composed()
    program, _, _, _ = case
    from pops.codegen.program_field_plan import _nodes
    affected = {"field_problem_load"} if damage == "load_boundary" else {
        "field_problem_coefficients", "field_problem_apply"}
    for node in _nodes(program):
        if node.op in affected:
            attrs = dict(node.attrs)
            assert attrs["physical_boundary"] == "periodic"
            attrs["physical_boundary"] = "homogeneous_neumann"
            object.__setattr__(node, "attrs", attrs)
    certificate, reason = prove(case)
    assert certificate is None and "registered physical boundary" in reason
