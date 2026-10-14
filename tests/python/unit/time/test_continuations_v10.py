"""Source/IR/codegen tests only; these do not qualify native persistence or rollback."""
import pops
import pytest
from pops.continuations import ContinuationSchema, Candidate, Assess, assess, accept, Advance, advance_once
from pops.graph_functions import graph
from pops.time.variant_types import Product, Float64, Bool, Result, Failure, type_of
from pops.time import FixedDt
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.time._program.detach import detach_compiled_program
from examples.api_v02.methods_T01_T12 import _storage
from examples.api_v02.models_M01_M14 import _initial


def fixture():
    parts = _storage('Continuation_source', ('u',), (1.,))
    case, layout, model, block, state, p, q, data = parts
    initial = p.product(state=q.n, time=p.float64(0.), step=p.int64(0), counter=p.int64(0))
    schema = ContinuationSchema(initial, fields=((q.next, ('state',)),), time=('time',), step=('step',))
    base = p.continuation_open(initial, schema)
    proposal = p.product(h=p.float64(.1))
    return parts, initial, schema, base, proposal


def candidate_fixture():
    parts, initial, schema, base, proposal = fixture()
    p, q = parts[5:7]
    current = p.continuation_read(base)
    result = p.value('trial', current['state'] * 2., at=q.next.point)
    continuation = p.product(state=result, time=current['time']+proposal['h'],
                             step=current['step']+p.int64(1), counter=current['counter']+p.int64(1))
    candidate = Candidate(p, base=base, proposal=proposal, continuation=continuation,
                          increment=p.product(amount=proposal['h']), observations=p.product(), checks=p.product())
    envelope = p.continuation_candidate_value(candidate)
    @graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal), candidate=type_of(envelope)),
           output=Product(accepted=Bool, checks=Product(endpoint_error=Float64)))
    def check(g, args):
        defect = args['candidate']['continuation']['time'] - (args['base']['time']+args['proposal']['h'])
        return g.product(accepted=defect == g.float64(0.), checks=g.product(endpoint_error=defect))
    return parts, schema, base, proposal, candidate, Assess(check)


def resolve(parts):
    case, layout, model, block, state, p, q, data = parts
    p.step_strategy(FixedDt(.1));case.program(p);_initial(case,block[state],data)
    return pops.resolve(pops.validate(case),layout=layout)


def test_complete_candidate_assessment_publication_survives_codegen_and_detach():
    parts,schema,base,proposal,candidate,policy=candidate_fixture();p=parts[5]
    approved=p.require_success(accept(p,candidate,assess(p,candidate,policy)))
    p.continuation_publish(approved)
    plan=resolve(parts)
    cpp=emit_cpp_program(plan.time,model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert 'ctx.continuation_publish(' in cpp
    assert 'ctx.commit_many(' not in cpp
    assert 'continuation configuration construction' in cpp
    assert '.freeze(' in cpp and '.attest(' in cpp and '.accept(' in cpp
    assert 'ctx.continuation_control_consensus();' in cpp
    detached=detach_compiled_program(plan.time)
    assert detached._ir_hash()==plan.time._ir_hash()
    assert detached.to_graph().to_data()==plan.time.to_graph().to_data()


def test_wrong_field_point_and_unbound_field_fail_closed():
    parts,initial,schema,base,proposal=fixture();p=parts[5]
    with pytest.raises(ValueError, match='exact structural'):
        Candidate(p,base=base,proposal=proposal,continuation=initial,increment=p.product(),observations=p.product(),checks=p.product())
    with pytest.raises(NotImplementedError, match='every persistent field'):
        ContinuationSchema(initial,fields=(),time=('time',),step=('step',))


def test_assessment_requires_closed_equation_graph_and_capabilities_cannot_persist():
    parts,schema,base,proposal,candidate,policy=candidate_fixture();p=parts[5]
    with pytest.raises(TypeError, match='GraphFunction'):
        p.continuation_attest(candidate,lambda *a: True)
    with pytest.raises(TypeError, match='persist'):
        ContinuationSchema(p.product(time=p.float64(0.),step=p.int64(0),cap=candidate),fields=(),time=('time',),step=('step',))


def test_transform_retains_native_supersede_and_exact_assessment_identity():
    parts,schema,base,proposal,candidate,policy=candidate_fixture();p=parts[5]
    receipt=assess(p,candidate,policy)
    transformed=p.continuation_transform(candidate,p.continuation_candidate_value(candidate))
    # Structurally identical old receipt deliberately reaches native exact-identity guard.
    approved=p.continuation_accept(transformed,receipt)
    p.continuation_publish(approved)
    plan=resolve(parts);cpp=emit_cpp_program(plan.time,model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert '.supersede(' in cpp and '.accept(' in cpp
    assert policy.body.identity in cpp


def test_child_field_frontier_is_explicitly_unqualified():
    parts,schema,base,proposal,candidate,policy=candidate_fixture();p=parts[5]
    approved=p.continuation_accept(candidate,assess(p,candidate,policy))
    with pytest.raises(NotImplementedError, match='field-frontier'):
        p.continuation_child_base(approved)


def method_fixture():
    parts,schema,base,proposal,candidate,policy=candidate_fixture();p,q=parts[5:7]
    envelope=p.continuation_candidate_value(candidate)
    result_type=Result(type_of(envelope),Failure())
    endpoint=q.next.point
    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal)),output=result_type)
    def trial(g,args):
        before=args['base'];h=args['proposal']['h']
        value=g.value('trial_body',before['state']*2.,at=endpoint)
        after=g.product(state=value,time=before['time']+h,step=before['step']+g.int64(1),counter=before['counter']+g.int64(1))
        image=g.product(continuation=after,increment=g.product(amount=h),observations=g.product(),checks=g.product())
        return g.variant(result_type,'ok',image)
    return parts,schema,base,proposal,Advance(trial,policy)


def test_scoped_advance_promotes_complete_child_and_revokes_scope_on_failure():
    parts,schema,base,proposal,method=method_fixture();p=parts[5]
    accepted=p.require_success(advance_once(p,method,base=base,proposal=proposal))
    p.continuation_publish(accepted)
    plan=resolve(parts)
    cpp=emit_cpp_program(plan.time,model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert '.begin_scope()' in cpp
    assert '.promote(std::move(*continuation_scope_' in cpp
    assert '.reset();' in cpp
    assert 'ctx.continuation_local_phase' in cpp
    assert 'ctx.commit_many(' not in cpp


def test_bounded_retry_uses_native_loop_and_unchanged_scientific_base():
    from pops.continuations import advance
    from pops.time.variant_types import Int64
    parts,schema,base,proposal,method=method_fixture();p=parts[5]
    controller=p.product(reductions=p.int64(0))
    errors=Failure(assessment_rejected=Product(checks=Product(endpoint_error=Float64)))
    response_type=Result(Product(proposal=type_of(proposal),controller=type_of(controller)),Failure())
    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal),controller=type_of(controller),failure=errors),output=response_type)
    def retry(g,args):
        answer=g.product(proposal=g.product(h=args['proposal']['h']*.5),
                         controller=g.product(reductions=args['controller']['reductions']+g.int64(1)))
        return g.variant(response_type,'ok',answer)
    approved=p.require_success(advance(p,Advance(method.trial,method.assessment,retry),base=base,proposal=proposal,
                                      controller=controller,max_attempts=p.int64(3)))
    p.continuation_publish(approved)
    plan=resolve(parts);cpp=emit_cpp_program(plan.time,model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert 'iteration_stop_' in cpp
    assert cpp.count('.begin_scope()')==2
    assert 'continuation_budget' in str(plan.time.to_graph().to_data())


def test_legacy_history_and_amr_publication_are_refused():
    from pops.time._program.versioned_values import validate_continuations
    parts,schema,base,proposal,candidate,policy=candidate_fixture();p=parts[5]
    p.continuation_publish(p.continuation_accept(candidate,assess(p,candidate,policy)))
    with pytest.raises(NotImplementedError,match='Uniform'):
        validate_continuations(p,target='amr')
    p._histories['undeclared_sidecar']=1
    with pytest.raises(NotImplementedError,match='history'):
        validate_continuations(p)


def test_schema_projection_change_is_not_structural_shape_equality():
    parts,schema,base,proposal,candidate,policy=candidate_fixture();p=parts[5]
    accepted=p.continuation_accept(candidate,assess(p,candidate,policy));p.continuation_publish(accepted)
    # Tamper with a commit after authoring: the strict post-load validator checks provenance too.
    p._commits[parts[6].next.state]=p.value('foreign_same_shape',parts[6].n*3.,at=parts[6].next.point)
    from pops.time._program.versioned_values import validate_continuations
    with pytest.raises(ValueError,match='authenticated projection'):
        validate_continuations(p)


@pytest.mark.parametrize("mode",("success","retry","assessment_reject","stale_transform","wrong_candidate","parent_reject","duplicate_increment","loop_identity"))
def test_deferred_native_receivers_resolve_real_continuation_cpp(mode):
    from examples.api_v02.continuation_checks import build_continuation
    witness=build_continuation(mode=mode,steps=3)
    plan=pops.resolve(pops.validate(witness.case),layout=witness.layout)
    source=emit_cpp_program(plan.time,model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert "ctx.continuation_new_domain(" in source
    assert "ctx.continuation_publish(" in source
    assert "pops_finite_instance_version" not in source
    if mode=="duplicate_increment":assert source.count(".consume_increment(")==2
    if mode=="loop_identity":assert "for (std::int64_t" in source
