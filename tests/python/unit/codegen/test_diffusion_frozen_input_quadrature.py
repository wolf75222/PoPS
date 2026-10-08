"""Exact accepted-update convex proofs do not classify independent observations as RK stages."""
from fractions import Fraction
from types import SimpleNamespace
import pytest

from pops.codegen.program_diffusion_exchanges import (
    _single_forward_euler_with_frozen_inputs, accepted_diffusive_quadrature,
)
from tests.python.integration.runtime.test_public_drift_diffusion_matrix import _solved_potential_case


def _clone(value, **changes):
    fields = {name:getattr(value,name) for name in ("op","attrs","inputs","block","point","state_ref")}
    return SimpleNamespace(**{**fields,**changes})


def test_exact_euler_face_quadrature_survives_an_independent_unchanged_driver():
    case, _ = _solved_potential_case(16,1e-4)
    program = case._time
    rows = accepted_diffusive_quadrature(program)
    assert len(rows) == 1 and rows[0][1] == {1:Fraction(1)}
    assert _single_forward_euler_with_frozen_inputs(program,rows)
    assert any(value.op == "solve_outcome" for value in program._values)


@pytest.mark.parametrize("fault",("changed_driver","wrong_coefficient","wrong_seed","extra_rhs"))
def test_frozen_input_euler_proof_refuses_different_accepted_equations(fault):
    case, _ = _solved_potential_case(16,1e-4)
    program = case._time
    rows = accepted_diffusive_quadrature(program)
    rate = rows[0][0]
    commits = dict(program._commits)
    selected = next(key for key,value in commits.items() if any(item is rate for item in value.inputs))
    driver = next(key for key in commits if key != selected)
    end = commits[selected]
    if fault == "changed_driver":
        value = commits[driver]
        commits[driver] = _clone(value, attrs={**value.attrs,"coeffs":({0:2},)})
    elif fault == "wrong_coefficient":
        commits[selected] = _clone(end, attrs={**end.attrs,"coeffs":({0:1},{1:Fraction(1,2)})})
    elif fault == "wrong_seed":
        from pops.time.points import TimePoint
        seed = _clone(rate.inputs[0],point=TimePoint(program.clock,step=1))
        replacement = _clone(rate,inputs=(seed,*rate.inputs[1:]))
        commits[selected] = _clone(end,inputs=(end.inputs[0],replacement))
        rows = ((replacement,{1:Fraction(1)}),)
    else:
        extra = _clone(rate,op="rhs")
        commits[selected] = _clone(end,inputs=(*end.inputs,extra),
            attrs={**end.attrs,"coeffs":(*end.attrs["coeffs"],{1:1})})
    assert not _single_forward_euler_with_frozen_inputs(SimpleNamespace(_commits=commits),rows)


def _observed_step(composition="expanded", *, weights=(Fraction(1,2),Fraction(1,2)),
                   last_point=Fraction(1), suffix="", hidden_feedback=False,
                   observe_before=False, second_coefficient=None, primitive=False):
    import pops
    from pops import math
    from pops.time import Program, FailRun
    from pops.fields import FieldDiscretization, CellCenteredSecondOrder
    from pops.solvers.elliptic import GeometricMG
    from test_field_rhs_auxiliary_inputs import _field_model
    model,state,operator=_field_model(name="independent"+suffix)
    if hidden_feedback:
        from pops.domain import Rectangle
        from pops.frames import Cartesian2D
        from pops.fields import FieldOutput
        frame=Rectangle("feedback-box",(0.,0.),(1.,1.)).frame(Cartesian2D())
        model=pops.Model("typed-feedback"+suffix,frame=frame)
        state=model.state("density",components=("rho",))
        potential=model.field("potential")
        operator=model.field_operator("fields",unknown=potential,
            equation=-math.laplacian(potential)+math.Reaction(potential,1)==state[0],
            outputs=(FieldOutput("observed",potential),))
    coefficient=(model.aux("observed") if hidden_feedback else Fraction(1,10))
    quantity=model.primitive("owned-derived"+suffix,state[0]**3) if primitive else state[0]
    flux=model.diffusive_flux("conduct"+suffix,state=state,value=coefficient*math.grad(quantity))
    rate=model.rate("balance"+suffix,equation=math.ddt(state)==math.div(flux))
    second_rate=rate
    if second_coefficient is not None:
        second_flux=model.diffusive_flux("other-expression"+suffix,state=state,
            value=second_coefficient*math.grad(quantity))
        second_rate=model.rate("other-balance"+suffix,equation=math.ddt(state)==math.div(second_flux))
    case=pops.Case("effect-closure"+suffix);block=case.block("material"+suffix,model)
    field=case.field(operator,FieldDiscretization(
        method=CellCenteredSecondOrder(),boundaries=(),solver=GeometricMG()))
    p=Program("data-expression"+suffix);u=p.state(block[state])
    if observe_before:p.record_scalar("independent-read",p.norm2(u.n))
    initial_fields=field(u.n).consume(action=FailRun()) if hidden_feedback else None
    first=rate(u.n,initial_fields) if hidden_feedback else rate(u.n)
    y=p.value("predictor"+suffix,u.n+p.dt*first,at=p.stage("stage"+suffix,c=1))
    fields=field(y).consume(action=FailRun())
    p.store_history("observed-state"+suffix,y,depth=1)
    if not observe_before:p.record_scalar("independent-read",p.norm2(u.n))
    if last_point!=1:
        y=p.value("sample"+suffix,1*y,at=p.stage("different-point"+suffix,c=last_point))
    second=second_rate(y,fields) if hidden_feedback else second_rate(y)
    if composition=="expanded":
        final=p.value("endpoint"+suffix,u.n+p.dt*weights[0]*first+p.dt*weights[1]*second,
                      at=u.next.point)
    else:
        final=p.value("endpoint"+suffix,Fraction(1,2)*u.n+Fraction(1,2)*y+
                      p.dt*Fraction(1,2)*second,at=u.next.point)
    p.commit(u.next,final)
    return p,model,first,second,y,fields


@pytest.mark.parametrize("composition",["expanded","retained"])
def test_exact_convex_update_preserves_both_authored_compositions_and_exchange_weights(composition):
    from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp
    p,_,first,second,_,_=_observed_step(composition)
    before=p._ir_hash();certificate,reason=prove_accepted_update_ssp(p)
    assert not reason and certificate.coefficient==1
    assert certificate.A==((Fraction(0),Fraction(0)),(Fraction(1),Fraction(0)))
    assert certificate.b==(Fraction(1,2),Fraction(1,2))
    assert certificate.base_weights==(Fraction(1),Fraction(0),Fraction(1,2))
    assert [(v.id,w) for v,w in accepted_diffusive_quadrature(p)]==[
        (first.id,{1:Fraction(1,2)}),(second.id,{1:Fraction(1,2)})]
    assert p._ir_hash()==before
    from pops.time import certify_program_graph, UnknownOrder
    assert isinstance(certify_program_graph(p.to_graph()).properties.order,UnknownOrder)


@pytest.mark.parametrize("weights",[(Fraction(-1,2),Fraction(3,2)),
                                    (Fraction(1,4),Fraction(3,4))])
def test_perturbed_correctly_typed_coefficients_change_their_proof_not_the_expression(weights):
    from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp
    p,_,_,_,_,_=_observed_step(weights=weights)
    before=p._ir_hash();certificate,reason=prove_accepted_update_ssp(p)
    assert certificate is None and "convex weights are negative" in reason
    with pytest.raises(ValueError,match="SSP coefficient-one convex proof"):
        accepted_diffusive_quadrature(p)
    assert p._ir_hash()==before


def test_wrong_stage_coordinate_is_a_separate_acceptance_obligation():
    from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp
    p,*_=_observed_step(last_point=Fraction(1,2))
    certificate,reason=prove_accepted_update_ssp(p)
    assert certificate is None and "stage coordinate" in reason


def test_shared_auxiliary_feedback_is_not_mistaken_for_an_independent_field():
    from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp
    p,*_=_observed_step(hidden_feedback=True)
    certificate,reason=prove_accepted_update_ssp(p)
    assert certificate is None and "dependent Field/evaluation inputs" in reason


def test_debug_model_operator_stage_and_ssa_spellings_do_not_select_a_certificate():
    from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp
    a,*_=_observed_step();b,*_=_observed_step(suffix="arbitrary")
    one,_=prove_accepted_update_ssp(a);two,_=prove_accepted_update_ssp(b)
    assert one is not None and two is not None
    assert (one.A,one.b,one.c,one.base_weights,one.forward_euler_weights)==(
        two.A,two.b,two.c,two.base_weights,two.forward_euler_weights)
    assert a._ir_hash()!=b._ir_hash()


def test_different_rate_expressions_keep_generic_coefficient_arithmetic():
    from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp
    p,*_=_observed_step(second_coefficient=Fraction(1,5))
    certificate,reason=prove_accepted_update_ssp(p)
    assert not reason and certificate.b==(Fraction(1,2),Fraction(1,2))


def test_independent_observation_reorders_actual_reference_ids_without_changing_math_proof():
    from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp
    first,*_=_observed_step();second,*_=_observed_step(observe_before=True)
    a,_=prove_accepted_update_ssp(first);b,_=prove_accepted_update_ssp(second)
    assert a is not None and b is not None and a.contributing_rates!=b.contributing_rates
    assert (a.A,a.b,a.c,a.base_weights,a.forward_euler_weights)==(
        b.A,b.b,b.c,b.base_weights,b.forward_euler_weights)


def test_closed_state_only_primitive_recipes_remain_ordinary_mathematical_expressions():
    from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp
    p,*_=_observed_step(primitive=True)
    certificate,reason=prove_accepted_update_ssp(p)
    assert not reason and certificate.coefficient==1


def test_independently_owned_updates_have_independent_convex_certificates():
    import pops
    from pops import math
    from pops.time import Program
    from pops.codegen.program_accepted_ssp import AcceptedSSPPlan,prove_accepted_update_ssp
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    model=pops.Model("independent-owners",frame=Rectangle("owners-box",(0.,0.),(1.,1.)).frame(Cartesian2D()))
    state=model.state("quantity",components=("q",))
    flux=model.diffusive_flux("law",state=state,value=Fraction(1,10)*math.grad(state[0]))
    rate=model.rate("balance",equation=math.ddt(state)==math.div(flux))
    case=pops.Case("two-equations");left=case.block("left",model);right=case.block("right",model)
    p=Program("two-expressions")
    for block in [left,right]:
        u=p.state(block[state]);rhs=rate(u.n)
        endpoint=p.value("next-"+block.local_id,u.n+p.dt*rhs,at=u.next.point)
        p.commit(u.next,endpoint)
    certificate,reason=prove_accepted_update_ssp(p)
    assert not reason and isinstance(certificate,AcceptedSSPPlan)
    assert len(certificate.updates)==2 and all(item.coefficient==1 for item in certificate.updates)


def test_coefficient_perturbation_still_lowers_its_actual_arithmetic_before_acceptance_proof():
    from pops.codegen.program_emit_ops import _emit_op
    from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp
    actual=[]
    for weights in [(Fraction(1,2),Fraction(1,2)),(Fraction(1,4),Fraction(3,4))]:
        p,*_=_observed_step(weights=weights);node=next(iter(p._commits.values()))
        variables={v.id:"actual_%d"%v.id for v in p._values};lines=[]
        base=next(v for v in p._values if v.op=='state')
        _emit_op(p,node,base,set(),variables,None,lines,prelude=[],block_idx=p._block_indices())
        certificate,reason=prove_accepted_update_ssp(p)
        actual.append(("\n".join(lines),certificate,reason))
    assert actual[0][0]!=actual[1][0]
    assert actual[0][1] is not None and actual[1][1] is None
    assert "pops::Real(3) / pops::Real(4)" in actual[1][0]


@pytest.mark.parametrize("fault",["cycle","detached_observation","detached_binder"])
def test_frozen_mathematical_reads_do_not_authenticate_corrupt_reference_or_binder_objects(fault):
    from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp
    case,_=_solved_potential_case(16,1e-4);p=case._time
    publication=next(v for v in p._values if v.op=='field_publication')
    def clone(value):
        result=object.__new__(type(value))
        for cls in reversed(type(value).__mro__):
            slots=cls.__dict__.get('__slots__',())
            for name in (slots,) if isinstance(slots,str) else slots:
                if name not in {'__dict__','__weakref__'} and hasattr(value,name):
                    object.__setattr__(result,name,getattr(value,name))
        return result
    if fault=='cycle':
        object.__setattr__(publication,'inputs',(publication,*publication.inputs[1:]))
    elif fault=='detached_observation':
        object.__setattr__(publication,'inputs',(clone(publication.inputs[0]),*publication.inputs[1:]))
    else:
        operator=next(v for v in p._values if v.op=='matrix_free_operator')
        object.__setattr__(operator,'attrs',{**operator.attrs,'apply_out':clone(operator.attrs['apply_out'])})
    certificate,reason=prove_accepted_update_ssp(p)
    assert certificate is None and reason


def test_boolean_scalar_metadata_cannot_supply_a_real_convex_coefficient():
    from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp
    p,*_=_observed_step();accepted=next(iter(p._commits.values()))
    object.__setattr__(accepted,'attrs',{'coeffs':({0:True},{1:Fraction(1,2)},{1:Fraction(1,2)})})
    certificate,reason=prove_accepted_update_ssp(p)
    assert certificate is None and reason


def test_unused_input_mutation_does_not_disappear_from_the_acceptance_proof():
    from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp
    p,*_=_observed_step();state=next(v for v in p._values if v.op=='state')
    p.project(state)
    certificate,reason=prove_accepted_update_ssp(p)
    assert certificate is None and "nonmutation" in reason


@pytest.mark.parametrize("action_name",["FailRun","RejectAttempt"])
def test_terminal_guards_keep_actual_conditions_and_success_path_value_algebra(action_name):
    from pops.time import solve_outcome
    from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp
    p,*_=_observed_step();reference,accepted=next(iter(p._commits.items()))
    condition=p.norm2(accepted)<0
    guarded=p.guard("actual-late-condition",accepted,condition,action=getattr(solve_outcome,action_name)())
    # Replace the authoring commit with its conditional alias; the actual guard
    # still executes and may withdraw the entire attempt before publication.
    p._commits[reference]=guarded
    certificate,reason=prove_accepted_update_ssp(p)
    assert not reason and certificate.b==(Fraction(1,2),Fraction(1,2))
    assert [weight for _,weight in accepted_diffusive_quadrature(p)]==[
        {1:Fraction(1,2)},{1:Fraction(1,2)}]
    assert guarded.inputs[1] is condition


def test_terminal_guard_cannot_authenticate_an_arbitrary_action_or_nonboolean_condition():
    from pops.time.solve_outcome import RejectAttempt
    from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp
    p,*_=_observed_step();accepted=next(iter(p._commits.values()))
    guarded=p.guard("actual-condition",accepted,p.norm2(accepted)<0,action=RejectAttempt())
    object.__setattr__(guarded,'attrs',{**guarded.attrs,'action':object()})
    certificate,reason=prove_accepted_update_ssp(p)
    assert certificate is None and reason
