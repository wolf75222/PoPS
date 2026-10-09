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


@pytest.mark.parametrize('composition',['weighted_rates','weighted_predictor'])
def test_detached_resolved_public_compositions_retain_closed_source_read_authority(composition,tmp_path):
    import pops
    from tests.python.support.public_diffusion_field_case import author_case,one_level_amr_layout
    from pops.fields import CompositeHierarchySolve
    from pops.solvers.elliptic import GeometricMG
    from pops.solvers.tolerances import Relative,AbsoluteFloor
    from pops.codegen._orchestration_compile import build_program_model_graph
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp
    from pops.time._program.detach import detach_compiled_program
    authored=author_case('ssprk2',rk2_composition=composition,layout_factory=one_level_amr_layout,
        field_solver=GeometricMG(tolerance=Relative(1e-13,floor=AbsoluteFloor(1e-14)),max_cycles=100),hierarchy_policy=CompositeHierarchySolve())
    resolved=pops.resolve(pops.validate(authored.case),layout=authored.layout)
    authority=build_program_model_graph(resolved)
    detached=detach_compiled_program(resolved.time)
    assert detached._ir_hash()==resolved.time._ir_hash()
    assert detached._compiled_detached and not detached._operator_registries
    assert all(v.block._instance_registry is None for v in detached._values if v.block is not None)
    original=emit_cpp_program(resolved.time,model_graph=authority,field_plans=resolved.field_plans,target='amr_system')
    rebuilt=emit_cpp_program(detached,model_graph=authority,field_plans=resolved.field_plans,target='amr_system')
    certificate,reason=prove_accepted_update_ssp(detached,model_authority=authority)
    assert certificate is not None,reason
    assert certificate.b==(Fraction(1,2),Fraction(1,2))
    assert [w for _,w in accepted_diffusive_quadrature(detached,model_authority=authority)]==[{1:Fraction(1,2)},{1:Fraction(1,2)}]
    # Authored handle presentation and compiled canonical provenance differ.
    # Retain complete original outputs and compare the actual step lambda body.
    (tmp_path/'authored.cpp').write_text(original)
    (tmp_path/'detached.cpp').write_text(rebuilt)
    start='      [=](double dt) {'
    end='\n      }\n    };'
    def step_body(source):
        assert source.count(start)==1
        return source.split(start,1)[1].split(end,1)[0]
    assert step_body(rebuilt)==step_body(original)
    no_certificate,reason=prove_accepted_update_ssp(detached)
    assert no_certificate is None and 'ProgramModelGraph source authority' in reason


def test_detached_constitutive_module_cannot_be_borrowed_from_a_different_owner():
    import pops
    from tests.python.support.public_diffusion_field_case import author_case,one_level_amr_layout
    from pops.fields import CompositeHierarchySolve
    from pops.solvers.elliptic import GeometricMG
    from pops.solvers.tolerances import Relative,AbsoluteFloor
    from pops.codegen._orchestration_compile import build_program_model_graph
    from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp
    from pops.time._program.detach import detach_compiled_program
    plans=[]
    for labels in (None,{'model':'different-constitutive-owner'}):
        case=author_case('ssprk2',labels=labels,layout_factory=one_level_amr_layout,
            field_solver=GeometricMG(tolerance=Relative(1e-13,floor=AbsoluteFloor(1e-14)),max_cycles=100),hierarchy_policy=CompositeHierarchySolve())
        plans.append(pops.resolve(pops.validate(case.case),layout=case.layout))
    certificate,reason=prove_accepted_update_ssp(detach_compiled_program(plans[0].time),model_authority=build_program_model_graph(plans[1]))
    assert certificate is None and ('route' in reason or 'owner' in reason)


# Independently authored two-owner Field publication regression cases.
from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp


def _guarded_frozen_coefficient(**kwargs):
    from tests.python.support.guarded_field_diffusion_observation_case import (
        author_frozen_coefficient_case,
    )
    from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp

    p, case = author_frozen_coefficient_case(**kwargs)
    before = p._ir_hash()
    certificate, reason = prove_accepted_update_ssp(p)
    result = {
        "certificate": certificate is not None,
        "reason": reason,
        "unchanged_by_proof": before == p._ir_hash(),
    }
    result["quadrature"] = [
        (v.id, {str(k): str(w) for k, w in weights.items()})
        for v, weights in accepted_diffusive_quadrature(p)
    ]
    return result, p, case


@pytest.mark.parametrize(
    "driver_guard,input_guard,suffix",
    [
        (False, False, ""),
        (True, False, ""),
        (False, True, ""),
        (True, True, ""),
        (False, False, "_renamed"),
        (True, False, "_renamed"),
        (False, True, "_renamed"),
    ],
)
def test_successful_frozen_value_aliases_keep_the_actual_elliptic_diffusion_equation(
    driver_guard, input_guard, suffix
):
    result, p, _ = _guarded_frozen_coefficient(
        guard_driver=driver_guard, guard_field_input=input_guard, suffix=suffix
    )
    assert result["certificate"], result["reason"]
    assert result["unchanged_by_proof"]
    assert len(result["quadrature"]) == 1 and result["quadrature"][0][1] == {"1": "1"}
    certificate, reason = prove_accepted_update_ssp(p)
    assert not reason and certificate.b == (1,) and certificate.coefficient == 1
    assert any(v.op == "matrix_free_operator" for v in p._values)
    if driver_guard or input_guard:
        assert any(v.op == "acceptance_guard" for v in p._values)


@pytest.mark.parametrize(
    "fault", ["unproved_action", "input_mutation", "changed_mathematical_driver"]
)
def test_alias_proof_does_not_drop_actual_conditions_effects_or_driver_evolution(fault):
    import pops

    result, p, _ = _guarded_frozen_coefficient(
        guard_driver=True, guard_field_input=True
    )
    if fault == "unproved_action":
        guard = next(v for v in p._values if v.op == "acceptance_guard")
        object.__setattr__(guard, "attrs", {**guard.attrs, "action": object()})
    elif fault == "input_mutation":
        p.project(next(v for v in p._values if v.op == "state"))
    else:
        reference = next(
            ref for ref in p._commits if "fixed-owner" in ref.block_ref.local_id
        )
        initial = next(
            v for v in p._values if v.op == "state" and v.state_ref == reference
        )
        endpoint = p._commits[reference]
        changed = p.value("actual-changed-donor", 2 * initial, at=endpoint.point)
        p._commits[reference] = changed
    certificate, reason = prove_accepted_update_ssp(p)
    assert certificate is None and reason
    with pytest.raises(ValueError, match="SSP coefficient-one convex proof"):
        accepted_diffusive_quadrature(p)


def _guarded_portable_resolved(suffix="", **math_inputs):
    import pops
    from pops.solvers import CompositeFieldGMRES
    from tests.python.support.guarded_field_diffusion_case import author_case

    case, layout, arrays, oracle = author_case(
        solver=CompositeFieldGMRES(max_iter=200, rel_tol=1e-12, abs_tol=1e-14),
        suffix=suffix,
        **math_inputs,
    )
    resolved = pops.resolve(pops.validate(case), layout=layout)
    from pops.codegen._orchestration_compile import build_program_model_graph

    return resolved, build_program_model_graph(resolved), oracle


@pytest.mark.parametrize("suffix", ["", "_renamed"])
def test_independent_portable_amr_math_resolves_detaches_and_emits_identical_actual_cpp(
    suffix,
):
    from pops.time._program.detach import detach_compiled_program
    from pops.codegen.program_codegen import emit_cpp_program

    resolved, authority, oracle = _guarded_portable_resolved(suffix)
    original = resolved.time
    detached = detach_compiled_program(original)
    assert original._ir_hash() == detached._ir_hash()
    assert oracle["dt"] * oracle["forward_euler_frequency"] < 1
    a, why_a = prove_accepted_update_ssp(original, model_authority=authority)
    b, why_b = prove_accepted_update_ssp(detached, model_authority=authority)
    assert a is not None and b is not None, (why_a, why_b)
    assert (a.A, a.b, a.c, a.base_weights, a.forward_euler_weights) == (
        b.A,
        b.b,
        b.c,
        b.base_weights,
        b.forward_euler_weights,
    )
    cpp = emit_cpp_program(
        original,
        model_graph=authority,
        field_plans=resolved.field_plans,
        target="amr_system",
    )
    detached_cpp = emit_cpp_program(
        detached,
        model_graph=authority,
        field_plans=resolved.field_plans,
        target="amr_system",
    )
    assert (
        cpp == detached_cpp
    )  # Includes actual guard conditions and canonical storage witness.
    assert "ctx.store_global_field_history(" in cpp
    assert "#authoring=" not in cpp


@pytest.mark.parametrize(
    "fault", ["no_source_authority", "foreign_model_graph", "foreign_source_module"]
)
def test_detached_publication_never_borrows_an_unknown_or_foreign_source_authority(
    fault,
):
    from pops.time._program.detach import detach_compiled_program
    from pops.codegen.program_models import ProgramModelGraph

    resolved, authority, _ = _guarded_portable_resolved()
    detached = detach_compiled_program(resolved.time)
    selected = authority
    if fault == "no_source_authority":
        selected = None
    elif fault == "foreign_model_graph":
        _, selected, _ = _guarded_portable_resolved("_foreign")
    else:
        modules = dict(authority.source_modules_by_owner)
        publication = next(
            value for value in detached._values if value.op == "field_publication"
        )
        receiver_owner = authority.owner_for_block(
            publication.attrs["bindings"][0]["target"].block_ref
        )
        donor_owner = next(owner for owner in modules if owner != receiver_owner)
        modules[receiver_owner] = modules[donor_owner]
        selected = ProgramModelGraph(
            models_by_owner=authority.models_by_owner,
            source_modules_by_owner=modules,
            owners_by_block=authority.owners_by_block,
            authorities_by_owner=authority._authorities_by_owner,
            models_by_block=authority._models_by_block,
            numerics_by_block=authority._numerics_by_block,
        )
    certificate, reason = prove_accepted_update_ssp(detached, model_authority=selected)
    assert certificate is None and reason


def test_publication_declaration_cannot_borrow_a_different_known_block_owner():
    from pops.time._program.detach import detach_compiled_program
    from pops.codegen.program_field_publication import publication_target_space

    resolved, authority, _ = _guarded_portable_resolved()
    detached = detach_compiled_program(resolved.time)
    publication = next(
        value for value in detached._values if value.op == "field_publication"
    )
    target = publication.attrs["bindings"][0]["target"]
    donor = next(
        value.block
        for value in detached._values
        if value.op == "state" and value.block.local_id == "donor"
    )
    corrupt = object.__new__(type(target))
    for cls in reversed(type(target).__mro__):
        slots = cls.__dict__.get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name not in {"__dict__", "__weakref__"} and hasattr(target, name):
                object.__setattr__(corrupt, name, getattr(target, name))
    donor_module = authority.source_module_for_owner(authority.owner_for_block(donor))
    donor_declaration = donor_module.field_handle(donor_module.field_spaces()["fields"])
    assert donor_declaration.kind == "field"
    object.__setattr__(corrupt, "_declaration_ref", donor_declaration)
    # Both source owners are real graph members; keep the destination block unchanged.
    authority.model_for_block(corrupt.block_ref)
    authority.source_module_for_owner(donor_declaration.owner_path)
    with pytest.raises(ValueError, match="different block model"):
        publication_target_space(authority, corrupt)


def test_independent_fv_oracle_matches_separable_discrete_eigenvalues():
    import numpy as np
    from tests.python.support.guarded_field_diffusion_case import independent_oracle

    arrays, oracle = independent_oracle()
    q0 = arrays["receiver"][0]
    nx, ny = 16, 12
    dx, dy = 2 / nx, 3 / ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    mode_x = 0.08 * np.sinc(1 / nx) * np.cos(np.pi * x)[None, :]
    mode_y = 0.03 * np.sinc(1 / ny) * np.cos(2 * np.pi * y / 3)[:, None]
    eigen_x = 4 * np.sin(np.pi / nx) ** 2 / dx**2
    eigen_y = 4 * np.sin(np.pi / ny) ** 2 / dy**2
    coefficient = 2 / 7 + (0.6 / 3) ** 2
    spectral = q0 - oracle["dt"] * coefficient * (eigen_x * mode_x + eigen_y * mode_y)
    np.testing.assert_allclose(oracle["receiver"], spectral, rtol=0, atol=3e-16)
    np.testing.assert_array_equal(oracle["phi"], np.full((ny, nx), 0.2))


def test_changed_guarded_publication_equations_change_actual_numerical_cpp_and_oracle(
    tmp_path,
):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.time._program.detach import detach_compiled_program

    original, old_graph, old_oracle = _guarded_portable_resolved()
    changed, new_graph, new_oracle = _guarded_portable_resolved(
        reaction=5, diffusion_offset=Fraction(3, 11)
    )
    old_cpp = emit_cpp_program(
        detach_compiled_program(original.time),
        model_graph=old_graph,
        field_plans=original.field_plans,
        target="amr_system",
    )
    new_cpp = emit_cpp_program(
        detach_compiled_program(changed.time),
        model_graph=new_graph,
        field_plans=changed.field_plans,
        target="amr_system",
    )
    assert old_oracle["coefficient"] != new_oracle["coefficient"]
    assert old_oracle["phi"][0, 0] == 0.2 and new_oracle["phi"][0, 0] == 0.12
    assert not (old_oracle["receiver"] == new_oracle["receiver"]).all()
    # These are generated physical value assignments, not provenance hashes.
    (tmp_path / "original.cpp").write_text(old_cpp)
    (tmp_path / "changed.cpp").write_text(new_cpp)
    old_values = [
        line.strip() for line in old_cpp.splitlines() if "pops::Real cse" in line
    ]
    new_values = [
        line.strip() for line in new_cpp.splitlines() if "pops::Real cse" in line
    ]
    assert old_values and new_values and old_values != new_values
    for program, authority in [(original.time, old_graph), (changed.time, new_graph)]:
        certificate, reason = prove_accepted_update_ssp(
            program, model_authority=authority
        )
        assert certificate is not None and certificate.b == (1,), reason


def _guarded_native_read_support_emitter(monkeypatch, *, suffix="", tensor=False,
                                          **math_inputs):
    import pops
    from pops.solvers import CompositeFieldGMRES
    from pops.numerics import TensorDiffusion
    from pops.codegen.module_lowering import lower_and_validate
    import tests.python.support.guarded_field_diffusion_case as authored

    with monkeypatch.context() as numerical_choice:
        if tensor:
            numerical_choice.setattr(authored, "Diffusion", TensorDiffusion)
        case, layout, _, _ = authored.author_case(
            solver=CompositeFieldGMRES(max_iter=200, rel_tol=1e-12, abs_tol=1e-14),
            suffix=suffix, **math_inputs)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    block = resolved.blocks[1]
    emitter, module = lower_and_validate(block.model,
        resolved_operations=block.resolved_operations, numerics=block.numerics)
    return emitter, module, block.resolved_operations, resolved


@pytest.mark.parametrize("suffix,reaction,offset,tensor,depth", [
    ("", 3, Fraction(2, 7), False, 1),
    ("_different_names", 3, Fraction(2, 7), False, 1),
    ("", 5, Fraction(3, 11), False, 1),
    ("_tensor", 3, Fraction(2, 7), True, 2),
])
def test_selected_constitutive_reads_determine_native_provider_support(
    monkeypatch, suffix, reaction, offset, tensor, depth,
):
    from pops.codegen._native_auxiliary_shapes import native_auxiliary_halos
    from pops.codegen.component_provider_packs import require_emitter_provider_carrier
    from pops.codegen._compile_emit import _emit_auxiliary_route_registration
    from pops.codegen._orchestration_compile import build_program_model_graph
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_emit_kernels import ProgramProviderPlans, program_provider_consumer_qid
    from pops.codegen.program_emit_model_kernels import _provider_binding
    from pops.codegen.program_emit_diffusion import _selected, _law_expressions

    emitter, module, plan, resolved = _guarded_native_read_support_emitter(monkeypatch,
        suffix=suffix, reaction=reaction, diffusion_offset=offset, tensor=tensor)
    implementation = emitter._m
    assert emitter._resolved_operations is implementation._resolved_operations is plan
    packs = implementation._resolved_operations.require_provider_packs(module)
    assert packs.auxiliary.to_data() == implementation._auxiliary_provider_pack.to_data()
    require_emitter_provider_carrier(implementation)
    # This field publication is not a hyperbolic flux or a model auxiliary recipe.
    assert not implementation._component_flux_consumer_plan
    assert not implementation._auxiliary_provider_routes
    demands = native_auxiliary_halos(implementation)
    assert len(demands) == 1 and set(demands.values()) == {depth}
    key = next(iter(implementation._auxiliary_provider_pack))
    assert demands[(key.owner_qid, key.space_kind, key.space_name, key.component)] == depth
    registration = _emit_auxiliary_route_registration(implementation, target="amr_system")
    assert "halo[axis] = %d;" % depth in registration
    evaluation = next(value for value in resolved.time._values if value.op == "diffusive_rhs")
    _, selected_law, _ = _selected(evaluation, emitter)
    consumers = ProgramProviderPlans(target="amr_system", provider_halos=demands)
    _provider_binding(implementation, _law_expressions(selected_law), consumers,
        program_provider_consumer_qid(emitter, evaluation.id, evaluation.block))
    consumer_cpp = consumers.cpp_install("amr_system")
    assert "halo[axis] = %d;" % depth in consumer_cpp
    graph = build_program_model_graph(resolved)
    if tensor:
        with pytest.raises(ValueError, match="proven composite stability bound"):
            emit_cpp_program(resolved.time, model_graph=graph,
                             field_plans=resolved.field_plans, target="amr_system")
    else:
        program_cpp = emit_cpp_program(resolved.time, model_graph=graph,
                                      field_plans=resolved.field_plans, target="amr_system")
        rows = [line for line in program_cpp.splitlines() if "ConsumerValue{Dependency{" in line]
        assert rows and all("halo[axis] = %d;" % depth in line for line in rows)


def test_provider_support_matches_typed_components_and_canonical_owner(monkeypatch):
    from dataclasses import replace
    from pops.model import Handle, OwnerPath
    from pops.model.provider_pack import ProviderPack
    from pops.codegen._native_auxiliary_shapes import native_auxiliary_halos

    emitter, _, plan, _ = _guarded_native_read_support_emitter(monkeypatch)
    implementation = emitter._m
    probe = SimpleNamespace(**vars(implementation))
    probe._resolved_operations = SimpleNamespace(operations=tuple(
        replace(operation, stencil_radius=0) for operation in plan.operations))
    assert native_auxiliary_halos(probe) == {}

    selected = next(iter(implementation._auxiliary_provider_pack))
    foreign = Handle(selected.space_name, kind=selected.space_kind,
        owner=OwnerPath.model("independent-foreign-owner").canonical()).qualified_id
    probe._resolved_operations = SimpleNamespace(operations=tuple(replace(operation,
        inputs=tuple(replace(read, reference=foreign) if read.kind == "field" else read
                     for read in operation.inputs)) for operation in plan.operations))
    assert native_auxiliary_halos(probe) == {}

    # A three-cell reconstructed State still takes pointwise provider traces
    # from the two cells adjacent to a face; those sampling contracts differ.
    probe._resolved_operations = SimpleNamespace(operations=tuple(replace(operation,
        stencil_radius=3,
        inputs=tuple(replace(read, sampling="right_face_trace")
                     if read.kind == "field" else read for read in operation.inputs))
        for operation in plan.operations))
    assert set(native_auxiliary_halos(probe).values()) == {1}

    probe._resolved_operations = plan
    unread = replace(selected, component="unread-neighbor")
    original = implementation._auxiliary_provider_pack
    probe._auxiliary_provider_pack = ProviderPack([
        (key, original.contract(key), original.declared_entry(key)) for key in original
    ] + [(unread, original.contract(selected), original.declared_entry(selected))])
    demands = native_auxiliary_halos(probe)
    assert len(demands) == 1
    assert (unread.owner_qid, unread.space_kind, unread.space_name, unread.component) not in demands


def test_selected_provider_support_propagates_to_native_prerequisites_and_boundaries(
    monkeypatch,
):
    from dataclasses import replace
    from pops.codegen._native_auxiliary_shapes import native_auxiliary_halos

    emitter, _, _, _ = _guarded_native_read_support_emitter(monkeypatch, tensor=True)
    implementation = emitter._m
    selected = next(iter(implementation._auxiliary_provider_pack))
    dependency = replace(selected, component="prerequisite")
    probe = SimpleNamespace(**vars(implementation))
    probe._auxiliary_provider_routes = {selected: {"dependencies": (dependency,)}}
    demands = native_auxiliary_halos(probe)
    assert demands[(dependency.owner_qid, dependency.space_kind,
                    dependency.space_name, dependency.component)] == 2
    probe._auxiliary_provider_routes = {
        selected: {"boundary": SimpleNamespace(width=3), "dependencies": (dependency,)}}
    demands = native_auxiliary_halos(probe)
    assert set(demands.values()) == {3}


def test_weno_state_reconstruction_keeps_pointwise_provider_face_support():
    import pops
    from pops import math
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.numerics import FiniteVolume, DiscretizationPlan, variables, riemann, reconstruction
    from pops.time import Program, FixedDt
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen._native_auxiliary_shapes import native_auxiliary_halos
    from pops.codegen._compile_emit import _emit_auxiliary_route_registration
    from pops.codegen.program_codegen import emit_cpp_program

    frame = Rectangle("fv-support-domain", (0, 0), (1, 1)).frame(Cartesian2D())
    model = pops.Model("variable-velocity-material", frame=frame)
    state = model.state("inventory", components=("inventory",))
    speed = model.aux("imposed_speed")
    flux = model.flux("physical-transport", frame=frame, state=state,
        components={frame.x: (speed * state[0],), frame.y: (0 * state[0],)},
        waves={frame.x: (speed,), frame.y: (0,)})
    rate = model.rate("conservation", equation=math.ddt(state) == -math.div(flux))
    case = pops.Case("provider-face-support")
    block = case.block("material", model)
    methods = DiscretizationPlan()
    methods.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.WENO5(), riemann=riemann.Rusanov()))
    case.numerics(methods, block=block)
    program = Program("advance-material")
    q = program.state(block[state])
    fields = program.input_fields(q.n, for_rate=rate)
    rhs = rate(q.n, fields)
    endpoint = program.value("actual-endpoint", q.n + program.dt * rhs, at=q.next.point)
    program.commit(q.next, endpoint)
    program.step_strategy(FixedDt(1e-5))
    case.program(program)
    resolved = pops.resolve(pops.validate(case), layout=Uniform(CartesianGrid(
        frame=frame, cells=(16, 16), periodic=PeriodicAxes(frame.axes))))
    selected = resolved.blocks[0]
    emitter, _ = lower_and_validate(selected.model,
        resolved_operations=selected.resolved_operations, numerics=selected.numerics)
    reads = [(operation.stencil_radius, read.sampling)
             for operation in selected.resolved_operations.operations
             for read in operation.inputs if read.kind == "field"]
    assert reads and all(radius == 3 and sampling == "face_trace"
                         for radius, sampling in reads)
    assert set(native_auxiliary_halos(emitter._m).values()) == {1}
    registration = _emit_auxiliary_route_registration(emitter._m)
    assert "halo[axis] = 1;" in registration and "halo[axis] = 3;" not in registration
    program_cpp = emit_cpp_program(resolved.time, model=emitter)
    consumers = [line for line in program_cpp.splitlines() if "ConsumerValue{Dependency{" in line]
    assert consumers and all("halo[axis] = 1;" in line for line in consumers)
