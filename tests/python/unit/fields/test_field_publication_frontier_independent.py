"""Independent adversarial checks of the current field-equation frontier."""
import pytest
from pops.time import FailRun
from pops.numerics.terms import SourceTerm
from pops.fields._program_publication import publication_states
from tests.python.unit.fields.test_program_field_problem import field_case


def setup():
    case,field,problem,program,values,point=field_case(publication_fields=True,publication_source=True)
    handles=tuple(values)
    model=case._block_registry.spec("first")["model"]
    first=case.blocks()["first"]
    carrier=first[model.module.field_handle(model.module.field_spaces()["fields"])]
    def observe(bound,at):
        return field.observe(program.solve(field,values=bound,at=at).consume(action=FailRun()))
    def publish(observed,**kwargs):
        return observed.publish({(carrier,"observed_phi"):observed[field[problem.unknowns[0]]]},**kwargs)
    return case,program,values,handles,point,observe,publish


def test_real_field_driven_stage_stops_old_provenance_but_rejects_stale_context():
    case,program,values,handles,point,observe,publish=setup()
    old_context=publish(observe(values,point))
    model=case._block_registry.spec("first")["model"]
    rate=program.rhs(state=values[handles[0]],fields=old_context,terms=[
        SourceTerm(case.blocks()["first"][model.module.operator_handle("published_force")])])
    fresh_point=program.stage("same date, new exact state",c=0)
    fresh=program.value("field driven stage",values[handles[0]]+program.dt*rate,at=fresh_point)
    other=program.value("second current stage",1*values[handles[1]],at=fresh_point)
    context=publish(observe({handles[0]:fresh,handles[1]:other},fresh_point))
    assert dict(publication_states(context)) == {case.blocks()["first"]:fresh,case.blocks()["second"]:other}
    assert context.field_context != old_context.field_context
    with pytest.raises(ValueError,match="incompatible field context"):
        program.rhs(state=fresh,fields=old_context,terms=[])
    # Same physical date does not allow a stale field context to masquerade as current.
    with pytest.raises(ValueError,match="conflicting or repeated"):
        publish(observe({handles[0]:fresh,handles[1]:other},fresh_point),states={handles[0]:values[handles[0]]})


def test_current_load_and_coefficient_conflict_is_still_rejected():
    case,program,values,handles,point,observe,publish=setup()
    observed=observe(values,point)
    coefficient=next(v for v in program._values if v.op=="field_problem_coefficients")
    changed=program.value("conflicting coefficient state",2*values[handles[0]],at=point)
    # Adversarial alteration of otherwise genuine field IR: the coefficient and load
    # read different current versions of the same exact block. No fabricated solver.
    object.__setattr__(coefficient,"inputs",tuple(changed if v.block==changed.block else v
                                                 for v in coefficient.inputs))
    with pytest.raises(ValueError,match="ambiguous current state provenance"):
        publish(observed)


def test_actual_stale_equation_input_is_not_relabelled_by_supplemental_state():
    case,program,values,handles,point,observe,publish=setup()
    observed=observe(values,point)
    fresh=program.value("new state at same solve point",1.1*values[handles[0]],at=point)
    with pytest.raises(ValueError,match="conflicting or repeated"):
        publish(observed,states={handles[0]:fresh})
    context=publish(observed)
    with pytest.raises(ValueError,match="incompatible field context"):
        program.rhs(state=fresh,fields=context,terms=[])
