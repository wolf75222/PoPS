import pytest

from pops import time as adctime
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.physics._facade import Model
from pops.solvers import DenseLU
from pops.solvers.nonlinear import LocalNewton
from tests.python.unit.time.typed_program_support import typed_state


def _source(kind, action, *, provider=True):
    model = Model("local_auxiliary_" + kind)
    u, = model.conservative_vars("u")
    model.primitive_vars(u)
    model.conservative_from([u])
    model.flux(x=[0 * u], y=[0 * u])
    model.eigenvalues(x=[0 * u], y=[0 * u])
    coefficient = model.aux("coefficient") if provider else 2.0
    if kind == "linear":
        model.linear_source("decay", [[-coefficient]])
    else:
        model.source_term("decay", [-coefficient * u * u])
    program = adctime.Program("local_auxiliary_" + kind)
    state = typed_state(program, "block", model=model)
    endpoint = typed_state(program, "block", state_name="U", model=model).next
    guess = program.value("guess", state, at=endpoint.point)
    if kind == "linear":
        operator = model.module.operator_handle("decay")
        solved = program.solve(
            adctime.LocalLinear(operator=program.I - program.dt * program.linear_source(operator),
                                rhs=guess), solver=DenseLU(), name="solved")
    else:
        def residual(program, iterate, reference):
            source = program._source("decay", state=iterate)
            return program.value("residual", iterate - reference - program.dt * source,
                                 at=iterate.point)

        solved = program.solve(adctime.LocalResidual(residual, guess),
                               solver=LocalNewton(), name="solved")
    program.commit(endpoint, solved.consume(action=action))
    return emit_cpp_program(program, model=lower_and_validate(model)[0])


@pytest.mark.parametrize("kind", ("linear", "nonlinear"))
@pytest.mark.parametrize("action,expected", (
    (adctime.RejectAttempt(), "kRejectAttempt"),
    (adctime.FailRun(), "kFailRun"),
    (adctime.RejectAttempt(statuses=("singular",)), "kFailRun"),
))
def test_numerical_prerequisite_uses_consuming_local_solve_action(kind, action, expected):
    source = _source(kind, action)
    assert source.count("ctx.prepare_provider_values_for_solve(") == 1
    assert "ctx.prepare_provider_values(" not in source
    start = source.index("ctx.prepare_provider_values_for_solve(")
    kernel = source.index("provider_values_view<1>(", start)
    prerequisite = source[start:kernel]
    assert "AuxiliaryPublicationStatus::nonfinite_candidate" in prerequisite
    assert ('mark_failed(pops::SolveStatus::kInvalidEvaluation, pops::SolveAction::%s, '
            '"auxiliary_nonfinite_candidate")' % expected) in prerequisite
    assert "pops::SolveOutcome::collective_lane(" in prerequisite
    assert ".consume(" in prerequisite
    assert "catch (" not in prerequisite
    if expected == "kRejectAttempt":
        assert "pops::runtime::program::StepAttemptRejected(" in prerequisite


@pytest.mark.parametrize("kind", ("linear", "nonlinear"))
def test_provider_free_local_solve_has_no_publication_or_synthetic_outcome(kind):
    source = _source(kind, adctime.RejectAttempt(), provider=False)
    assert "prepare_provider_values_for_solve(" not in source
    assert "auxiliary_nonfinite_candidate" not in source
