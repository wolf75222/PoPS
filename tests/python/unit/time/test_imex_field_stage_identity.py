"""IMEX field reads retain the exact stage SSA and each partition's time."""
from fractions import Fraction

import pytest

from pops.lib.time import IMEX, IMEX_ARS222_TABLEAU, IMEX_EULER_TABLEAU
from pops.time import FailRun, Program, StagePoint, TimePoint
from pops.time._evaluation_point import evaluation_stage_fraction
from pops.time._methods.tableau import AdditiveRungeKuttaTableau, RungeKuttaTableau
from test_time_std_imex_lie_ab import _authoring


def test_public_field_read_rejects_a_numerically_equal_different_stage():
    state, explicit, _, field = _authoring("imex-exact-public")
    program = Program("imex-exact-public")
    temporal = program.state(state)
    point = StagePoint("split", {
        "explicit": TimePoint(program.clock, 0),
        "implicit": TimePoint(program.clock, 1),
    })
    stage = program.value("stage", 1 * temporal.n, at=point)
    field_state = program.value("field_state", stage, at=point.time_for("implicit"))
    fields = field(field_state).consume(action=FailRun())
    assert field_state.id != stage.id
    assert fields.field_context.stage_sources == ((stage.block, field_state.id),)
    with pytest.raises(ValueError, match="incompatible field context"):
        explicit(stage, fields)
    # Merely using field_state makes the read legal, but evaluates at c_implicit.
    naive_rate = explicit(field_state, fields)
    assert evaluation_stage_fraction(naive_rate, ark_partition="explicit") == 1


_SPLIT_TABLEAU = AdditiveRungeKuttaTableau(
    RungeKuttaTableau(A=[[], [1]], b=[Fraction(1, 2), Fraction(1, 2)], c=[0, 1]),
    implicit_A=[[Fraction(1, 2)], [0, Fraction(1, 2)]],
    implicit_b=[Fraction(1, 2), Fraction(1, 2)],
    implicit_c=[Fraction(1, 2), Fraction(1, 2)],
    name="split-field-stage",
)


@pytest.mark.parametrize("tableau", [IMEX_EULER_TABLEAU, IMEX_ARS222_TABLEAU, _SPLIT_TABLEAU],
                         ids=["euler", "ars222", "distinct-two-stage"])
def test_imex_solves_and_reads_the_same_stage_at_the_explicit_coordinate(tableau):
    state, explicit, implicit, field = _authoring("imex-field-" + tableau.name)
    program = IMEX(state, explicit_operator=explicit, implicit_operator=implicit,
                   fields_operator=field, tableau=tableau)
    assert program.validate() is True
    solves = [value for value in program._values if value.op == "solve_fields"]
    rates = [value for value in program._values if value.op == "rhs"]
    implicit_rates = [value for value in program._values if value.op == "apply"]
    assert len(solves) == len(rates) == len(implicit_rates) == tableau.stages
    for index, (solve, rate, implicit_rate) in enumerate(zip(solves, rates, implicit_rates)):
        (solved_state,) = solve.inputs
        read_state, fields = rate.inputs
        assert read_state is solved_state
        assert fields.field_context == solve.field_context
        assert fields.field_context.stage_sources == ((solved_state.block, solved_state.id),)
        assert evaluation_stage_fraction(solve) == tableau.explicit.c[index]
        assert evaluation_stage_fraction(rate, ark_partition="explicit") == tableau.explicit.c[index]
        assert evaluation_stage_fraction(implicit_rate) == tableau.implicit_c[index]
        # The aligned state is exactly Y_i, with coefficient one, not a different stage.
        assert solved_state.op == "linear_combine"
        (joint_stage,) = solved_state.inputs
        assert tuple(dict(coefficient) for coefficient in solved_state.attrs["coeffs"]) == ({0: 1},)
        assert implicit_rate.inputs == (joint_stage,)
        with pytest.raises(ValueError, match="incompatible field context"):
            fields.field_context.require_read(field, joint_stage.block, joint_stage.id)
