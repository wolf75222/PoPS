from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace

import pytest

import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_flux_lowering import _stage
from pops.codegen.program_flux_lowering import flux_table_budgets, lower_amr_flux_tables
from pops.codegen.program_persistent_plan import get_program_resource_plan
from pops.lib import time as libtime
from pops.lib.time import IMEX, IMEX_EULER_TABLEAU
from pops.physics._facade import Model
from pops.problem import Case
from pops.solvers import LocalNewton
from pops.time import Clock, StagePoint, TimePoint
from pops.time import FailRun
from pops.time.references import block_name


def _value(point):
    return SimpleNamespace(point=point)


class _LoweringProgram:
    def __init__(self, values, commit):
        self._values = tuple(values)
        self._commits = {_CommitKey("owner.state"): commit}
        self._block = next((value.block for value in self._values if value.block != "owner"), "owner")

    def _block_indices(self):
        return {self._block: 0}

    @staticmethod
    def cell_local_time_contract():
        return None


class _LoweringPlan:
    @staticmethod
    def row_for_value(value, *, occurrence_path=None):
        _stage_fraction, clock = _stage(value)
        return SimpleNamespace(
            slot=value.id,
            key=SimpleNamespace(owner=block_name(value.block), clock=clock, level=None),
        )


@dataclass(frozen=True)
class _CommitKey:
    qualified_id: str


def _history_value(value_id, op, *, history, inputs=(), flux=True, block="owner"):
    return SimpleNamespace(
        id=value_id,
        op=op,
        attrs={"history": history, "flux": flux},
        block=block,
        inputs=tuple(inputs),
        point=None,
    )


def test_flux_stage_lowers_an_exact_time_point_and_clock() -> None:
    clock = Clock("macro")

    stage, clock_id = _stage(_value(TimePoint(clock, Fraction(1, 3))))

    assert stage == Fraction(1, 3)
    assert clock_id == clock.qualified_id


def test_flux_stage_selects_the_named_explicit_imex_partition() -> None:
    clock = Clock("macro")
    point = StagePoint(
        "imex-stage",
        {
            "explicit": TimePoint(clock, Fraction(1, 3)),
            "implicit": TimePoint(clock, Fraction(2, 3)),
        },
    )

    stage, clock_id = _stage(_value(point))

    assert stage == Fraction(1, 3)
    assert clock_id == clock.qualified_id


def test_flux_stage_accepts_a_single_unambiguous_stage_coordinate() -> None:
    clock = Clock("macro")
    point = StagePoint("single-stage", {"implicit": TimePoint(clock, Fraction(1, 2))})

    stage, _clock_id = _stage(_value(point))

    assert stage == Fraction(1, 2)


def test_flux_stage_refuses_an_ambiguous_stage_without_explicit_partition() -> None:
    clock = Clock("macro")
    point = StagePoint(
        "unqualified-stage",
        {
            "left": TimePoint(clock, Fraction(1, 3)),
            "right": TimePoint(clock, Fraction(2, 3)),
        },
    )

    with pytest.raises(NotImplementedError, match="exact explicit StagePoint partition"):
        _stage(_value(point))


def test_flux_stage_refuses_an_opaque_evaluation_point() -> None:
    with pytest.raises(TypeError, match="exact TimePoint or StagePoint"):
        _stage(_value(object()))


def test_history_without_an_effective_flux_expression_lowers_to_an_empty_expression() -> None:
    history_name = "owner.state"
    rhs = _history_value(1, "rhs", history=history_name, flux=False)
    store = _history_value(2, "store_history", history=history_name, inputs=(rhs,))
    read = _history_value(3, "history", history=history_name)

    bases, terms = lower_amr_flux_tables(
        _LoweringProgram((rhs, store, read), read), _LoweringPlan()
    )

    assert bases == ()
    assert terms == ()


def test_history_rehydrating_an_effective_flux_expression_fails_closed() -> None:
    clock = Clock("macro")
    model = Model("history-flux")
    model.conservative_vars("u")
    block = Case("history-flux-case").block("fluid", model)
    rhs = _history_value(1, "rhs", history="owner.rate", block=block)
    rhs.point = TimePoint(clock, Fraction(0))
    store = _history_value(2, "store_history", history="owner.rate", inputs=(rhs,), block=block)
    read = _history_value(3, "history", history="owner.rate", block=block)

    with pytest.raises(NotImplementedError, match="effective.*conservative flux expression"):
        lower_amr_flux_tables(_LoweringProgram((rhs, store, read), read), _LoweringPlan())


def test_flux_budget_uses_only_retained_basis_rows_and_final_terms() -> None:
    bases = (
        (0, 5, 1, -1, 17, 0, 0, 1, "basis:marker", "root/marker", "qualified:marker", "clock"),
        (1, 6, 1, -1, 18, 0, 0, 1, "basis:marker-cancelled", "root/cancelled", "qualified:marker", "clock"),
    )
    terms = (
        (0, 0, 9, 1, 1, 1, "term:marker", "root/marker/final", "qualified:marker", "clock"),
    )

    assert flux_table_budgets(2, bases, terms) == ((0, 0), (2, 1))


def test_flux_budget_refuses_a_stage_without_a_retained_basis() -> None:
    stage = (0, 3, 9, 1, 1, 1, "term", "root/final", "qualified", "clock")

    with pytest.raises(ValueError, match="missing retained basis"):
        flux_table_budgets(1, (), (stage,))


def test_candidate_flux_rows_use_fully_qualified_abi_symbols() -> None:
    model = Model("qualified-flux-table")
    model.conservative_vars("u")
    rate = model.rate("transport", flux=True, sources=())
    state = next(item for item in model.declaration_index().records() if item.kind == "state")
    block = pops.Case("qualified-flux-case").block("fluid", model)
    program = libtime.ForwardEuler(block[state], rate=rate)

    source = emit_cpp_program(program, model=model, target="amr_system")

    assert (
        "sizeof(pops::runtime::program::ProgramFluxBasisOccurrenceRecord), "
        "pops::runtime::program::kProgramFluxBasisOccurrenceSchemaVersion"
    ) in source
    assert (
        "sizeof(pops::runtime::program::ProgramFaceFluxStageRecord), "
        "pops::runtime::program::kProgramFaceFluxStageSchemaVersion"
    ) in source
    assert ", sizeof(ProgramFluxBasisOccurrenceRecord)," not in source
    assert ", kProgramFluxBasisOccurrenceSchemaVersion," not in source
    assert ", sizeof(ProgramFaceFluxStageRecord)," not in source
    assert ", kProgramFaceFluxStageSchemaVersion," not in source


def test_nonlinear_imex_flux_lowering_uses_explicit_stage_coordinate() -> None:
    model = Model("flux-lowering-imex")
    (u,) = model.conservative_vars("u")
    explicit = model.rate("transport", flux=True, sources=())
    implicit = model.source_term("decay", (-u * u,))
    state = next(item for item in model.declaration_index().records() if item.kind == "state")
    block = Case("flux-lowering-imex-case").block("fluid", model)

    program = IMEX(
        block[state],
        explicit_operator=explicit,
        implicit_operator=implicit,
        tableau=IMEX_EULER_TABLEAU,
        implicit_solver=LocalNewton(tolerance=1.0e-12, max_iterations=4),
        solve_action=FailRun(),
    )

    plan = get_program_resource_plan(program, target="amr_system")
    bases, terms = lower_amr_flux_tables(program, plan)

    assert len(bases) == 1
    assert bases[0][6:8] == (0, 1)
    basis_resource = plan.entries[bases[0][1]]
    final_resource = plan.entries[terms[0][2]]
    assert bases[0][10] == basis_resource.key.owner
    assert bases[0][11] == basis_resource.key.clock == program.clock.qualified_id
    assert bases[0][3] == (-1 if basis_resource.key.level is None else basis_resource.key.level)
    assert terms[0][8:10] == (final_resource.key.owner, final_resource.key.clock)
    assert terms
