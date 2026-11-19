"""The decision to regrid written in the Program: ``Regrid(criterion, when)`` (spec 0.3, DM13b, N12).

``program.regrid(Regrid(criterion, when))`` writes the refinement criterion and the decision to
regrid in the Program; the adaptive layout then writes neither ``tagging=`` nor ``regrid=``. The
``when`` body is a closed body of the accepted continuation (T33), traced once and generated into
the compiled Program with the criterion. These tests cover the traced surface, the resolution and
the C++; the native runs are in ``tests/python/integration/amr/test_regrid_decision_native.py``.
"""
from __future__ import annotations

import re

import pops
import pytest
from pops.amr import AMRRegrid, Coarsen, Regrid, Tag, largest_cells, smallest_cells
from pops.lib.time import every
from pops.mesh._amr import RegridWhen, ResolvedRefinementCriterion
from pops.mesh._amr.regrid_decision import (
    AcceptedTime, CellIndicator, EveryAcceptedSteps, RegridBodyError,
)
from tests.python.support.refinement_criterion_cases import advection_case


def _value_rules(names):
    return (
        Tag(lambda cell: cell.value(names.state) > names.value("refine")),
        Coarsen(lambda cell: cell.value(names.state) < names.value("coarsen")),
    )


def _peak(names):
    return lambda B: largest_cells(lambda cell: cell.value(names.state)) > names.value("refine")


def _resolved(when, rules=_value_rules):
    case, layout, names = advection_case(rules, when=when)
    return pops.resolve(pops.validate(case), layout=layout), names


def test_every_is_the_body_true_after_every_kth_accepted_step() -> None:
    body = RegridWhen.trace(every(4))
    assert type(body.expression) is EveryAcceptedSteps
    assert body.expression.k == 4
    assert RegridWhen.trace(lambda B: B.steps % 4 == 0).key_data() == body.key_data()
    for k in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="every"):
            every(k)


def test_a_body_reads_steps_time_indicators_and_parameters() -> None:
    _, _, names = advection_case(_value_rules)
    body = RegridWhen.trace(
        lambda B: ((largest_cells(lambda cell: cell.value(names.state)) > names.value("refine"))
                   & (B.steps % 2 == 0))
        | (smallest_cells(lambda cell: cell.value(names.state)) <= names.value("coarsen"))
        | ~(B.time < 0.05))
    from pops.mesh._amr.regrid_decision import _leaves

    kinds = {type(leaf).__name__ for leaf in _leaves((body.expression,))}
    assert {"CellIndicator", "EveryAcceptedSteps", "AcceptedTime", "ValueExpr"} <= kinds
    indicators = [leaf for leaf in _leaves((body.expression,)) if type(leaf) is CellIndicator]
    assert [indicator.kind for indicator in indicators] == ["largest", "smallest"]


@pytest.mark.parametrize("body, message", [
    (lambda B: True, "Python value"),
    (lambda B: B.state, "largest_cells"),
    (lambda B: B.steps % 2 == 1, "B.steps % k == 0"),
    (lambda B: B.steps > 3, "B.steps % k == 0"),
    (lambda B: B.steps % 2 != 0, "~(B.steps % k == 0)"),
    (lambda B: 0 != B.steps % 2, "~(B.steps % k == 0)"),
    (lambda B: (B.time > 0.5) & (B.steps % 2 != 0), "~(B.steps % k == 0)"),
    (lambda B: B.time + 1.0, "returns a Bool"),
    (lambda B: largest_cells(lambda cell: 1.0), "returns a Bool"),
])
def test_a_body_that_reads_outside_t33_or_returns_no_bool_is_refused(body, message) -> None:
    with pytest.raises((RegridBodyError, TypeError), match=re.escape(message)):
        RegridWhen.trace(body)


def test_python_control_flow_cannot_evaluate_a_body() -> None:
    def body(B):
        if B.steps:
            return every(1)(B)
        return every(2)(B)

    with pytest.raises(RegridBodyError, match="no number"):
        RegridWhen.trace(body)


def test_an_indicator_reads_the_cell_itself_in_this_lot() -> None:
    _, _, names = advection_case(_value_rules)
    x = names.frame.axes[0]
    with pytest.raises(NotImplementedError, match="cell.shift"):
        RegridWhen.trace(lambda B: largest_cells(
            lambda cell: cell.shift(x, 1).value(names.state)) > 0.5)
    with pytest.raises(NotImplementedError, match="over="):
        largest_cells(lambda cell: cell.value(names.state), over=object())
    with pytest.raises(TypeError, match="scalar"):
        largest_cells(lambda cell: cell.value(names.state) > 0.5)


def test_regrid_takes_a_criterion_written_as_functions_of_cell() -> None:
    from pops.math import ValueExpr

    case, layout, names = advection_case(_value_rules)
    assert type(Regrid(layout.tagging, every(2)).when) is RegridWhen
    # The tag graph that read an expression of the state went with N12 lot 3c-2b: a rule is a
    # region of cell, and the refusal shows that form.
    value = ValueExpr(names.state)
    for rule in (Tag, Coarsen):
        with pytest.raises(TypeError, match=r"no longer takes an expression of the state.*"
                                            r"lambda cell: cell\.value\(q\)"):
            rule(value > names.value("refine"))
    with pytest.raises(TypeError, match="AMRTagging"):
        Regrid(object(), every(2))


def test_a_program_takes_one_regrid() -> None:
    case, layout, names = advection_case(_value_rules)
    program = names.program
    regrid = Regrid(layout.tagging, every(2))
    assert program.regrid(regrid) is program
    assert program.regrid_contract() is regrid
    with pytest.raises(ValueError, match="only once"):
        program.regrid(Regrid(layout.tagging, every(3)))
    with pytest.raises(TypeError, match="Regrid"):
        type(program)("other").regrid(layout.tagging)


def test_a_program_regrid_resolves_with_the_criterion() -> None:
    plan, _ = _resolved(lambda names: every(2))
    criterion = plan.bootstrap_plan.tagging
    assert type(criterion) is ResolvedRefinementCriterion
    decision = criterion.decision
    assert decision is not None
    assert decision.indicators == () and decision.fields == () and decision.parameters == ()
    assert criterion.canonical_identity()["regrid_decision"] == decision.canonical_identity()

    plan, _ = _resolved(_peak)
    decision = plan.bootstrap_plan.tagging.decision
    assert [indicator.kind for indicator in decision.indicators] == ["largest"]
    assert [row.block for row in decision.fields] == ["tracer"]
    assert [row.variables for row in decision.fields] == [("u",)]
    assert [handle.local_id for handle in decision.parameters] == ["refine_threshold"]


def test_writing_the_regrid_in_the_program_keeps_the_criterion_identity() -> None:
    in_layout, _ = advection_case(_value_rules)[0:2], None
    case, layout = in_layout
    layout_criterion = pops.resolve(pops.validate(case), layout=layout).bootstrap_plan.tagging
    plan, _ = _resolved(lambda names: every(2))
    program_criterion = plan.bootstrap_plan.tagging
    assert layout_criterion.decision is None
    assert "regrid_decision" not in layout_criterion.canonical_identity()
    assert program_criterion._criterion_identity() == layout_criterion.canonical_identity()
    assert program_criterion.qualified_id != layout_criterion.qualified_id


def test_the_regrid_is_written_in_one_place() -> None:
    case, layout, names = advection_case(_value_rules)
    names.program.regrid(Regrid(layout.tagging, every(2)))
    with pytest.raises(ValueError, match="written twice"):
        pops.resolve(pops.validate(case), layout=layout)

    from pops.layouts import AMR

    case, layout, names = advection_case(_value_rules)
    bare = AMR(grid=layout.grid, hierarchy=layout.hierarchy, transfer=layout.transfer,
               execution=layout.execution)
    with pytest.raises(ValueError, match="program.regrid"):
        pops.resolve(pops.validate(case), layout=bare)
    with pytest.raises(TypeError, match="both tagging= and regrid="):
        AMR(grid=layout.grid, hierarchy=layout.hierarchy, transfer=layout.transfer,
            execution=layout.execution, tagging=layout.tagging).options()
    with pytest.raises(TypeError, match="exact AMRRegrid"):
        AMR(grid=layout.grid, hierarchy=layout.hierarchy, transfer=layout.transfer,
            execution=layout.execution, tagging=layout.tagging, regrid=object()).options()
    assert type(layout.regrid) is AMRRegrid


def test_the_layout_names_the_program_as_the_author_of_its_regrid() -> None:
    case, layout, names = advection_case(_value_rules, when=lambda names: every(2))
    assert layout.regrid_in_program
    options = layout.options()
    assert options["regrid"] == {"schema_version": 1, "authority_type": "amr_regrid",
                                 "mode": "program"}
    assert options["tagging"]["mode"] == "program"
    from pops.runtime._amr_bind_lowering import _regrid_every

    assert _regrid_every(layout.runtime_layout_data()) == 1


def test_the_program_identity_holds_its_when_body() -> None:
    from pops.identity.semantic import program_semantic_data

    identities = set()
    for k in (2, 3):
        _, _, names = advection_case(_value_rules, when=lambda names, k=k: every(k))
        data = program_semantic_data(names.program)
        assert data["regrid"]["authority_type"] == "program_regrid"
        identities.add(repr(data["regrid"]))
    assert len(identities) == 2


def test_the_decision_is_generated_with_two_entries_and_its_identity() -> None:
    from pops.codegen.program_emit_regrid_decision import emit_regrid_decision

    plan, names = _resolved(lambda names: lambda B: (
        (largest_cells(lambda cell: cell.value(names.state)) > names.value("refine"))
        & (B.steps % 3 == 0))
        | (smallest_cells(lambda cell: cell.value(names.state)) < names.value("coarsen")))
    decision = plan.bootstrap_plan.tagging.decision
    source = emit_regrid_decision(decision, native_dimension=2)
    assert 'extern "C" int pops_program_regrid_indicator_amr(' in source
    assert 'extern "C" int pops_program_regrid_decision_amr(' in source
    assert decision.qualified_id in source
    assert "request->steps % 3 == 0" in source
    # The smallest value is generated as the largest of the opposite one.
    assert "(-request->indicators[1])" in source
    assert "return pops::Real(-(" in source
    with pytest.raises(ValueError, match="dimension"):
        emit_regrid_decision(decision, native_dimension=3)


def test_a_when_body_is_not_a_tag_predicate() -> None:
    # every(k) is a Boolean of a when body. A region that joins it is refused at the author's line,
    # since it is not a strict comparison; one that reads it in an operand is refused by the
    # criterion, which reads cells, widths and parameters only.
    _, _, names = advection_case(_value_rules)
    with pytest.raises(TypeError, match="strict < or >"):
        Tag(lambda cell: (cell.value(names.state) > names.value("refine"))
            & EveryAcceptedSteps(2))
    with pytest.raises(TypeError, match="EveryAcceptedSteps"):
        _resolved(None, rules=lambda names: (
            Tag(lambda cell: cell.value(names.state) + EveryAcceptedSteps(2)
                > names.value("refine")),
            Coarsen(lambda cell: cell.value(names.state) < names.value("coarsen"))))
    assert type(RegridWhen.trace(lambda B: B.time > 0.5).expression.a) is AcceptedTime
