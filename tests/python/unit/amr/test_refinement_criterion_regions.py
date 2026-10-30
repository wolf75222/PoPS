"""Refinement criteria written as functions of ``cell`` (spec 0.3, DM13b, Linear N12).

A Tag or Coarsen predicate may be a function of ``cell``: PoPS traces it once, resolves its reads
into field and parameter tables, and generates it into the compiled Program. These tests cover the
traced surface, the resolved criterion and its C++; the native parity with the tag-graph VM is in
``tests/python/integration/amr/test_refinement_criterion_parity.py``.
"""
from __future__ import annotations

import re

import pops
import pytest
from pops.amr import Coarsen, ConflictPolicy, EqualityPolicy, Hysteresis, Tag
from pops.lib.amr import gradient_norm
from pops.math import ValueExpr, grad, norm
from pops.mesh._amr import CriterionCell, RefinementRegion, ResolvedRefinementCriterion
from tests.python.support.refinement_criterion_cases import advection_case


def _gradient_rules(names):
    axes = names.frame.axes

    def magnitude(cell):
        return gradient_norm(cell, names.state, axes=axes, stencil=names.gradient)

    return (
        Tag(lambda cell: magnitude(cell) > names.value("refine")),
        Coarsen(lambda cell: magnitude(cell) < names.value("coarsen")),
    )


def _value_rules(names):
    return (
        Tag(lambda cell: cell.value(names.state) > names.value("refine")),
        Coarsen(lambda cell: cell.value(names.state) < names.value("coarsen")),
    )


def _resolved(rules, **options):
    case, layout, _ = advection_case(rules, **options)
    return pops.resolve(pops.validate(case), layout=layout)


def _key(expression):
    from pops._ir.visitors import _key as structural_key

    return structural_key(expression)


def test_cell_shift_is_one_cell_whatever_the_order_of_its_steps() -> None:
    _, _, names = advection_case(_value_rules)
    x, y = names.frame.axes
    cell = CriterionCell()

    first = cell.shift(x, 1).shift(y, -2).value(names.state)
    second = cell.shift(y, -2).shift(x, 1).value(names.state)
    assert _key(first) == _key(second)
    assert first.offset == ((0, 1), (1, -2))
    assert _key(cell.shift(x, 1).shift(x, -1).value(names.state)) == _key(
        cell.value(names.state))
    assert _key(cell.value(names.state)) == _key(cell.value(ValueExpr(names.state)["u"]))


def test_cell_refuses_reads_that_are_not_typed_cell_reads() -> None:
    _, _, names = advection_case(_value_rules)
    x, _ = names.frame.axes
    cell = CriterionCell()

    with pytest.raises(TypeError, match="axis of a Cartesian frame"):
        cell.shift(0, 1)
    with pytest.raises(TypeError, match="integer number of steps"):
        cell.shift(x, 1.0)
    with pytest.raises(TypeError, match="integer number of steps"):
        cell.shift(x, True)
    with pytest.raises(TypeError, match="axis of a Cartesian frame"):
        cell.width("x")
    with pytest.raises(TypeError, match="scalar block state or one typed component"):
        cell.value(names.params["refine"])

    model = pops.Model("two_components")
    pair = model.state("Q", components=("a", "b"))
    block = pops.Case("two_components").block("pair", model)
    with pytest.raises(TypeError, match="scalar block state or one typed component"):
        cell.value(block[pair])
    assert cell.value(ValueExpr(block[pair])["b"]).component == "b"


def test_tag_traces_a_function_of_cell_once_and_keeps_its_boolean_region() -> None:
    _, _, names = advection_case(_value_rules)
    calls = []

    def region(cell):
        calls.append(cell)
        return cell.value(names.state) > names.value("refine")

    rule = Tag(region)
    assert type(rule.predicate) is RefinementRegion
    assert len(calls) == 1 and type(calls[0]) is CriterionCell
    assert rule.inspect()["predicate"]["protocol"] == "pops.amr.refinement-region.v1"
    assert rule.inspect() == Tag(region).inspect()

    with pytest.raises(TypeError, match="Boolean expression"):
        Tag(lambda cell: cell.value(names.state))
    with pytest.raises(TypeError, match="function of cell"):
        Coarsen(0.5)


def test_tagging_writes_every_predicate_the_same_way() -> None:
    def mixed(names):
        return (
            Tag(lambda cell: cell.value(names.state) > names.value("refine")),
            Coarsen(ValueExpr(names.state) < names.value("coarsen")),
        )

    with pytest.raises(TypeError, match="the same way"):
        advection_case(mixed)


def test_resolution_builds_the_field_and_parameter_tables_of_the_regions() -> None:
    resolved = _resolved(_gradient_rules)
    criterion = resolved.bootstrap_plan.tagging

    assert type(criterion) is ResolvedRefinementCriterion
    assert criterion.dimension == 2
    assert len(criterion.refine) == len(criterion.coarsen) == 1
    (field,) = criterion.fields
    assert field.block == "tracer"
    assert field.variables == ("u",)
    assert field.components == (0,)
    assert field.reach == (1, 1)
    assert [handle.local_id for handle in criterion.parameters] == [
        "refine_threshold", "coarsen_threshold"]
    inspected = criterion.inspect()
    assert inspected["decision"] == "refine_then_coarsen"
    assert inspected["qualified_id"] == criterion.qualified_id
    # The bootstrap receipts name the indicator provider: the region generated into the Program.
    assert [row["lowering"]["qualified_id"] for row in inspected["lowerings"]] == [
        "pops.handle.v1::shared:pops.amr.refinement-criterion.v1::tag_lowering::program_region"]

    again = _resolved(_gradient_rules).bootstrap_plan.tagging
    assert again.qualified_id == criterion.qualified_id
    values = _resolved(_value_rules).bootstrap_plan.tagging
    assert values.qualified_id != criterion.qualified_id
    assert values.fields[0].reach == (0, 0)


def test_region_bound_values_come_from_the_runtime_parameters() -> None:
    criterion = _resolved(
        _value_rules, thresholds={"refine": 0.30, "coarsen": 0.20}).bootstrap_plan.tagging
    refine, coarsen = criterion.parameters

    data = criterion.runtime_criterion_data({refine: 0.5, coarsen: 0.25})
    assert data["parameters"] == [0.5, 0.25]
    assert data["identity"] == criterion.qualified_id
    assert data["fields"][0]["state"] == criterion.fields[0].state.canonical_identity()[
        "qualified_id"]
    with pytest.raises(ValueError, match="is not bound"):
        criterion.runtime_criterion_data({refine: 0.5})
    with pytest.raises(TypeError, match="finite scalars"):
        criterion.runtime_criterion_data({refine: float("nan"), coarsen: 0.25})


@pytest.mark.parametrize("hysteresis, conflict", [
    (Hysteresis(0, EqualityPolicy.HOLD), ConflictPolicy.ERROR),
    (Hysteresis(0, EqualityPolicy.HOLD), ConflictPolicy.COARSEN_WINS),
    (Hysteresis(0, EqualityPolicy.REFINE), ConflictPolicy.REFINE_WINS),
])
def test_regions_decide_refine_before_coarsen_and_never_meet_an_equality(
        hysteresis, conflict) -> None:
    with pytest.raises(ValueError, match="REFINE_WINS and EqualityPolicy.HOLD"):
        _resolved(_value_rules, hysteresis=hysteresis, conflict_policy=conflict)


def test_region_reads_the_level_only_through_cell() -> None:
    def direct(names):
        return (Tag(lambda cell: ValueExpr(names.state) > names.value("refine")),)

    with pytest.raises(TypeError, match="through cell.value, cell.width and runtime parameters"):
        _resolved(direct)


def test_a_region_compares_strictly_as_a_leaf_of_the_tag_graph() -> None:
    _, _, names = advection_case(_value_rules)
    threshold = names.value("refine")

    for region in (lambda cell: cell.value(names.state) >= threshold,
                   lambda cell: cell.value(names.state) <= threshold,
                   lambda cell: cell.value(names.state) == threshold,
                   lambda cell: ~(cell.value(names.state) >= threshold)):
        with pytest.raises(TypeError, match="at exact equality decides nothing"):
            Tag(region)


def test_a_region_acts_where_it_is_certainly_true() -> None:
    from pops.mesh._amr.refinement_criterion import certainly

    _, _, names = advection_case(_value_rules)
    x, _ = names.frame.axes
    cell = CriterionCell()
    v, w = cell.value(names.state), cell.shift(x, 1).value(names.state)
    low, high = names.value("refine"), names.value("coarsen")

    # Kleene's logic with a comparison undecided at equality, as the VM evaluates a rule: a negated
    # comparison is certainly true only where the opposite strict comparison holds.
    assert _key(certainly(~(v < low))) == _key(v > low)
    assert _key(certainly(~(v < low), False)) == _key(v < low)
    assert _key(certainly((v > low) & ~(v > high))) == _key((v > low) & (v < high))
    assert _key(certainly(~((v > low) | (w < high)))) == _key((v < low) & (w > high))
    assert _key(certainly(~((v > low) & (w < high)))) == _key((v < low) | (w > high))
    assert _key(certainly(~~(v > low))) == _key(v > low)
    composite = (v > low) & ((w < high) | (v < high))
    assert _key(certainly(composite)) == _key(composite)


def test_every_comparison_is_evaluated_before_the_rules_join_them() -> None:
    from pops.codegen.program_emit_refinement_criterion import emit_refinement_criterion

    def band(names):
        x, _ = names.frame.axes
        return (
            Tag(lambda cell: (cell.value(names.state) > names.value("refine"))
                & ~(cell.shift(x, 2).value(names.state) > names.value("coarsen"))),
            Coarsen(lambda cell: ~(cell.value(names.state) < names.value("coarsen"))),
        )

    source = emit_refinement_criterion(_resolved(band).bootstrap_plan.tagging,
                                       native_dimension=2)
    body = source[source.index("operator()"):source.index("const bool finite")]
    # No negation is left, so a comparison at equality decides nothing under either rule.
    assert "!" not in body
    value = re.search(r"const pops::Real (cse\d+_) = fields\[0\]\(index, 0\);", body).group(1)
    shifted = re.search(r"const pops::Real (cse\d+_) = fields\[0\]\(at0_, 0\);", body).group(1)
    below = re.search(r"const pops::Real (cse\d+_) = \(%s < cse\d+_\);" % shifted, body).group(1)
    above = re.search(r"const pops::Real (cse\d+_) = \(%s > cse\d+_\);" % value, body)
    assert above
    # The right operand of && was read and compared before the join, which only returns it, so a
    # non-finite read there marks the cell even where the left operand is false.
    assert re.search(r"\(cse\d+_ && \(\[&\]\(\) \{\s+return %s; \}\(\)\)\)" % below, body)
    assert "Kokkos::isfinite(%s)" % shifted in source
    assert "Kokkos::isfinite(%s)" % below in source


def test_gradient_norm_is_the_region_written_by_hand() -> None:
    # CF4: the library object and the same region written in a case have one identity.
    from pops.math import sqrt

    def by_hand(cell, names):
        total = None
        for axis, row in zip(names.frame.axes, names.gradient.axes, strict=True):
            derivative = None
            for offset, coefficient in zip(row.offsets, row.coefficients, strict=True):
                term = coefficient * cell.shift(axis, offset).value(names.state)
                derivative = term if derivative is None else derivative + term
            derivative = derivative / cell.width(axis)
            square = derivative * derivative
            total = square if total is None else total + square
        return sqrt(total)

    def library(cell, names):
        return gradient_norm(cell, names.state, axes=names.frame.axes, stencil=names.gradient)

    def rules_of(magnitude):
        def rules(names):
            return (Tag(lambda cell: magnitude(cell, names) > names.value("refine")),
                    Coarsen(lambda cell: magnitude(cell, names) < names.value("coarsen")))
        return rules

    _, _, names = advection_case(_value_rules)
    cell = CriterionCell()
    assert _key(library(cell, names)) == _key(by_hand(cell, names))
    assert (_resolved(rules_of(library)).bootstrap_plan.tagging.qualified_id
            == _resolved(rules_of(by_hand)).bootstrap_plan.tagging.qualified_id)


def test_emitted_kernel_keeps_the_arithmetic_order_of_the_vm_gradient() -> None:
    from pops.codegen.program_emit_refinement_criterion import (
        CRITERION_ENTRY_SYMBOL,
        CRITERION_IDENTITY_SYMBOL,
        emit_refinement_criterion,
    )

    criterion = _resolved(_gradient_rules).bootstrap_plan.tagging
    source = emit_refinement_criterion(criterion, native_dimension=2)

    assert 'extern "C" int %s(' % CRITERION_ENTRY_SYMBOL in source
    assert 'extern "C" const char* %s()' % CRITERION_IDENTITY_SYMBOL in source
    assert '"%s"' % criterion.qualified_id in source
    assert "patch->field_count != 1" in source and "patch->parameter_count != 2" in source
    # (c0 * v(-1) + c1 * v(+1)) / h per axis, then the sum of the squares and its square root: the
    # order of PreparedTaggingPatchKernel, one operation per statement.
    order = [r"\(-0\.5 \* cse\d+_\)", r"\(0\.5 \* cse\d+_\)",
             r"\(cse\d+_ \+ cse\d+_\)", r"\(cse\d+_ / cse\d+_\)", r"\(cse\d+_ \* cse\d+_\)",
             r"std::sqrt\(cse\d+_\)", r"\(cse\d+_ > cse\d+_\)", r"\(cse\d+_ < cse\d+_\)"]
    positions = [re.search(pattern, source) for pattern in order]
    assert all(positions), [p for p, m in zip(order, positions) if m is None]
    starts = [match.start() for match in positions]
    assert starts == sorted(starts)
    assert source.count("Kokkos::isfinite(") >= 8

    with pytest.raises(ValueError, match="native dimension 3"):
        emit_refinement_criterion(criterion, native_dimension=3)


def test_gradient_norm_reads_a_function_of_cell_at_each_stencil_cell() -> None:
    _, _, names = advection_case(_value_rules)
    x, y = names.frame.axes
    cell = CriterionCell()

    def doubled(at):
        return 2.0 * at.value(names.state)

    expression = gradient_norm(cell, doubled, axes=(x, y), stencil=names.gradient)
    from pops.mesh._amr.refinement_criterion import CellRead, criterion_inputs

    reads = [leaf for leaf in criterion_inputs((expression > 0.0,)) if type(leaf) is CellRead]
    assert sorted(leaf.offset for leaf in reads) == [((0, -1),), ((0, 1),), ((1, -1),), ((1, 1),)]
    with pytest.raises(TypeError, match="discrete gradient stencil"):
        gradient_norm(cell, names.state, axes=(x,), stencil=(-1, 1))
    with pytest.raises(ValueError, match="one stencil row per axis"):
        gradient_norm(cell, names.state, axes=(x,), stencil=names.gradient)


def test_tag_graph_route_is_unchanged_for_expression_predicates() -> None:
    def graph(names):
        magnitude = norm(grad(ValueExpr(names.state)))
        return (Tag(magnitude > names.value("refine")),
                Coarsen(magnitude < names.value("coarsen")))

    tagging = _resolved(graph).bootstrap_plan.tagging
    assert type(tagging) is not ResolvedRefinementCriterion
    assert tagging.runtime_tagging_data()["graph_type"] == "amr_tagging_runtime"
