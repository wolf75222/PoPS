"""Refinement criteria written as functions of ``cell`` (spec 0.3, DM13b, Linear N12).

A Tag or Coarsen predicate may be a function of ``cell``: PoPS traces it once, resolves its reads
into field and parameter tables, and generates it into the compiled Program. These tests cover the
traced surface, the resolved criterion and its C++; the native comparison with the recorded oracles of
the tag-graph VM, removed in N12 lot 3c-2b, is in
``tests/python/integration/amr/test_refinement_criterion_parity.py``.
"""
from __future__ import annotations

import operator
import re

import pops
import pytest
from pops.amr import Coarsen, ConflictPolicy, EqualityPolicy, Hysteresis, Tag
from pops.lib.amr import gradient_norm
from pops.math import ValueExpr, grad, norm, where
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


def test_a_rule_refuses_an_expression_of_the_state() -> None:
    # The tag graph read an expression of the state; without it a rule is a region of cell, and
    # the refusal shows that form.
    def mixed(names):
        return (
            Tag(lambda cell: cell.value(names.state) > names.value("refine")),
            Coarsen(ValueExpr(names.state) < names.value("coarsen")),
        )

    def gradient(names):
        return (Tag(norm(grad(ValueExpr(names.state))) > names.value("refine")),)

    for rules, where in ((mixed, "Coarsen"), (gradient, "Tag")):
        with pytest.raises(TypeError, match=r"%s no longer takes an expression of the state: .*"
                                            r"%s\(lambda cell: cell\.value\(q\) > "
                                            r"case\.value\(t\)\)" % (where, where)):
            advection_case(rules)


def test_the_tagging_policies_are_explicit() -> None:
    from pops.amr import AMRTagging, Buffer

    assert Hysteresis(min_cycles=2, equality=EqualityPolicy.HOLD).canonical_identity() == {
        "schema_version": 1, "hysteresis_type": "min_cycles", "min_cycles": 2,
        "equality": "hold"}
    with pytest.raises(TypeError, match="EqualityPolicy"):
        Hysteresis(min_cycles=1, equality="hold")
    with pytest.raises(ValueError, match=">= 0"):
        Hysteresis(min_cycles=-1, equality=EqualityPolicy.HOLD)
    _, layout, _ = advection_case(_value_rules)
    with pytest.raises(TypeError, match="ConflictPolicy"):
        AMRTagging(rules=layout.tagging.rules, hysteresis=layout.tagging.hysteresis,
                   conflict_policy="error")
    assert AMRTagging(rules=layout.tagging.rules, hysteresis=layout.tagging.hysteresis,
                      conflict_policy=ConflictPolicy.ERROR).rules == layout.tagging.rules
    assert sum(type(rule) is Buffer for rule in layout.tagging.rules) == 1


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


def test_a_region_compares_strictly() -> None:
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


def test_a_where_inside_an_operand_is_the_cond_of_ir4() -> None:
    from pops.codegen.program_emit_refinement_criterion import emit_refinement_criterion
    from pops.mesh._amr.refinement_criterion import certainly

    _, _, names = advection_case(_value_rules)

    def flag(test):
        return lambda cell: where(test(cell.value(names.state), names.value("refine")),
                                  lambda: 1.0, lambda: 0.0) > 0.5

    # The strict rule of the comparisons does not reach the test of a where: that test is the
    # two-valued Boolean of the cond of IR4, which may compare at equality.
    for test in (operator.ge, operator.le, operator.eq):
        region = Tag(flag(test)).predicate.expression
        assert _key(certainly(region)) == _key(region)

    def guarded(names):
        x, _ = names.frame.axes
        return (Tag(lambda cell: where(cell.value(names.state) == names.value("refine"),
                                       lambda: cell.shift(x, 1).value(names.state),
                                       lambda: 0.0) > names.value("coarsen")),)

    source = emit_refinement_criterion(_resolved(guarded).bootstrap_plan.tagging,
                                       native_dimension=2)
    body = source[source.index("operator()"):source.index("const bool finite")]
    test = re.search(r"const pops::Real (cse\d+_) = \(cse\d+_ == cse\d+_\);", body).group(1)
    guard = re.search(r"pops::Real (guard\d+_cse\d+_) = 0;", body).group(1)
    # Only the branch that the test selects is evaluated: the shifted read happens inside it, and
    # where the test is false its value keeps the zero it starts with, so it marks no cell.
    assert body.count("fields[0](at0_, 0)") == 1
    assert re.search(r"\(%s \? \(\[&\]\(\) \{ %s = fields\[0\]\(at0_, 0\); return %s; \}\(\)\)"
                     % (test, guard, guard), body)
    assert "Kokkos::isfinite(%s)" % guard in source

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
    # order of the gradient of the removed tag-graph VM, one operation per statement.
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


@pytest.mark.parametrize("selection", [None, "refine-requests"])
def test_a_region_criterion_keeps_the_selection_of_its_buffer(selection) -> None:
    # Buffer(cells=n) is master's: its cells join the transition buffer, max(n, derived), and the
    # native tag selection 2 grows the whole target coverage. selection="refine-requests" is
    # pops.amr.tag-selection@1: n cells around each refine request, and derived nesting alone in
    # the hierarchy. Regions of cell keep either choice, with the identity, native configuration and
    # transition buffers that the tag graph of the same cells had (recorded before it was removed).
    from pops.amr import Buffer
    from pops.mesh._amr.hierarchy_native import lower_native_hierarchy
    from pops.runtime._amr_bind_lowering import _install_native_tag_selection_config

    class ConfigProjection:
        tag_selection_contract_version = None
        tag_selection_buffer = None

    def route(rules):
        resolved = _resolved(rules, buffer=Buffer(cells=3, selection=selection))
        (local,) = resolved.layout_amr_authorities.values()
        tagging = local.authorities.tagging
        config = ConfigProjection()
        _install_native_tag_selection_config(config, tagging, dimension=2)
        identity = {key: value for key, value in tagging.canonical_identity().items()
                    if key != "graph"}
        native = (config.tag_selection_contract_version, config.tag_selection_buffer)
        buffers = lower_native_hierarchy(resolved.resolved_hierarchy).transition_buffers
        return tagging, identity, native, buffers

    region, region_identity, region_native, region_buffers = route(_value_rules)

    assert type(region.graph) is ResolvedRefinementCriterion
    assert region.buffer_cells == 3
    assert region.selection == selection
    if selection is None:
        assert region_identity == {
            "schema_version": 1, "authority_type": "resolved_amr_tagging", "buffer_cells": 3}
        assert region_native == (2, (0, 0))
        assert region_buffers == ((3, 3),)
    else:
        assert region_identity == {
            "schema_version": 2, "authority_type": "resolved_amr_tagging",
            "selection_contract": "pops.amr.tag-selection@1", "buffer_cells": 3}
        assert region_native == (1, (3, 3))
        assert region_buffers == ((2, 2),)
