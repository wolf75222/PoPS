"""G03 lot M1a (ROM-21): the members X and V of a moving configuration, source-only checks.

Note §7.2, §7.4, §7.6, §7.7, §7.9: the node rate of form 1 and of form 2 (∂ₜΦ, derived
symbolically), the members' block and its rates, zero node rate normalized to the base (K13), and
the refusals of a step rule (lot T), a hierarchy (G11) and a node rate that reads a field (R26).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import pops
from pops.analytic import CellBounds, coordinate, cos, sin, time, where
from pops.codegen._moving_members_consumers import MovingMembersNotYetElaborated
from pops.domain import CartesianDomain, Rectangle
from pops.frames import Cartesian1D, Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, GeometryEvolution, MovingControlVolumes
from pops.mesh.moving_configuration import (
    MEMBER_COMPONENTS, MovingConfiguration, evaluate, members_of, time_derivative)
from pops.surface import NotYetElaborated
from pops.time import AdaptiveCFL

from examples.api_v02.models_G03_moving_members import AMPLITUDE, CELLS, build_members
from examples.api_v02.models_static_geometry import build_graded_diffusion, graded_nodes


def _frame():
    return CartesianDomain("g03_m1a_unit", (0.0,), (1.0,)).frame(Cartesian1D())


def _grid(frame, cells=CELLS):
    return Uniform(CartesianGrid(frame=frame, cells=(cells,)))


def test_the_node_rate_of_a_map_is_its_time_derivative_at_the_reference_coordinate():
    """Form 2: w = ∂ₜΦ(ξ, t), derived symbolically; checked against a centred difference."""
    frame, clock = _frame(), pops.Program("g03_m1a_clock").clock
    x, t = coordinate(frame, frame.axes[0]), time(clock)
    law = x + AMPLITUDE*sin(2*math.pi*x)*sin(2*math.pi*t)/(2*math.pi) + 0.01*t*t*x
    derivative = time_derivative(law)
    xi = np.linspace(0.0, 1.0, 9)
    for moment in (0.0, 0.3, 0.71):
        h = 1e-5
        centred = (evaluate(law, xi, moment + h) - evaluate(law, xi, moment - h))/(2*h)
        np.testing.assert_allclose(evaluate(derivative, xi, moment), centred, atol=1e-9)
        exact = AMPLITUDE*np.sin(2*np.pi*xi)*np.cos(2*np.pi*moment) + 0.02*moment*xi
        np.testing.assert_allclose(evaluate(derivative, xi, moment), exact, rtol=0, atol=1e-15)
    assert time_derivative(x*x + 3) is None


def test_zero_over_a_proven_nonzero_denominator_has_no_moving_members():
    frame = _frame()
    x = coordinate(frame, frame.axes[0])
    t = time(pops.Program("geometry_defined_zero").clock)
    base = _grid(frame, cells=4)
    law = x + 0/(1 + t*t)
    assert time_derivative(law) is None
    configuration = MovingConfiguration(base, node_rate=GeometryEvolution((law,)))
    assert configuration.at_rest and configuration.layout is base


@pytest.mark.parametrize("case", ["zero-over-zero", "stationary-pole", "erased-pole"])
def test_partial_map_fails_before_normalization_or_differentiation(case):
    frame = _frame()
    x = coordinate(frame, frame.axes[0])
    t = time(pops.Program("geometry_partial_map").clock)
    base = _grid(frame, cells=4)
    law = {
        "zero-over-zero": x + 0/(x - x),
        "stationary-pole": x + 1/(x - 0.5),
        "erased-pole": x + t + 0*(1/(x - 0.5)),
    }[case]
    with pytest.raises(ValueError, match="zero denominator|undefined at a reference node"):
        MovingConfiguration(base, node_rate=GeometryEvolution((law,)))


def test_stationary_map_can_select_away_a_quotient_pole():
    frame = _frame()
    x = coordinate(frame, frame.axes[0])
    base = _grid(frame, cells=4)
    law = x + where(x == 0.5, 0, 0/(x - 0.5))
    assert MovingConfiguration(base, node_rate=GeometryEvolution((law,))).layout is base


def test_differentiation_retains_a_dynamic_quotient_domain():
    frame = _frame()
    x = coordinate(frame, frame.axes[0])
    t = time(pops.Program("geometry_dynamic_domain").clock)
    configuration = MovingConfiguration(
        _grid(frame, cells=4), node_rate=GeometryEvolution((x + t + 0/(t - 0.5),)))
    xi = configuration.reference_nodes()
    np.testing.assert_array_equal(configuration.node_rate_at(xi, xi, 0.0), np.ones_like(xi))
    with np.errstate(divide="ignore", invalid="ignore"):
        assert not np.isfinite(configuration.node_rate_at(xi, xi, 0.5)).any()


def test_mapped_displacement_is_anchored_to_the_base_nodes():
    frame = _frame()
    x = coordinate(frame, frame.axes[0])
    t = time(pops.Program("geometry_anchored_map").clock)
    base = _grid(frame, cells=4)
    law = x + x*(1 - x)*(0.3 + t)
    configuration = MovingConfiguration(base, node_rate=GeometryEvolution((law,)))
    xi = configuration.reference_nodes()
    np.testing.assert_array_equal(configuration.base_nodes(), xi)
    np.testing.assert_allclose(configuration.node_rate_at(xi, xi, 0.0), xi*(1-xi))
    assert not np.array_equal(configuration.base_nodes(), evaluate(law, xi, 0.0))


@pytest.mark.parametrize("form", ["expression", "map"])
def test_the_members_are_the_state_of_a_block_with_one_rate(form):
    case, layout, values, configuration = build_members(form=form)
    assert configuration.form == form and not configuration.at_rest
    plan = pops.resolve(pops.validate(case), layout=layout)
    (resolved,) = plan.blocks
    assert members_of(resolved.model) is configuration.declared
    assert resolved.state_spaces == ("configuration",)
    assert tuple(item.component for item in configuration.declared.state) == MEMBER_COMPONENTS
    (initial,) = values.values()
    nodes = configuration.reference_nodes()
    np.testing.assert_array_equal(initial, np.stack((nodes[:-1], nodes[1:], np.diff(nodes))))


def test_moving_trials_slice_without_relaxing_global_guard_refusal():
    from pops.codegen.program_slicing import slice_program

    case, layout, _, _ = build_members()
    plan = pops.resolve(pops.validate(case), layout=layout)
    sliced = slice_program(plan.time, ("configuration",))
    guards = [value for value in sliced._values if value.op == "acceptance_guard"]
    assert len(guards) == 4
    assert all(value.attrs["trial_site"]["variant"] == "NotAdmissible" for value in guards)
    assert tuple(sliced._commits.values()) == (guards[-1],)
    assert [value.inputs[0] for value in sliced._values if value.op == "source"] == guards[:-1]
    assert not case._time._acceptance_guards

    case, layout, _, _ = build_members(reject=True)
    plan = pops.resolve(pops.validate(case), layout=layout)
    with pytest.raises(ValueError, match="does not support dt bounds or guards"):
        slice_program(plan.time, ("configuration",))


def test_layout_guard_composition_keeps_exact_strategies_and_unique_sites():
    from dataclasses import replace

    from pops.runtime._multi_layout_executor import _compose_transaction_plans
    from pops.time import FixedDt

    case, layout, _, _ = build_members()
    plan = pops.resolve(pops.validate(case), layout=layout).time.transaction_plan()
    ordinary = replace(plan, guards=())
    assert _compose_transaction_plans((ordinary, ordinary)) is ordinary
    assert _compose_transaction_plans((ordinary, plan)) == plan
    with pytest.raises(ValueError, match="exact transaction strategy"):
        _compose_transaction_plans((replace(ordinary, strategy=FixedDt(0.5)), plan))
    with pytest.raises(ValueError, match="guard names must be unique"):
        _compose_transaction_plans((plan, plan))


def test_a_graded_base_gives_the_members_its_nodes_and_resolves_on_its_logical_grid():
    """The base gives the initial positions; in motion the geometry is the members, so the case
    resolves on the logical grid, not on the static arrays of the base at rest."""
    case, layout, values, configuration = build_members(base="graded")
    assert type(layout) is Uniform
    (initial,) = values.values()
    np.testing.assert_allclose(initial[0], graded_nodes(CELLS)[:-1], rtol=0, atol=4e-16)
    np.testing.assert_array_equal(initial[2], initial[1] - initial[0])
    pops.resolve(pops.validate(case), layout=layout)


@pytest.mark.parametrize("spelling", ["zero", "analytic zero", "map without a clock"])
def test_a_zero_node_rate_normalizes_to_the_base(spelling):
    """K13: the same layout object, no member, and the same resolved plan identity as the base."""
    first = build_graded_diffusion()
    case, base = first[0], first[1]
    frame = base.layout.mesh.frame
    x = coordinate(frame, frame.axes[0])
    value = {"zero": 0, "analytic zero": 0*x*cos(2*math.pi*time(pops.Program("g03_m1a_rest").clock)),
             "map without a clock": GeometryEvolution((x*x,))}[spelling]
    configuration = MovingConfiguration(base, node_rate=value)
    assert configuration.at_rest and configuration.layout is base
    assert configuration.members(case) is None
    moving = pops.resolve(pops.validate(case), layout=configuration.layout)
    still = pops.resolve(pops.validate(build_graded_diffusion()[0]), layout=base)
    assert moving.plan_identity == still.plan_identity
    assert not any(members_of(block.model) for block in moving.blocks)


def test_a_zero_node_rate_normalizes_a_base_of_rank_two():
    frame = Rectangle("g03_m1a_rest_2d", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    base = Uniform(CartesianGrid(frame=frame, cells=(4, 4)))
    assert MovingConfiguration(base, node_rate=0).layout is base
    with pytest.raises(NotYetElaborated, match="G03 lot M2"):
        MovingConfiguration(base, node_rate=coordinate(frame, frame.axes[0]))


def test_a_step_rule_is_refused_naming_lot_t_and_a_fixed_step():
    strategy = AdaptiveCFL(cfl=0.5, max_dt=0.1)
    case, layout, _, _ = build_members(strategy=strategy)
    with pytest.raises(MovingMembersNotYetElaborated, match="G03 lot T") as refused:
        pops.resolve(pops.validate(case), layout=layout)
    assert refused.value.reason == "moving_step_rule" and "dt= a fixed step" in str(refused.value)


def test_a_hierarchy_and_its_regrid_are_refused_naming_g11():
    from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer, Buffer,
                          ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
    from pops.layouts import AMR
    from pops.lib.amr import StateTransfer
    from pops.math import ValueExpr
    from pops.time import every

    def amr_layout(case, members, clock):
        transfer = AMRTransfer()
        transfer.state(members.block[members.state], StateTransfer())
        field = ValueExpr(members.block[members.state])
        return AMR(grid=CartesianGrid(frame=members.configuration.frame, cells=(CELLS,)),
                   hierarchy=AMRHierarchy(max_levels=1, ratios=()),
                   tagging=AMRTagging(rules=(Tag(lambda cell: cell.value(field["volume"]) > 1.0),
                                                 Buffer(cells=0, selection="refine-requests")),
                                      hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
                                      conflict_policy=ConflictPolicy.REFINE_WINS),
                   regrid=AMRRegrid(schedule=every(10, clock=clock)), transfer=transfer,
                   execution=AMRExecution.synchronous())

    case, _, _, configuration = build_members()
    members = configuration.declared
    layout = amr_layout(case, members, case._time.clock)
    with pytest.raises(MovingMembersNotYetElaborated, match="G11") as refused:
        pops.resolve(pops.validate(case), layout=layout)
    assert refused.value.reason == "moving_regrid"


def test_a_node_rate_that_reads_a_field_is_refused_naming_r26():
    frame = _frame()
    model = pops.Model("g03_m1a_field_reader", frame=frame)
    U = model.state("U", components=("u",))
    with pytest.raises(NotYetElaborated, match="R26"):
        MovingConfiguration(_grid(frame), node_rate=0.1*U[0])
    lower = CellBounds(frame).lower(frame.axes[0])
    with pytest.raises(NotYetElaborated, match="R26"):
        MovingConfiguration(_grid(frame), node_rate=lower*1.0)


def test_a_periodic_base_in_motion_waits_for_the_certificate_of_the_seam():
    from pops.mesh import PeriodicAxes
    frame = _frame()
    grid = Uniform(CartesianGrid(frame=frame, cells=(8,), periodic=PeriodicAxes(frame.axes)))
    with pytest.raises(NotYetElaborated, match="M1c"):
        MovingConfiguration(grid, node_rate=0.1*coordinate(frame, frame.axes[0]))
