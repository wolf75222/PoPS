"""G03 lot M1a (ROM-21): the members X and V of a moving configuration, source-only checks.

Note §7.2, §7.4, §7.6, §7.7, §7.9: the velocity of form 1 and of form 2 (∂ₜΦ, derived
symbolically), the members' block and its rates, zero velocity normalized to the base (K13), and
the refusals of a step rule (lot T), a hierarchy (G11) and a velocity that reads a field (R26).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import pops
from pops.analytic import CellBounds, coordinate, cos, sin, time
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


def test_the_velocity_of_a_map_is_its_time_derivative_at_the_reference_coordinate():
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


def test_a_graded_base_gives_the_members_its_nodes_and_resolves_on_its_logical_grid():
    """The base gives the initial positions; in motion the geometry is the members, so the case
    resolves on the logical grid, not on the static arrays of the base at rest."""
    case, layout, values, configuration = build_members(base="graded")
    assert type(layout) is Uniform
    (initial,) = values.values()
    np.testing.assert_allclose(initial[0], graded_nodes(CELLS)[:-1], rtol=0, atol=4e-16)
    np.testing.assert_array_equal(initial[2], initial[1] - initial[0])
    pops.resolve(pops.validate(case), layout=layout)


@pytest.mark.parametrize("velocity", ["zero", "analytic zero", "map without a clock"])
def test_zero_velocity_normalizes_to_the_base(velocity):
    """K13: the same layout object, no member, and the same resolved plan identity as the base."""
    first = build_graded_diffusion()
    case, base = first[0], first[1]
    frame = base.layout.mesh.frame
    x = coordinate(frame, frame.axes[0])
    value = {"zero": 0, "analytic zero": 0*x*cos(2*math.pi*time(pops.Program("g03_m1a_rest").clock)),
             "map without a clock": GeometryEvolution((x*x,))}[velocity]
    configuration = MovingConfiguration(base, velocity=value)
    assert configuration.at_rest and configuration.layout is base
    assert configuration.members(case) is None
    moving = pops.resolve(pops.validate(case), layout=configuration.layout)
    still = pops.resolve(pops.validate(build_graded_diffusion()[0]), layout=base)
    assert moving.plan_identity == still.plan_identity
    assert not any(members_of(block.model) for block in moving.blocks)


def test_zero_velocity_normalizes_a_base_of_rank_two():
    frame = Rectangle("g03_m1a_rest_2d", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    base = Uniform(CartesianGrid(frame=frame, cells=(4, 4)))
    assert MovingConfiguration(base, velocity=0).layout is base
    with pytest.raises(NotYetElaborated, match="G03 lot M2"):
        MovingConfiguration(base, velocity=coordinate(frame, frame.axes[0]))


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


def test_a_velocity_that_reads_a_field_is_refused_naming_r26():
    frame = _frame()
    model = pops.Model("g03_m1a_field_reader", frame=frame)
    U = model.state("U", components=("u",))
    with pytest.raises(NotYetElaborated, match="R26"):
        MovingConfiguration(_grid(frame), velocity=0.1*U[0])
    lower = CellBounds(frame).lower(frame.axes[0])
    with pytest.raises(NotYetElaborated, match="R26"):
        MovingConfiguration(_grid(frame), velocity=lower*1.0)


def test_a_periodic_base_in_motion_waits_for_the_certificate_of_the_seam():
    from pops.mesh import PeriodicAxes
    frame = _frame()
    grid = Uniform(CartesianGrid(frame=frame, cells=(8,), periodic=PeriodicAxes(frame.axes)))
    with pytest.raises(NotYetElaborated, match="M1c"):
        MovingConfiguration(grid, velocity=0.1*coordinate(frame, frame.axes[0]))
