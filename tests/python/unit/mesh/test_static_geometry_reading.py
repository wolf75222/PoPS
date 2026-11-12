"""Source-only tests of the static reading of the geometry evolution contract (N17, C1).

A coordinate map that reads no clock is read at resolution as static geometry arrays: node
positions when component a reads the logical coordinate a alone, a chart otherwise, whose
measures need a stated rule (D0b). What the arrays do not serve is refused before any C++, with
its exact reason; the clock-reading maps keep the refusals of the moving route
(test_moving_geometry_contract.py).
"""
import functools
import math

import numpy as np
import pops
import pytest
from pops import math as pm
from pops.analytic import constant, coordinates, cos, sin
from pops.boundary.transport import Inflow, NoFlux, SlipWall
from pops.codegen._native_spatial_layout import NativeSpatialLayoutError, native_spatial_layouts
from pops.codegen._static_geometry_consumers import (
    StaticGeometryConsumerError, static_geometry_layouts)
from pops.domain import CartesianDomain
from pops.frames import Cartesian2D, Cartesian3D
from pops.layouts import Uniform
from pops.lib.quadrature import GaussLegendre
from pops.mesh import (
    CartesianGrid, GeometryEvolution, MovingControlVolumes, PeriodicAxes, normalize_layout_plan)
from pops.mesh._static_geometry import is_static_coordinate_map
from pops.numerics import (
    Diffusion, DiscretizationPlan, StateStorage, TensorDiffusion, reconstruction, riemann)
from pops.numerics.stencil_flux import StencilFlux
from pops.physics.diffusion import DiffusiveBoundary
from pops.time import AdaptiveCFL, FixedDt

from examples.api_v02.models_static_geometry import (
    _initial, _ssprk2, annulus_map, build_annulus_free_stream, build_graded_diffusion)


def _refusal(layout, owner="pops-static-geometry"):
    plan = normalize_layout_plan(layout, owner=pops.Case(owner).owner_path.canonical())
    with pytest.raises(NativeSpatialLayoutError) as caught:
        native_spatial_layouts(plan)
    return caught.value.code


def test_a_clock_free_map_is_static_and_a_clocked_one_is_not():
    _, layout, _, _ = build_annulus_free_stream()
    assert is_static_coordinate_map(layout.evolution.to_data())
    from tests.python.unit.mesh.test_moving_geometry_contract import request
    _, moving = request()
    assert not is_static_coordinate_map(moving.evolution.to_data())


def test_a_chart_without_its_rule_is_refused():
    """D0b, SU9: the measures of a chart are generated with a stated quadrature; omitted is
    refused, never defaulted."""
    _, layout, _, _ = build_annulus_free_stream(geometry=None)
    assert _refusal(layout) == "static_geometry_rule_missing"


def test_node_positions_refuse_a_rule_that_would_state_nothing():
    _, layout, _, _ = build_graded_diffusion(geometry=GaussLegendre(2))
    assert _refusal(layout) == "static_geometry_rule_unused"


def test_a_component_that_reads_no_coordinate_is_refused():
    frame = CartesianDomain("static-degenerate", (1.0, 0.0), (2.0, 1.0)).frame(Cartesian2D())
    r, _ = coordinates(frame)
    layout = MovingControlVolumes(Uniform(CartesianGrid(frame=frame, cells=(4, 4))),
                                  evolution=GeometryEvolution((r*r, constant(2.0))),
                                  geometry=GaussLegendre(2))
    assert _refusal(layout) == "static_geometry_degenerate_map"


def test_a_graded_box_of_rank_three_is_read_as_node_positions():
    """N17 L1b: node positions are served at every rank to three (a graded box, which
    pops.Graded elaborates to); the base refused rank three."""
    frame = CartesianDomain("static-rank-three", (0.0,)*3, (1.0,)*3).frame(Cartesian3D())
    x, y, z = coordinates(frame)
    layout = MovingControlVolumes(Uniform(CartesianGrid(frame=frame, cells=(2, 2, 2))),
                                  evolution=GeometryEvolution((x*x/2 + x/2, y, z*z*z)))
    owner = pops.Case("static-rank-three").owner_path.canonical()
    plan = normalize_layout_plan(layout, owner=owner)
    assert native_spatial_layouts(plan)
    (reading,) = static_geometry_layouts(plan).values()
    assert (reading.kind, reading.rank) == ("nodes", 3)


def test_a_chart_of_rank_three_is_planned_not_covered():
    """A chart has rank two (design note N17, 3.10); one of rank three is refused at resolution,
    with its stated rule, before any C++."""
    frame = CartesianDomain("static-chart-three", (0.0,)*3, (1.0,)*3).frame(Cartesian3D())
    x, y, z = coordinates(frame)
    layout = MovingControlVolumes(Uniform(CartesianGrid(frame=frame, cells=(2, 2, 2))),
                                  evolution=GeometryEvolution((x + y/4, y, z)),
                                  geometry=GaussLegendre(2))
    owner = pops.Case("static-chart-three").owner_path.canonical()
    plan = normalize_layout_plan(layout, owner=owner)
    with pytest.raises(NativeSpatialLayoutError) as caught:
        native_spatial_layouts(plan)
    assert caught.value.code == "static_geometry_rank_unavailable"
    assert "a chart of rank 3 is planned, not covered" in str(caught.value)


def test_the_polar_chart_of_graph_one_reads_both_axes_in_each_component():
    frame = CartesianDomain("static-chart", (1.0, 0.0), (2.0, 2*math.pi)).frame(Cartesian2D())
    r, theta = coordinates(frame)
    layout = MovingControlVolumes(Uniform(CartesianGrid(frame=frame, cells=(4, 8))),
                                  evolution=GeometryEvolution((r*cos(theta), r*sin(theta))),
                                  geometry=GaussLegendre(2))
    plan = normalize_layout_plan(layout, owner=pops.Case("static-chart").owner_path.canonical())
    assert native_spatial_layouts(plan)


@pytest.mark.parametrize("variant,code", [
    (dict(traces=reconstruction.MUSCL()), "static_geometry_reconstruction_unavailable"),
    (dict(faces=riemann.HLLC()), "static_geometry_riemann_unavailable"),
    (dict(faces=riemann.HLL()), "static_geometry_riemann_unavailable"),
    (dict(step=AdaptiveCFL(0.4)), "static_geometry_step_unavailable"),
    (dict(walls=SlipWall), "static_geometry_boundary_unavailable"),
    (dict(walls=NoFlux), "static_geometry_boundary_unavailable"),
    (dict(walls=functools.partial(Inflow, value=(1.3, 0.7, -0.4))),
     "static_geometry_boundary_unavailable"),
])
def test_consumers_that_read_the_logical_spacings_are_refused_at_resolution(variant, code):
    """The arrays are read by face laws whose declared requirements are the normal flux, a bound
    of its normal spectral radius and the provider pack, on traces of formal order one, by
    boundaries that extend the interior state alone, by the scalar diffusion of node positions and
    by a fixed step; every other consumer of a mapped layout is refused before any C++ (a wall
    reads the logical axis, HLL reads signed wave speeds, a prescribed inflow state is not on the
    allowlist of this lot)."""
    case, layout, _, _ = build_annulus_free_stream(**variant)
    with pytest.raises(StaticGeometryConsumerError) as caught:
        pops.resolve(pops.validate(case), layout=layout)
    assert caught.value.code == code


def test_source_authored_traces_are_refused_at_resolution():
    """A trace authored in source compiles to a stencil of its own; on a mapped layout the faces
    are one resolved catalogue route that reads the arrays, so the source route is refused even at
    formal order one."""
    traces = reconstruction.User(lambda sample: sample(0), formal_order=1)
    case, layout, _, _ = build_annulus_free_stream(traces=traces)
    with pytest.raises(StaticGeometryConsumerError) as caught:
        pops.resolve(pops.validate(case), layout=layout)
    assert caught.value.code == "static_geometry_source_numerics_unavailable"
    assert caught.value.evidence == {"part": "reconstruction"}


def _chart_diffusion():
    """The scalar diffusion of the annulus of graph (1), conormal walls, theta periodic."""
    frame = CartesianDomain("n17_chart_diffusion", (1.0, 0.0), (2.0, 2*math.pi)).frame(Cartesian2D())
    model = pops.Model("N17_chart_diffusion", frame=frame)
    U = model.state("heat", components=("u",))
    walls = (DiffusiveBoundary(0, "lower", "conormal", 0.0),
             DiffusiveBoundary(0, "upper", "conormal", 0.0),
             DiffusiveBoundary(1, "lower", "periodic"), DiffusiveBoundary(1, "upper", "periodic"))
    flux = model.diffusive_flux("conduction", state=U,
                                value=pm.CoeffGradient(U[0], ((0.01, 0.0), (0.0, 0.01))),
                                boundaries=walls)
    rate = model.rate("heat_balance", equation=pm.ddt(U) == pm.div(flux))
    case = pops.Case("N17_chart_diffusion")
    block = case.block("ring", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, Diffusion(flux=flux))
    case.numerics(plan, block=block)
    _ssprk2(case, block, U, rate, 1e-4, "N17_chart_diffusion_SSPRK2")
    periodic = PeriodicAxes((frame.axes[1],))
    layout = MovingControlVolumes(Uniform(CartesianGrid(frame=frame, cells=(4, 8),
                                                        periodic=periodic)),
                                  evolution=annulus_map(frame), geometry=GaussLegendre(2))
    _initial(case, block[U], np.ones((1, 8, 4)))
    return case, layout


def test_the_diffusion_of_a_chart_is_refused_at_resolution():
    """The two-point flux reads the distance of the centroids along the face normal, which holds
    on node positions, whose faces are orthogonal to the line of the centroids, and not on a chart
    (its faces are chords); a chart is refused at every rank."""
    case, layout = _chart_diffusion()
    with pytest.raises(StaticGeometryConsumerError) as caught:
        pops.resolve(pops.validate(case), layout=layout)
    assert caught.value.code == "static_geometry_diffusion_unavailable"
    assert "diffuses on a chart of rank 2" in str(caught.value)


def _plane_diffusion(method):
    """The scalar diffusion of a graded plane (node positions of rank two), both axes periodic."""
    frame = CartesianDomain("n17_plane_diffusion", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    model = pops.Model("N17_plane_diffusion", frame=frame)
    U = model.state("U", components=("u",))
    flux = model.diffusive_flux("conduction", state=U,
                                value=pm.CoeffGradient(U[0], ((0.01, 0.0), (0.0, 0.01))),
                                boundaries=tuple(DiffusiveBoundary(axis, side, "periodic")
                                                 for axis in (0, 1) for side in ("lower", "upper")))
    rate = model.rate("heat_balance", equation=pm.ddt(U) == pm.div(flux))
    case = pops.Case("N17_plane_diffusion")
    block = case.block("plane", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, method(flux=flux))
    case.numerics(plan, block=block)
    _ssprk2(case, block, U, rate, 1e-4, "N17_plane_diffusion_SSPRK2")
    x, y = coordinates(frame)
    periodic = PeriodicAxes(tuple(frame.axes))
    layout = MovingControlVolumes(Uniform(CartesianGrid(frame=frame, cells=(4, 4),
                                                        periodic=periodic)),
                                  evolution=GeometryEvolution((x*x/2 + x/2, y)))
    _initial(case, block[U], np.ones((1, 4, 4)))
    return case, layout


def test_the_two_point_diffusion_of_a_graded_plane_resolves_and_a_wider_face_is_refused():
    """The static arrays serve a diffusion by capabilities (CF2): a face that reads its two cells
    alone (ghost depth one), one component and a diagonal tensor. The tensor realization of the
    same diagonal law reads a centred cell gradient through a face two cells wide, so it is refused
    before any C++, on node positions of rank two as on any other."""
    case, layout = _plane_diffusion(Diffusion)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    assert resolved is not None
    case, layout = _plane_diffusion(TensorDiffusion)
    with pytest.raises(StaticGeometryConsumerError) as caught:
        pops.resolve(pops.validate(case), layout=layout)
    assert caught.value.code == "static_geometry_diffusion_unavailable"
    assert "through a wider face" in str(caught.value)


def _stencil_body(read):
    """The two-point face -k (u_R - u_L) / h of a StencilFlux and its incident frequency
    (k / h) / h; ``read`` names a variant that divides by the spacing of the other axis or reads
    the cell diagonal to the face instead of its left cell."""
    def body(sample, spacing, axis):
        h = spacing[1 - axis] if read == "other spacing" else spacing[axis]
        left = [-1 if d == axis else 0 for d in range(2)]
        if read == "diagonal":
            left[1 - axis] = 1
        gradient = (sample((0, 0), 0) - sample(tuple(left), 0)) / h
        return (-(0.01 * gradient),), (0.01 / h) / h

    return body


def _stencil_plane(read="two cells", chart=False):
    """A scalar of the periodic plane advanced by the divergence of a StencilFlux (N3 lot 1, the
    realization of two_point()), on node positions of rank two (x x / 2 + x / 2, y) or, with
    ``chart=True``, on the chart (x + y / 4, y), whose first component reads both coordinates."""
    frame = CartesianDomain("n17_stencil_plane", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    model = pops.Model("N17_stencil_plane", frame=frame)
    U = model.state("U", components=("u",))
    case = pops.Case("N17_stencil_plane")
    block = case.block("plane", model)
    plan = DiscretizationPlan()
    plan.states.add(U, StateStorage(ghost_depth=1))
    case.numerics(plan, block=block)
    program = pops.Program("N17_stencil_plane_step")
    u = program.state(block[U])
    flux = StencilFlux(dimension=2, components=1, radius=1, body=_stencil_body(read))
    rate = program.stencil_flux_divergence("two_point", u.n, flux=flux)
    program.commit(u.next, program.value("candidate", u.n + program.dt * rate, at=u.next.point))
    program.step_strategy(FixedDt(1e-4))
    case.program(program)
    x, y = coordinates(frame)
    periodic = PeriodicAxes(tuple(frame.axes))
    layout = MovingControlVolumes(
        Uniform(CartesianGrid(frame=frame, cells=(4, 4), periodic=periodic)),
        evolution=GeometryEvolution((x + y/4, y) if chart else (x*x/2 + x/2, y)),
        geometry=GaussLegendre(2) if chart else None)
    _initial(case, block[U], np.ones((1, 4, 4)))
    return case, layout


@pytest.mark.parametrize("read,chart,reason", [
    ("two cells", False, None),
    ("other spacing", False, "the spacing of another axis"),
    ("diagonal", False, "a cell beyond the two cells of its face"),
    ("two cells", True, "a chart of rank 2"),
])
def test_a_stencil_flux_reads_its_two_cells_and_the_distance_across_its_face(read, chart, reason):
    """The divergence of a StencilFlux (two_point() since N3 lot 1) is served on node positions:
    the row of axis a reads the two cells of its face and the spacing of a, which the arrays give
    as the distance across the face, and its faces are orthogonal to the line of the centroids. A
    row that reads the spacing of another axis or a cell beyond its two cells, and any stencil on
    a chart, are refused at resolution, before any C++."""
    case, layout = _stencil_plane(read, chart)
    if reason is None:
        assert pops.resolve(pops.validate(case), layout=layout) is not None
        return
    with pytest.raises(StaticGeometryConsumerError) as caught:
        pops.resolve(pops.validate(case), layout=layout)
    assert caught.value.code == "static_geometry_stencil_unavailable"
    assert reason in str(caught.value)


def test_diffusive_boundary_data_that_reads_a_coordinate_is_refused():
    """The trace value + slope.x of a diffusive boundary would read the logical position x."""
    walls = (DiffusiveBoundary(0, "lower", "value", 1.0, slope=(0.5,)),
             DiffusiveBoundary(0, "upper", "conormal", 0.0))
    case, layout, _, _ = build_graded_diffusion(walls=walls)
    with pytest.raises(StaticGeometryConsumerError) as caught:
        pops.resolve(pops.validate(case), layout=layout)
    assert caught.value.code == "static_geometry_boundary_data_unavailable"
