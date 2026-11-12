"""Source-only tests of the static reading of the geometry evolution contract (N17, C1).

A coordinate map that reads no clock is read at resolution as static geometry arrays: node
positions when component a reads the logical coordinate a alone, a chart otherwise, whose
measures need a stated rule (D0b). What the arrays do not serve is refused before any C++, with
its exact reason; the clock-reading maps keep the refusals of the moving route
(test_moving_geometry_contract.py).
"""
import functools
import math

import pops
import pytest
from pops.analytic import constant, coordinates, cos, sin
from pops.boundary.transport import Inflow, NoFlux, SlipWall
from pops.codegen._native_spatial_layout import NativeSpatialLayoutError, native_spatial_layouts
from pops.codegen._static_geometry_consumers import StaticGeometryConsumerError
from pops.domain import CartesianDomain
from pops.frames import Cartesian2D, Cartesian3D
from pops.layouts import Uniform
from pops.lib.quadrature import GaussLegendre
from pops.mesh import CartesianGrid, GeometryEvolution, MovingControlVolumes, normalize_layout_plan
from pops.mesh._static_geometry import is_static_coordinate_map
from pops.numerics import reconstruction, riemann
from pops.physics.diffusion import DiffusiveBoundary
from pops.time import AdaptiveCFL

from examples.api_v02.models_static_geometry import (
    build_annulus_free_stream, build_graded_diffusion)


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


def test_rank_three_waits_for_the_graded_box():
    frame = CartesianDomain("static-rank-three", (0.0,)*3, (1.0,)*3).frame(Cartesian3D())
    x, y, z = coordinates(frame)
    layout = MovingControlVolumes(Uniform(CartesianGrid(frame=frame, cells=(2, 2, 2))),
                                  evolution=GeometryEvolution((x*x + x, y, z)))
    assert _refusal(layout) == "static_geometry_rank_unavailable"


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


def test_diffusive_boundary_data_that_reads_a_coordinate_is_refused():
    """The trace value + slope.x of a diffusive boundary would read the logical position x."""
    walls = (DiffusiveBoundary(0, "lower", "value", 1.0, slope=(0.5,)),
             DiffusiveBoundary(0, "upper", "conormal", 0.0))
    case, layout, _, _ = build_graded_diffusion(walls=walls)
    with pytest.raises(StaticGeometryConsumerError) as caught:
        pops.resolve(pops.validate(case), layout=layout)
    assert caught.value.code == "static_geometry_boundary_data_unavailable"
