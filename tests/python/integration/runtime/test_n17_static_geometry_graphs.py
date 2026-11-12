"""The two failing graphs of C1, static geometry arrays of a structured configuration (N17, CF15).

Spec 0.3 D0b: the measures, face vectors and normals of a mapped configuration are geometry arrays
generated « as for a moving configuration (D18, D21) at zero velocity ». At the base the geometry
evolution contract (``pops.mesh.moving``) has one provider, the periodic Uniform1D carrier of a
moving interval, so a clock-free map of rank one or two is refused at resolution, before any C++,
with ``native_geometry_evolution_unavailable``. The two graphs are written below the surface with
the descriptors that exist there (design note N17, section 4, DN14), so that each fails for C1's
reason and for no missing word of C2:

* graph (1), a vector balance in Cartesian components on a charted support (the structure of the
  diocotron on a disk, probe 2): isothermal Euler on a 4 x 8 annulus 1 <= r <= 2, theta periodic,
  the map (r cos theta, r sin theta), which reads no clock, Rusanov faces, a constant state. The
  rule of its measures is stated (D0b), GaussLegendre(2) on each logical axis;
* graph (2), graded nodes (the structure of the wall-graded sheath, probe 3): diffusion with
  no-flux ends on 32 cells of a non-periodic 1-D layout whose nodes are x = tanh(3 s) / tanh(3),
  written with exp, which reads no clock either. A map of rank one is read as node positions with
  exact node differences, so it states no rule.

The reference box of each layout is logical and the physical domain is the image of the map, fixed
in time (note, section 4, « How C1 reads the geometry-evolution contract »).
"""
from __future__ import annotations

import math

import numpy as np
import pops
import pytest
from pops import math as pm
from pops.analytic import coordinates, cos, exp, sin
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Outflow
from pops.codegen._layout_resolution import resolve_native_spatial_layouts
from pops.codegen._native_spatial_layout import native_spatial_layouts
from pops.codegen._static_geometry_consumers import static_geometry_layouts
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D, Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.quadrature import GaussLegendre
from pops.mesh import (CartesianGrid, GeometryEvolution, MovingControlVolumes, PeriodicAxes,
                       normalize_layout_plan)
from pops.numerics import Diffusion, DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.physics.diffusion import DiffusiveBoundary
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt

# Graph (1): the annulus and its constant state.
ANNULUS_CELLS = (4, 8)
SOUND_SPEED = 1.0
FREE_STREAM = (1.3, 0.7, -0.4)            # rho, mx, my: a constant Cartesian state
ANNULUS_DT = 0.01
ANNULUS_STEPS = 4

# Graph (2): the graded segment and its diffusion.
GRADED_CELLS = 32
GRADING = 3.0
CONDUCTIVITY = 0.01
GRADED_DT = 1.0e-5
GRADED_STEPS = 8


def _ssprk2(case, block, state, rate, dt, name, step=None):
    """Two authored stages, the SSPRK2 of the corpus builders."""
    program = pops.Program(name)
    temporal = program.state(block[state])
    u0 = temporal.n
    k0 = program.value(name + "_k0", rate(u0), at=program.stage(name + "_stage0", c=0))
    point1 = program.stage(name + "_stage1", c=1)
    u1 = program.value(name + "_predictor", u0 + program.dt*k0, at=point1)
    k1 = program.value(name + "_k1", rate(u1), at=point1)
    program.commit(temporal.next, program.value(name + "_accepted", .5*u0 + .5*(u1 + program.dt*k1),
                                                at=temporal.next.point))
    program.step_strategy(FixedDt(dt) if step is None else step)
    case.program(program)


def _initial(case, subject, values):
    case.initials.add(InitialCondition(state=subject, value=BindArray(),
                                       projection=ConservativeCellAverage()))
    return {subject: np.ascontiguousarray(values, dtype=float)}


def annulus_map(frame):
    """The polar map of graph (1): it reads the logical coordinates only, no clock."""
    r, theta = coordinates(frame)
    return GeometryEvolution((r*cos(theta), r*sin(theta)))


def build_graph_1(*, geometry=GaussLegendre(2), faces=riemann.Rusanov(),
                  traces=reconstruction.FirstOrder(), step=None, walls=Outflow):
    """Graph (1): isothermal Euler in Cartesian components on the 4 x 8 annulus.

    The keywords name the variants that the refusal tests of the static reading use."""
    frame = CartesianDomain("n17_annulus_logical", (1.0, 0.0),
                            (2.0, 2*math.pi)).frame(Cartesian2D())
    x, y = frame.axes          # below the surface the axes of the frame name the image components
    model = pops.Model("N17_isothermal_euler", frame=frame)
    U = model.state("conserved", components=("rho", "mx", "my"))
    rho, mx, my = U
    vx, vy = mx/rho, my/rho
    c2 = SOUND_SPEED*SOUND_SPEED
    flux = model.flux("isothermal_euler_flux", state=U, frame=frame,
                      components={x: (mx, mx*vx + c2*rho, my*vx),
                                  y: (my, mx*vy, my*vy + c2*rho)},
                      waves={x: (vx - SOUND_SPEED, vx, vx + SOUND_SPEED),
                             y: (vy - SOUND_SPEED, vy, vy + SOUND_SPEED)})
    rate = model.rate("isothermal_euler", equation=pm.ddt(U) == -pm.div(flux))
    case = pops.Case("N17_graph_1_annulus_free_stream")
    block = case.block("gas", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(U),
                                      reconstruction=traces, riemann=faces))
    boundaries = frame.boundaries
    periodic = PeriodicAxes((frame.axes[1],))
    plan.boundaries.add(TransportBoundarySet({
        boundaries.x_min: walls(state=block[U]),
        boundaries.x_max: walls(state=block[U]),
    }, periodic=periodic))
    case.numerics(plan, block=block)
    _ssprk2(case, block, U, rate, ANNULUS_DT, "N17_graph_1_SSPRK2", step)
    grid = CartesianGrid(frame=frame, cells=ANNULUS_CELLS, periodic=periodic)
    layout = MovingControlVolumes(Uniform(grid), evolution=annulus_map(frame), geometry=geometry)
    initial = np.empty((3, ANNULUS_CELLS[1], ANNULUS_CELLS[0]))
    for component, value in enumerate(FREE_STREAM):
        initial[component] = value
    return case, layout, _initial(case, block[U], initial), initial


def graded_nodes(cells=GRADED_CELLS, beta=GRADING):
    """The nodes tanh(beta s) / tanh(beta) at s = k / cells, in binary64."""
    s = np.arange(cells + 1)/cells
    return np.tanh(beta*s)/np.tanh(beta)


def graded_map(frame, beta=GRADING):
    """x = tanh(beta s) / tanh(beta), written with exp (pops.analytic has no tanh)."""
    s, = coordinates(frame)
    growth = exp(2*beta*s)
    return GeometryEvolution(((growth - 1)/((growth + 1)*math.tanh(beta)),))


def build_graph_2(*, geometry=None):
    """Graph (2): diffusion with no-flux ends on 32 graded cells."""
    frame = CartesianDomain("n17_graded_logical", (0.0,), (1.0,)).frame(Cartesian1D())
    model = pops.Model("N17_graded_diffusion", frame=frame)
    U = model.state("heat", components=("u",))
    walls = (DiffusiveBoundary(0, "lower", "conormal", 0.0),
             DiffusiveBoundary(0, "upper", "conormal", 0.0))
    flux = model.diffusive_flux("conduction", state=U,
                                value=pm.CoeffGradient(U[0], ((CONDUCTIVITY,),)),
                                boundaries=walls)
    rate = model.rate("heat_balance", equation=pm.ddt(U) == pm.div(flux))
    case = pops.Case("N17_graph_2_graded_diffusion")
    block = case.block("rod", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, Diffusion(flux=flux))
    case.numerics(plan, block=block)
    _ssprk2(case, block, U, rate, GRADED_DT, "N17_graph_2_SSPRK2")
    layout = MovingControlVolumes(Uniform(CartesianGrid(frame=frame, cells=(GRADED_CELLS,))),
                                  evolution=graded_map(frame), geometry=geometry)
    nodes = graded_nodes()
    centres = (nodes[:-1] + nodes[1:])/2
    initial = (1.0 + 0.5*np.cos(math.pi*centres))[None]
    return case, layout, _initial(case, block[U], initial), initial


GRAPHS = {"graph (1) annulus free stream": build_graph_1,
          "graph (2) graded diffusion": build_graph_2}


@pytest.mark.parametrize("label,kind", [("graph (1) annulus free stream", "chart"),
                                        ("graph (2) graded diffusion", "nodes")])
def test_the_graph_resolves_to_static_geometry_arrays(label, kind):
    """CF15, first half of the flip: the clock-free map of each graph is now read as static
    geometry arrays at resolution (a chart with its stated rule, or node positions), instead of
    the refusal native_geometry_evolution_unavailable of the base."""
    case, layout, _, _ = GRAPHS[label]()
    plan = normalize_layout_plan(layout, owner=case.owner_path.canonical())
    evolution = plan.layouts[0].requirements["geometry_evolution"]
    assert evolution["domain_motion"] == "fixed_physical_domain"
    assert not any(value.get("op") == "time" for value in _nodes(evolution["coordinate_map"]))
    assert native_spatial_layouts(plan)
    assert resolve_native_spatial_layouts(plan)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    (reading,) = static_geometry_layouts(resolved.layout_plan).values()
    assert reading.kind == kind and reading.rank == len(layout.evolution.coordinate_map)


def _nodes(data):
    """Every node of the serialized analytic expressions."""
    pending = list(data)
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            yield node
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
