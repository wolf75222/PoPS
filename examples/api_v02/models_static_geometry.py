"""The two cases of C1, static geometry arrays of a structured configuration (N17, CF15).

Spec 0.3 D0b: the measures, face vectors and normals of a mapped configuration are geometry arrays
generated « as for a moving configuration (D18, D21) at zero velocity ». Both cases are written
below the surface, with the descriptors of the geometry-evolution contract (design note N17,
section 4, DN14): the reference box of each layout is logical, and the physical domain is the image
of a coordinate map that reads no clock, fixed in time.

* ``build_annulus_free_stream``, a vector balance in Cartesian components on a charted support
  (the structure of the diocotron on a disk): isothermal Euler on a 4 x 8 annulus 1 <= r <= 2,
  theta periodic, the map (r cos theta, r sin theta), Rusanov faces, a constant state. The rule of
  its measures is stated (D0b), GaussLegendre(2) on each logical axis;
* ``build_graded_diffusion``, graded nodes (the structure of the wall-graded sheath): diffusion
  with no-flux ends on 32 cells of a non-periodic 1-D layout whose nodes are
  x = tanh(3 s) / tanh(3), written with exp. A map of rank one is read as node positions with exact
  node differences, so it states no rule.

Each builder returns the Case, its layout, the initial values that ``pops.bind`` receives and the
initial array; the oracles live in the tests that run them.
"""
from __future__ import annotations

import math

import numpy as np
import pops
from pops import math as pm
from pops.analytic import coordinates, cos, exp, sin
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Outflow
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D, Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.quadrature import GaussLegendre
from pops.mesh import CartesianGrid, GeometryEvolution, MovingControlVolumes, PeriodicAxes
from pops.numerics import Diffusion, DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.physics.diffusion import DiffusiveBoundary
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt

# The annulus and its constant state.
ANNULUS_CELLS = (4, 8)
SOUND_SPEED = 1.0
FREE_STREAM = (1.3, 0.7, -0.4)            # rho, mx, my: a constant Cartesian state
ANNULUS_DT = 0.01
ANNULUS_STEPS = 4

# The graded segment and its diffusion.
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
    """The polar map of the annulus: it reads the logical coordinates only, no clock."""
    r, theta = coordinates(frame)
    return GeometryEvolution((r*cos(theta), r*sin(theta)))


def build_annulus_free_stream(*, geometry=GaussLegendre(2), faces=riemann.Rusanov(),
                              traces=reconstruction.FirstOrder(), step=None, walls=Outflow,
                              state=None):
    """Isothermal Euler in Cartesian components on the 4 x 8 annulus, a constant state.

    The keywords name the variants that the refusal tests of the static reading use; ``state``
    replaces the constant state by cell averages of shape (3, 8, 4), (rho, mx, my) by (theta, r).
    ``faces`` is a face descriptor, or a function of the flux and the state that returns one, as a
    face authored in source (``riemann.User``) is written."""
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
    if callable(faces):
        faces = faces(flux, U)
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
    if state is None:
        initial = np.empty((3, ANNULUS_CELLS[1], ANNULUS_CELLS[0]))
        for component, value in enumerate(FREE_STREAM):
            initial[component] = value
    else:
        initial = np.array(state, dtype=float)
        if initial.shape != (3, ANNULUS_CELLS[1], ANNULUS_CELLS[0]):
            raise ValueError("the annulus state holds (rho, mx, my) on 8 x 4 cells")
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


def build_graded_diffusion(*, geometry=None, walls=None):
    """Diffusion with no-flux ends on 32 graded cells.

    The keywords name the variants that the refusal tests of the static reading use."""
    frame = CartesianDomain("n17_graded_logical", (0.0,), (1.0,)).frame(Cartesian1D())
    model = pops.Model("N17_graded_diffusion", frame=frame)
    U = model.state("heat", components=("u",))
    if walls is None:
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
