#!/usr/bin/env python3
"""Compiler une fois puis binder deux vitesses d'advection differentes.

Les RuntimeParam appartiennent au modele. Le meme artefact C++ est installe deux fois avec des
valeurs qualifiees differentes ; aucun flux ni operateur n'est recompile entre les deux runs.
"""

# ruff: noqa: E402

import numpy as np

import pops

pops.set_threads(7)

from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Inflow, Outflow
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import SSPRK2
from pops.math import ddt, div
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.params import Positive, RuntimeParam
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import AdaptiveCFL


NX = 64
NY = 64
CFL = 0.45
MAX_DT = 1.0e-2
T_END = 0.10
MAX_STEPS = 10_000


# 1. Domaine et grille uniformes.
domain = Rectangle(
    "unit_square",
    lower=(0.0, 0.0),
    upper=(1.0, 1.0),
).tag("fluid")
frame = domain.frame(Cartesian2D())
x_axis, y_axis = frame.axes
grid = CartesianGrid(frame=frame, cells=(NX, NY))


# 2. Les vitesses sont des declarations typees, pas des noms de dictionnaire.
model = pops.Model("parameterized_scalar_advection", frame=frame)
U = model.state(
    "U",
    components=("u",),
    representation=Conservative(),
    space=CellState(frame=frame),
)
(u,) = U

a_x_declaration = model.param(
    RuntimeParam("a_x", default=1.0, domain=Positive())
)
a_y_declaration = model.param(
    RuntimeParam("a_y", default=0.25, domain=Positive())
)
a_x = model.value(a_x_declaration)
a_y = model.value(a_y_declaration)

velocity = model.vector(
    "a",
    frame=frame,
    components={x_axis: a_x, y_axis: a_y},
)
physical_flux = model.flux(
    "advection_flux",
    frame=frame,
    state=U,
    components={x_axis: (a_x * u,), y_axis: (a_y * u,)},
    waves={x_axis: (a_x,), y_axis: (a_y,)},
)
advection_rate = model.rate(
    "advection_rate",
    equation=ddt(U) == -div(physical_flux),
)


# 3. Methode spatiale et bloc qualifie.
finite_volume = FiniteVolume(
    flux=physical_flux,
    variables=variables.Conservative(U),
    reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
    riemann=riemann.ScalarUpwind(velocity=velocity),
)
numerics = DiscretizationPlan()
numerics.rates.add(advection_rate, finite_volume)

case = pops.Case("tutorial_runtime_parameters")
tracer = case.block("tracer", model=model)
tracer_U = tracer[U]

boundaries = frame.boundaries
numerics.boundaries.add(TransportBoundarySet({
    boundaries.x_min: Inflow(state=tracer_U, value=0.0),
    boundaries.x_max: Outflow(state=tracer_U),
    boundaries.y_min: Inflow(state=tracer_U, value=0.0),
    boundaries.y_max: Outflow(state=tracer_U),
}))
case.numerics(numerics, block=tracer)

program = SSPRK2(tracer_U, rate=advection_rate)
program.step_strategy(AdaptiveCFL(cfl=CFL, max_dt=MAX_DT))
case.program(program)


# 4. La meme condition initiale sera donnee aux deux installations.
x = (np.arange(NX, dtype=np.float64) + 0.5) / NX
y = (np.arange(NY, dtype=np.float64) + 0.5) / NY
xx, yy = np.meshgrid(x, y, indexing="xy")
initial_u = 0.05 + 0.95 * np.exp(
    -120.0 * ((xx - 0.30) ** 2 + (yy - 0.35) ** 2)
)
initial_state = np.ascontiguousarray(initial_u[np.newaxis, :, :])


# 5. Validation, resolution des handles puis une seule compilation.
validated = pops.validate(case)
a_x_param = validated.resolve(a_x_declaration)
a_y_param = validated.resolve(a_y_declaration)
resolved = pops.resolve(validated, layout=Uniform(grid))
artifact = pops.compile(resolved)


# 6. Premier bind : transport lent.
slow = pops.bind(
    artifact,
    params={a_x_param: 0.50, a_y_param: 0.10},
    initial_state={"tracer": initial_state.copy()},
)
slow_report = pops.run(slow, t_end=T_END, max_steps=MAX_STEPS)
slow_state = np.asarray(slow.state_global("tracer"), dtype=np.float64).copy()


# 7. Second bind frais du meme artefact : transport plus rapide.
fast = pops.bind(
    artifact,
    params={a_x_param: 1.00, a_y_param: 0.25},
    initial_state={"tracer": initial_state.copy()},
)
fast_report = pops.run(fast, t_end=T_END, max_steps=MAX_STEPS)
fast_state = np.asarray(fast.state_global("tracer"), dtype=np.float64).copy()

difference = float(np.max(np.abs(fast_state - slow_state)))

print("PoPS RuntimeParam tutorial finished")
print("  artifact identity: %s" % artifact.artifact_identity.hexdigest[:12])
print("  slow bind identity: %s" % slow.bind_identity.hexdigest[:12])
print("  fast bind identity: %s" % fast.bind_identity.hexdigest[:12])
print("  slow steps        : %d" % slow_report.accepted_steps)
print("  fast steps        : %d" % fast_report.accepted_steps)
print("  max state delta   : %.6e" % difference)
