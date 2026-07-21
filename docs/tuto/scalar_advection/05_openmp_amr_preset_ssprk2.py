#!/usr/bin/env python3
"""Advection scalaire 2D avec AMR, OpenMP et le preset SSPRK2.

Le script reste top-level et lineaire. Python decrit la physique, la discretisation et les
regles AMR ; les cellules, les regrids et les pas de temps sont executes en C++/Kokkos OpenMP.
"""

# ruff: noqa: E402

import pops

pops.set_threads(7)

from pops.amr import (
    AMRClockRelation,
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    AMRTransfer,
    Buffer,
    Coarsen,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    Tag,
)
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Inflow, Outflow
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import Gaussian
from pops.lib.time import SSPRK2
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import AdaptiveCFL, every


NX = 32
NY = 32
AX = 1.0
AY = 0.25
FAR_FIELD = 0.05
CFL = 0.45
MAX_DT = 1.0e-2
T_END = 0.10
MAX_STEPS = 10_000


# 1. Domaine, repere et grille grossiere de la hierarchie adaptative.
domain = Rectangle(
    "unit_square",
    lower=(0.0, 0.0),
    upper=(1.0, 1.0),
).tag("fluid")

frame = domain.frame(Cartesian2D())
x_axis, y_axis = frame.axes
grid = CartesianGrid(frame=frame, cells=(NX, NY))


# 2. Physique : d_t U + div(a U) = 0.
model = pops.Model("scalar_advection_amr", frame=frame)

U = model.state(
    "U",
    components=("u",),
    representation=Conservative(),
    space=CellState(frame=frame),
)
(u,) = U

velocity = model.vector(
    "a",
    frame=frame,
    components={x_axis: AX, y_axis: AY},
)

physical_flux = model.flux(
    "advection_flux",
    frame=frame,
    state=U,
    components={x_axis: (AX * u,), y_axis: (AY * u,)},
    waves={x_axis: (AX,), y_axis: (AY,)},
)

advection_rate = model.rate(
    "advection_rate",
    equation=ddt(U) == -div(physical_flux),
)


# 3. Volumes finis : l'ordre et les halos sont derives des briques choisies.
finite_volume = FiniteVolume(
    flux=physical_flux,
    variables=variables.Conservative(U),
    reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
    riemann=riemann.ScalarUpwind(velocity=velocity),
)

numerics = DiscretizationPlan()
numerics.rates.add(advection_rate, finite_volume)


# 4. Bloc qualifie et conditions aux limites de transport.
case = pops.Case("tutorial_scalar_advection_amr")
tracer = case.block("tracer", model=model)
tracer_U = tracer[U]

boundaries = frame.boundaries
transport_boundaries = TransportBoundarySet({
    boundaries.x_min: Inflow(state=tracer_U, value=FAR_FIELD),
    boundaries.x_max: Outflow(state=tracer_U),
    boundaries.y_min: Inflow(state=tracer_U, value=FAR_FIELD),
    boundaries.y_max: Outflow(state=tracer_U),
})

numerics.boundaries.add(transport_boundaries)
case.numerics(numerics, block=tracer)


# 5. Programme temporel preimplemente.
program = SSPRK2(tracer_U, rate=advection_rate)
program.step_strategy(AdaptiveCFL(cfl=CFL, max_dt=MAX_DT))
case.program(program)


# 6. Condition initiale analytique, projetee conservativement sur chaque niveau AMR.
case.initials.add(InitialCondition(
    state=tracer_U,
    value=Gaussian(
        frame=frame,
        center={x_axis: 0.30, y_axis: 0.35},
        background=FAR_FIELD,
        amplitude=0.95,
        inverse_width=120.0,
    ),
    projection=ConservativeCellAverage(),
))


# 7. Deux seuils font suivre le paquet par l'AMR sans conserver sa trainee raffinee.
refine_threshold = case.param(RuntimeParam("refine_u", default=0.30))
coarsen_threshold = case.param(RuntimeParam("coarsen_u", default=0.20))

tagging = AMRTagging(
    rules=(
        Tag(ValueExpr(tracer_U) > case.value(refine_threshold)),
        Coarsen(ValueExpr(tracer_U) < case.value(coarsen_threshold)),
        Buffer(cells=2),
    ),
    hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
    conflict_policy=ConflictPolicy.REFINE_WINS,
)

transfer = AMRTransfer()
transfer.state(tracer_U, StateTransfer())

layout = AMR(
    grid=grid,
    hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
    tagging=tagging,
    regrid=AMRRegrid(schedule=every(2, clock=program.clock)),
    transfer=transfer,
    execution=AMRExecution.subcycled((
        AMRClockRelation(0, 1, 2),
    )),
)


# 8. Cycle public final.
validated = pops.validate(case)
resolved = pops.resolve(validated, layout=layout)
artifact = pops.compile(resolved)
simulation = pops.bind(artifact)

pops.run(simulation, t_end=T_END, max_steps=MAX_STEPS)


# 9. Les rapports AMR restent legers : aucune copie de champ par cellule n'est faite en Python.
patches = simulation.amr.patch_table()
regrid = simulation.amr.explain_regrid()

print("PoPS OpenMP AMR scalar-advection tutorial finished")
print("  program          : pops.lib.time.SSPRK2")
print("  AMR levels       : %d" % simulation.n_levels())
print("  fine patches     : %d" % patches.n_patches)
print("  completed regrids: %d" % regrid.regrid_count)
print("  topology epoch   : %d" % regrid.topology_epoch)
