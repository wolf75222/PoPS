#!/usr/bin/env python3
"""Advection scalaire 2D avec AMR, OpenMP et SSPRK2 ecrit dans pops.Program.

Le fichier est autonome et lineaire. Le programme explicite produit le meme graphe temporel que
le preset ; le calcul adaptatif est compile puis execute en C++/Kokkos OpenMP.
"""

# ruff: noqa: E402

from fractions import Fraction
import time

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
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.runtime_environment import runtime_environment_report
from pops.spaces import CellState
from pops.time import AdaptiveCFL, StagePoint, TimePoint, every


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


# 3. Methode spatiale identique a la variante preset.
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


# 5. SSPRK2 ecrit avec les operations generiques de Program.
program = pops.Program("SSPRK2")
q = program.state(tracer_U)

stage_0 = StagePoint(
    "ssprk2_stage_0",
    {"main": TimePoint(program.clock, 0)},
)
k0 = program.value(
    "ssprk2_k_0",
    advection_rate(q.n),
    at=stage_0,
)

stage_1 = StagePoint(
    "ssprk2_stage_1",
    {"main": TimePoint(program.clock, 1)},
)
q_stage = program.value(
    "ssprk2_U1",
    q.n + program.dt * k0,
    at=stage_1,
)
k1 = program.value(
    "ssprk2_k_1",
    advection_rate(q_stage),
    at=stage_1,
)

half = Fraction(1, 2)
q_next = program.value(
    "ssprk2_step",
    q.n + program.dt * half * k0 + program.dt * half * k1,
    at=q.next.point,
)

program.commit(q.next, q_next)
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


# 7. Meme hierarchie, meme tagging et meme subcycling que la variante preset.
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

communicator = artifact.platform_manifest.communicator.require(
    "OpenMP AMR scalar-advection tutorial communicator"
)
backend = str(runtime_environment_report()["kokkos_backend"])
simulation = pops.bind(artifact)

start = time.perf_counter()
report = pops.run(simulation, t_end=T_END, max_steps=MAX_STEPS)
elapsed_seconds = time.perf_counter() - start


# 9. Inspection legere de la hierarchie native.
patches = simulation.amr.patch_table()
regrid = simulation.amr.explain_regrid()

print("PoPS OpenMP AMR scalar-advection tutorial finished")
print("  program          : explicit pops.Program SSPRK2")
print("  Kokkos backend   : %s" % backend)
print("  requested threads: 7 via pops.set_threads")
print("  communicator     : %s" % communicator)
print("  accepted steps   : %d" % report.accepted_steps)
print("  final time       : %.6f" % simulation.time())
print("  AMR levels       : %d" % simulation.n_levels())
print("  fine patches     : %d" % patches.n_patches)
print("  completed regrids: %d" % regrid.regrid_count)
print("  topology epoch   : %d" % regrid.topology_epoch)
print("  elapsed          : %.6f s" % elapsed_seconds)
