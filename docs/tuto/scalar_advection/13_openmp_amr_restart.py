#!/usr/bin/env python3
"""AMR OpenMP avec checkpoint, restart frais et continuation bit-identique.

Cette variante ajoute uniquement le restart. Une trajectoire est checkpointée a mi-parcours,
restauree dans un nouveau bind, puis comparee a la continuation sans interruption.
"""

# ruff: noqa: E402

from pathlib import Path
import shutil

import numpy as np
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
CFL = 0.45
MAX_DT = 1.0e-2
SPLIT_TIME = 0.10
FINAL_TIME = 0.20
MAX_STEPS = 10_000

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "results" / "13_openmp_amr_restart"

shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)


# 1. Domaine, etat conservatif et flux physique.
domain = Rectangle(
    "unit_square",
    lower=(0.0, 0.0),
    upper=(1.0, 1.0),
).tag("fluid")
frame = domain.frame(Cartesian2D())
x_axis, y_axis = frame.axes
grid = CartesianGrid(frame=frame, cells=(NX, NY))

model = pops.Model("scalar_advection_amr_restart", frame=frame)
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


# 2. Methode spatiale et bloc qualifie.
finite_volume = FiniteVolume(
    flux=physical_flux,
    variables=variables.Conservative(U),
    reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
    riemann=riemann.ScalarUpwind(velocity=velocity),
)
numerics = DiscretizationPlan()
numerics.rates.add(advection_rate, finite_volume)

case = pops.Case("tutorial_scalar_advection_amr_restart")
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


# 3. Programme et condition initiale analytique.
program = SSPRK2(tracer_U, rate=advection_rate)
program.step_strategy(AdaptiveCFL(cfl=CFL, max_dt=MAX_DT))
case.program(program)

case.initials.add(InitialCondition(
    state=tracer_U,
    value=Gaussian(
        frame=frame,
        center={x_axis: 0.30, y_axis: 0.35},
        background=0.05,
        amplitude=0.95,
        inverse_width=120.0,
    ),
    projection=ConservativeCellAverage(),
))


# 4. AMR a deux niveaux avec transfert conservatif et subcycling 2:1.
refine_threshold = case.param(RuntimeParam("refine_u", default=0.30))
tagging = AMRTagging(
    rules=(
        Tag(ValueExpr(tracer_U) > case.value(refine_threshold)),
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


# 5. Une trajectoire avance jusqu'au point de restart.
validated = pops.validate(case)
resolved = pops.resolve(validated, layout=layout)
artifact = pops.compile(resolved)
simulation = pops.bind(artifact)
first_report = pops.run(
    simulation,
    t_end=SPLIT_TIME,
    max_steps=MAX_STEPS,
)
checkpoint_path = Path(simulation.checkpoint(OUTPUT_ROOT / "restart"))


# 6. Un bind frais restaure l'etat, la topologie AMR et les horloges.
resumed = pops.bind(artifact)
resumed.restart(checkpoint_path)


# 7. Les trajectoires continue et restauree avancent jusqu'au meme temps final.
continuous_report = pops.run(
    simulation,
    t_end=FINAL_TIME,
    max_steps=MAX_STEPS,
)
restarted_report = pops.run(
    resumed,
    t_end=FINAL_TIME,
    max_steps=MAX_STEPS,
)


# 8. Etats et geometrie AMR doivent rester bit-identiques.
continuous_level_0 = np.asarray(
    simulation.block_level_state_global("tracer", 0), dtype=np.float64
)
restarted_level_0 = np.asarray(
    resumed.block_level_state_global("tracer", 0), dtype=np.float64
)
continuous_level_1 = np.asarray(
    simulation.block_level_state_global("tracer", 1), dtype=np.float64
)
restarted_level_1 = np.asarray(
    resumed.block_level_state_global("tracer", 1), dtype=np.float64
)
continuous_patches = np.asarray(simulation.patch_boxes(), dtype=np.int64)
restarted_patches = np.asarray(resumed.patch_boxes(), dtype=np.int64)

np.testing.assert_array_equal(continuous_level_0, restarted_level_0)
np.testing.assert_array_equal(continuous_level_1, restarted_level_1)
np.testing.assert_array_equal(continuous_patches, restarted_patches)
np.testing.assert_equal(simulation.time(), resumed.time())
np.testing.assert_equal(simulation.macro_step(), resumed.macro_step())

print("PoPS AMR checkpoint/restart tutorial finished")
print("  first run steps  : %d" % first_report.accepted_steps)
print("  continuous steps : %d" % continuous_report.accepted_steps)
print("  restarted steps  : %d" % restarted_report.accepted_steps)
print("  final time       : %.6f" % simulation.time())
print("  AMR levels       : %d" % simulation.n_levels())
print("  fine patches     : %d" % simulation.amr.patch_table().n_patches)
print("  restart identity : %s" % resumed.last_restart_identity.hexdigest[:12])
print("  checkpoint       : %s" % checkpoint_path)
