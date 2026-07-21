#!/usr/bin/env python3
"""AMR OpenMP avec sorties scientifiques HDF5 et ParaView.

Cette variante ajoute uniquement les publications scientifiques. Le ConsumerGraph ecrit les deux
formats a la fin acceptee du run, puis leurs lecteurs publics rouvrent les artefacts produits.
"""

# ruff: noqa: E402

from pathlib import Path
import shutil

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
from pops.output import (
    ConsumerGraph,
    HDF5,
    ParaView,
    ParallelMode,
    ScientificOutput,
    read_hdf5,
    read_paraview,
)
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import AdaptiveCFL, every, on_end


NX = 32
NY = 32
AX = 1.0
AY = 0.25
FAR_FIELD = 0.05
CFL = 0.45
MAX_DT = 1.0e-2
T_END = 0.10
MAX_STEPS = 10_000

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "results" / "12_openmp_amr_outputs"

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

model = pops.Model("scalar_advection_amr_outputs", frame=frame)
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

case = pops.Case("tutorial_scalar_advection_amr_outputs")
tracer = case.block("tracer", model=model)
tracer_U = tracer[U]

boundaries = frame.boundaries
numerics.boundaries.add(TransportBoundarySet({
    boundaries.x_min: Inflow(state=tracer_U, value=FAR_FIELD),
    boundaries.x_max: Outflow(state=tracer_U),
    boundaries.y_min: Inflow(state=tracer_U, value=FAR_FIELD),
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
        background=FAR_FIELD,
        amplitude=0.95,
        inverse_width=120.0,
    ),
    projection=ConservativeCellAverage(),
))


# 4. Une seule autorite publie HDF5 et ParaView a la fin acceptee du run.
end_schedule = on_end(clock=program.clock)
case.consumers(ConsumerGraph.from_consumers((
    ScientificOutput(
        format=HDF5(mode=ParallelMode.SERIAL),
        schedule=end_schedule,
        fields=(tracer_U,),
        target="state/tracer",
    ),
    ScientificOutput(
        format=ParaView(mode=ParallelMode.SERIAL),
        schedule=end_schedule,
        fields=(tracer_U,),
        target="solution/tracer",
    ),
)))


# 5. AMR a deux niveaux avec transfert conservatif et subcycling 2:1.
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


# 6. Cycle public final et publication des deux formats.
validated = pops.validate(case)
resolved = pops.resolve(validated, layout=layout)
artifact = pops.compile(resolved)
simulation = pops.bind(artifact)
report = pops.run(
    simulation,
    t_end=T_END,
    max_steps=MAX_STEPS,
    output_dir=OUTPUT_ROOT,
)


# 7. Les lecteurs publics authentifient les fichiers juste produits.
(hdf5_path,) = tuple(sorted(OUTPUT_ROOT.rglob("*.h5")))
(paraview_path,) = tuple(sorted(OUTPUT_ROOT.rglob("*.vtu")))
hdf5_output = read_hdf5(hdf5_path)
paraview_output = read_paraview(paraview_path)

print("PoPS AMR scientific-output tutorial finished")
print("  accepted steps   : %d" % report.accepted_steps)
print("  final time       : %.6f" % simulation.time())
print("  AMR levels       : %d" % simulation.n_levels())
print("  HDF5 arrays      : %d" % len(hdf5_output.arrays))
print("  ParaView arrays  : %d" % len(paraview_output.arrays))
print("  HDF5 identity    : %s" % hdf5_output.output_identity.hexdigest[:12])
print("  ParaView identity: %s" % paraview_output.output_identity.hexdigest[:12])
print("  output root      : %s" % OUTPUT_ROOT)
