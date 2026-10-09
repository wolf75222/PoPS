#!/usr/bin/env python3
"""Public Field -> inferred Inflow -> initial/accepted Ghost, Serial Dim2.

Prepare-only until this exact script is executed with an authenticated installed SDK.
"""
import math
import pops
from pops.amr import AcceptedHaloPreparation,AMRExecution,AMRHierarchy,AMRRegrid,AMRTagging,AMRTransfer,Buffer,Tag,Hysteresis,EqualityPolicy,ConflictPolicy
from pops.analytic import cos,x
from pops.boundary import TransportBoundarySet,interior_trace
from pops.boundary.transport import Inflow,Outflow
from pops.domain import Rectangle
from pops.fields import CellCenteredSecondOrder,CompositeHierarchySolve,FieldDiscretization,FieldOutput
from pops.fields.bcs import AllPhysicalBoundaries,BoundaryCondition,Neumann
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import EllipticRecompute,StateTransfer
from pops.lib.initial import Analytic
from pops.lib.time import ForwardEuler
from pops.math import ValueExpr,ddt,div,laplacian,unknown
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan,reconstruction,riemann,variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.solvers.elliptic import GeometricMG
from pops.time import FailRun,FixedDt,every
from pops.fields.boundary_values import logical_time

DT=1/64
ALPHA=1/8

# Physical equations first: dc/dt=0, dm/dt=m, flux=0.
# Screened Field: phi - ALPHA*laplacian(phi) = m, homogeneous Neumann.
frame=Rectangle('initial field ghost',(0.,0.),(1.,1.)).frame(Cartesian2D())
model=pops.Model('growth with screened field',frame=frame)
threshold=model.param(RuntimeParam('tag_threshold',default=.7))
state=model.state('U',components=('c','m'),sampling='cell_average')
c,m=state
flux=model.flux('zero',frame=frame,state=state,components={a:(0*c,0*m) for a in frame.axes},waves={a:(0*c,0*m) for a in frame.axes})
source=model.source('growth',on=state,value=(0*c,m))
rate=model.rate('evolution',equation=ddt(state)==-div(flux)+source)
phi=model.field('phi')
operator=model.field_operator('screened',unknown=phi,equation=-laplacian(unknown(phi))+(1/ALPHA)*unknown(phi)==m/ALPHA,outputs=(FieldOutput('phi',phi),))
case=pops.Case('accepted initial Field/Ghost scientific board')
block=case.block('marker',model)
field=case.field(operator,FieldDiscretization(method=CellCenteredSecondOrder(),
    boundaries=(BoundaryCondition(AllPhysicalBoundaries(),Neumann(0.)),),
    solver=GeometricMG(),hierarchy_policy=CompositeHierarchySolve()))
numerics=DiscretizationPlan()
numerics.rates.add(rate,FiniteVolume(flux=flux,variables=variables.Conservative(state),reconstruction=reconstruction.FirstOrder(),riemann=riemann.Rusanov()))
conditions={boundary:Outflow(state=block[state]) for boundary in (frame.boundaries.x_min,frame.boundaries.x_max,frame.boundaries.y_min,frame.boundaries.y_max)}
conditions[frame.boundaries.x_min]=Inflow(state=block[state],value=(interior_trace(block[state],'c'),ValueExpr(block[phi])+1+logical_time()))
numerics.boundaries.add(TransportBoundarySet(conditions))
case.numerics(numerics,block=block)
program=ForwardEuler(block[state],rate=rate,fields=field,solve_action=FailRun())
program.step_strategy(FixedDt(DT));case.program(program)
case.initials.add(InitialCondition(state=block[state],value=Analytic(frame=frame,components=(cos(2*math.pi*x(frame)),2+0*x(frame))),projection=ConservativeCellAverage()))
transfer=AMRTransfer();transfer.state(block[state],StateTransfer());transfer.field(field,EllipticRecompute())
layout=AMR(grid=CartesianGrid(frame=frame,cells=(8,8)),hierarchy=AMRHierarchy(max_levels=2,ratios=(2,)),
    tagging=AMRTagging(rules=(Tag(ValueExpr(block[state])['c']>model.value(threshold)),Buffer(cells=0)),hysteresis=Hysteresis(0,EqualityPolicy.HOLD),conflict_policy=ConflictPolicy.REFINE_WINS),
    regrid=AMRRegrid(schedule=every(1000,clock=program.clock)),transfer=transfer,
    execution=AMRExecution.synchronous(accepted_halo=AcceptedHaloPreparation(cells=(1,1))))

# Execution begins: no model-specific component assembly is needed.
from pathlib import Path
import hashlib
import json
import os
import sys
import numpy as np
from pops._native_selector import select_native_dimension

native = select_native_dimension(2)  # verifies the installed native variant manifest
prefix = Path(sys.prefix).resolve()
for package_file in (pops.__file__, native.__file__):
    if not Path(package_file).resolve().is_relative_to(prefix):
        raise RuntimeError("Use the installed SDK with env -u PYTHONPATH; Source import is refused")
if native.__abi_version__ != 8 or native.mpi_world().size != 1:
    raise RuntimeError("This example requires ABI8, Dim2 and a one-rank launch")
output = Path(os.environ.get("POPS_EXAMPLE_OUTPUT", "initial-field-ghost-output")).resolve()
output.mkdir(parents=True, exist_ok=False)  # retain earlier evidence; never overwrite it

validated = pops.validate(case)
resolved = pops.resolve(validated, layout=layout)
component, = resolved.component_inputs
assert component.component_manifest.signature["inferred_boundary_expression"]["schema"] == "inferred-boundary-expression-component@1"
artifact = pops.compile(resolved)
resources = ({"execution_context": pops.ExecutionContext.mpi_world(artifact)}
             if artifact.platform_manifest.communicator.require("example.communicator") == "MPI_COMM_WORLD" else {})
simulation = pops.bind(artifact, resources=resources)
initial_checkpoint = Path(simulation.checkpoint(output / "initial"))
report = pops.run(simulation, t_end=DT, max_steps=1, console=False)
assert report.accepted_steps == 1 and report.rejected_steps == 0
accepted_checkpoint = Path(simulation.checkpoint(output / "accepted"))
resumed = pops.bind(artifact, resources=resources)
resumed.restart(accepted_checkpoint)
reloaded_checkpoint = Path(resumed.checkpoint(output / "reloaded"))

# Restart images, topology and accepted clock must agree exactly.
for level in range(2):
    before = np.asarray(simulation.block_level_state_global("marker", level))
    after = np.asarray(resumed.block_level_state_global("marker", level))
    assert before.dtype == after.dtype and before.shape == after.shape
    assert before.tobytes() == after.tobytes()
assert tuple(simulation.patch_boxes()) == tuple(resumed.patch_boxes())
assert simulation.time() == resumed.time() == DT
proof = {"scope": "example execution/restart; not full-grown scientific reception",
         "python_path": pops.__file__, "native_path": native.__file__,
         "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
         "artifact_identity": artifact.artifact_identity.token,
         "component_manifest_identity": component.component_manifest.manifest_digest.token,
         "checkpoints": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for path in (initial_checkpoint, accepted_checkpoint, reloaded_checkpoint)]}
(output / "example-receipt.json").write_text(json.dumps(proof, indent=2) + "\n")
print(output / "example-receipt.json")
