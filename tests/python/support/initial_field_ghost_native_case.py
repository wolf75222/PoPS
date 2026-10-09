"""Separate scientific Field/Ghost board; never changes the growth5/2 fixture.

Public Ghost expression mghost=phi+1+t distinguishes it from Outflow and stale Field.
The zero flux PDE and original growth fixtures are unchanged.
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
from pops.solvers.options import CompositeFAC
from tests.python.support.initial_field_ghost_native_oracle import ORIGINAL_F_BOUND
from pops.time import FailRun,FixedDt,every
from pops.fields.boundary_values import logical_time

DT=1/64
ALPHA=1/8
# Static FAC measures Original R(0) in global Linf. This board covers initial
# and one FE step: max |m/alpha|=2*(1+DT)/alpha. Reserve 7/8 of the independent
# residual guard for postsolve operator/reflux/roundoff; do not loosen that guard.
FIELD_FORCING_BOUND=2*(1+DT)/ALPHA
FIELD_RESIDUAL_BUDGET=ORIGINAL_F_BOUND/8
FIELD_RELATIVE_TOL=FIELD_RESIDUAL_BUDGET/FIELD_FORCING_BOUND

def build(*,ghost=True,boundary_composer=None):
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
        solver=GeometricMG(fac=CompositeFAC(rel_tol=FIELD_RELATIVE_TOL)),hierarchy_policy=CompositeHierarchySolve()))
    numerics=DiscretizationPlan()
    numerics.rates.add(rate,FiniteVolume(flux=flux,variables=variables.Conservative(state),reconstruction=reconstruction.FirstOrder(),riemann=riemann.Rusanov()))
    conditions={boundary:Outflow(state=block[state]) for boundary in (frame.boundaries.x_min,frame.boundaries.x_max,frame.boundaries.y_min,frame.boundaries.y_max)}
    if ghost:conditions[frame.boundaries.x_min]=Inflow(state=block[state],value=(interior_trace(block[state],'c'),ValueExpr(block[phi])+1+logical_time()))
    boundaries=TransportBoundarySet(conditions)
    numerics.boundaries.add(boundaries if boundary_composer is None else boundary_composer(boundaries))
    case.numerics(numerics,block=block)
    program=ForwardEuler(block[state],rate=rate,fields=field,solve_action=FailRun())
    program.step_strategy(FixedDt(DT));case.program(program)
    case.initials.add(InitialCondition(state=block[state],value=Analytic(frame=frame,components=(cos(2*math.pi*x(frame)),2+0*x(frame))),projection=ConservativeCellAverage()))
    transfer=AMRTransfer();transfer.state(block[state],StateTransfer());transfer.field(field,EllipticRecompute())
    layout=AMR(grid=CartesianGrid(frame=frame,cells=(8,8)),hierarchy=AMRHierarchy(max_levels=2,ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(ValueExpr(block[state])['c']>model.value(threshold)),Buffer(cells=0)),hysteresis=Hysteresis(0,EqualityPolicy.HOLD),conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(1000,clock=program.clock)),transfer=transfer,
        execution=AMRExecution.synchronous(accepted_halo=AcceptedHaloPreparation(cells=(1,1))))
    return case,layout
