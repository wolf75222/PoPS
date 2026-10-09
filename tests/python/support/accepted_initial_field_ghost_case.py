"""Separate scientific Field/Ghost board; never changes the growth5/2 fixture.

The typed Field resolves today. The dynamic Inflow edge still needs a genuine
compiled boundary component/emitter; ghost=True intentionally exposes that gap.
"""
import math
import pops
from pops.amr import AcceptedHaloPreparation,AMRExecution,AMRHierarchy,AMRRegrid,AMRTagging,AMRTransfer,Buffer,Tag,Hysteresis,EqualityPolicy,ConflictPolicy
from pops.analytic import cos,x
from pops.boundary import TransportBoundarySet
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

DT=1/64
ALPHA=1/8

def build(*,ghost=False):
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
    if ghost:conditions[frame.boundaries.x_min]=Inflow(state=block[state],value=(ValueExpr(block[state])['c'],ValueExpr(block[phi])))
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
    return case,layout
