"""Two public independent transports and independent vectorized FV oracle."""
import numpy as np
import pops
from pops.amr import AMRExecution,AMRHierarchy,AMRRegrid,AMRTagging,AMRTransfer,Buffer,ConflictPolicy,EqualityPolicy,Hysteresis,Tag
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import BindArray
from pops.math import ValueExpr,ddt,div
from pops.mesh import CartesianGrid,PeriodicAxes
from pops.numerics import DiscretizationPlan,reconstruction,riemann,variables
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt,every
from pops.params import RuntimeParam
N=8
DT=1/256
PARTITION=('early_transport','late_transport')
NAMES=(('early_left','early_right'),('late_momentum','late_marker','late_density'))
VELOCITIES=((.5,-.25),(-.375,.625))


def build():
    frame=Rectangle('scoped_wave_box',(0.,0.),(1.,1.)).frame(Cartesian2D())
    model=pops.Model('independent_wave_transports',frame=frame)
    states=[model.species(label,state=names) for label,names in zip(PARTITION,NAMES)]
    case=pops.Case('scoped_wave_case');program=pops.Program('two_independent_FE_transports')
    subjects={};transfer=AMRTransfer();fluxes=[];rates=[]
    for k,(label,state,velocity) in enumerate(zip(PARTITION,states,VELOCITIES)):
        wave=2. if k==0 else 1.+.125*state[2]
        flux=model.flux('flux_'+label,state=state,frame=frame,
            components={axis:tuple(v*q for q in state) for axis,v in zip(frame.axes,velocity)},
            waves={axis:tuple(wave for _ in state) for axis in frame.axes})
        rate=model.rate('rate_'+label,equation=ddt(state)==-div(flux))
        fluxes.append(flux);rates.append(rate)
    for label,state,flux,rate in zip(PARTITION,states,fluxes,rates):
        block=case.block(label,model,states=(state,));subjects[label]=block[state]
        plan=DiscretizationPlan();plan.rates.add(rate,FiniteVolume(flux=flux,variables=variables.Conservative(state),reconstruction=reconstruction.FirstOrder(),riemann=riemann.Rusanov()))
        case.numerics(plan,block=block)
        q=program.state(subjects[label]);program.commit(q.next,program.value('FE_'+label,q.n+program.dt*rate(q.n),at=q.next.point))
        case.initials.add(InitialCondition(state=subjects[label],value=BindArray(),projection=ConservativeCellAverage()))
        transfer.state(subjects[label],StateTransfer())
    program.step_strategy(FixedDt(DT));case.program(program)
    threshold=case.param(RuntimeParam('unused_refinement_threshold',default=1000.))
    layout=AMR(grid=CartesianGrid(frame=frame,cells=(N,N),periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=1,ratios=()),
        tagging=AMRTagging(rules=(Tag(ValueExpr(subjects[PARTITION[1]])[NAMES[1][2]] >case.value(threshold)),Buffer(cells=0)),
            hysteresis=Hysteresis(0,EqualityPolicy.HOLD),conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(100,clock=program.clock)),transfer=transfer,execution=AMRExecution.synchronous())
    return case,layout,subjects


def initial():
    y,x=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    first=np.stack((20.+x/8+y/16,-10.+x/16-y/8))
    second=np.stack((.25+x/16-y/32,-.5+x/32+y/16,2.+((x+2*y)%5)/8))
    return {PARTITION[0]:np.ascontiguousarray(first),PARTITION[1]:np.ascontiguousarray(second)}


def step(values,k,*,wrong_global=False,wrong_state=False):
    # Positive-oriented face i+1/2; x is final NumPy axis, y is penultimate.
    q=np.asarray(values);rhs=np.zeros_like(q)
    speed=np.full(q.shape[1:],2.) if k==0 or wrong_global else 1.+.125*q[0 if wrong_state else 2]
    for axis,v in zip((-1,-2),VELOCITIES[k]):
        right=np.roll(q,-1,axis=axis)
        bound=np.maximum(speed,np.roll(speed,-1,axis=axis))
        face=.5*v*(q+right)-.5*bound[None,...]*(right-q)
        rhs-=N*(face-np.roll(face,1,axis=axis))
    return q+DT*rhs
