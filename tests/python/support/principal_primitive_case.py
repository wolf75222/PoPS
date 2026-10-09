"""A genuinely cross-state primitive coordinate map for principal FV receipts."""
import numpy as np
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.math import ddt,div
from pops.mesh import CartesianGrid,PeriodicAxes
from pops.numerics import FiniteVolume,DiscretizationPlan,reconstruction,riemann,variables,limiters
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt

CELLS,DT=8,.0005


def primitive_case(widths=(1,1),*,reverse=False,coordinates=True,map_reverse=False,sample_offsets=None):
    count=sum(widths)
    order=tuple(reversed(range(count))) if reverse else tuple(range(count))
    frame=Rectangle('primitive_box',(0.,0.),(1.,1.)).frame(Cartesian2D())
    model=pops.Model('joint_primitive_physics',frame=frame)
    starts=np.cumsum((0,*widths))
    states=tuple(model.species('row%d'%i,state=tuple('c%d'%order[j] for j in range(starts[i],starts[i+1])))
                 for i in range(len(widths)))
    q=tuple(component for state in states for component in state)
    # Both coordinates and conservative inverse use the same ordinary expression IR.
    parameter=model.param(RuntimeParam('coordinate_beta',default=.02))
    beta=model.value(parameter)
    p=(q[0],*(model.primitive('v%d'%j,q[j]/q[0]+beta*q[0]*q[0]) for j in range(1,count)))
    inverse=(p[0],*(p[0]*(p[j]-beta*p[0]*p[0]) for j in range(1,count)))
    if coordinates:
        selected=tuple(reversed(range(len(widths)))) if map_reverse else tuple(range(len(widths)))
        positions=tuple(j for i in selected for j in range(starts[i],starts[i+1]))
        map_states=tuple(states[i] for i in selected)
        model.primitive_state(*(p[j] for j in positions),states=map_states,
                              conservative=tuple(inverse[j] for j in positions))
        model.recovery_admissibility(states=map_states,**{q[0].component:q[0]>0})
    row,col=np.indices((count,count))
    matrices=tuple(a[np.ix_(order,order)] for a in (
        np.diag(1.+.1*np.arange(count))+.025*(row+col+1),
        np.diag(-.5-.05*np.arange(count))-.015*(row+col+1)))
    waves=tuple(float(np.max(np.sum(abs(a),axis=1))) for a in matrices)
    # One joint principal group has one explicitly authored common bound.
    # Its value is the same full-matrix row-sum bound as the historical witness.
    model.module.eigenvalues(**{axis:(bound,)*count for axis,bound in zip(("x","y"),waves)})
    fluxes=[];rates=[]
    for i,state in enumerate(states):
        flux=model.flux('flux%d'%i,frame=frame,state=state,
            components={axis:tuple(sum(float(a[k,j])*(inverse[j] if coordinates else q[j]) for j in range(count))
                                   for k in range(starts[i],starts[i+1]))
                        for axis,a in zip(frame.axes,matrices)})
        fluxes.append(flux)
        rates.append(model.rate('rate%d'%i,equation=ddt(state)==-div(flux)))
    case=pops.Case('joint_primitive')
    rows=tuple(reversed(range(len(widths)))) if reverse else tuple(range(len(widths)))
    blocks={i:case.block('block%d'%i,model,states=(states[i],)) for i in rows}
    for i in rows:
        reconstruction_policy = (reconstruction.MUSCL(limiter=limiters.MC())
            if sample_offsets is None else reconstruction.User(
                lambda sample, offset=sample_offsets[i]: sample(offset), formal_order=1))
        method=FiniteVolume(flux=fluxes[i],variables=variables.Primitive(states[i]),
            reconstruction=reconstruction_policy,riemann=riemann.Rusanov(),
            sampling=tuple(states[j] for j in rows if j!=i))
        plan=DiscretizationPlan();plan.rates.add(rates[i],method);case.numerics(plan,block=blocks[i])
        case.initials.add(InitialCondition(state=blocks[i][states[i]],value=BindArray(),
                                           projection=ConservativeCellAverage()))
    program=pops.Program('primitive_euler')
    temporal={i:program.state(blocks[i][states[i]]) for i in rows}
    bindings={states[i]:temporal[i].n for i in rows}
    for i in rows:
        rhs=rates[i](temporal[i].n,bindings=bindings)
        program.commit(temporal[i].next,program.value('accepted',temporal[i].n+program.dt*rhs,
                                                    at=temporal[i].next.point))
    program.step_strategy(FixedDt(DT));case.program(program)
    layout=Uniform(CartesianGrid(frame=frame,cells=(CELLS,CELLS),periodic=PeriodicAxes(frame.axes)))
    subjects=tuple(case.resolve(blocks[i][states[i]]) for i in range(len(widths)))
    return case,layout,subjects,matrices,tuple(blocks[i][parameter] for i in range(len(widths)))
