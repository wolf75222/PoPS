"""Public arbitrary Aux/component spellings; equations remain ordinary local sources."""
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.fields import AnalyticAux,AuxiliaryBoundary
from pops.analytic import coordinate
from pops.math import Var
from pops.model import PhysicalSupport,PhysicalDimension
from pops.layouts import Uniform
from pops.mesh import CartesianGrid,PeriodicAxes
from pops.time import FixedDt

NAMES=('a b','a_b','κ','λ','providers','pops_cell_symbol_617578_612062')

def build(names=NAMES, *, state_names=("heat energy","ρ"), recovery=False, primitive=False):
    frame=Rectangle('symbol frame',(0,0),(1,1)).frame(Cartesian2D())
    model=pops.Model('public symbols',frame=frame)
    support=PhysicalSupport((('x','material'),('y','material')));unit=PhysicalDimension()
    state=model.species('public carrier',state=state_names,support=support,
                        units=(unit,unit),sampling='cell_average')
    value=sum((i+1)*Var(name,'aux') for i,name in enumerate(names))
    if primitive:
        velocity=model.primitive('derived gain',state[1]/state[0])
        model.primitive_state(state[0],velocity,states=(state,),conservative=(state[0],state[0]*velocity))
    if recovery:
        model.recovery_admissibility(states=(state,),**{state_names[0]:state[0]>0})
    rate=model.source('weighted local source',on=state,value=(value,-value))
    module=model.module
    for i,name in enumerate(names):
        aux=module.aux_field(name,'cell_scalar',frame=frame.canonical_id,unit='1')
        module.aux_provider(AnalyticAux(module.aux_handle(aux),(i+1)+coordinate(frame,frame.axes[0]),frame=frame,
                                      boundary=AuxiliaryBoundary(width=1,kind='foextrap')))
    case=pops.Case('exact public symbol authority');block=case.block('actual block',model,states=(state,))
    program=pops.Program('named source');current=program.state(block[state])
    rhs=module.operator_handle(rate.reg_name)(current.n)
    program.commit(current.next,program.value('candidate',current.n+program.dt*rhs,at=current.next.point))
    program.step_strategy(FixedDt(1/64));case.program(program)
    return pops.resolve(pops.validate(case),layout=Uniform(CartesianGrid(frame=frame,cells=(4,7),periodic=PeriodicAxes(frame.axes))))
