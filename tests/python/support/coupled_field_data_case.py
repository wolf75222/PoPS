"""Public exchange with nontrivial provider data at two stages.

The default retains the original static witness. Optional temporal/SSPRK2 profiles
exercise the same public composition with its exact consuming physical Clock.
"""
from fractions import Fraction
import pops
from pops.model import PhysicalSupport,PhysicalDimension
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.fields import AnalyticAux,DerivedAux,AuxiliaryBoundary
from pops.analytic import coordinate
from pops.math import Var,ValueExpr
from pops.time import FixedDt
from pops.mesh import CartesianGrid,PeriodicAxes
from pops.layouts import Uniform

DT=Fraction(1,64)
FRACTIONS=(Fraction(1,4),Fraction(3,4))
NAMES=('z_exchange','a_exchange','m_catalyst')
SUPPORT=PhysicalSupport((('first-coordinate','ordinary spatial rectangle'),
                         ('second-coordinate','ordinary spatial rectangle')))
UNIT=PhysicalDimension()

def build(*,cells=(7,3),reverse=False,permuted=True,time_dependent=False,ssprk2=False):
    frame=Rectangle('different non-square spatial frame',(0,0),(1,1)).frame(Cartesian2D())
    model=pops.Model('independent spatially modulated exchange',frame=frame)
    a=model.species('alpha',state=('amount','response'),support=SUPPORT,units=(UNIT,UNIT),sampling='cell_average')
    b=model.species('beta',state=('amount','response'),support=SUPPORT,units=(UNIT,UNIT),sampling='cell_average')
    catalyst=model.species('catalyst',state=('material',),support=SUPPORT,units=(UNIT,),sampling='cell_average')
    module=model.module
    p=(pops.Program('two fractional evaluations and shared exchange')
       if time_dependent else None)
    seed=module.aux_field('spatial_seed','cell_scalar',frame=frame.canonical_id,unit='1')
    gain=module.aux_field('nonlinear_gain','cell_scalar',frame=frame.canonical_id,unit='1')
    seed_expression=1+coordinate(frame,frame.axes[0])+2*coordinate(frame,frame.axes[1])
    if time_dependent:
        from pops.analytic import time as analytic_time
        seed_expression=seed_expression+analytic_time(p.clock)
    module.aux_provider(AnalyticAux(module.aux_handle(seed),
        seed_expression,frame=frame,
        boundary=AuxiliaryBoundary(width=1,kind='foextrap')))
    module.aux_provider(DerivedAux(module.aux_handle(gain),
        2+ValueExpr(module.aux_handle(seed))**2,
        boundary=AuxiliaryBoundary(width=1,kind='foextrap')))
    coefficient=Var('nonlinear_gain','aux')
    q=coefficient*catalyst[0]*(b[0]-a[0])
    inputs=(catalyst,b,a) if permuted else (a,b,catalyst)
    coupling=model.coupled_rate('reversible exchange',inputs=inputs,
        outputs={a:(q,coefficient),b:(-q,coefficient)})
    case=pops.Case('three species consume declared provider data')
    definitions=tuple(zip(NAMES,(a,b,catalyst),strict=True))
    blocks={};subjects={}
    for name,state in reversed(definitions) if reverse else definitions:
        blocks[name]=case.block(name,model,states=(state,));subjects[name]=blocks[name][state]
    if p is None:p=pops.Program('two fractional evaluations and shared exchange')
    states={name:p.state(subjects[name]) for name in NAMES}
    def evaluate(c,alpha,beta):
        point=p.stage('fractional exchange %s'%c,c=c)
        values=(p.value('alpha %s'%c,alpha,at=point),p.value('beta %s'%c,beta,at=point),
                p.value('readonly catalyst %s'%c,states[NAMES[2]].n,at=point))
        called=(values[2],values[1],values[0]) if permuted else values
        bundle=coupling(*called)
        return bundle[blocks[NAMES[0]]],bundle[blocks[NAMES[1]]]
    fractions=(Fraction(0),Fraction(1)) if ssprk2 else FRACTIONS
    intermediate=Fraction(1) if ssprk2 else Fraction(1,2)
    r0a,r0b=evaluate(fractions[0],states[NAMES[0]].n,states[NAMES[1]].n)
    r1a,r1b=evaluate(fractions[1],states[NAMES[0]].n+intermediate*p.dt*r0a,
                                 states[NAMES[1]].n+intermediate*p.dt*r0b)
    accepted=(Fraction(1,2)*(r0a+r1a),Fraction(1,2)*(r0b+r1b)) if ssprk2 else (r1a,r1b)
    for name,rhs in ((NAMES[0],accepted[0]),(NAMES[1],accepted[1])):
        state=states[name];p.commit(state.next,p.value('accepted '+name,state.n+p.dt*rhs,at=state.next.point))
    state=states[NAMES[2]];p.commit(state.next,p.value('unchanged catalyst',state.n,at=state.next.point))
    p.step_strategy(FixedDt(float(DT)));case.program(p)
    layout=Uniform(CartesianGrid(frame=frame,cells=cells,periodic=PeriodicAxes(frame.axes)))
    return pops.resolve(pops.validate(case),layout=layout,compile_options={'model_source_policy':'require'})
