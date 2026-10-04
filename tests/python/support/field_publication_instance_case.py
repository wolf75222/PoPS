"""Public exchange with a declared, stage-varying solved-field provider.

The forcing carrier obeys ds/dt=cos(2*pi*x). Thus its exact stage value
is (1+t)*cos(2*pi*x); no AnalyticAux time extension or Python cell callback.
"""
from fractions import Fraction
from math import pi
import pops
from pops.model import PhysicalSupport,PhysicalDimension,Handle,OwnerPath
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.fields import (AnalyticAux,DerivedAux,AuxiliaryBoundary,FieldProblem,
                        FieldBoundary,FieldDiscretization,SharedMeanGauge,bcs)
from pops.fields.methods import CellCenteredSecondOrder
from pops.analytic import coordinate,cos
from pops.math import Var,ValueExpr,laplacian,Reaction
from pops.solvers import CG
from pops.time import FixedDt,FailRun
from pops.mesh import CartesianGrid,PeriodicAxes
from pops.layouts import Uniform

DT=Fraction(1,64)
FRACTIONS=(Fraction(1,4),Fraction(3,4))
NAMES=('z_exchange','a_exchange','m_catalyst','separate_forcing')
SUPPORT=PhysicalSupport((('first-coordinate','ordinary spatial rectangle'),
                         ('second-coordinate','ordinary spatial rectangle')))
UNIT=PhysicalDimension()

def build(*,cells=(7,3),reverse=False,permuted=True,different_values=False,first_beta=False):
    frame=Rectangle('different non-square spatial frame',(0,0),(1,1)).frame(Cartesian2D())
    exchange=pops.Model('independent field-modulated exchange',frame=frame)
    a=exchange.species('alpha',state=('amount','response'),support=SUPPORT,units=(UNIT,UNIT),sampling='cell_average')
    b=exchange.species('beta',state=('amount','response'),support=SUPPORT,units=(UNIT,UNIT),sampling='cell_average')
    catalyst=exchange.species('catalyst',state=('material',),support=SUPPORT,units=(UNIT,),sampling='cell_average')
    module=exchange.module
    incoming=module.field_space('imposed solved field',('published_potential',),support=SUPPORT,units=(UNIT,),sampling='cell')
    gain=module.aux_field('nonlinear_gain','cell_scalar',frame=frame.canonical_id,unit='1')
    module.aux_provider(DerivedAux(module.aux_handle(gain),
        2+ValueExpr(module.field_handle(incoming))**2,
        boundary=AuxiliaryBoundary(width=1,kind='foextrap')))
    coefficient=Var('nonlinear_gain','aux')
    q=coefficient*catalyst[0]*(b[0]-a[0])
    inputs=(b,catalyst,a) if first_beta else ((catalyst,b,a) if permuted else (a,b,catalyst))
    coupling=exchange.coupled_rate('reversible exchange',inputs=inputs,
        outputs={a:(q,coefficient),b:(-q,coefficient)})
    driver=pops.Model('independent forcing carrier',frame=frame)
    signal=driver.species('forcing',state=('load',),support=SUPPORT,units=(UNIT,),sampling='cell_average')
    mode=Var("spatial_mode","aux")
    derivative=driver.source("forcing ramp",on=signal,value=(mode,))
    auxiliary=driver.module.aux_field('spatial_mode','cell_scalar',frame=frame.canonical_id,unit='1')
    mode=Var('spatial_mode','aux')
    driver.module.aux_provider(AnalyticAux(driver.module.aux_handle(auxiliary),
        cos(2*pi*coordinate(frame,frame.axes[0])),frame=frame,
        boundary=AuxiliaryBoundary(width=1,kind='foextrap')))
    case=pops.Case('three species consume candidate solved-field data')
    definitions=((NAMES[0],exchange,a),(NAMES[1],exchange,b),
                 (NAMES[2],exchange,catalyst),(NAMES[3],driver,signal))
    blocks={};subjects={}
    for name,model,state in reversed(definitions) if reverse else definitions:
        blocks[name]=case.block(name,model,states=(state,));subjects[name]=blocks[name][state]
    potential=Handle('field potential',kind='field',owner=OwnerPath.model('separate field equation'))
    second=Handle("second field potential",kind="field",owner=OwnerPath.model("separate field equation"))
    unknowns=(potential,second) if different_values else (potential,)
    # The separate joint witness has exactly one shared constant kernel. Independent
    # uncoupled Laplacians would have two constants and cannot use SharedMeanGauge.
    equations=(-laplacian(potential)+Reaction(potential,2)-Reaction(second,2)==signal[0],
               -laplacian(second)+Reaction(second,2)-Reaction(potential,2)==2*signal[0]) if different_values else (-laplacian(potential)==signal[0],)
    boundaries=tuple(FieldBoundary(h,bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(),bcs.Periodic())) for h in unknowns)
    problem=FieldProblem('periodic candidate Poisson',unknowns=unknowns,
        equations=equations,
        boundaries=boundaries,
        gauge=SharedMeanGauge(unknowns,0))
    field=case.field(problem,FieldDiscretization(method=CellCenteredSecondOrder(),boundaries=(),
        solver=CG(max_iter=200,rel_tol=1e-12,abs_tol=1e-12)))
    p=pops.Program('two fractional field evaluations and shared exchange')
    states={name:p.state(subjects[name]) for name in NAMES}
    source_rate=driver.module.operator_handle(derivative.reg_name)(states[NAMES[3]].n)
    field_values=[]
    def evaluate(c,alpha,beta):
        point=p.stage('fractional field %s'%c,c=c)
        ramp=p.value('candidate ramp %s'%c,states[NAMES[3]].n+c*p.dt*source_rate,at=point)
        values=(p.value('alpha %s'%c,alpha,at=point),p.value('beta %s'%c,beta,at=point),
                p.value('readonly catalyst %s'%c,states[NAMES[2]].n,at=point))
        observed=field.observe(p.solve(field,values={subjects[NAMES[3]]:ramp},at=point).consume(action=FailRun()))
        phi=observed[field[potential]];field_values.append(phi)
        targets={(blocks[name][module.field_handle(incoming)],'published_potential'):(observed[field[second]] if different_values and name==NAMES[1] else phi) for name in NAMES[:3]}
        observed.publish(targets,states={subjects[name]:value for name,value in zip(NAMES[:3],values,strict=True)})
        called=(values[1],values[2],values[0]) if first_beta else ((values[2],values[1],values[0]) if permuted else values)
        bundle=coupling(*called)
        return bundle[blocks[NAMES[0]]],bundle[blocks[NAMES[1]]]
    r0a,r0b=evaluate(FRACTIONS[0],states[NAMES[0]].n,states[NAMES[1]].n)
    r1a,r1b=evaluate(FRACTIONS[1],states[NAMES[0]].n+Fraction(1,2)*p.dt*r0a,
                                 states[NAMES[1]].n+Fraction(1,2)*p.dt*r0b)
    for name,rhs in ((NAMES[0],r1a),(NAMES[1],r1b),(NAMES[3],source_rate)):
        state=states[name];p.commit(state.next,p.value('accepted '+name,state.n+p.dt*rhs,at=state.next.point))
    state=states[NAMES[2]];p.commit(state.next,p.value('unchanged catalyst',state.n,at=state.next.point))
    for index,value in enumerate(field_values):p.store_history('candidate_field_%d'%index,value,depth=1)
    p.step_strategy(FixedDt(float(DT)));case.program(p)
    layout=Uniform(CartesianGrid(frame=frame,cells=cells,periodic=PeriodicAxes(frame.axes)))
    return pops.resolve(pops.validate(case),layout=layout,compile_options={'model_source_policy':'require'})
