"""Short original Fan--Li15 support, preserving the original Gauss4 script."""
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
N=16
DT=1e-4
STEPS=2


@lru_cache(None)
def original():
    directory=ROOT/'examples/migration/scientific'
    sys.path.insert(0,str(directory))
    try:
        spec=importlib.util.spec_from_file_location('m17_original_scientific_support',directory/'api040_m17_fan_li.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(directory))
    if module.N!=N or module.DT!=DT or module.STEPS!=8:
        raise ValueError('original M17 corpus constants changed')
    return module


def make_case(order):
    # Same original equations and guarded SSPRK2; shared directional expressions
    # are explicit because the path authenticates the exact retained AST.
    import pops
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid,PeriodicAxes
    from pops.math import ddt,div,minimum
    from pops.moments import CartesianMonomialBasis,fan_li15_path
    from pops.moments.fan_li import fan_li15_expressions
    from pops.numerics import DiscretizationPlan,PathConservativeFiniteVolume,reconstruction,riemann,variables
    from pops.representations import Conservative
    from pops.spaces import CellState
    from pops.time import FailRun,FixedDt
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.projection import ConservativeCellAverage
    module=original()
    if len(order)!=15 or set(order)!=set(module.INDICES):raise ValueError('exact original degree-four basis required')
    frame=Rectangle('fan_li_periodic_square',lower=(0.,0.),upper=(1.,1.)).frame(Cartesian2D())
    x_axis,y_axis=frame.axes;covectors={x_axis:(1,0),y_axis:(0,1)}
    model=pops.Model('authored_fan_li15',frame=frame)
    labels=tuple('M%d%d'%index for index in order)
    state=model.state('raw_moments',components=labels,representation=Conservative(),space=CellState(frame=frame))
    slot={index:k for k,index in enumerate(order)}
    physical=fan_li15_expressions(tuple(state[slot[index]] for index in module.INDICES))
    flux=model.flux('hermite_transport',state=state,frame=frame,components={
        axis:tuple(physical.directional_flux(g)[module.INDICES.index(index)] for index in order)
        for axis,g in covectors.items()})
    def matrix(g):
        rows=physical.directional_nonconservative_matrix(g)
        return tuple(tuple(rows[module.INDICES.index(index)][module.INDICES.index(other)] for other in order) for index in order)
    product=model.nonconservative_product('full_temperature_regularization',state=state,
        matrices={axis:matrix(g) for axis,g in covectors.items()},
        conservative_components=tuple('M%d%d'%index for index in order if sum(index)<4))
    path=fan_li15_path(product,frame=frame,covectors=covectors,basis=CartesianMonomialBasis(order))
    rate=model.rate('complete_fan_li_balance',equation=ddt(state)==-div(flux)-product)
    numerics=DiscretizationPlan();numerics.rates.add(rate,PathConservativeFiniteVolume(
        flux=flux,path=path,variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(),riemann=riemann.Rusanov()))
    case=pops.Case('api040_M17_fan_li15_short_composition');block=case.block('gas',model);case.numerics(numerics,block=block)
    program=pops.Program('FanLi_SSPRK2_with_domain');temporal=program.state(block[state])
    def guarded(candidate,tag):
        raw=tuple(candidate[slot[index]] for index in module.INDICES)
        rho=raw[0];u,v=raw[1]/rho,raw[5]/rho
        a,b,c=raw[2]/rho-u*u,raw[6]/rho-u*v,raw[9]/rho-v*v
        margin=minimum(minimum(rho,a),minimum(c,a*c-b*b))
        margin_field=program.value(tag+'_domain_margin',tuple(margin if k==0 else 0*candidate[k] for k in range(15)),at=candidate.point)
        return program.guard(tag+'_positive_SPD',candidate,program.min(margin_field)>0.,action=FailRun())
    stage0=program.stage('fan_li_stage_0',c=0);stage1=program.stage('fan_li_stage_1',c=1)
    current=guarded(temporal.n,'current');k0=program.value('fan_li_rhs_0',rate(current),at=stage0)
    first=program.value('fan_li_first_stage',temporal.n+program.dt*k0,at=stage1);first=guarded(first,'stage_1')
    k1=program.value('fan_li_rhs_1',rate(first),at=stage1)
    endpoint=program.value('fan_li_ssprk2_endpoint',temporal.n+.5*program.dt*(k0+k1),at=temporal.next.point)
    program.commit(temporal.next,guarded(endpoint,'endpoint'));program.step_strategy(FixedDt(DT));case.program(program)
    case.initials.add(InitialCondition(state=block[state],value=BindArray(),projection=ConservativeCellAverage()))
    return case,Uniform(CartesianGrid(frame=frame,cells=(N,N),periodic=PeriodicAxes(frame.axes)))


def initial(order):
    return original().initial_state(order,n=N)


def canonical(values,order):
    return np.asarray(values).reshape(15,N,N)[[order.index(index) for index in original().INDICES]]


def rhs(values,*,points=48,nonconservative=True):
    oracle=sys.modules['api040_m17_oracle']
    line=np.asarray(values).reshape(15,N)
    right=np.roll(line,-1,axis=1)
    flux=np.stack([oracle.flux(line[:,k],(1.,0.)) for k in range(N)],axis=1)
    speed=np.sqrt(6+np.sqrt(10))*np.sqrt(line[2]/line[0])
    face=.5*(flux+np.roll(flux,-1,axis=1)-np.maximum(speed,np.roll(speed,-1))[None,:]*(right-line))
    integral=oracle.independent_path(line,right,(1.,0.),points=points) if nonconservative else np.zeros_like(line)
    return -N*(face-np.roll(face,1,axis=1))-.5*N*(integral+np.roll(integral,1,axis=1))


def ssprk2(values,*,points=48,nonconservative=True):
    values=np.asarray(values)
    first=rhs(values,points=points,nonconservative=nonconservative)
    stage=values+DT*first
    return values+.5*DT*(first+rhs(stage,points=points,nonconservative=nonconservative))


def reference_proof(values):
    from tests.python.support.fan_li15_oracle import generating_hermite
    oracle=sys.modules['api040_m17_oracle']
    line=np.asarray(values).reshape(15,N)
    coefficients=[generating_hermite(line[:,k])[0] for k in range(N)]
    h3=max(abs(value) for row in coefficients for index,value in row.items() if sum(index)==3)
    h4=max(abs(value) for row in coefficients for index,value in row.items() if sum(index)==4)
    product=oracle.independent_path(line,np.roll(line,-1,axis=1),(1.,0.),points=48)
    active=ssprk2(line);disabled=ssprk2(line,nonconservative=False)
    return {'h3_max':float(h3),'h4_max':float(h4),'regularized_face_max':float(np.max(np.abs(product))),
            'active_vs_distinct_B0_step_gap':float(np.max(np.abs(active-disabled))),
            'quadrature24_48_step_gap':float(np.max(np.abs(active-ssprk2(line,points=24))))}
