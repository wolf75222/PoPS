"""Heterogeneous 5/10-moment plasma and evolved Maxwell, on native spatial cells.

S10 https://arxiv.org/html/1409.0262, equations (3)--(5) and (10)--(19).
The ten-moment closure here is zero CENTRAL third moment (Gaussian closure),
not the paper's nonlocal collisionless heat-flux fit. All quantities are
normalized with epsilon0=mu0=c=1 and species charge/mass ratios (-1,+1).
"""
from __future__ import annotations

import math
import numpy as np
import pops
from pops import math as pm
from pops.fields import SpatialInteractionKernel, CellVolumeMeasure, CellMidpoint, DirectSpatialInteraction
from pops.model import FieldSpace, PhysicalSupport
from pops.numerics import DiscretizationPlan, StateStorage
from pops.physics.roles import Custom
from pops.time import FixedDt
from .common import CaseSpec, state_array, non_owner_metrics
from .models_M01_M14 import _domain, _initial
from .models_M22_M25_M28 import _materialize


PAIRS=((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))


def _cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def _central_kernel(cells,lengths,axis):
    """Fourier representation of the EXACT nearest-neighbour centered D_h.

    This is a dense evaluation of a local matrix, not a spectral derivative.
    The transverse cardinal kernel gives Kronecker delta at the actual cell
    centers. CellVolumeMeasure supplies the physical quadrature factors.
    """
    n=cells[axis];L=lengths[axis];h=L/n;other=1-axis
    def kernel(x,y):
        delta=x[axis]-y[axis]
        derivative=sum(-2/L*math.sin(2*math.pi*k/n)/h*pm.sin(2*math.pi*k*delta/L)
                       for k in range(1,(n-1)//2+1))
        m=cells[other];width=lengths[other];distance=x[other]-y[other]
        cardinal=1+sum(2*pm.cos(2*math.pi*k*distance/width) for k in range(1,(m-1)//2+1))
        if m%2==0:cardinal=cardinal+pm.cos(math.pi*m*distance/width)
        return derivative*cardinal/width
    return SpatialInteractionKernel(2,kernel)


def reference_rhs(values,spacing):
    """Independent NumPy tensor equations using local roll stencils, never the DSL."""
    five,ten,field=values
    def D(a,axis):
        return (np.roll(a,-1,axis=-1-axis)-np.roll(a,1,axis=-1-axis))/(2*spacing[axis])
    rho=five[0];m=five[1:4];velocity=m/rho
    pressure=(2/3)*(five[4]-.5*np.sum(m*velocity,axis=0))
    result_five=np.zeros_like(five)
    for axis in range(2):
        flux=np.empty_like(five);flux[0]=m[axis]
        flux[1:4]=m*velocity[axis];flux[axis+1]+=pressure
        flux[4]=(five[4]+pressure)*velocity[axis]
        result_five-=D(flux,axis)
    ri=ten[0];mi=ten[1:4];ui=mi/ri
    raw=np.empty((3,3,*ri.shape))
    for component,(a,b) in enumerate(PAIRS):raw[a,b]=raw[b,a]=ten[component+4]
    result_ten=np.zeros_like(ten)
    for axis in range(2):
        flux=np.empty_like(ten);flux[0]=mi[axis];flux[1:4]=raw[:,axis]
        for component,(a,b) in enumerate(PAIRS):
            flux[component+4]=(ui[a]*raw[b,axis]+ui[b]*raw[a,axis]
                +ui[axis]*raw[a,b]-2*ri*ui[a]*ui[b]*ui[axis])
        result_ten-=D(flux,axis)
    E=field[:3];B=field[3:]
    epsilon=np.zeros((3,3,3))
    epsilon[0,1,2]=epsilon[1,2,0]=epsilon[2,0,1]=1
    epsilon[0,2,1]=epsilon[2,1,0]=epsilon[1,0,2]=-1
    magnetic=np.einsum("ikl,lyx->ikyx",epsilon,B)
    result_five[1:4]-=rho*E+np.einsum("ikyx,kyx->iyx",magnetic,m)
    result_five[4]-=np.sum(m*E,axis=0)
    result_ten[1:4]+=ri*E+np.einsum("ikyx,kyx->iyx",magnetic,mi)
    second=(mi[:,None]*E[None,:]+E[:,None]*mi[None,:]
            +np.einsum("ikyx,kjyx->ijyx",magnetic,raw)
            +np.einsum("jkyx,ikyx->ijyx",magnetic,raw))
    for component,(a,b) in enumerate(PAIRS):result_ten[component+4]+=second[a,b]
    current=-m+mi
    electric=np.stack((D(B[2],1),-D(B[2],0),D(B[1],0)-D(B[0],1)))-current
    magnetic_rate=-np.stack((D(E[2],1),-D(E[2],0),D(E[1],0)-D(E[0],1)))
    return result_five,result_ten,np.concatenate((electric,magnetic_rate),axis=0)


def reference_rk4(initial,spacing,dt,steps):
    """Finite reception oracle only; never used by the PoPS time Program."""
    values=tuple(value.copy() for value in initial)
    def stage(base,rhs,factor):
        return tuple(u+factor*dt*r for u,r in zip(base,rhs,strict=True))
    for _ in range(steps):
        a=reference_rhs(values,spacing)
        b=reference_rhs(stage(values,a,.5),spacing)
        c=reference_rhs(stage(values,b,.5),spacing)
        d=reference_rhs(stage(values,c,1),spacing)
        values=tuple(u+dt*(r1+2*r2+2*r3+r4)/6
                     for u,r1,r2,r3,r4 in zip(values,a,b,c,d,strict=True))
    return values


def build_m12(cells=(8,8),*,dt=.0005,steps=2):
    """Two real species blocks (5 and 10 components) plus a six-component EM block.

    Shared centered derivatives satisfy D C=0 and summation by parts on this
    torus. The same stage current J is used in Ampere and the work audit.
    Classical RK4 preserves the linear Gauss constraints; energy conservation
    is a semidiscrete property, with measured finite temporal error. No global
    positivity theorem, reconnection qualification or AMR claim is implied.
    """
    if not 0<dt<=.0005 or type(steps) is not int or not 2<=steps<=8:
        raise ValueError("this smooth small-data reception requires dt<=.0005 and 2..8 steps")
    lengths=(2*np.pi,2*np.pi);frame,layout=_domain("M12_torus",cells,lengths=lengths)
    support=PhysicalSupport((("x","M12_torus"),("y","M12_torus")))
    declarations=[];case=pops.Case("M12_heterogeneous_Maxwell")
    for label,names in (("five",("rho","mx","my","mz","energy")),
                        ("ten",("rho","mx","my","mz","Mxx","Myy","Mzz","Mxy","Mxz","Myz")),
                        ("EM",("Ex","Ey","Ez","Bx","By","Bz"))):
        model=pops.Model(label+"_equations",frame=frame)
        state=model.state(label,components=names,roles={name:Custom("m12_"+name) for name in names},
                          support=support,sampling="cell_average")
        block=case.block(label,model)
        plan=DiscretizationPlan();plan.states.add(state,StateStorage())
        case.numerics(plan,block=block);declarations.append((block,state))
    p=pops.Program("M12_RK4_compatible_Maxwell")
    temporal=tuple(p.state(block[state]) for block,state in declarations)
    initial_states=tuple(t.n for t in temporal)
    kernels=tuple(_central_kernel(cells,lengths,axis) for axis in range(2))
    def derivative(value,axis,label):
        space=FieldSpace(label+"_differential",components=value.space.components,
            representation="field",layout="cell",centering="cell",sampling="cell_center",
            frame=value.space.frame,clock=value.space.clock,support=value.space.support,units=value.space.units)
        result=p.spatial_interaction(value,kernels[axis],output_space=space,
            measure=CellVolumeMeasure(),quadrature=CellMidpoint(),
            realization=DirectSpatialInteraction(64<<20),name=label+"_D")
        return p.cell_mean(result,template=value,quadrature=CellMidpoint(),name=label+"_mean")
    def materialize(label,entries,template):
        return _materialize(p,label,tuple(entries),template,template.point)
    def five_flux(q,axis):
        rho=q[0];m=tuple(q[i+1] for i in range(3));u=tuple(v/rho for v in m)
        pressure=(2/3)*(q[4]-sum(v*v for v in m)/(2*rho))
        return (m[axis],*(m[i]*u[axis]+(pressure if i==axis else 0.) for i in range(3)),
                (q[4]+pressure)*u[axis])
    def ten_data(q):
        rho=q[0];m=tuple(q[i+1] for i in range(3));u=tuple(v/rho for v in m)
        indices={pair:index+4 for index,pair in enumerate(PAIRS)}
        raw=lambda i,j:q[indices[tuple(sorted((i,j)))]]
        pressure=lambda i,j:raw(i,j)-m[i]*m[j]/rho
        return rho,m,u,raw,pressure
    def ten_flux(q,axis):
        rho,m,u,raw,P=ten_data(q)
        third=tuple(u[axis]*raw(i,j)+u[i]*P(j,axis)+u[j]*P(i,axis) for i,j in PAIRS)
        return (m[axis],*(raw(i,axis) for i in range(3)),*third)
    def rhs(states,label):
        e,i,em=states;E=tuple(em[k] for k in range(3));B=tuple(em[k+3] for k in range(3))
        me=tuple(e[k+1] for k in range(3));rho_i,mi,ui,M,Pi=ten_data(i)
        # One shared current SSA: same moment values as species mass fluxes.
        current=materialize(label+"_current",tuple(-me[k]+mi[k] for k in range(3))+(0.,0.,0.),em)
        derivatives=[]
        for species,formula,name in ((e,five_flux,"five"),(i,ten_flux,"ten")):
            axes=[]
            for axis in range(2):
                flux=materialize(label+"_"+name+"_flux%d"%axis,formula(species,axis),species)
                axes.append(derivative(flux,axis,label+"_"+name+"_axis%d"%axis))
            derivatives.append(axes)
        de,di=derivatives
        eb=_cross(me,B);ib=_cross(mi,B)
        force_e=tuple(-(e[0]*E[k]+eb[k]) for k in range(3))
        force_i=tuple(rho_i*E[k]+ib[k] for k in range(3))
        second=[]
        for a,b in PAIRS:
            # Magnetic action on each raw-tensor index; no half-symmetrization.
            column_a=tuple(M(k,a) for k in range(3));column_b=tuple(M(k,b) for k in range(3))
            magnetic=_cross(column_b,B)[a]+_cross(column_a,B)[b]
            second.append(mi[a]*E[b]+mi[b]*E[a]+magnetic)
        source_e=(0.,*force_e,-sum(me[k]*E[k] for k in range(3)))
        source_i=(0.,*force_i,*second)
        re=materialize(label+"_five_rhs",(source_e[k]-de[0][k]-de[1][k] for k in range(5)),e)
        ri=materialize(label+"_ten_rhs",(source_i[k]-di[0][k]-di[1][k] for k in range(10)),i)
        dx=derivative(em,0,label+"_EM_x");dy=derivative(em,1,label+"_EM_y")
        # curl B=(Dy Bz,-Dx Bz,Dx By-Dy Bx), likewise for E.
        rem=materialize(label+"_Maxwell_rhs",(
            dy[5]-current[0],-dx[5]-current[1],dx[4]-dy[3]-current[2],
            -dy[2],dx[2],-dx[1]+dy[0]),em)
        return re,ri,rem
    def shifted(base,rate,factor,point,label):
        return tuple(p.value(label+str(index),u+factor*p.dt*r,at=point)
                     for index,(u,r) in enumerate(zip(base,rate,strict=True)))
    point_half=p.stage("half",c=.5);point_end=p.stage("end",c=1)
    k1=rhs(initial_states,"k1")
    k2=rhs(shifted(initial_states,k1,.5,point_half,"u2"),"k2")
    k3=rhs(shifted(initial_states,k2,.5,point_half,"u3"),"k3")
    k4=rhs(shifted(initial_states,k3,1.,point_end,"u4"),"k4")
    for index,t in enumerate(temporal):
        candidate=p.value("accepted_"+str(index),t.n+p.dt/6*(k1[index]+2*k2[index]+2*k3[index]+k4[index]),at=t.next.point)
        p.commit(t.next,candidate)
    p.step_strategy(FixedDt(dt));case.program(p)
    x,y=np.meshgrid((np.arange(cells[0])+.5)*lengths[0]/cells[0],(np.arange(cells[1])+.5)*lengths[1]/cells[1])
    amplitude=.02*np.sinc(1/cells[0]);re=1+amplitude*np.sin(x);ri=np.ones_like(x)
    ue=(.03,.01,.02);ui=(-.01,.02,0.)
    e=np.stack((re,*(re*u for u in ue),.15+.5*re*sum(u*u for u in ue)))
    pressures={(0,0):.15+0*x,(1,1):.12+0*x,(2,2):.1+0*x,
               (0,1):.01*np.cos(x),(0,2):.005*np.sin(y),(1,2):.003*np.cos(x+y)}
    ion=np.stack((ri,*(ri*u for u in ui),*(pressures[a,b]+ri*ui[a]*ui[b] for a,b in PAIRS)))
    sigma=math.sin(2*math.pi/cells[0])/(lengths[0]/cells[0])
    em=np.stack((.01+amplitude*np.cos(x)/sigma,0*x,.02*np.sin(x)+.01*np.sin(y),
                 .01*np.sin(y),-.02*np.sin(x),.1+0*x))
    initial=(e,ion,em);bindings={};subjects={}
    for (block,state),values,label in zip(declarations,initial,("five","ten","EM"),strict=True):
        subject=block[state];bindings.update(_initial(case,subject,values));subjects[label]=subject
    def d(values,axis):
        return (np.roll(values,-1,axis=1-axis)-np.roll(values,1,axis=1-axis))/(2*lengths[axis]/cells[axis])
    def constraints(values):
        five,ten,field=values
        return d(field[0],0)+d(field[1],1)-(-five[0]+ten[0]), d(field[3],0)+d(field[4],1)
    def energy(values):
        five,ten,field=values
        return float(np.mean(five[4]+.5*(ten[4]+ten[5]+ten[6])+.5*np.sum(field*field,axis=0)))
    gauss0,divb0=constraints(initial);energy0=energy(initial)
    expected=reference_rk4(initial,tuple(L/n for L,n in zip(lengths,cells)),dt,steps)
    def check(runtime):
        actual=tuple(state_array(runtime,label,values.shape) for label,values in zip(("five","ten","EM"),initial,strict=True))
        if actual[0] is None:return non_owner_metrics()
        five,ten,field=actual;gauss,divb=constraints(actual)
        tensor=np.empty((*ten.shape[1:],3,3))
        for k,(a,b) in enumerate(PAIRS):
            tensor[...,a,b]=tensor[...,b,a]=ten[k+4]-ten[a+1]*ten[b+1]/ten[0]
        eigenvalue=float(np.min(np.linalg.eigvalsh(tensor)))
        pressure=(2/3)*(five[4]-np.sum(five[1:4]**2,axis=0)/(2*five[0]))
        gauss_error=float(np.max(abs(gauss-gauss0)));divb_error=float(np.max(abs(divb-divb0)))
        energy_error=abs(energy(actual)-energy0)
        exchange=float(np.mean(.5*np.sum(field*field,axis=0))-
                       np.mean(.5*np.sum(initial[2]*initial[2],axis=0)))
        change=max(float(np.max(abs(a-b))) for a,b in zip(actual,initial,strict=True))
        mass_error=max(abs(float(np.mean(actual[k][0]-initial[k][0]))) for k in (0,1))
        component_errors=tuple(float(np.max(abs(a-b))) for a,b in zip(actual,expected,strict=True))
        initial_constraint=max(float(np.max(abs(gauss0))),float(np.max(abs(divb0))))
        passed=bool(all(np.isfinite(v).all() for v in actual) and min(np.min(five[0]),np.min(ten[0]))>0
            and np.min(pressure)>0 and eigenvalue>0 and gauss_error<2.e-11 and divb_error<2.e-11
            and energy_error<2.e-10 and mass_error<2.e-12 and abs(exchange)>1.e-9 and change>1.e-7
            and max(component_errors)<2.e-10 and initial_constraint<2.e-13)
        return dict(metrics_owner=True,passed=passed,component_widths=(5,10,6),Gauss_defect=gauss_error,
            divB_defect=divb_error,total_energy_defect=energy_error,field_energy_exchange=exchange,
            mass_defect=mass_error,minimum_ten_pressure_eigenvalue=eigenvalue,state_change=change,
            initial_constraint_defect=initial_constraint,full_state_reference_errors=component_errors)
    return CaseSpec(case,layout,bindings,subjects,steps*dt,dt,check,
        "M12: periodic 2D3V heterogeneous five-moment electron / ten-moment ion / six-component evolved Maxwell; Qcentral=0 and gamma5/3, common centered FV derivative evaluated by dense native spatial quadrature, shared stage current, native RK4; Gauss/divB and semidiscrete work pairing, finite temporal energy error, no reconnection/AMR/global positivity claim")
