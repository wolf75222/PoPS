"""Authored radiation, nonconservative continuum and moving-volume examples.

This is a scientific library, not an engine extension. Python elaborates all
equations; NumPy below supplies initial data and independent reception checks.
Each builder states its closed physical/numerical variant and remaining limits.
"""
from __future__ import annotations

from functools import reduce
import numpy as np
import pops
from pops import math as pm
from pops.linalg.finite import FiniteSupport
from pops.numerics import (DiscretizationPlan, PathConservativeFiniteVolume,
    SymbolicPath, reconstruction, riemann, variables)
from pops.time import FixedDt, FailRun
from .common import CaseSpec, MissingCapabilityError, state_array, non_owner_metrics
from .models_M01_M14 import _domain, _initial, _fv, _ssprk2, _traced_ssprk2


LIMITS = {
    "M22": ("diffusion-limit asymptotic preservation", "Marshak boundary layer",
            "relativistic or comoving emission corrections"),
    "M24": ("disappearance of a phase", "stiff pressure/velocity relaxation",
            "proof of positivity and exact material-contact preservation"),
    "M23": ("multidimensional constrained-transport div B guarantee",
            "general magnetic boundary conditions", "Hall stiffness implicit solve"),
    "M25": ("stiff viscous/conduction asymptotic limits", "exact discrete det A=rho/rho0 constraint",
            "general-state hyperbolic speed enclosure outside the specified small-data profile"),
    "M28": ("multidimensional moving-face provider", "non-centered moving-face quadrature",
            "topology-changing moving meshes"),
}


def _materialize(p, name, entries, template, at):
    """Explicit ordered component support and co-located spatial map."""
    from pops.time.expressions import component_names
    return FiniteSupport(name + "_support", component_names(template)).bind(
        entries).materialize(p, name, template=template, at=at)


def build_m22(cells=(12, 8), *, opacity=3., dt=.0005, steps=4):
    """Gray lab-frame absorption/emission, c=a=cv=1, ideal gas gamma=1.4.

    Euler material U=(rho,mx,my,Etot), M1 radiation R=(Er,Fx,Fy).
    G=opacity*(T**4-Er), H=opacity*F. Radiation gets (G,-H),
    material gets (0,H,-G). This defines a closed lab-frame model with
    total energy/momentum exchange exact; no claim of a mixed-frame AP method.
    """
    if opacity < 0 or dt <= 0 or steps < 1:
        raise ValueError("opacity>=0, dt>0 and steps>=1 required")
    frame, layout = _domain("M22_periodic", cells)
    gas = pops.Model("M22_material_equations", frame=frame)
    u = gas.state("material", components=("rho", "mx", "my", "Etot"))
    rho, mx, my, energy = u
    pressure = .4*(energy-(mx*mx+my*my)/(2*rho))
    gas_flux = gas.flux("Euler_flux", frame=frame, state=u,
        components={axis: (u[d+1], mx*u[d+1]/rho+(pressure if d == 0 else 0),
                           my*u[d+1]/rho+(pressure if d == 1 else 0),
                           (energy+pressure)*u[d+1]/rho)
                    for d,axis in enumerate(frame.axes)},
        waves={axis: (u[d+1]/rho-pm.sqrt(1.4*pressure/rho),
                       u[d+1]/rho, u[d+1]/rho,
                       u[d+1]/rho+pm.sqrt(1.4*pressure/rho))
               for d,axis in enumerate(frame.axes)})
    gas_rate = gas.rate("Euler_transport", equation=pm.ddt(u) == -pm.div(gas_flux))
    rad = pops.Model("M22_radiation_equations", frame=frame)
    r = rad.state("radiation", components=("Er", "Fx", "Fy"))
    er, fx, fy = r
    f2 = (fx*fx+fy*fy)/(er*er)
    chi = (3+4*f2)/(5+2*pm.sqrt(4-3*f2))
    # Algebraically rationalized (3chi-1)/(2f^2), regular at f=0.
    anisotropy = 3/(2+pm.sqrt(4-3*f2))
    def tensor(i, j):
        return (er*(1-chi)/2 if i == j else 0) + anisotropy*r[i+1]*r[j+1]/er
    radiation_flux = rad.flux("M1_flux", frame=frame, state=r,
        components={axis:(r[d+1], tensor(0,d), tensor(1,d))
                    for d,axis in enumerate(frame.axes)},
        waves={axis:(-1.,0.,1.) for axis in frame.axes})
    rad_rate = rad.rate("radiation_transport", equation=pm.ddt(r) == -pm.div(radiation_flux))
    case = pops.Case("M22_shared_energy_momentum_exchange")
    material, radiation = case.block("material", gas), case.block("radiation", rad)
    for block,state,flux,rate in ((material,u,gas_flux,gas_rate),(radiation,r,radiation_flux,rad_rate)):
        numerics=DiscretizationPlan();numerics.rates.add(rate,_fv(flux,state));case.numerics(numerics,block=block)
    from pops.contributions import Contribution, Delivery, Increment
    p=pops.Program("M22_coupled_SSPRK2")
    m,n=p.state(material[u]),p.state(radiation[r])
    window=p.interval()
    def stage(a,b,point,label):
        temperature=(a[3]-(a[1]*a[1]+a[2]*a[2])/(2*a[0]))/a[0]
        exchange=_materialize(p,label+"_one_exchange",(
            p.dt*opacity*(temperature**4-b[0]),p.dt*opacity*b[1],p.dt*opacity*b[2]),b,b.point)
        km,kn=gas_rate(a),rad_rate(b)
        am=p.value(label+"_material_transport",a+p.dt*km,at=point)
        bn=p.value(label+"_radiation_transport",b+p.dt*kn,at=point)
        # One native SSA increment, one authentic temporal window, two explicit
        # incidence maps. No separately evaluated opposite physical source.
        contribution=Contribution(label+"_radiation_matter",exchange,window)
        increment=Increment((contribution,),(
            Delivery(contribution,am,((0,0,0),(0,1,0),(0,0,1),(-1,0,0))),
            Delivery(contribution,bn,((1,0,0),(0,-1,0),(0,0,-1))),
        ))
        am=increment.apply(p,am,name=label+"_material",at=point)
        bn=increment.apply(p,bn,name=label+"_radiation",at=point)
        increment.require_application(am);increment.require_application(bn)
        return am,bn
    first=p.stage("predictor",c=1)
    a,b=stage(m.n,n.n,first,"stage0")
    a2,b2=stage(a,b,m.next.point,"stage1")
    mf=p.value("accepted_material",.5*m.n+.5*a2,at=m.next.point)
    rf=p.value("accepted_radiation",.5*n.n+.5*b2,at=n.next.point)
    p.commit_many({m.next:mf,n.next:rf});p.step_strategy(FixedDt(dt));case.program(p)
    x,y=np.meshgrid((np.arange(cells[0])+.5)/cells[0],(np.arange(cells[1])+.5)/cells[1])
    initial_m=np.array([np.ones_like(x),np.zeros_like(x),np.zeros_like(x),.8+.02*np.cos(2*np.pi*y)])
    initial_r=np.array([.6+.08*np.cos(2*np.pi*x),.03*np.sin(2*np.pi*x),.02*np.cos(2*np.pi*y)])
    initials={**_initial(case,material[u],initial_m),**_initial(case,radiation[r],initial_r)}
    from .continuum_oracles import advance_m22
    expected_m, expected_r = advance_m22(initial_m, initial_r, opacity=opacity, dt=dt, steps=steps)
    def check(runtime):
        actual_m=state_array(runtime,"material",initial_m.shape)
        actual_r=state_array(runtime,"radiation",initial_r.shape)
        if actual_m is None:return non_owner_metrics()
        before=np.array([np.mean(initial_m[3]+initial_r[0]),np.mean(initial_m[1]+initial_r[1]),np.mean(initial_m[2]+initial_r[2])])
        after=np.array([np.mean(actual_m[3]+actual_r[0]),np.mean(actual_m[1]+actual_r[1]),np.mean(actual_m[2]+actual_r[2])])
        defect=float(np.max(abs(after-before)))
        internal=actual_m[3]-(actual_m[1]**2+actual_m[2]**2)/(2*actual_m[0])
        margin=float(np.min(actual_r[0]-np.sqrt(actual_r[1]**2+actual_r[2]**2)))
        change=float(np.max(abs(actual_r-initial_r)))
        error=float(max(np.max(abs(actual_m-expected_m)),np.max(abs(actual_r-expected_r))))
        return dict(passed=error<2e-11 and defect<2e-11 and margin>0 and np.min(internal)>0 and change>1e-8,
                    discrete_reference_max_error=error,
                    total_energy_momentum_defect=defect,radiative_margin=margin,
                    material_min_internal_energy=float(np.min(internal)),state_change=change)
    return CaseSpec(case,layout,initials,{"material":material[u],"radiation":radiation[r]},dt*steps,dt,check,
        f"M22: spatial 2D gray lab-frame M1 + distinct Euler material, c=a=cv=1, shared energy/momentum exchange, periodic FV/Rusanov/SSPRK2; opacity={opacity:g}, dt={dt:g}, steps={steps}; final radiation-cone margin and material internal energy checked, no general dt-opacity guard or AP/Marshak qualification")


def bn_equations(q):
    """BN convective seven-equation 1D variant: pI=p1,uI=u2."""
    a,r1,r2,m1,m2,e1,e2=q
    v1,v2=m1/r1,m2/r2
    p1,p2=.4*(e1-m1*m1/(2*r1))/a,.6*(e2-m2*m2/(2*r2))/(1-a)
    flux=(0.,m1,m2,m1*v1+a*p1,m2*v2+(1-a)*p2,
          (e1+a*p1)*v1,(e2+(1-a)*p2)*v2)
    column=(v2,0.,0.,-p1,p1,-p1*v2,p1*v2)
    matrix=tuple((entry,0.,0.,0.,0.,0.,0.) for entry in column)
    return flux,matrix,(v1,v2,p1,p2)


def hall_face(*, hall=.1, ohmic=0., gamma=1.4):
    """Full 2.5D MHD + Hall/Ohmic conservative flux, mu0=q=1.

    Hall reads a gradient but supplies an explicit dispersive frequency bound;
    it does not claim a dissipative operator contract. Central Hall flux and
    Rusanov ideal flux are evaluated on one shared face in the real runtime.
    """
    from pops.numerics.stencil_flux import StencilFlux
    def cross(a,b):
        return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    def body(sample,h,axis):
        left=tuple(-int(d==axis) for d in range(2));right=(0,0)
        def raw(offset):return tuple(sample(offset,i) for i in range(8))
        l,r=raw(left),raw(right)
        mean=tuple((a+b)/2 for a,b in zip(l,r))
        def primitive(q):
            rho=q[0];v=tuple(q[i]/rho for i in (1,2,3));b=q[5:8]
            p=(gamma-1)*(q[4]-.5*rho*sum(a*a for a in v)-.5*sum(a*a for a in b))
            return rho,v,b,p
        def ideal(q):
            rho,v,b,p=primitive(q);pt=p+.5*sum(x*x for x in b)
            flux=(q[axis+1],*(rho*v[i]*v[axis]+(pt if i==axis else 0)-b[i]*b[axis] for i in range(3)),
                  (q[4]+pt)*v[axis]-sum(v[i]*b[i] for i in range(3))*b[axis],
                  *(v[axis]*b[i]-v[i]*b[axis] for i in range(3)))
            speed=abs(v[axis])+pm.sqrt((gamma*p+sum(x*x for x in b))/rho)
            return flux,speed
        def derivative(component,direction):
            if direction==2:return 0.
            if direction==axis:return (r[component]-l[component])/h[direction]
            def shift(offset,sign):return tuple(offset[d]+sign*int(d==direction) for d in range(2))
            return (sample(shift(right,1),component)-sample(shift(right,-1),component)
                    +sample(shift(left,1),component)-sample(shift(left,-1),component))/(4*h[direction])
        current=(derivative(7,1),-derivative(7,0),derivative(6,0)-derivative(5,1))
        rho,velocity,b,pressure=primitive(mean)
        jxb=cross(current,b)
        electric=tuple(hall*jxb[i]/rho+ohmic*current[i] for i in range(3))
        poynting=cross(electric,b)[axis]
        normal=tuple(float(i==axis) for i in range(3))
        induction=cross(normal,electric)
        fl,al=ideal(l);fr,ar=ideal(r);speed=pm.maximum(al,ar)
        correction=(0.,0.,0.,0.,poynting,*induction)
        value=tuple(.5*(fl[i]+fr[i]-speed*(r[i]-l[i]))+correction[i] for i in range(8))
        # Incidence contribution is supplied per face to the native dt guard.
        frequency=speed/h[axis]+4*(abs(hall)*pm.sqrt(sum(x*x for x in b))/rho+ohmic)/h[axis]**2
        return value,frequency
    return StencilFlux(dimension=2,components=8,radius=1,body=body)


def build_m23(cells=(32,4), *, hall=.1, ohmic=0., dt=None, steps=8):
    """Small circular Hall-Alfven wave in full MHD equations, periodic x."""
    from pops.numerics import StateStorage
    frame,layout=_domain("M23_Hall_MHD",cells)
    model=pops.Model("M23_full_MHD_equations",frame=frame)
    from pops.physics.roles import Custom
    names=("rho","mx","my","mz","E","Bx","By","Bz")
    # The three physical velocity components are independent of mesh rank.
    # No legacy name-to-role inference is allowed to impose a 2-component fluid.
    u=model.state("fluid_and_induction",components=names,roles={n:Custom(n) for n in names})
    case=pops.Case("M23_planar_whistler");block=case.block("magnetofluid",model)
    flux=hall_face(hall=hall,ohmic=ohmic)
    plan=DiscretizationPlan();plan.states.add(u,StateStorage(ghost_depth=max(flux.ghost_extent)));case.numerics(plan,block=block)
    maximum_dt=min(.0001,.01/(max(cells)**2*(abs(hall)+ohmic+.01)))
    dt=maximum_dt if dt is None else float(dt)
    if not 0<dt<=maximum_dt or type(steps) is not int or steps<1:
        raise ValueError("Hall reception requires a positive stable step and positive step count")
    _ssprk2(case,block,u,lambda p,q,point,label:p.stencil_flux_divergence("MHD_"+label,q,flux=flux),dt,name="M23_SSPRK2")
    x,y=np.meshgrid((np.arange(cells[0])+.5)/cells[0],(np.arange(cells[1])+.5)/cells[1])
    k=2*np.pi;dx=1/cells[0];kd=np.sin(k*dx)/dx;kh2=4*np.sin(k*dx/2)**2/dx**2
    omega=.5*(hall*kh2+np.sqrt((hall*kh2)**2+4*kd**2))
    amplitude=1.e-4;by=amplitude*np.cos(k*x);bz=amplitude*np.sin(k*x)
    vy=kd/omega*by;vz=kd/omega*bz
    rho=np.ones_like(x);zero=np.zeros_like(x)
    initial=np.array([rho,zero,vy,vz,2.5+.5*(vy*vy+vz*vz)+.5*(1+by*by+bz*bz),rho,by,bz])
    def check(runtime):
        q=state_array(runtime,"magnetofluid",initial.shape)
        if q is None:return non_owner_metrics()
        defect=float(max(abs(np.mean(q[i])-np.mean(initial[i])) for i in range(8)))
        divb=float(np.max(abs((np.roll(q[5],-1,axis=1)-np.roll(q[5],1,axis=1))/(2*dx))))
        mode=np.mean((q[6]+1j*q[7])*np.exp(-1j*k*x))
        measured=float(np.angle(mode/amplitude)/(dt*steps))
        # Spatial central-difference dispersion, before finite-time RK error;
        # The amplitude is the total damping observable, separate from Hall phase.
        # A companion reception contrasts eta>0 with eta=0 at the same interval;
        # neither this metric nor total-energy conservation splits the numerical
        # and physical damping into independently measured quantitative budgets.
        relative=abs(measured-omega)/omega
        return dict(passed=bool(defect<2e-11 and divb<2e-12 and relative<.03),
            inventory_defect=defect,div_B=divb,frequency=measured,semidiscrete_frequency=omega,
            relative_frequency_error=relative,amplitude=float(abs(mode)),hall=hall,ohmic=ohmic,
            energy_loss=float(np.mean(initial[4]-q[4])),planar_topology=True)
    return CaseSpec(case,layout,_initial(case,block[u],initial),{"magnetofluid":block[u]},dt*steps,dt,check,
        "M23: full mass/momentum/energy/induction Hall-MHD 2.5D equations; planar circular wave, central Hall/Ohmic + Rusanov ideal face algebra, SSPRK2; div B qualified only for this planar topology")


def gpr_equations(q, *, tau_distortion=.1, tau_heat=.1):
    """GPR rho,m,E,A,rhoJ with gamma=1.4,cv=cs=alpha=rho0=T0=1.

    Internal energy follows the conserved total energy. Relaxation changes A/J
    and hence heats matter; no independent energy source duplicates that heat.
    """
    rho=q[0];velocity=tuple(q[i]/rho for i in (1,2,3));energy=q[4]
    a=tuple(tuple(q[5+3*i+j] for j in range(3)) for i in range(3))
    j=tuple(q[14+i]/rho for i in range(3))
    g=tuple(tuple(sum(a[k][i]*a[k][j] for k in range(3)) for j in range(3)) for i in range(3))
    trace=sum(g[i][i] for i in range(3))
    dev=tuple(tuple(g[i][j]-(trace/3 if i==j else 0) for j in range(3)) for i in range(3))
    elastic=.25*sum(x*x for row in dev for x in row)
    thermal=.5*sum(x*x for x in j)
    temperature=energy/rho-.5*sum(v*v for v in velocity)-elastic-thermal
    pressure=.4*rho*temperature
    psi=tuple(tuple(sum(a[i][k]*dev[k][j] for k in range(3)) for j in range(3)) for i in range(3))
    stress=tuple(tuple(-rho*sum(a[k][i]*psi[k][j] for k in range(3)) for j in range(3)) for i in range(3))
    det=(a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
         -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
         +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))
    theta1=tau_distortion*det**(-5/3)/3
    theta2=tau_heat*rho/temperature
    source=(0.,0.,0.,0.,0.,*(-psi[i][k]/theta1 for i in range(3) for k in range(3)),
            *(-rho*j[i]/theta2 for i in range(3)))
    fluxes=[];matrices=[]
    for axis in range(2):
        flux=(q[axis+1],*(rho*velocity[i]*velocity[axis]+(pressure if i==axis else 0)-stress[i][axis] for i in range(3)),
              (energy+pressure)*velocity[axis]-sum(velocity[i]*stress[i][axis] for i in range(3))+temperature*j[axis],
              *(sum(a[i][m]*velocity[m] for m in range(3)) if k==axis else 0. for i in range(3) for k in range(3)),
              *(rho*j[i]*velocity[axis]+(temperature if i==axis else 0.) for i in range(3)))
        matrix=[[0.]*17 for _ in range(17)]
        for i in range(3):
            for k in range(3):
                for ell in range(3):
                    matrix[5+3*i+k][5+3*i+ell]=(velocity[axis] if k==ell else 0.)-(velocity[ell] if k==axis else 0.)
        fluxes.append(flux);matrices.append(tuple(map(tuple,matrix)))
    return tuple(fluxes),tuple(matrices),source,(temperature,det,elastic,thermal)


def build_m25(cells=(12,4), *, tau_distortion=.1, tau_heat=.1):
    from pops.numerics import StateStorage
    frame,layout=_domain("M25_periodic",cells)
    model=pops.Model("M25_GPR_equations",frame=frame)
    names=("rho","mx","my","mz","E",*(f"A{i}{j}" for i in range(3) for j in range(3)),"rhoJx","rhoJy","rhoJz")
    from pops.physics.roles import Custom
    u=model.state("thermomechanical",components=names,roles={n:Custom(n) for n in names})
    fluxes,matrices,source_values,_=gpr_equations(tuple(u),tau_distortion=tau_distortion,tau_heat=tau_heat)
    flux=model.flux("thermomechanical_flux",state=u,frame=frame,components=dict(zip(frame.axes,fluxes)))
    nc=model.nonconservative_product("distortion_transport",state=u,matrices=dict(zip(frame.axes,matrices)),
        conservative_components=(*names[:5],*names[14:]))
    source=model.source("constitutive_relaxation",on=u,value=source_values)
    balance=model.rate("GPR_balance",equation=pm.ddt(u)==-pm.div(flux)-nc+source)
    transport=balance.select(flux,nc);relaxation=balance.select(source)
    # Deliberately explicit engineering envelope for this small-data reception.
    # This is not a generic GPR eigenvalue oracle or a positivity certificate.
    path=SymbolicPath(nc,frame=frame,quadrature=((.5-1/(2*np.sqrt(3)),.5),(.5+1/(2*np.sqrt(3)),.5)),
                      speed=lambda left,right,axis:64.)
    method=PathConservativeFiniteVolume(flux=flux,path=path,variables=variables.Conservative(u),
        reconstruction=reconstruction.FirstOrder(),riemann=riemann.Rusanov())
    case=pops.Case("M25_evolved_distortion_and_heat");block=case.block("continuum",model)
    plan=DiscretizationPlan();plan.rates.add(transport,method);plan.rates.add(relaxation,StateStorage());case.numerics(plan,block=block)
    dt=1.e-5;steps=8
    _traced_ssprk2(case,block,u,balance,dt,name="M25_SSPRK2")
    x,y=np.meshgrid((np.arange(cells[0])+.5)/cells[0],(np.arange(cells[1])+.5)/cells[1])
    initial=np.zeros((17,cells[1],cells[0]));initial[0]=1
    for i in range(3):initial[5+3*i+i]=1
    initial[6]=.001*np.sin(2*np.pi*x);initial[2]=.001*np.sin(2*np.pi*x)
    initial[14]=.003*np.cos(2*np.pi*x)
    # Set E so the initial internal temperature is exactly one.
    initial[4]=1
    _,_,_,(temporary,_,elastic,thermal)=gpr_equations(tuple(initial),tau_distortion=tau_distortion,tau_heat=tau_heat)
    initial[4]+=1-temporary
    from .continuum_oracles import advance_m25
    expected=advance_m25(initial,dt=dt,steps=steps,tau_distortion=tau_distortion,tau_heat=tau_heat)
    def check(runtime):
        q=state_array(runtime,"continuum",initial.shape)
        if q is None:return non_owner_metrics()
        _,_,_,(temperature,det,elastic_final,thermal_final)=gpr_equations(tuple(q),tau_distortion=tau_distortion,tau_heat=tau_heat)
        defect=float(max(abs(np.mean(q[i])-np.mean(initial[i])) for i in range(5)))
        change=float(np.max(abs(q[5:]-initial[5:])))
        compatibility=float(np.max(abs(det-q[0])))
        error=float(np.max(abs(q-expected)))
        return dict(passed=error<2e-11 and defect<2e-11 and np.min(temperature)>0 and np.min(det)>0 and change>1e-9 and compatibility<1e-5,
                    discrete_reference_max_error=error,
                    conservative_inventory_defect=defect,distortion_heat_change=change,
                    min_temperature=float(np.min(temperature)),min_det_A=float(np.min(det)),
                    det_A_density_defect=compatibility,
                    internal_mode_energy_before=float(np.mean(elastic+thermal)),
                    internal_mode_energy_after=float(np.mean(elastic_final+thermal_final)))
    return CaseSpec(case,layout,_initial(case,block[u],initial),{"continuum":block[u]},dt*steps,dt,check,
        "M25: full 17-component GPR thermo-mechanical equations (three velocity/internal axes, two spatial axes), evolved A and rhoJ, straight Gauss2 path, explicit relaxation SSPRK2; small planar perturbation, fixed speed64, no stiff-limit qualification")


def build_m28(cells=24, *, amplitude=.04, nonuniform=False):
    """Actual moving control volumes, endpoint swept measures, native GCL."""
    from pops.analytic import coordinate, sin, time
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid,GeometryEvolution,MovingControlVolumes,PeriodicAxes
    from pops.time import MovingFieldProjection
    frame=CartesianDomain("M28_interval",(0.,),(1.,)).frame(Cartesian1D())
    model=pops.Model("M28_transport_equation",frame=frame)
    u=model.state("density",components=("a","b"))
    flux=model.flux("transport",state=u,frame=frame,
        components={frame.axes[0]:(.7*u[0],.7*u[1])},waves={frame.axes[0]:(.7,.7)})
    rate=model.rate("conservation",equation=pm.ddt(u)==-pm.div(flux))
    case=pops.Case("M28_oscillating_control_volumes");block=case.block("moving",model)
    plan=DiscretizationPlan();plan.rates.add(rate,_fv(flux,u));case.numerics(plan,block=block)
    p=pops.Program("M28_reynolds_interval");state=p.state(block[u])
    x=coordinate(frame,frame.axes[0])
    evolution=GeometryEvolution((x+amplitude*sin(2*np.pi*x)*sin(2*np.pi*time(p.clock)),))
    geometry=p.geometry_state(state.n,evolution=evolution)
    update=p.reynolds_update(geometry,physical_rate=rate(state.n),
        projection=MovingFieldProjection((.5,.5),(1,0)),geometry_tolerance=1e-13,at=state.next.point)
    p.commit(state.next,update);dt=.001;steps=8;p.step_strategy(FixedDt(dt));case.program(p)
    layout=MovingControlVolumes(Uniform(CartesianGrid(frame=frame,cells=(cells,),periodic=PeriodicAxes(frame.axes))),evolution=evolution)
    centers=(np.arange(cells)+.5)/cells
    initial=np.stack([np.full(cells,1.3),np.full(cells,.8)])
    if nonuniform:initial[0]+=.05*np.cos(2*np.pi*centers)*np.sinc(1/cells)
    # Independent finite-volume Reynolds update. For this profile .7-w>0
    # at every face; the authored Rusanov flux is therefore exactly upwind.
    reference_nodes=np.arange(cells+1)/cells
    expected=initial.copy()
    previous_nodes=reference_nodes.copy()
    for step in range(steps):
        next_nodes=reference_nodes+amplitude*np.sin(2*np.pi*reference_nodes)*np.sin(2*np.pi*(step+1)*dt)
        mesh_speed=(next_nodes[:-1]-previous_nodes[:-1])/dt
        relative_speed=.7-mesh_speed
        if np.min(relative_speed)<=0:
            raise ValueError("M28 reference profile requires positive relative transport speed")
        relative_flux=relative_speed*np.roll(expected,1,axis=1)
        inventory=expected*np.diff(previous_nodes)+dt*(relative_flux-np.roll(relative_flux,-1,axis=1))
        expected=inventory/np.diff(next_nodes)
        previous_nodes=next_nodes
    def check(runtime):
        q=state_array(runtime,"moving",initial.shape)
        from pops.codegen.program_lowerability import all_ops
        from pops.time.references import canonical_handle
        declaration=next(v for v in all_ops(runtime._install_plan.artifact.program.program) if v.op=="geometry_state")
        identity="pops.moving:"+canonical_handle(declaration.state_ref).qualified_id
        # Read-only inspection of the actual native geometry, never a substitute
        # coordinate law used by the solver or its state update.
        snapshot=runtime._executor._s._output_moving_geometry_snapshot(identity,declaration.space.frame)
        if q is None:return non_owner_metrics()
        volumes=np.asarray(snapshot["cell_volumes"]).reshape(-1)
        nodes=np.asarray(snapshot["node_coordinates"]).reshape(-1)
        expected_nodes=reference_nodes+amplitude*np.sin(2*np.pi*reference_nodes)*np.sin(2*np.pi*runtime.time())
        geometry_error=float(np.max(abs(nodes-expected_nodes)))
        conservation=float(np.max(abs(q@volumes-np.mean(initial,axis=1))))
        uniform_error=float(np.max(abs(q[1]-.8)))
        if not nonuniform:uniform_error=max(uniform_error,float(np.max(abs(q[0]-1.3))))
        error=float(np.max(abs(q-expected)))
        return dict(passed=error<2e-11 and geometry_error<2e-12 and conservation<2e-11 and uniform_error<2e-11,
                    discrete_reference_max_error=error,
                    geometry_error=geometry_error,conservation_defect=conservation,
                    uniform_state_error=uniform_error,min_volume=float(np.min(volumes)),
                    geometry_generation=int(snapshot["generation"]),spatial_cells=cells,
                    moving_amplitude=amplitude,nonuniform=nonuniform)
    return CaseSpec(case,layout,_initial(case,block[u],initial),{"moving":block[u]},dt*steps,dt,check,
        "M28: Dim1 real ALE volumes with oscillating analytic nodes, endpoint swept measures and one shared Reynolds interval; GCL/uniform-state and total inventory checks, no AMR-remap surrogate")


def build_m24(cells=(16,4), *, pressure_relaxation=0., velocity_relaxation=0.):
    """Seven-equation BN with an explicitly chosen interphase relaxation law.

    S18 supplies the convective variant pI=p1,uI=u2. Nonzero coefficients
    below select an additional pressure/drag law; it is not attributed to S18.
    Interphase momentum and total energy are delivered with opposite signs.
    """
    if any(not np.isfinite(c) or c < 0 for c in (pressure_relaxation, velocity_relaxation)):
        raise ValueError("relaxation coefficients must be finite and nonnegative")
    frame,layout=_domain("M24_periodic_planar",cells)
    model=pops.Model("M24_Baer_Nunziato_convective",frame=frame)
    names=("alpha","r1","r2","m1","m2","e1","e2")
    u=model.state("two_phase",components=names)
    flux_values,bx,(v1s,v2s,p1s,p2s)=bn_equations(tuple(u))
    flux=model.flux("phase_fluxes",state=u,frame=frame,
        components={frame.axes[0]:flux_values,frame.axes[1]:(0.,)*7})
    product=model.nonconservative_product("interface_products",state=u,
        matrices={frame.axes[0]:bx,frame.axes[1]:((0.,)*7,)*7},
        conservative_components=("r1","r2"))
    def speed(left,right,axis):
        if axis==1:return 0.
        # Convective eigenvalues uI,u1±c1,u2±c2. Along a straight
        # admissible path, phase velocity is a mass-weighted average;
        # e/r bounds specific internal energy from above everywhere.
        result=0.
        for r,m,e,gamma in ((1,3,5,1.4),(2,4,6,1.6)):
            vmax=pm.maximum(abs(left[m]/left[r]),abs(right[m]/right[r]))
            emax=pm.maximum(left[e]/left[r],right[e]/right[r])
            result=pm.maximum(result,vmax+pm.sqrt(gamma*(gamma-1)*emax))
        return result
    path=SymbolicPath(product,frame=frame,quadrature=((.5-1/(2*np.sqrt(3)),.5),(.5+1/(2*np.sqrt(3)),.5)),speed=speed)
    volume_exchange = pressure_relaxation*(p1s-p2s)
    drag = velocity_relaxation*(v2s-v1s)
    work = -p1s*volume_exchange+v2s*drag
    exchange = model.source("interphase_relaxation", on=u,
        value=(volume_exchange, 0., 0., drag, -drag, work, -work))
    rate=model.rate("two_phase_balance",equation=pm.ddt(u)==-pm.div(flux)-product+exchange)
    transport, relaxation = rate.select(flux, product), rate.select(exchange)
    method=PathConservativeFiniteVolume(flux=flux,path=path,variables=variables.Conservative(u),
        reconstruction=reconstruction.FirstOrder(),riemann=riemann.Rusanov())
    case=pops.Case("M24_path_conservative");block=case.block("mixture",model)
    from pops.numerics import StateStorage
    plan=DiscretizationPlan();plan.rates.add(transport,method)
    plan.rates.add(relaxation,StateStorage());case.numerics(plan,block=block)
    dt=.0005;steps=4
    _traced_ssprk2(case,block,u,rate,dt,name="M24_SSPRK2")
    x,y=np.meshgrid((np.arange(cells[0])+.5)/cells[0],(np.arange(cells[1])+.5)/cells[1])
    a=.45+.08*np.sin(2*np.pi*x);r1=a*(1+.03*np.cos(2*np.pi*x));r2=(1-a)*.8
    v1=.2+.02*np.cos(2*np.pi*x);v2=-.1+.03*np.sin(2*np.pi*x)
    p1=1+.02*np.cos(2*np.pi*x);p2=.9+.03*np.sin(2*np.pi*x)
    initial=np.array([a,r1,r2,r1*v1,r2*v2,a*p1/.4+.5*r1*v1*v1,(1-a)*p2/.6+.5*r2*v2*v2])
    from .continuum_oracles import advance_m24
    coefficients=dict(pressure_relaxation=pressure_relaxation,velocity_relaxation=velocity_relaxation)
    expected=advance_m24(initial,dt=dt,steps=steps,**coefficients)
    refined_path=advance_m24(initial,dt=dt,steps=steps,path_points=4,**coefficients)
    unrelaxed=advance_m24(initial,dt=dt,steps=steps)
    def check(runtime):
        q=state_array(runtime,"mixture",initial.shape)
        if q is None:return non_owner_metrics()
        _,_,(_,_,p1,p2)=bn_equations(tuple(q))
        before=np.array([np.mean(initial[1]),np.mean(initial[2]),np.mean(initial[3]+initial[4]),np.mean(initial[5]+initial[6])])
        after=np.array([np.mean(q[1]),np.mean(q[2]),np.mean(q[3]+q[4]),np.mean(q[5]+q[6])])
        defect=float(np.max(abs(after-before)));change=float(np.max(abs(q-initial)))
        margin=float(min(np.min(q[0]),np.min(1-q[0]),np.min(q[1]),np.min(q[2]),np.min(p1),np.min(p2)))
        error=float(np.max(abs(q-expected)))
        path_error=float(np.max(abs(q-refined_path)))
        relaxation_gap=float(np.max(abs(q-unrelaxed)))
        return dict(passed=error<2e-11 and path_error<2e-10 and defect<2e-11 and margin>0 and change>1e-8,
            discrete_reference_max_error=error,refined_path_max_error=path_error,
            mixture_inventory_defect=defect,admissibility_margin=margin,state_change=change,
            relaxation_gap=relaxation_gap,pressure_relaxation=pressure_relaxation,
            velocity_relaxation=velocity_relaxation)
    return CaseSpec(case,layout,_initial(case,block[u],initial),{"mixture":block[u]},dt*steps,dt,check,
        "M24: seven-equation BN, pI=p1,uI=u2, gamma=(1.4,1.6), straight conserved-state path Gauss2, "
        "periodic FV/SSPRK2; authored pressure/drag relaxation coefficients="+str(coefficients))


def build_model(identifier, *, profile=None, require_complete=False, **kwargs):
    from .profile_dispatch import build_profile
    key = str(identifier).upper()
    if key not in {"M22", "M23", "M24", "M25", "M28"}:
        raise MissingCapabilityError(key, required=("unknown continuum model identifier",),
            profile="M22/M23/M24/M25/M28 selected continuum and geometry profiles")
    return build_profile(key, profile=profile, require_complete=require_complete, **kwargs)
