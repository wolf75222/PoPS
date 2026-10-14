"""Native PoPS equation builders for the precisely stated M01--M14 profiles.

All evolving expressions use pops.math and ordinary expression composition.
NumPy and Python scalar mathematics prepare data and inspect results only.
No compile, bind, run, registered physical model, or host timestep at import.
The unavailable families raise MissingCapabilityError; a smaller ODE witness
is never returned under the name of the missing PDE.
"""
from __future__ import annotations

import math as reference_math

import numpy as np
import pops
from pops import math as pm
from pops.domain import Rectangle
from pops.fields import (CellCenteredSecondOrder, FieldBoundary, FieldDiscretization,
                         FieldProblem, SharedMeanGauge, bcs)
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import (CoordinatedFace, CoordinatedFiniteVolume, Diffusion,
                           DiscretizationPlan, FaceBalance, ScharfetterGummel,
                           TensorDiffusion, reconstruction, riemann, variables)
from pops.numerics.spatial import FiniteVolume
from pops.physics.diffusion import DiffusiveBoundary
from pops.projection import ConservativeCellAverage
from pops.solvers import CG
from pops.time import FailRun, FixedDt

from .common import CaseSpec, MissingCapabilityError, non_owner_metrics, state_array


READY = ("M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "M10", "M11", "M12", "M13", "M14")
ALL_MODELS = tuple(f"M{i:02d}" for i in range(1, 15))


def _domain(name, cells, *, lengths=(1., 1.), periodic=(0, 1)):
    if len(cells) != 2 or any(type(n) is not int or n < 4 for n in cells):
        raise ValueError("these Dim2 PDE cases require at least four cells on each axis")
    frame = Rectangle(name, (0., 0.), lengths).frame(Cartesian2D())
    layout = Uniform(CartesianGrid(frame=frame, cells=cells,
        periodic=PeriodicAxes(tuple(frame.axes[i] for i in periodic))))
    return frame, layout


def _initial(case, subject, values):
    case.initials.add(InitialCondition(state=subject, value=BindArray(),
                                      projection=ConservativeCellAverage()))
    return {subject: np.ascontiguousarray(values, dtype=float)}


def _fv(flux, state, face=None):
    return FiniteVolume(flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.Rusanov() if face is None else face)


def _ssprk2(case, block, state, evaluate, dt, *, name, trace_quantity=False):
    """Two explicitly authored stages; evaluate constructs native rate graphs."""
    program = pops.Program(name)
    temporal = program.state(block[state])
    u0 = temporal.n
    k0 = evaluate(program, u0, program.stage(name + "_stage0", c=0), "stage0")
    point1 = program.stage(name + "_stage1", c=1)
    u1 = program.value(name + "_predictor", u0 + program.dt*k0, at=point1)
    k1 = evaluate(program, u1, point1, "stage1")
    accepted = program.value(name + "_accepted", .5*u0 + .5*(u1+program.dt*k1),
                             at=temporal.next.point)
    if trace_quantity:
        quantity = program.integral_state(name + "_boundary_heat", initial=0.)
        for rate in (k0, k1):
            for side in (0, 1):
                program.accept_external_trace(quantity, rate=rate, axis=0,
                                              side=side, component=0)
    else:
        quantity = None
    program.commit(temporal.next, accepted)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    return program, quantity


def _periodic_metrics(initial, *, block="fluid", components=(0,), tolerance=2.e-10,
                      minimum=None, reference=None, error_limit=None, changed=True):
    def check(runtime):
        actual = state_array(runtime, block, initial.shape)
        if actual is None:
            return non_owner_metrics()
        mass = float(max(abs(np.mean(actual[i])-np.mean(initial[i])) for i in components))
        difference = float(np.max(np.abs(actual-initial)))
        out = {"mass_defect": mass, "state_change": difference,
               "finite": bool(np.all(np.isfinite(actual))), "metrics_owner": True}
        passed = out["finite"] and mass <= tolerance
        if changed:
            passed = passed and difference > 1.e-10
        if minimum is not None:
            out["minimum"] = float(np.min(actual[0]))
            passed = passed and out["minimum"] >= minimum
        if reference is not None:
            out["reference_l1"] = float(np.mean(np.abs(actual-reference)))
            passed = passed and out["reference_l1"] <= error_limit
        out["passed"] = bool(passed)
        return out
    return check


def build_m01(cells=(16, 8)):
    frame, layout = _domain("M01_periodic", cells)
    model = pops.Model("M01_authored_transport", frame=frame)
    U = model.state("amount", components=("u",))
    speeds = (.7, .3)
    flux = model.flux("transport", state=U, frame=frame,
        components={axis: (a*U[0],) for axis, a in zip(frame.axes, speeds)},
        waves={axis: (a,) for axis, a in zip(frame.axes, speeds)})
    rate = model.rate("conservation", equation=pm.ddt(U) == -pm.div(flux))
    case = pops.Case("M01_periodic_transport")
    block = case.block("fluid", model)
    plan = DiscretizationPlan(); plan.rates.add(rate, _fv(flux, U)); case.numerics(plan, block=block)
    dt, end = .05/max(cells), .1/max(cells)
    _ssprk2(case, block, U, lambda p, u, point, label: rate(u), dt, name="M01_SSPRK2")
    x, y = np.meshgrid((np.arange(cells[0])+.5)/cells[0], (np.arange(cells[1])+.5)/cells[1])
    amp = .2*np.sinc(1/cells[0])*np.sinc(1/cells[1])
    initial = (1+amp*np.cos(2*np.pi*(x+y)))[None]
    exact = (1+amp*np.cos(2*np.pi*(x+y-sum(speeds)*end)))[None]
    subject = block[U]
    return CaseSpec(case, layout, _initial(case, subject, initial), {"fluid": subject}, end, dt,
        _periodic_metrics(initial, minimum=.799, reference=exact, error_limit=.012),
        "M01: 2D periodic scalar advection, authored flux, rectangular cell means, first-order FV/SSPRK2; short-time accuracy and mass reception")


def _godunov_burgers(left, right, _fl, _fr, _speed):
    l, r = left[0], right[0]
    return (pm.where(l <= r, lambda: pm.where(l >= 0, lambda: l*l/2,
        lambda: pm.where(r <= 0, lambda: r*r/2, lambda: 0)),
        lambda: pm.where(l+r >= 0, lambda: l*l/2, lambda: r*r/2)),)


def build_m02(cells=(16, 8)):
    # y has a zero physical flux; the same authored scalar Godunov body would
    # introduce a spurious y flux. Use equal x/y Burgers PDE, with y-invariant data.
    frame, layout = _domain("M02_periodic", cells)
    model = pops.Model("M02_Burgers", frame=frame)
    U = model.state("scalar", components=("u",))
    flux = model.flux("Burgers_flux", state=U, frame=frame,
        components={a: (U[0]*U[0]/2,) for a in frame.axes},
        waves={a: (U[0],) for a in frame.axes})
    rate = model.rate("entropy_balance", equation=pm.ddt(U) == -pm.div(flux))
    face = riemann.User(body=_godunov_burgers, state=U,
        stability=lambda l, r, fl, fr, speed: pm.maximum(abs(l[0]), abs(r[0])))
    case = pops.Case("M02_Godunov_shock_rarefaction"); block = case.block("fluid", model)
    plan = DiscretizationPlan(); plan.rates.add(rate, _fv(flux, U, face)); case.numerics(plan, block=block)
    dt, end = .04/max(cells), .08/max(cells)
    _ssprk2(case, block, U, lambda p,u,point,label: rate(u), dt, name="M02_SSPRK2")
    line = np.where(np.arange(cells[0]) < cells[0]//2, 2., 1.)
    initial = np.broadcast_to(line, (1, cells[1], cells[0])).copy()
    def rate_reference(u):
        r = np.roll(u, -1, axis=-1)
        f = np.where(u <= r, np.where(u >= 0, u*u/2, np.where(r <= 0,r*r/2,0)),
                     np.where(u+r >= 0,u*u/2,r*r/2))
        return -cells[0]*(f-np.roll(f,1,axis=-1))
    reference = initial.copy()
    for _ in range(2):
        predictor = reference+dt*rate_reference(reference)
        reference = .5*reference+.5*(predictor+dt*rate_reference(predictor))
    subject = block[U]
    return CaseSpec(case, layout, _initial(case,subject,initial), {"fluid":subject}, end,dt,
        _periodic_metrics(initial,minimum=1.-1.e-12,reference=reference,error_limit=2.e-11),
        "M02: u_t+(u²/2)_x+(u²/2)_y=0, periodic y-invariant discontinuous data with shock and rarefaction; exact authored Godunov face, two SSPRK2 steps versus independent FV oracle")


def _euler(model, frame, *, gamma, pressure_offset=0., electromagnetic=False):
    U = model.state("conserved", components=("rho","mx","my","energy"))
    rho,mx,my,energy = U
    vx,vy = mx/rho,my/rho
    pressure = (gamma-1)*(energy-(mx*mx+my*my)/(2*rho))-gamma*pressure_offset
    sound = pm.sqrt(gamma*(pressure+pressure_offset)/rho)
    x,y = frame.axes
    flux = model.flux("Euler_flux", state=U, frame=frame,
        components={x:(mx,mx*vx+pressure,my*vx,(energy+pressure)*vx),
                    y:(my,mx*vy,my*vy+pressure,(energy+pressure)*vy)},
        waves={x:(vx-sound,vx,vx,vx+sound),y:(vy-sound,vy,vy,vy+sound)})
    if electromagnetic:
        phi=model.field("potential")
        gradient=model.vector("electric_potential_gradient",frame=frame,
            components={x:pm.grad(phi).x,y:pm.grad(phi).y})
        ex,ey=-gradient.x,-gradient.y
        bz=.3
        force=model.source("Lorentz",on=U,value=(0*rho,rho*ex+my*bz,
                                                 rho*ey-mx*bz,mx*ex+my*ey))
        rate=model.rate("Euler_Poisson_Lorentz",equation=pm.ddt(U)==-pm.div(flux)+force)
    else:
        phi=None
        rate=model.rate("Euler_balance",equation=pm.ddt(U)==-pm.div(flux))
    return U,flux,rate,phi


def build_m03(cells=(16,8), *, gamma=1.4, pressure_offset=.2):
    if gamma <= 1 or pressure_offset < 0:
        raise ValueError("the selected gas domain requires gamma>1,p_inf>=0")
    frame,layout=_domain("M03_periodic",cells)
    model=pops.Model("M03_authored_stiffened_Euler",frame=frame)
    U,flux,rate,_=_euler(model,frame,gamma=gamma,pressure_offset=pressure_offset)
    case=pops.Case("M03_full_Euler_entropy_wave");block=case.block("fluid",model)
    plan=DiscretizationPlan();plan.rates.add(rate,_fv(flux,U));case.numerics(plan,block=block)
    dt,end=.02/max(cells),.04/max(cells)
    _ssprk2(case,block,U,lambda p,u,point,label:rate(u),dt,name="M03_SSPRK2")
    x=(np.arange(cells[0])+.5)/cells[0]
    def data(time):
        rho=np.broadcast_to(1+.1*np.sinc(1/cells[0])*np.cos(2*np.pi*(x-.3*time)),(cells[1],cells[0]))
        return np.stack((rho,.3*rho,.1*rho,
                         (1+gamma*pressure_offset)/(gamma-1)+.05*rho))
    initial=data(0);subject=block[U]
    base=_periodic_metrics(initial,components=(0,1,2,3),minimum=.899,reference=data(end),error_limit=.004)
    def check(runtime):
        out=base(runtime);actual=state_array(runtime,"fluid",initial.shape)
        if actual is not None:
            p=(gamma-1)*(actual[3]-(actual[1]**2+actual[2]**2)/(2*actual[0]))-gamma*pressure_offset
            out["pressure_error"]=float(np.max(abs(p-1)))
            out["passed"]=bool(out["passed"] and out["pressure_error"]<2.e-10)
        return out
    return CaseSpec(case,layout,_initial(case,subject,initial),{"fluid":subject},end,dt,check,
        f"M03: full compressible 2D Euler, stiffened EOS gamma={gamma},p_inf={pressure_offset}; periodic entropy wave, all conservative fluxes/energy, FV Rusanov/SSPRK2")


def build_m04(cells=(16,8)):
    frame,layout=_domain("M04_periodic",cells)
    model=pops.Model("M04_oblique_diffusion_transport",frame=frame);U=model.state("amount",components=("u",))
    velocity=(.4,-.2);K=((.02,.006),(.006,.015))
    flux=model.flux("transport",state=U,frame=frame,
        components={axis:(a*U[0],) for axis,a in zip(frame.axes,velocity)},
        waves={axis:(a,) for axis,a in zip(frame.axes,velocity)})
    diffusion=model.diffusive_flux("anisotropic_conduction",state=U,value=pm.CoeffGradient(U[0],K))
    balance=model.rate("advection_diffusion",equation=pm.ddt(U)==-pm.div(flux)+pm.div(diffusion))
    advection,conduction=balance.select(flux),balance.select(diffusion)
    case=pops.Case("M04_anisotropic_advection_diffusion");block=case.block("fluid",model)
    plan=DiscretizationPlan();plan.rates.add(advection,_fv(flux,U));plan.rates.add(conduction,TensorDiffusion(flux=diffusion))
    case.numerics(plan,block=block)
    dt,end=.03/(max(cells)**2),.06/(max(cells)**2)
    _ssprk2(case,block,U,lambda p,u,point,label:advection(u)+conduction(u),dt,name="M04_combined_SSPRK2")
    x,y=np.meshgrid((np.arange(cells[0])+.5)/cells[0],(np.arange(cells[1])+.5)/cells[1])
    amp=.2*np.sinc(1/cells[0])*np.sinc(1/cells[1]);initial=(1+amp*np.cos(2*np.pi*(x+y)))[None]
    exact=(1+amp*np.exp(-4*np.pi**2*sum(map(sum,K))*end)*np.cos(2*np.pi*(x+y-sum(velocity)*end)))[None]
    subject=block[U]
    return CaseSpec(case,layout,_initial(case,subject,initial),{"fluid":subject},end,dt,
        _periodic_metrics(initial,minimum=.79,reference=exact,error_limit=.002),
        "M04: full SPD oblique tensor [[.02,.006],[.006,.015]], 2D periodic transport+conduction; separate spatial realizations combined in each SSPRK2 stage, no positivity theorem for TensorDiffusion")


def navier_stokes_face(*, gamma=1.4, viscosity=.01, prandtl=.72):
    """Complete compressible NS face flux; every normal/transverse derivative is Python algebra.

    Ideal gas R=1, Newtonian Stokes stress in a planar 3D fluid (w=0),
    mu=viscosity*T**.7 and Fourier k=mu*cp/Pr. The Rusanov inviscid
    part and centered viscous part share one face transfer. The authored
    frequency is a conservative engineering envelope, not an NS positivity proof.
    """
    from pops.numerics.stencil_flux import StencilFlux
    if gamma <= 1 or viscosity < 0 or prandtl <= 0:
        raise ValueError("NS requires gamma>1, viscosity>=0, Pr>0")
    cv, cp = 1/(gamma-1), gamma/(gamma-1)

    def body(sample, h, axis):
        right = (0, 0); left = tuple(-int(d == axis) for d in range(2))
        sampled_states = {}
        def data(offset):
            if offset in sampled_states:
                return sampled_states[offset]
            rho, mx, my, energy = (sample(offset, i) for i in range(4))
            u, v = mx/rho, my/rho
            pressure = (gamma-1)*(energy-(mx*mx+my*my)/(2*rho))
            sampled_states[offset] = rho, u, v, pressure/rho, pressure, energy
            return sampled_states[offset]
        l, r = data(left), data(right)
        def derivative(component, direction):
            if direction == axis:
                return (r[component]-l[component])/h[direction]
            def shifted(offset, sign):
                return tuple(offset[d]+sign*int(d == direction) for d in range(2))
            return (data(shifted(right, 1))[component]-data(shifted(right, -1))[component]
                    +data(shifted(left, 1))[component]-data(shifted(left, -1))[component])/(4*h[direction])
        grad_u = tuple(tuple(derivative(i+1, j) for j in range(2)) for i in range(2))
        divergence = grad_u[0][0]+grad_u[1][1]
        mu = viscosity*((l[3]+r[3])/2)**.7
        conductivity = cp*mu/prandtl
        tau = tuple(mu*(grad_u[i][axis]+grad_u[axis][i]
                        - (2/3)*int(i == axis)*divergence) for i in range(2))
        def inviscid(q):
            rho, u, v, temperature, pressure, energy = q
            velocity = (u, v); normal = velocity[axis]
            return (rho*normal, rho*u*normal+int(axis == 0)*pressure,
                    rho*v*normal+int(axis == 1)*pressure, (energy+pressure)*normal)
        fl, fr = inviscid(l), inviscid(r)
        speed = pm.maximum(abs(l[axis+1])+pm.sqrt(gamma*l[3]),
                           abs(r[axis+1])+pm.sqrt(gamma*r[3]))
        euler = tuple((fl[i]+fr[i])/2-speed*(sample(right, i)-sample(left, i))/2
                      for i in range(4))
        work = sum((l[i+1]+r[i+1])*tau[i]/2 for i in range(2))
        flux = (euler[0], euler[1]-tau[0], euler[2]-tau[1],
                euler[3]-work-conductivity*derivative(3, axis))
        rho_min = pm.minimum(l[0], r[0])
        frequency = (speed/(2*h[axis])
                     +2*((4/3)*mu+conductivity/cv)/(rho_min*h[axis]**2)
                     +2*mu/(rho_min*h[axis]*h[1-axis]))
        domain_floor = l[0]
        for q in sampled_states.values():
            domain_floor = pm.minimum(domain_floor, pm.minimum(q[0],q[3]))
        return flux, pm.where(domain_floor > 0, lambda: frequency, lambda: -1.)
    return StencilFlux(dimension=2, components=4, radius=1, body=body)


def build_m05(cells=(16, 8), *, viscosity=.01, two_dimensional=False):
    """A discrete NS balance, not an Euler declaration with an opaque correction."""
    from pops.numerics import StateStorage
    frame, layout = _domain("M05_NS", cells)
    model = pops.Model("M05_compressible_Navier_Stokes", frame=frame)
    U = model.state("conservative", components=("rho", "mx", "my", "E"))
    # The complete balance is the materialized face expression below. No fictitious
    # source or truncated Euler rate supplies an installation/coverage loophole.
    case = pops.Case("M05_conservative_NS_stencil"); block = case.block("fluid", model)
    flux = navier_stokes_face(viscosity=viscosity)
    plan = DiscretizationPlan(); plan.states.add(U, StateStorage(ghost_depth=max(flux.ghost_extent)))
    case.numerics(plan, block=block)
    dt, end = .005/max(cells), .01/max(cells)
    _ssprk2(case, block, U, lambda p,u,point,label: p.stencil_flux_divergence(
        "NS_"+label, u, flux=flux), dt, name="M05_SSPRK2")
    x, y = np.meshgrid((np.arange(cells[0])+.5)/cells[0], (np.arange(cells[1])+.5)/cells[1])
    rho = 1+.03*np.cos(2*np.pi*x)
    vx = .15+.03*np.sin(2*np.pi*x)
    vy = .02*np.cos(2*np.pi*x)
    temperature = 1+.08*np.cos(2*np.pi*x)
    if two_dimensional:
        rho += .02*np.sin(2*np.pi*y)
        vx += .02*np.sin(2*np.pi*y)
        vy += .03*np.cos(2*np.pi*y)
        temperature += .04*np.sin(2*np.pi*(x+y))
    initial = np.asarray((rho, rho*vx, rho*vy, rho*temperature/.4+.5*rho*(vx*vx+vy*vy)))
    base = _periodic_metrics(initial, components=(0,1,2,3))
    def check(runtime):
        metrics = base(runtime)
        actual = state_array(runtime, "fluid", initial.shape)
        if actual is None: return metrics
        internal = actual[3]-(actual[1]**2+actual[2]**2)/(2*actual[0])
        metrics.update(min_density=float(np.min(actual[0])), min_internal_energy=float(np.min(internal)))
        metrics["passed"] &= metrics["min_density"] > 0 and metrics["min_internal_energy"] > 0
        return metrics
    subject = block[U]
    profile = ("M05: full compressible ideal-gas NS discrete balance, "
        + ("2D" if two_dimensional else "1D planar data in Dim2")
        + ", variable viscosity, Newtonian stress incl. cross-gradients and viscous work, Fourier heat; "
        "shared Rusanov/centered face flux, periodic Uniform/SSPRK2. The face algebra is the discrete equation authority; continuum convergence and general positivity are not qualified.")
    return CaseSpec(case, layout, _initial(case,subject,initial), {"fluid":subject}, end,dt,check,profile)


def _stefan_lambda():
    # Classical one-phase Neumann solution, Ste=1. Numerical input/oracle only.
    low,high=0.,2.
    for _ in range(80):
        mid=(low+high)/2
        if mid*reference_math.exp(mid*mid)*reference_math.erf(mid)<1/reference_math.sqrt(reference_math.pi):
            low=mid
        else: high=mid
    return (low+high)/2


def stefan_cell_means(nx,time):
    """Exact enthalpy averages, latent=capacity=conductivity=hot T=1,Tm=0.

    Neumann solution s=2 lambda sqrt(t), lambda exp(lambda²)erf(lambda)=1/sqrt(pi).
    Source: Sadoun et al., JAFM 5(1), 2012, section 2, pp.106--107,
    https://www.jafmonline.net/article_1298_a51fb975227d6640e4fe47854476d133.pdf
    The chosen enthalpy FV method is NOT the paper's front-tracking method.
    """
    lam=_stefan_lambda();scale=2*reference_math.sqrt(time);front=lam*scale
    def primitive(x):
        return x*reference_math.erf(x/scale)+scale/reference_math.sqrt(reference_math.pi)*reference_math.exp(-(x/scale)**2)
    values=[]
    for i in range(nx):
        a,b=i/nx,min((i+1)/nx,front)
        values.append(0. if b<=a else nx*(2*(b-a)-(primitive(b)-primitive(a))/reference_math.erf(lam)))
    return np.asarray(values),front


def build_m06(cells=(24,8)):
    frame,layout=_domain("M06_Stefan_slab",cells,periodic=(1,))
    model=pops.Model("M06_enthalpy_conduction",frame=frame);U=model.state("enthalpy",components=("H",));H=U[0]
    temperature=pm.where(H<0,lambda:H,lambda:pm.where(H<=1,lambda:0.,lambda:H-1))
    boundaries=tuple(DiffusiveBoundary(axis,side,"periodic") if axis==1 else
        DiffusiveBoundary(axis,side,"value",1. if side=="lower" else 0.)
        for axis in range(2) for side in ("lower","upper"))
    flux=model.diffusive_flux("Fourier_heat_flux",state=U,value=pm.CoeffGradient(temperature,((1.,0.),(0.,0.))),boundaries=boundaries)
    rate=model.rate("enthalpy_balance",equation=pm.ddt(U)==pm.div(flux))
    case=pops.Case("M06_Stefan_moving_front");block=case.block("material",model)
    plan=DiscretizationPlan();plan.rates.add(rate,Diffusion(flux=flux));case.numerics(plan,block=block)
    dt=.15/cells[0]**2;end=8*dt;start=.1
    _,heat=_ssprk2(case,block,U,lambda p,u,point,label:rate(u),dt,name="M06_SSPRK2",trace_quantity=True)
    line,_=stefan_cell_means(cells[0],start);initial=np.broadcast_to(line,(1,cells[1],cells[0])).copy()
    exact_line,front=stefan_cell_means(cells[0],start+end)
    def check(runtime):
        actual=state_array(runtime,"material",initial.shape)
        exchanged=runtime.integral_state(heat)
        if actual is None:return non_owner_metrics()
        gain=float(np.mean(actual-initial));phase=np.clip(actual,0,1)
        front_error=abs(float(np.mean(phase))-front)
        energy_error=abs(gain-exchanged)
        error=float(np.mean(abs(actual-exact_line)))
        return {"metrics_owner":True,"enthalpy_gain":gain,"boundary_energy_defect":energy_error,
                "enthalpy_l1":error,"front_error":front_error,"exact_front":front,
                "passed":bool(np.isfinite(actual).all() and np.min(actual)>=-1.e-12 and gain>0
                               and energy_error<2.e-10 and error<2.5/cells[0] and front_error<2.5/cells[0])}
    subject=block[U]
    return CaseSpec(case,layout,_initial(case,subject,initial),{"material":subject},end,dt,check,
        "M06: one-phase Stefan melting front, H=T+fraction,L=c=k=1, latent plateau retained, heat Dirichlet traces, exact enthalpy initial cell averages at physical t=.1; genuine x conduction on a Dim2 y-invariant mesh, SSPRK2 and shared boundary heat ledger")


def build_m07(cells=(24,8)):
    frame,layout=_domain("M07_periodic_lake",cells)
    model=pops.Model("M07_dry_safe_Saint_Venant",frame=frame);U=model.state("lake",components=("h","q","z"));h,q,z=U
    velocity=pm.where(h>0,lambda:q/h,lambda:0.)
    flux=model.flux("shallow_water",state=U,frame=frame,
        components={frame.axes[0]:(q,q*velocity+h*h/2,0.),frame.axes[1]:(0.,0.,0.)})
    product=model.nonconservative_product("bed",state=U,
        matrices={frame.axes[0]:((0.,0.,0.),(0.,0.,h),(0.,0.,0.)),
                  frame.axes[1]:((0.,0.,0.),)*3},conservative_components=("h","z"))
    rate=model.rate("Saint_Venant_balance",equation=pm.ddt(U)==-pm.div(flux)-product)
    model.primitive_state(*U,conservative=tuple(U))
    dry_valid=pm.where(h>0,lambda:1.,lambda:pm.where(q==0,lambda:1.,lambda:0.))
    model.recovery_admissibility(h=h>=0,q=dry_valid>0)
    def body(left,right,axis):
        if axis==1:return FaceBalance((0.,0.,0.),(0.,0.,0.),(0.,0.,0.),0.)
        hl,ql,zl=left;hr,qr,zr=right
        ul=pm.where(hl>0,lambda:ql/hl,lambda:0.);ur=pm.where(hr>0,lambda:qr/hr,lambda:0.)
        crest=pm.maximum(zl,zr);a=pm.maximum(0.,hl+zl-crest);b=pm.maximum(0.,hr+zr-crest)
        speed=pm.maximum(abs(ul)+pm.sqrt(pm.maximum(hl,0.)),abs(ur)+pm.sqrt(pm.maximum(hr,0.)))
        fl=(a*ul,a*ul*ul+a*a/2);fr=(b*ur,b*ur*ur+b*b/2)
        shared=tuple((fl[i]+fr[i])/2-speed*((b,b*ur)[i]-(a,a*ul)[i])/2 for i in range(2))+(0.,)
        return FaceBalance(shared,(0.,-(hl*hl-a*a)/2,0.),(0.,(hr*hr-b*b)/2,0.),speed)
    face=CoordinatedFace(flux=flux,product=product,frame=frame,body=body)
    case=pops.Case("M07_wet_dry_lake");block=case.block("fluid",model)
    plan=DiscretizationPlan();plan.rates.add(rate,CoordinatedFiniteVolume(face=face));case.numerics(plan,block=block)
    dt,end=.02/max(cells),.04/max(cells)
    _ssprk2(case,block,U,lambda p,u,point,label:rate(u),dt,name="M07_SSPRK2")
    x=(np.arange(cells[0])+.5)/cells[0];bed=.4+.4*np.cos(2*np.pi*x);height=np.maximum(.6-bed,0.)
    initial=np.stack([np.broadcast_to(v,(cells[1],cells[0])) for v in (height,np.zeros_like(x),bed)])
    def check(runtime):
        actual=state_array(runtime,"fluid",initial.shape)
        if actual is None:return non_owner_metrics()
        defect=float(np.max(abs(actual-initial)))
        return {"metrics_owner":True,"equilibrium_defect":defect,"dry_cells":int(np.count_nonzero(initial[0]==0)),
                "minimum_depth":float(np.min(actual[0])),"passed":bool(np.isfinite(actual).all() and defect<3.e-11 and np.min(actual[0])>=-1.e-12)}
    subject=block[U]
    return CaseSpec(case,layout,_initial(case,subject,initial),{"fluid":subject},end,dt,check,
        "M07: Saint-Venant x transport with frozen bed, periodic Dim2 embedding, wet/dry lake at rest; explicit dry branch and hydrostatic Rusanov shared face plus two bed-source deliveries; no epsilon floor")


def _poisson(case,phi,rhs):
    problem=FieldProblem("live_Poisson",unknowns=(phi,),equations=(-pm.laplacian(phi)==rhs,),
        boundaries=(FieldBoundary(phi,bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(),bcs.Periodic())),),
        gauge=SharedMeanGauge((phi,)))
    return case.field(problem,FieldDiscretization(method=CellCenteredSecondOrder(),boundaries=(),
        solver=CG(max_iter=2000,rel_tol=1.e-12,abs_tol=1.e-13)))


def _live_field_evaluator(field,phi,model,block,U,rate,*,potential_target=None):
    module=model.module;carrier=block[module.field_handle(module.field_spaces()["fields"])]
    def evaluate(program,state,point,label):
        observation=field.observe(program.solve(field,values={block[U]:state},at=point).consume(action=FailRun()))
        potential=observation[field[phi]]
        if potential_target is None:
            gradient=observation.gradient(field[phi],dimension=2)
            context=observation.publish({(carrier,"potential_grad_x"):(gradient,0),
                                         (carrier,"potential_grad_y"):(gradient,1)})
        else:
            context=observation.publish({(carrier,potential_target):potential})
        program.store_history(label+"_state",state,depth=1)
        program.store_history(label+"_potential",potential,depth=1)
        return rate(state,context)
    return evaluate


def _field_metrics(initial,cells,lengths,*,rhs_offset,minimum=None):
    base=_periodic_metrics(initial,minimum=minimum)
    def check(runtime):
        out=base(runtime)
        state=np.asarray(runtime.history_global("stage1_state",1)) if out.get("metrics_owner") else runtime.history_global("stage1_state",1)
        phi=runtime.history_global("stage1_potential",1)
        if not out.get("metrics_owner"):return out
        state=np.asarray(state).reshape(initial.shape);phi=np.asarray(phi).reshape(cells[1],cells[0])
        positive_lap=sum((2*phi-np.roll(phi,1,axis=a)-np.roll(phi,-1,axis=a))/(lengths[1-a]/cells[1-a])**2 for a in (0,1))
        defect=float(np.max(abs(positive_lap-(state[0]-rhs_offset))))
        out["stage1_Poisson_residual"]=defect;out["passed"]=bool(out["passed"] and defect<2.e-8)
        return out
    return check


def build_m08(cells=(16,16)):
    lengths=(2*np.pi,2*np.pi);frame,layout=_domain("M08_periodic",cells,lengths=lengths)
    model=pops.Model("M08_guiding_center",frame=frame);U=model.state("density",components=("q",));phi=model.field("potential")
    gradient=model.vector("potential_gradient",frame=frame,
        components={frame.axes[0]:pm.grad(phi).x,frame.axes[1]:pm.grad(phi).y})
    gx,gy=gradient.x,gradient.y
    flux=model.flux("guiding_flux",state=U,frame=frame,
        components={frame.axes[0]:(-gy*U[0],),frame.axes[1]:(gx*U[0],)},
        waves={frame.axes[0]:(-gy,),frame.axes[1]:(gx,)})
    rate=model.rate("density_transport",equation=pm.ddt(U)==-pm.div(flux))
    case=pops.Case("M08_live_guiding_center");block=case.block("fluid",model)
    plan=DiscretizationPlan();plan.rates.add(rate,_fv(flux,U));case.numerics(plan,block=block)
    field=_poisson(case,phi,U[0]);dt,end=.002,.004
    _ssprk2(case,block,U,_live_field_evaluator(field,phi,model,block,U,rate),dt,name="M08_SSPRK2")
    x,y=np.meshgrid(2*np.pi*(np.arange(cells[0])+.5)/cells[0],2*np.pi*(np.arange(cells[1])+.5)/cells[1])
    initial=(np.sinc(1/cells[0])*np.sin(x)+.3*np.sinc(2/cells[1])*np.cos(2*y))[None];subject=block[U]
    return CaseSpec(case,layout,_initial(case,subject,initial),{"fluid":subject},end,dt,
        _field_metrics(initial,cells,lengths,rhs_offset=0.),
        "M08: 2D periodic guiding center q_t+div(q*(-phi_y,phi_x))=0,-Delta phi=q; q is a signed mean-zero density/charge fluctuation, not a received nonnegative total density; mean-zero shared gauge, fresh Poisson at both SSPRK2 stages, retained stage residual")


def build_m09(cells=(16,16)):
    lengths=(2*np.pi,2*np.pi);frame,layout=_domain("M09_periodic",cells,lengths=lengths)
    model=pops.Model("M09_Euler_Poisson_magnetic",frame=frame)
    U,flux,rate,phi=_euler(model,frame,gamma=1.4,electromagnetic=True)
    case=pops.Case("M09_complete_Euler_Poisson_PDE");block=case.block("fluid",model)
    plan=DiscretizationPlan();plan.rates.add(rate,_fv(flux,U));case.numerics(plan,block=block)
    field=_poisson(case,phi,U[0]-1);dt,end=.001,.002
    _ssprk2(case,block,U,_live_field_evaluator(field,phi,model,block,U,rate),dt,name="M09_SSPRK2")
    x,y=np.meshgrid(2*np.pi*(np.arange(cells[0])+.5)/cells[0],2*np.pi*(np.arange(cells[1])+.5)/cells[1])
    rho=1+.1*np.sinc(1/cells[0])*np.cos(x)+.05*np.sinc(1/cells[1])*np.sin(y)
    initial=np.stack((rho,.1*rho,.05*rho,2.5+.00625*rho));subject=block[U]
    base=_field_metrics(initial,cells,lengths,rhs_offset=1.,minimum=.8)
    def check(runtime):
        out=base(runtime);actual=state_array(runtime,"fluid",initial.shape)
        if actual is not None:
            internal=actual[3]-(actual[1]**2+actual[2]**2)/(2*actual[0])
            out["minimum_internal_energy"]=float(np.min(internal));out["passed"]=bool(out["passed"] and np.min(internal)>0)
        return out
    return CaseSpec(case,layout,_initial(case,subject,initial),{"fluid":subject},end,dt,check,
        "M09: compressible 2D ideal Euler-Poisson, fixed neutralizing background=1, alpha=1,Bz=.3, full Lorentz and electric work, live stage fields; explicit FV/SSPRK2, NOT the Hoffart midpoint/Schur method or an exact total-energy certificate")


def build_m10(cells=(16,8)):
    lengths=(2*np.pi,2*np.pi);frame,layout=_domain("M10_periodic",cells,lengths=lengths)
    model=pops.Model("M10_Nernst_Planck_Poisson",frame=frame);U=model.state("density",components=("n",))
    potential=model.aux("potential");phi=model.field("solved_potential");D=.1
    drift=model.drift_flux("electromigration",state=U,mobility=D,potential=potential)
    diffusion=model.diffusive_flux("Fick",state=U,value=D*pm.grad(U))
    rate=model.rate("number_balance",equation=pm.ddt(U)==-pm.div(drift)+pm.div(diffusion))
    case=pops.Case("M10_live_Nernst_Planck_Poisson");block=case.block("fluid",model)
    plan=DiscretizationPlan();plan.rates.add(rate,ScharfetterGummel(drift=drift,flux=diffusion));case.numerics(plan,block=block)
    field=_poisson(case,phi,U[0]-1);dt,end=.002,.004
    _ssprk2(case,block,U,_live_field_evaluator(field,phi,model,block,U,rate,potential_target="potential"),dt,name="M10_SSPRK2")
    x=(np.arange(cells[0])+.5)*2*np.pi/cells[0]
    initial=np.broadcast_to(1+.1*np.sinc(1/cells[0])*np.cos(x),(1,cells[1],cells[0])).copy();subject=block[U]
    return CaseSpec(case,layout,_initial(case,subject,initial),{"fluid":subject},end,dt,
        _field_metrics(initial,cells,lengths,rhs_offset=1.,minimum=.85),
        "M10: periodic Nernst-Planck-Poisson with D=mobility=.1 and fixed background1; joint Scharfetter-Gummel face, live Poisson at each SSPRK2 stage, nonuniform positive density on genuine Dim2 mesh")


def maxwell_stefan_face():
    """All face physics in Python; no Maxwell--Stefan branch in the engine."""
    from pops.numerics.stencil_flux import StencilFlux
    from .maxwell_stefan_equations import ternary_face
    def body(sample, spacing, axis):
        flux, residual, frequency = ternary_face(
            sample, lambda offset: sample(offset, 0, field=0), spacing, axis)
        left = tuple(-int(d == axis) for d in range(2))
        minimum = sample(left, 0)
        for offset in (left, (0, 0)):
            for i in range(3):
                minimum = pm.minimum(minimum, sample(offset, i))
        return flux, pm.where(minimum > 0, lambda: frequency, lambda: -1.)
    return StencilFlux(dimension=2, components=3, radius=1,
                       field_components=(1,), body=body)


def build_m11(cells=(12, 8)):
    """Ternary Poisson--Maxwell--Stefan, normalized mass density one.

    S09 equations (19),(22),(24); distinct molar masses (1,2,4), charges
    (1,-1,0), symmetric binary diffusivities (1,.8,1.2). The per-face 2x2
    elimination enforces sum(J)=0 exactly. Centered FV + SSPRK2 is the chosen
    method; the article's FE entropy/positivity theorem is not inherited.
    """
    from pops.model import FieldSpace, PhysicalDimension, PhysicalSupport
    from pops.numerics import StateStorage
    lengths=(2*np.pi, 2*np.pi)
    frame,layout=_domain("M11_periodic",cells,lengths=lengths)
    support=PhysicalSupport((("x", "M11_periodic"), ("y", "M11_periodic")))
    unit=PhysicalDimension()
    model=pops.Model("M11_ternary_Maxwell_Stefan",frame=frame)
    U=model.state("mass_fractions",components=("rho0","rho1","rho2"),
                  support=support,sampling="cell_average",units=(unit,)*3)
    phi=model.field("electric_potential")
    space=model.module.state_spaces()["mass_fractions"]
    potential_space=FieldSpace("electric_potential",components=("phi",),
        representation="field",layout="cell",centering="cell",sampling="cell",
        frame=space.frame,clock=space.clock,support=support,units=(unit,))
    problem=FieldProblem("M11_Poisson",unknowns=(phi,),
        equations=(-pm.laplacian(phi)==U[0]-U[1]/2,),
        boundaries=(FieldBoundary(phi,bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(),bcs.Periodic())),),
        gauge=SharedMeanGauge((phi,)),unknown_spaces={phi:potential_space},coordinate_units=(unit,unit))
    case=pops.Case("M11_complete_ternary_Poisson_Maxwell_Stefan")
    block=case.block("fluid",model)
    flux=maxwell_stefan_face()
    plan=DiscretizationPlan();plan.states.add(U,StateStorage(ghost_depth=max(flux.ghost_extent)))
    case.numerics(plan,block=block)
    field=case.field(problem,FieldDiscretization(method=CellCenteredSecondOrder(),boundaries=(),observation_axes=(0,1),
        solver=CG(max_iter=2000,rel_tol=1.e-12,abs_tol=1.e-13)))
    dt=2.e-5*(12/max(cells))**2;end=2*dt
    def evaluate(program,state,point,label):
        observation=field.observe(program.solve(field,values={block[U]:state},at=point).consume(action=FailRun()))
        potential=observation[field[phi]]
        program.store_history(label+"_state",state,depth=1)
        program.store_history(label+"_potential",potential,depth=1)
        return program.stencil_flux_divergence(label+"_mass_flux",state,flux=flux,fields=(potential,))
    _ssprk2(case,block,U,evaluate,dt,name="M11_SSPRK2")
    x,y=np.meshgrid((np.arange(cells[0])+.5)*lengths[0]/cells[0],
                    (np.arange(cells[1])+.5)*lengths[1]/cells[1])
    rho0=.2+.01*np.sinc(1/cells[0])*np.cos(x)
    rho1=.4+.015*np.sinc(1/cells[1])*np.sin(y)
    initial=np.stack((rho0,rho1,1-rho0-rho1));subject=block[U]
    from .reference_coupled_fields import ssprk2, maxwell_stefan_rhs
    expected=ssprk2(initial,lambda value:maxwell_stefan_rhs(value,lengths),dt)
    base=_periodic_metrics(initial,components=(0,1,2),minimum=.15)
    def check(runtime):
        out=base(runtime)
        stage=runtime.history_global("stage1_state",1)
        phi_values=runtime.history_global("stage1_potential",1)
        actual=state_array(runtime,"fluid",initial.shape)
        if actual is None:return out
        stage=np.asarray(stage).reshape(initial.shape)
        potential=np.asarray(phi_values).reshape(cells[1],cells[0])
        lap=sum((2*potential-np.roll(potential,1,axis=a)-np.roll(potential,-1,axis=a))
                /(lengths[1-a]/cells[1-a])**2 for a in (0,1))
        residual=float(np.max(abs(lap-(stage[0]-stage[1]/2))))
        total_defect=float(np.max(abs(np.sum(actual,axis=0)-1)))
        reference_error=float(np.max(abs(actual-expected)))
        out.update(stage1_Poisson_residual=residual,pointwise_total_mass_defect=total_defect,
                   full_state_reference_error=reference_error)
        out["passed"]=bool(out["passed"] and residual<2.e-8 and total_defect<2.e-12 and reference_error<2.e-10)
        return out
    return CaseSpec(case,layout,_initial(case,subject,initial),{"fluid":subject},end,dt,check,
        "M11: 2D periodic ternary Poisson-Maxwell-Stefan, distinct molar masses and charges; exact face mass constraint, centered friction elimination with live typed Poisson at both SSPRK2 stages; positive short-time reception, no inherited FE entropy theorem")


def build_m12(**kwargs):
    from .models_M12 import build_m12 as builder
    return builder(**kwargs)


def build_m13(**kwargs):
    from .models_M13 import build_m13 as builder
    return builder(**kwargs)


def build_m14(**kwargs):
    from .models_M14 import build_m14 as builder
    return builder(**kwargs)


_MISSING={}


def build_model(model, *, require_complete=False, **kwargs):
    """Build a real native case or fail explicitly before inventing a substitute."""
    if model in _MISSING:
        required,profile=_MISSING[model]
        raise MissingCapabilityError(model,required=required,profile=profile)
    if model not in READY:
        raise ValueError(f"unknown model id {model!r}")
    if require_complete:
        from .coverage import OBLIGATIONS
        required = OBLIGATIONS[model]
        if required:
            raise MissingCapabilityError(model, required=required,
                profile="all scientific obligations of the corpus model family")
    return globals()["build_"+model.lower()](**kwargs)
