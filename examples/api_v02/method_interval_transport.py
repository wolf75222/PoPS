"""Spatial BGK characteristic interval, authored entirely as scalar equations.

Two discrete velocities, periodic uniform mesh, centered linear reconstruction,
and a frozen density equilibrium. The face predictor integrates the exact
Duhamel solution for these reconstructions; the volume collision uses the exact
frozen-equilibrium increment. This precisely specified first-order time update
does not claim the full nonlinear Maxwellian UGKS/AP scheme of a paper.
"""
from __future__ import annotations

import numpy as np
import pops
from pops import math as pm
from pops.numerics import DiscretizationPlan, StateStorage
from pops.numerics.stencil_flux import StencilFlux
from pops.realization import Realize, Use
from pops.time import ComputedDt, FixedDt, RejectAttempt
from pops.time.expressions import ProgramScalar
from .common import CaseSpec, non_owner_metrics, state_array
from .models_M01_M14 import _domain, _initial


def _quadrature_step(q, h, dx, tau, *, flux_duration=None):
    """Independent nested quadrature of the space/time characteristics."""
    interval=h if flux_duration is None else flux_duration
    nodes,weights=np.polynomial.legendre.leggauss(14)
    nodes=(nodes+1)/2;weights=weights/2
    slope=(np.roll(q,-1,axis=-1)-np.roll(q,1,axis=-1))/(2*dx)
    face=[];initial=[];gradients=[]
    for c,v in enumerate((1.,-1.)):
        donor=np.roll(q[c],1,axis=-1) if v>0 else q[c]
        gradient=np.roll(slope[c],1,axis=-1) if v>0 else slope[c]
        initial.append(donor+v*.5*dx*gradient);gradients.append(gradient)
    equilibrium=.5*(initial[0]+initial[1]);gradient_g=.5*(gradients[0]+gradients[1])
    for c,v in enumerate((1.,-1.)):
        integral=np.zeros_like(q[c])
        for t,wt in zip(interval*nodes,interval*weights,strict=True):
            value=np.exp(-t/tau)*(initial[c]-v*t*gradients[c])
            for s,ws in zip(t*nodes,t*weights,strict=True):
                value+=ws/tau*np.exp(-(t-s)/tau)*(equilibrium-v*(t-s)*gradient_g)
            integral+=wt*v*value
        face.append(integral/interval)
    face=np.stack(face)
    transport=h/dx*(face-np.roll(face,-1,axis=-1))
    equilibrium_cell=.5*q.sum(axis=0)
    collision=-np.expm1(-h/tau)*(equilibrium_cell[None]-q)
    return q+transport+collision


def build_interval_transport(*, cells=(32,4), dt=.01, steps=3, retry=False):
    if not 0<dt<=.01 or type(steps) is not int or steps<1:
        raise ValueError("reception requires 0<dt<=.01 and positive steps")
    tau=.03
    frame,layout=_domain("T09_characteristic_interval",cells)
    model=pops.Model("two_velocity_BGK",frame=frame)
    U=model.state("f",components=("positive","negative"))
    transport=model.flux("free_flight",state=U,frame=frame,
        components={frame.axes[0]:(U[0],-U[1]),frame.axes[1]:(0.,0.)},
        waves={frame.axes[0]:(-1.,1.),frame.axes[1]:(0.,0.)})
    equilibrium=.5*(U[0]+U[1])
    collision=model.source("BGK",on=U,value=((equilibrium-U[0])/tau,(equilibrium-U[1])/tau))
    equation=model.rate("kinetic_balance",equation=pm.ddt(U)==-pm.div(transport)+collision)
    case=pops.Case("T09_spatial_Duhamel");block=case.block("method",model)

    def face(sample,spacing,axis,h):
        if axis==1:return (0.,0.),0.
        dx=spacing[0]
        initial=[];slopes=[]
        for c,v in enumerate((1.,-1.)):
            offset=-1 if v>0 else 0
            slope=(sample((offset+1,0),c)-sample((offset-1,0),c))/(2*dx)
            initial.append(sample((offset,0),c)+v*.5*dx*slope);slopes.append(slope)
        g=.5*(initial[0]+initial[1]);sg=.5*(slopes[0]+slopes[1])
        e=pm.exp(-h/tau);one_minus_e=-pm.expm1(-h/tau)
        A=tau*one_minus_e
        B=tau*tau*one_minus_e-tau*h*e
        C=h-A
        D=tau*h-2*tau*tau*one_minus_e+tau*h*e
        return tuple(v*(A*initial[c]-v*B*slopes[c]+C*g-v*D*sg)/h
                     for c,v in enumerate((1.,-1.))),1/dx

    flux=StencilFlux(dimension=2,components=2,radius=2,parameters=("duration",),body=face)
    p=pops.Program("characteristic_interval_equations");q=p.state(block[U]);h=p.requested_dt()
    face_use=Use(equation.occurrences[0],"mesh_face","whole_interval","inventory","transport")
    volume_use=Use(equation.occurrences[1],"cell","whole_interval","inventory","collision")
    from pops.numerics import ProgramRealization
    plan=DiscretizationPlan();plan.rates.add(equation,ProgramRealization(
        uses=(face_use,volume_use),ghost_depth=max(flux.ghost_extent)))
    case.numerics(plan,block=block)

    def body(builder,old,duration):
        average=builder.stencil_flux_divergence("characteristic_average",old,flux=flux,parameters=(duration,))
        factor=-pm.expm1(-ProgramScalar(duration)/tau)/ProgramScalar(duration)
        g=.5*(old[0]+old[1])
        volume=builder.rate("collision_interval_average",(factor*(g-old[0]),factor*(g-old[1])),of=old)
        return {"face":average,"collision":volume}

    realized=Realize("joint_characteristic_and_volume",owns=(face_use,volume_use),reads=(volume_use,),
        body=body,bindings={"old":q.n,"duration":h},outputs={face_use:("face",),volume_use:("collision",)}).evaluate(p)
    candidate=p.value("conservative_interval",q.n+p.dt*realized[face_use]+p.dt*realized[volume_use],at=q.next.point)
    if retry:
        candidate=p.guard("rebuild_interval_after_retry",candidate,h<=dt*(1+1.e-12),action=RejectAttempt())
        p.reached_duration(h)
    p.commit(q.next,candidate)
    p.step_strategy(ComputedDt(2*dt,max_rejections=2,endpoint_ulps=4) if retry else FixedDt(dt))
    case.program(p)
    x,y=np.meshgrid((np.arange(cells[0])+.5)/cells[0],(np.arange(cells[1])+.5)/cells[1])
    initial=np.stack((1+.1*np.sin(2*np.pi*x)+.02*np.cos(4*np.pi*x),
                      .7+.08*np.cos(2*np.pi*x)))
    expected=initial.copy();frozen=initial.copy()
    for _ in range(steps):
        expected=_quadrature_step(expected,dt,1/cells[0],tau)
        frozen=_quadrature_step(frozen,dt,1/cells[0],tau,flux_duration=2*dt)

    def check(runtime):
        actual=state_array(runtime,"method",initial.shape)
        if actual is None:return non_owner_metrics()
        error=float(np.max(abs(actual-expected)));mass=float(abs(actual.sum()-initial.sum())/np.prod(cells))
        stale=float(np.max(abs(expected-frozen)))
        return dict(passed=bool(error<3.e-13 and mass<3.e-14 and stale>1.e-5 and np.min(actual)>0),
            metrics_owner=True,full_state_quadrature_error=error,mass_defect=mass,
            reused_wrong_interval_error=stale,minimum=float(np.min(actual)))
    return CaseSpec(case,layout,_initial(case,block[U],initial),{"method":block[U]},steps*dt,dt,check,
        "T09: two-velocity spatial BGK; linear characteristic reconstruction, exact Duhamel face time integral, frozen-equilibrium volume collision; shared native face ledger; duration is native SSA; nonlinear Maxwellian/AP UGKS and arbitrary ADER order unqualified")
