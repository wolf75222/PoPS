"""T12: a posteriori replacement of a true shared Cartesian mesh-face flux.

The scalar positivity detector is a finite reception profile, not a theorem for
arbitrary data or a complete iterative MOOD method. Every numerical equation is
authored below; the core sees sampled expressions and their exact SSA inputs.
"""
from __future__ import annotations

import numpy as np
import pops
from pops import math as pm
from pops.fields import CellMidpoint
from pops.model import PhysicalDimension
from pops.numerics import DiscretizationPlan, ProgramRealization
from pops.numerics.stencil_flux import StencilFlux
from pops.realization import Realize, Use
from pops.time import FixedDt
from .common import CaseSpec, non_owner_metrics, state_array
from .models_M01_M14 import _domain, _initial


def reference_step(initial, *, dt, dx, velocity=1.):
    """Independent NumPy face arrays: face j separates cells j-1 and j."""
    old = np.asarray(initial,dtype=float)
    left = np.roll(old,1,axis=-1)
    high = velocity*((left+old)/2-velocity*dt/(2*dx)*(old-left))
    low = velocity*left
    divergence = lambda faces: (np.roll(faces,-1,axis=-1)-faces)/dx
    provisional = old-dt*divergence(high)
    bad = provisional < 0
    bad_incident = np.roll(bad,1,axis=-1) | bad
    selected = np.where(bad_incident,low,high)
    corrected = old-dt*divergence(selected)
    # Counterexample: replacing a cell update alone fails to deliver the opposite
    # share of that same face transfer to its neighbor.
    unilateral = np.where(bad,old-dt*divergence(low),provisional)
    only_left_incident = old-dt*divergence(np.where(np.roll(bad,1,axis=-1),low,high))
    return dict(corrected=corrected,provisional=provisional,bad=bad,faces=selected,
                replaced=bad_incident,unilateral=unilateral,one_incident=only_left_incident)


def authored_flux(*, corrected, velocity=1.):
    def body(sample,spacing,axis,duration):
        if axis != 0:
            return (0.,),0.
        left,right = sample((-1,0),0),sample((0,0),0)
        high = velocity*((left+right)/2-velocity*duration/(2*spacing[0])*(right-left))
        if corrected:
            # BOTH incident candidates decide one face value. The backend applies
            # its oriented divergence to both cells and retains that exact face.
            bad = (sample((-1,0),0,field=0) < 0.) | (sample((0,0),0,field=0) < 0.)
            high = pm.where(bad,lambda:velocity*left,lambda:high)
        return (high,),abs(velocity)/spacing[0]
    return StencilFlux(dimension=2,components=1,radius=1,body=body,
                       field_components=(1,) if corrected else (),parameters=("duration",))


def build_t12_meshface(*, cells=(32,4), cfl=.4):
    if cells[0] < 16 or not 0 < cfl <= .5:
        raise ValueError("T12 profile requires at least 16 x cells and 0 < CFL <= 1/2")
    frame,layout = _domain("T12_face_domain",cells)
    model = pops.Model("T12_authored_advection",frame=frame)
    U = model.state("amount",components=("q",),sampling="cell_average",units=(PhysicalDimension(),))
    flux = model.flux("physical_transport",state=U,frame=frame,
        components={frame.axes[0]:(U[0],),frame.axes[1]:(0.,)},
        waves={frame.axes[0]:(1.,),frame.axes[1]:(0.,)})
    rate = model.rate("conservation",equation=pm.ddt(U)==-pm.div(flux))
    use = Use(rate.occurrences[0],"mesh_face","accepted_step","cell_average")
    case = pops.Case("T12_shared_mesh_face")
    block = case.block("method",model,states=(U,))
    plan = DiscretizationPlan()
    plan.rates.add(rate,ProgramRealization(uses=(use,),ghost_depth=1))
    case.numerics(plan,block=block)
    p = pops.Program("T12_predict_detect_replace")
    q = p.state(block[U]); old=q.n
    duration = p.requested_dt()
    high = p.stencil_flux_divergence("provisional_high_flux",old,flux=authored_flux(corrected=False),
                                   parameters=(duration,))
    provisional = p.value("provisional_high_state",old+p.dt*high,at=q.next.point)
    candidate_sample = p.cell_field(provisional,quadrature=CellMidpoint(),name="candidate_samples")
    selected_flux = authored_flux(corrected=True)
    selected = Realize("accepted_face_replacement",owns=(use,),
        body=lambda b,old,samples,h:b.stencil_flux_divergence("selected_face_flux",old,flux=selected_flux,
            fields=(samples,),parameters=(h,)),
        bindings={"old":old,"samples":candidate_sample,"h":duration},outputs={use:()}).evaluate(p)[use]
    p.commit(q.next,p.value("accepted",old+p.dt*selected,at=q.next.point))
    dx=1/cells[0];dt=cfl*dx
    p.step_strategy(FixedDt(dt));case.program(p)
    initial=np.full((1,cells[1],cells[0]),.15)
    initial[...,6:12]=np.array([.15,.06,.005,1.,.65,.3])
    reference=reference_step(initial,dt=dt,dx=dx)
    def check(runtime):
        actual=state_array(runtime,"method",initial.shape)
        if actual is None:
            return non_owner_metrics()
        error=float(np.max(np.abs(actual-reference["corrected"])))
        mass=float(abs(np.mean(actual)-np.mean(initial)))
        defect=float(abs(np.mean(reference["unilateral"])-np.mean(initial)))
        one_gap=float(np.max(np.abs(reference["one_incident"]-reference["corrected"])))
        metrics=dict(reference_error=error,mass_defect=mass,minimum=float(actual.min()),
            provisional_minimum=float(reference["provisional"].min()),
            troubled_cells=int(np.count_nonzero(reference["bad"])),
            replaced_faces=int(np.count_nonzero(reference["replaced"])),
            unilateral_mass_defect=defect,missing_incident_gap=one_gap,metrics_owner=True)
        metrics["passed"]=bool(np.isfinite(actual).all() and error<3.e-13 and mass<3.e-14
            and actual.min()>=0 and reference["provisional"].min()<0 and defect>1.e-5 and one_gap>1.e-4)
        return metrics
    return CaseSpec(case,layout,_initial(case,block[U],initial),{"method":block[U]},dt,dt,check,
        "T12: native periodic Cartesian faces, Lax-Wendroff provisional SSA, two-incident-cell positivity detector, "
        "one selected upwind/high flux delivered to both recipients, only final face ledger accepted; "
        "one-pass scalar reception, general iterative MOOD/AMR and dependent field re-solves unqualified")
