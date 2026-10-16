"""Declared constant-coefficient local-field streamer with nonlocal radiation.

This is a normalized low-ionization, neutral-excess profile of the balance
structure in S11, https://arxiv.org/html/1011.2871v1, equations (1a)--(1c).
The impact frequency is prescribed, not a fitted air Townsend coefficient.
Photoionization uses one declared plane-parallel Helmholtz absorption group,
not the exact Zhelezniak absorption function. No electron-energy equation or
closed total-energy claim is made by this local-field approximation.
"""
from __future__ import annotations

import math
import numpy as np
import pops
from pops import math as pm
from pops.fields import (FieldProblem, FieldBoundary, FieldDiscretization,
    CellCenteredSecondOrder, SharedMeanGauge, bcs)
from pops.model import PhysicalDimension, PhysicalSupport, FieldSpace
from pops.numerics import DiscretizationPlan, StateStorage
from pops.numerics.stencil_flux import StencilFlux
from pops.solvers import CG
from pops.time import FailRun
from .common import CaseSpec, state_array, non_owner_metrics
from .models_M01_M14 import _domain, _initial, _ssprk2


def build_m13(cells=(16, 8), *, photo_frequency=.03):
    """ne_t=-div(je)+I+P, ni_t=I+P, n0_t=-I-P, -Lap(phi)=ni-ne.

    je=mu*ne*grad(phi)-mu*ne*E0-D*grad(ne), I=nu*ne,
    P=nu_photo*G_lambda*ne. I and P create one electron/ion and consume one
    neutral with a single stoichiometric source. Fixed frequencies represent
    the undepleted neutral-gas approximation; the inventory variable n0 tracks
    its finite consumption and is checked positive over this short reception.
    G is the periodic Green function of 1-lambda^2*dxx, with transverse average.
    The native centered Helmholtz inverse preserves the constant mode; its
    nonlocal kernel is checked independently by the discrete Fourier symbol.
    """
    if not math.isfinite(photo_frequency) or photo_frequency < 0:
        raise ValueError("photo frequency must be finite and nonnegative")
    lengths=(2*np.pi,2*np.pi);mu=.2;diffusion=.02;E0=.3;impact_frequency=.08;absorption_length=.4
    frame,layout=_domain("M13_streamer",cells,lengths=lengths)
    support=PhysicalSupport((("x","M13_streamer"),("y","M13_streamer")))
    unit=PhysicalDimension()
    model=pops.Model("M13_local_field_streamer",frame=frame)
    U=model.state("species",components=("electron","ion","neutral"),
        support=support,sampling="cell_average",units=(unit,)*3)
    phi=model.field("potential")
    production=impact_frequency*U[0]
    model.source("impact",on=U,value=(production,production,-production))
    photon_production=model.aux("photo_production")
    radiation_unknown=model.field("radiation")
    model.source("photo",on=U,value=(photon_production,photon_production,-photon_production))
    case=pops.Case("M13_nonlocal_streamer");block=case.block("fluid",model)
    def body(sample,h,axis):
        left=tuple(-int(d==axis) for d in range(2));right=(0,0)
        gradient=(sample(right,0,field=0)-sample(left,0,field=0))/h[axis]
        velocity=mu*(gradient-(E0 if axis==0 else 0.))
        nl,nr=sample(left,0),sample(right,0)
        donor=pm.where(velocity>=0,lambda:nl,lambda:nr)
        current=velocity*donor-diffusion*(nr-nl)/h[axis]
        frequency=abs(velocity)/h[axis]+diffusion/h[axis]**2
        return (current,0.,0.),frequency
    flux=StencilFlux(dimension=2,components=3,radius=1,field_components=(1,),body=body)
    plan=DiscretizationPlan();plan.states.add(U,StateStorage(ghost_depth=max(flux.ghost_extent)))
    case.numerics(plan,block=block)
    space=model.module.state_spaces()["species"]
    phi_space=FieldSpace("potential",components=("phi",),layout="cell",representation="field",
        centering="cell",sampling="cell",frame=space.frame,clock=space.clock,support=support,units=(unit,))
    problem=FieldProblem("streamer_Poisson",unknowns=(phi,),equations=(-pm.laplacian(phi)==U[1]-U[0],),
        boundaries=(FieldBoundary(phi,bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(),bcs.Periodic())),),
        gauge=SharedMeanGauge((phi,)),unknown_spaces={phi:phi_space},coordinate_units=(unit,unit))
    field=case.field(problem,FieldDiscretization(method=CellCenteredSecondOrder(),boundaries=(),
        observation_axes=(0,1),solver=CG(max_iter=2000,rel_tol=1.e-12,abs_tol=1.e-13)))
    L,Ly=lengths;lam=absorption_length
    radiation_problem=FieldProblem("photon_Helmholtz",unknowns=(radiation_unknown,),
        equations=(pm.Reaction(radiation_unknown,1)-pm.div(lam**2*pm.grad(radiation_unknown))==photo_frequency*U[0],),
        boundaries=(FieldBoundary(radiation_unknown,bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(),bcs.Periodic())),))
    radiation=case.field(radiation_problem,FieldDiscretization(method=CellCenteredSecondOrder(),boundaries=(),
        solver=CG(max_iter=2000,rel_tol=1.e-12,abs_tol=1.e-13)))
    carrier=block[model.module.field_handle(model.module.field_spaces()["fields"])]
    dt=.002;end=2*dt
    def evaluate(p,state,point,label):
        observation=field.observe(p.solve(field,values={block[U]:state},at=point).consume(action=FailRun()))
        potential=observation[field[phi]]
        transport=p.stencil_flux_divergence(label+"_electron_flux",state,flux=flux,fields=(potential,))
        photons=radiation.observe(p.solve(radiation,values={block[U]:state},at=point).consume(action=FailRun()))
        photo_field=photons[radiation[radiation_unknown]]
        photo_context=photons.publish({(carrier,"photo_production"):photo_field})
        impact=p.source(model.module.operator_handle("impact"),state)
        photo=p.source(model.module.operator_handle("photo"),state,photo_context)
        p.store_history(label+"_state",state,depth=1)
        p.store_history(label+"_potential",potential,depth=1)
        p.store_history(label+"_photo",photo_field,depth=1)
        return transport+impact+photo
    _ssprk2(case,block,U,evaluate,dt,name="M13_SSPRK2")
    x,y=np.meshgrid((np.arange(cells[0])+.5)*L/cells[0],(np.arange(cells[1])+.5)*Ly/cells[1])
    electron=.02*(1+.15*np.sinc(1/cells[0])*np.cos(x)+.05*np.sinc(2/cells[0])*np.cos(2*x))
    ion=electron+.003*np.sinc(1/cells[0])*np.sin(x)
    initial=np.stack((electron,ion,1-ion));subject=block[U]
    from .reference_coupled_fields import ssprk2, streamer_rhs
    expected=ssprk2(initial,lambda value:streamer_rhs(value,lengths,photo=photo_frequency),dt)
    def check(runtime):
        actual=state_array(runtime,"fluid",initial.shape)
        stage=runtime.history_global("stage1_state",1)
        phi_data=runtime.history_global("stage1_potential",1)
        photo_data=runtime.history_global("stage1_photo",1)
        if actual is None:return non_owner_metrics()
        stage=np.asarray(stage).reshape(initial.shape);potential=np.asarray(phi_data).reshape(cells[1],cells[0])
        photo_values=np.asarray(photo_data).reshape(cells[1],cells[0])
        kx=np.arange(cells[0]);ky=np.arange(cells[1])
        symbol=1+4*lam**2*((np.sin(np.pi*kx/cells[0])/(L/cells[0]))[None,:]**2
                           +(np.sin(np.pi*ky/cells[1])/(Ly/cells[1]))[:,None]**2)
        reference=np.fft.ifft2(photo_frequency*np.fft.fft2(stage[0])/symbol).real
        radiation_error=float(np.max(abs(photo_values-reference)))
        lap=sum((2*potential-np.roll(potential,1,axis=d)-np.roll(potential,-1,axis=d))
                /(lengths[1-d]/cells[1-d])**2 for d in (0,1))
        poisson_error=float(np.max(abs(lap-(stage[1]-stage[0]))))
        charge_error=float(abs(np.mean(actual[1]-actual[0])-np.mean(initial[1]-initial[0])))
        nuclei_error=float(np.max(abs(actual[1]+actual[2]-1)))
        growth=float(np.mean(actual[0])-np.mean(initial[0]))
        # The Helmholtz inverse preserves constants. All transport sums
        # telescope, hence the global electron inventory obeys N'=a*N exactly
        # at the semidiscrete level and this SSPRK2 polynomial exactly in time.
        factor=1+(impact_frequency+photo_frequency)*dt+.5*((impact_frequency+photo_frequency)*dt)**2
        inventory_error=float(abs(np.mean(actual[0])-np.mean(initial[0])*factor**2))
        reference_error=float(np.max(abs(actual-expected)))
        passed=bool(reference_error<2.e-10 and np.isfinite(actual).all() and np.min(actual)>0 and charge_error<2.e-12
            and nuclei_error<2.e-12 and radiation_error<2.e-12 and poisson_error<2.e-8
            and growth>1.e-7 and inventory_error<2.e-12)
        return dict(metrics_owner=True,passed=passed,charge_defect=charge_error,
            nuclei_defect=nuclei_error,minimum_species=float(np.min(actual)),electron_growth=growth,
            discrete_Green_convolution_defect=radiation_error,Poisson_residual=poisson_error,
            electron_inventory_SSPRK2_defect=inventory_error,full_state_reference_error=reference_error)
    return CaseSpec(case,layout,_initial(case,subject,initial),{"fluid":subject},end,dt,check,
        "M13: periodic plane-parallel local-field streamer, live Poisson and electron drift-diffusion, immobile ions, tracked neutral depletion; constant impact coefficient, one nonlocal native Helmholtz photo group; charge/nuclei/convolution receipt, no electron-energy equation or calibrated-air/front-speed claim")
