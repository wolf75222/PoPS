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
from pops.time import FailRun, FixedDt
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
    return _build_meshface(cells=cells, cfl=cfl)


def reference_screened_poisson(rhs, *, cells, reaction=1.):
    """Independent Fourier inverse of reaction - Delta_h on the unit torus."""
    rhs = np.asarray(rhs, dtype=float)
    ky = np.arange(cells[1])[:, None]
    kx = np.arange(cells[0])[None, :]
    symbol = (reaction + 4*cells[0]**2*np.sin(np.pi*kx/cells[0])**2
              + 4*cells[1]**2*np.sin(np.pi*ky/cells[1])**2)
    return np.fft.ifft2(np.fft.fft2(rhs, axes=(-2, -1))/symbol,
                        axes=(-2, -1)).real


def build_t12_corrected_field(*, cells=(32, 4), cfl=.4, field_input="accepted"):
    """Replace faces, solve again, and advance another state with the new field.

    ``field_input`` selects deliberate stale-input counterexamples for reception.
    All three versions run both field solves at the same requested end time;
    only ``accepted`` supplies the corrected state to the consumed solve.
    """
    if field_input not in ("accepted", "provisional", "old"):
        raise ValueError("field_input must be accepted, provisional or old")
    return _build_meshface(cells=cells, cfl=cfl, field_input=field_input)


def _build_meshface(*, cells, cfl, field_input=None):
    if cells[0] < 16 or not 0 < cfl <= .5:
        raise ValueError("T12 profile requires at least 16 x cells and 0 < CFL <= 1/2")
    frame,layout = _domain("T12_face_domain",cells)
    model = pops.Model("T12_authored_advection",frame=frame)
    U = model.state("amount",components=("q",),sampling="cell_average",units=(PhysicalDimension(),))
    flux = model.flux("physical_transport",state=U,frame=frame,
        components={frame.axes[0]:(U[0],),frame.axes[1]:(0.,)},
        waves={frame.axes[0]:(1.,),frame.axes[1]:(0.,)})
    rate = model.rate("conservation",equation=pm.ddt(U)==-pm.div(flux))
    potential = model.field("screened_potential") if field_input is not None else None
    use = Use(rate.occurrences[0],"mesh_face","accepted_step","cell_average")
    case = pops.Case("T12_shared_mesh_face")
    block = case.block("method",model,states=(U,))
    plan = DiscretizationPlan()
    plan.rates.add(rate,ProgramRealization(uses=(use,),ghost_depth=1))
    case.numerics(plan,block=block)
    response_block = response_state = response_rate = field = None
    if field_input is not None:
        from pops._ir.elliptic import Reaction
        from pops.fields import FieldBoundary, FieldDiscretization, FieldProblem, bcs
        from pops.fields.methods import CellCenteredSecondOrder
        from pops.numerics import StateStorage
        from pops.solvers import CG
        response_model = pops.Model("field_response", frame=frame)
        response_state = response_model.state("response", components=("v",))
        forcing = response_model.aux("accepted_potential")
        source = response_model.source("field_forcing", on=response_state, value=(forcing,))
        response_rate = response_model.rate("response_balance",
                                            equation=pm.ddt(response_state) == source)
        response_block = case.block("response", response_model)
        response_plan = DiscretizationPlan()
        response_plan.rates.add(response_rate, StateStorage())
        case.numerics(response_plan, block=response_block)
        problem = FieldProblem("correction_dependent_field", unknowns=(potential,),
            equations=(-pm.laplacian(potential)+Reaction(potential, 1.) == U[0],),
            boundaries=(FieldBoundary(potential, bcs.BoundaryCondition(
                bcs.AllPhysicalBoundaries(), bcs.Periodic())),))
        field = case.field(problem, FieldDiscretization(method=CellCenteredSecondOrder(),
            boundaries=(), solver=CG(max_iter=2000, rel_tol=1.e-12, abs_tol=1.e-13)))
    p = pops.Program("T12_predict_detect_replace")
    q = p.state(block[U]); old=q.n
    duration = p.requested_dt()
    high = p.stencil_flux_divergence("provisional_high_flux",old,flux=authored_flux(corrected=False),
                                   parameters=(duration,))
    provisional = p.value("provisional_high_state",old+p.dt*high,at=q.next.point)
    if field is not None:
        provisional_observation = field.observe(p.solve(field, values={block[U]: provisional},
            at=q.next.point).consume(action=FailRun()))
        p.store_history("provisional_potential", provisional_observation[field[potential]], depth=1)
    candidate_sample = p.cell_field(provisional,quadrature=CellMidpoint(),name="candidate_samples")
    selected_flux = authored_flux(corrected=True)
    selected = Realize("accepted_face_replacement",owns=(use,),
        body=lambda b,old,samples,h:b.stencil_flux_divergence("selected_face_flux",old,flux=selected_flux,
            fields=(samples,),parameters=(h,)),
        bindings={"old":old,"samples":candidate_sample,"h":duration},outputs={use:()}).evaluate(p)[use]
    accepted = p.value("accepted",old+p.dt*selected,at=q.next.point)
    p.commit(q.next,accepted)
    if field is not None:
        field_state = {"accepted": accepted, "provisional": provisional, "old": old}[field_input]
        if field_input == "old":
            field_state = p.value("old_state_at_end", 1*old, at=q.next.point)
        observation = field.observe(p.solve(field, values={block[U]: field_state},
            at=q.next.point).consume(action=FailRun()))
        final_potential = observation[field[potential]]
        p.store_history("accepted_potential", final_potential, depth=1)
        p.store_history("accepted_field_input", field_state, depth=1)
        p.store_history("corrected_amount", accepted, depth=1)
        module = response_model.module
        carrier = response_block[module.field_handle(module.field_spaces()["fields"])]
        response = p.state(response_block[response_state])
        stage = p.value("response_at_end", 1*response.n, at=response.next.point)
        context = observation.publish({(carrier, "accepted_potential"): final_potential},
            states={response_block[response_state]: stage})
        candidate = p.value("forced_response", response.n+p.dt*response_rate(stage, context),
                            at=response.next.point)
        p.commit(response.next, candidate)
    dx=1/cells[0];dt=cfl*dx
    p.step_strategy(FixedDt(dt));case.program(p)
    initial=np.full((1,cells[1],cells[0]),.15)
    initial[...,6:12]=np.array([.15,.06,.005,1.,.65,.3])
    reference=reference_step(initial,dt=dt,dx=dx)
    def check(runtime):
        actual=state_array(runtime,"method",initial.shape)
        if field is not None:
            response_actual = state_array(runtime, "response", initial.shape)
            raw_histories = {name: runtime.history_global(name, 1) for name in
                ("provisional_potential", "accepted_potential", "accepted_field_input", "corrected_amount")}
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
        if field is not None:
            histories = {name: np.asarray(value).reshape(initial.shape)
                         for name, value in raw_histories.items()}
            expected_potential = reference_screened_poisson(reference["corrected"], cells=cells)
            provisional_potential = reference_screened_poisson(reference["provisional"], cells=cells)
            old_potential = reference_screened_poisson(initial, cells=cells)
            observed = histories["accepted_potential"]
            operator_value = observed + sum(cells[1-axis]**2*(
                2*observed-np.roll(observed, 1, axis+1)-np.roll(observed, -1, axis+1))
                for axis in (0, 1))
            field_metrics = dict(
                accepted_field_error=float(np.max(abs(observed-expected_potential))),
                accepted_field_residual=float(np.max(abs(operator_value-actual))),
                field_input_error=float(np.max(abs(histories["accepted_field_input"]-actual))),
                retained_corrected_error=float(np.max(abs(histories["corrected_amount"]-actual))),
                provisional_field_error=float(np.max(abs(histories["provisional_potential"]-provisional_potential))),
                response_error=float(np.max(abs(response_actual-dt*expected_potential))),
                response_uses_executed_field_error=float(np.max(abs(response_actual-dt*observed))),
                stale_provisional_response_gap=float(np.max(abs(dt*(provisional_potential-expected_potential)))),
                stale_old_response_gap=float(np.max(abs(dt*(old_potential-expected_potential)))),
                field_input=field_input)
            metrics.update(field_metrics)
            metrics["passed"] = bool(metrics["passed"]
                and field_metrics["accepted_field_error"] < 2.e-11
                and field_metrics["accepted_field_residual"] < 2.e-9
                and field_metrics["field_input_error"] < 3.e-13
                and field_metrics["retained_corrected_error"] < 3.e-13
                and field_metrics["provisional_field_error"] < 2.e-11
                and field_metrics["response_error"] < 2.e-12
                and field_metrics["response_uses_executed_field_error"] < 2.e-12
                and field_metrics["stale_provisional_response_gap"] > 1.e-8
                and field_metrics["stale_old_response_gap"] > 1.e-7)
        return metrics
    initial_values = _initial(case, block[U], initial)
    subjects = {"method": block[U]}
    if field is not None:
        initial_values.update(_initial(case, response_block[response_state], np.zeros_like(initial)))
        subjects["response"] = response_block[response_state]
        return CaseSpec(case, layout, initial_values, subjects, dt, dt, check,
            "T12: conservative shared-face replacement followed by a fresh screened-Poisson solve "
            "and an independently advanced state consuming that field; provisional and final solves "
            "share the end time and retain distinct state-version inputs; selected field input="+field_input)
    return CaseSpec(case,layout,initial_values,subjects,dt,dt,check,
        "T12: native periodic Cartesian faces, Lax-Wendroff provisional SSA, two-incident-cell positivity detector, "
        "one selected upwind/high flux delivered to both recipients, only final face ledger accepted; "
        "one-pass scalar reception, general iterative MOOD/AMR and dependent field re-solves unqualified")
