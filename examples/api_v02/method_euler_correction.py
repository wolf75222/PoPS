"""Euler T12: replace one face flux using both incident provisional states.

This is a finite, one-pass reception profile on a periodic Cartesian 2D grid.
The high flux is a Richtmyer half-time face predictor, including transverse
centered derivatives; the low flux is Rusanov. A failed density/pressure test
in either recipient selects the same low flux for both deliveries. It is not
a general positivity theorem, an iterative MOOD algorithm, or an AMR method.

NumPy is used only for initial values and an independent validation oracle.
All executed numerical equations are authored as ordinary PoPS expressions.
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


def reference_pressure(state, *, gamma=1.4, pressure_offset=.2):
    q = np.asarray(state, dtype=float)
    return ((gamma-1)*(q[3]-.5*np.sum(q[1:3]**2, axis=0)/q[0])
            - gamma*pressure_offset)


def reference_step(initial, *, dt, spacing, gamma=1.4,
                   pressure_offset=.2, admissibility_floor=1.e-10):
    """Independent array FV oracle; face i separates cell i-1 from cell i.

    Array axes are (component, y, x). Each directional flux is constructed
    once, and its single value enters both oriented divergence deliveries.
    """
    old = np.asarray(initial, dtype=float)
    pressure = reference_pressure(old, gamma=gamma, pressure_offset=pressure_offset)
    sound = np.sqrt(gamma*(pressure+pressure_offset)/old[0])
    speed = tuple(abs(old[1+d]/old[0])+sound for d in range(2))

    def physical_flux(q, direction):
        p = reference_pressure(q, gamma=gamma, pressure_offset=pressure_offset)
        normal_velocity = q[1+direction]/q[0]
        result = q*normal_velocity
        result[1+direction] += p
        result[3] += p*normal_velocity
        return result

    high, low, half_states = [], [], []
    for direction in range(2):
        normal_axis, transverse_axis = 2-direction, 1+direction
        transverse = 1-direction
        left, right = np.roll(old, 1, axis=normal_axis), old
        fl, fr = physical_flux(left, direction), physical_flux(right, direction)
        transverse_difference = (
            physical_flux(np.roll(left, -1, axis=transverse_axis), transverse)
            + physical_flux(np.roll(right, -1, axis=transverse_axis), transverse)
            - physical_flux(np.roll(left, 1, axis=transverse_axis), transverse)
            - physical_flux(np.roll(right, 1, axis=transverse_axis), transverse))
        half = ((left+right)/2-dt*(fr-fl)/(2*spacing[direction])
                -dt*transverse_difference/(8*spacing[transverse]))
        half_states.append(half)
        high.append(physical_flux(half, direction))
        envelope = np.maximum(speed[direction],
                              np.roll(speed[direction], 1, axis=1-direction))
        low.append((fl+fr)/2-.5*envelope*(right-left))

    def advance(faces):
        return old-dt*sum((np.roll(faces[d], -1, axis=2-d)-faces[d])/spacing[d]
                          for d in range(2))

    provisional = advance(high)
    candidate_pressure = reference_pressure(provisional, gamma=gamma,
                                            pressure_offset=pressure_offset)
    bad = ((provisional[0] <= admissibility_floor)
           | (candidate_pressure <= admissibility_floor))
    replaced = tuple(bad | np.roll(bad, 1, axis=1-d) for d in range(2))
    selected = tuple(np.where(replaced[d][None], low[d], high[d]) for d in range(2))
    corrected = advance(selected)
    # Deliberately wrong: replacing only a troubled cell's net update loses the
    # opposite delivery to an untroubled neighbor of that very same face.
    unilateral = np.where(bad[None], advance(low), provisional)
    one_incident = advance(tuple(np.where(np.roll(bad, 1, axis=1-d)[None],
                                          low[d], high[d]) for d in range(2)))
    without_y_delivery = old-dt*(np.roll(selected[0], -1, axis=2)-selected[0])/spacing[0]
    return dict(corrected=corrected, provisional=provisional, bad=bad, faces=selected,
                replaced=replaced, unilateral=unilateral, one_incident=one_incident,
                without_y_delivery=without_y_delivery, half_states=tuple(half_states))


def authored_flux(*, corrected, gamma=1.4, pressure_offset=.2,
                  admissibility_floor=1.e-10):
    """Pure Python algebra for the native face kernel; no model opcode."""
    def body(sample, spacing, axis, duration):
        left = tuple(-int(d == axis) for d in range(2))
        right = (0, 0)
        transverse = 1-axis

        def state(offset, *, field=None):
            return tuple(sample(offset, i) if field is None else sample(offset, i, field=field)
                         for i in range(4))

        def pressure(q):
            return ((gamma-1)*(q[3]-(q[1]*q[1]+q[2]*q[2])/(2*q[0]))
                    -gamma*pressure_offset)

        def flux(q, direction):
            normal = q[1+direction]/q[0]
            p = pressure(q)
            return (q[1+direction], q[1]*normal+int(direction == 0)*p,
                    q[2]*normal+int(direction == 1)*p, (q[3]+p)*normal)

        def shifted(offset, sign):
            return tuple(offset[d]+sign*int(d == transverse) for d in range(2))

        ql, qr = state(left), state(right)
        fl, fr = flux(ql, axis), flux(qr, axis)
        tlp, trp = flux(state(shifted(left, 1)), transverse), flux(state(shifted(right, 1)), transverse)
        tlm, trm = flux(state(shifted(left, -1)), transverse), flux(state(shifted(right, -1)), transverse)
        half = tuple((ql[i]+qr[i])/2-duration*(fr[i]-fl[i])/(2*spacing[axis])
                     -duration*(tlp[i]+trp[i]-tlm[i]-trm[i])/(8*spacing[transverse])
                     for i in range(4))
        high = flux(half, axis)
        envelope = pm.maximum(
            abs(ql[1+axis]/ql[0])+pm.sqrt(gamma*(pressure(ql)+pressure_offset)/ql[0]),
            abs(qr[1+axis]/qr[0])+pm.sqrt(gamma*(pressure(qr)+pressure_offset)/qr[0]))
        if corrected:
            def troubled(q):
                # For rho>floor this is exactly rho*(p-floor)<=0.
                # The separate density predicate rejects the other domain,
                # without ever dividing by a possibly zero candidate rho.
                pressure_margin = ((gamma-1)*(q[0]*q[3]-(q[1]*q[1]+q[2]*q[2])/2)
                                   -(gamma*pressure_offset+admissibility_floor)*q[0])
                return (q[0] <= admissibility_floor) | (pressure_margin <= 0.)
            bad = troubled(state(left, field=0)) | troubled(state(right, field=0))
            low = tuple((fl[i]+fr[i])/2-envelope*(qr[i]-ql[i])/2 for i in range(4))
            high = tuple(pm.where(bad, lambda i=i: low[i], lambda i=i: high[i])
                         for i in range(4))
        return high, envelope/spacing[axis]

    return StencilFlux(dimension=2, components=4, radius=1, body=body,
                       field_components=(4,) if corrected else (), parameters=("duration",))


def build_euler_correction(*, cells=(32, 8), cfl=.4):
    """One conservative step of an asymmetric periodic Euler Riemann profile.

    The transverse density and velocity variation makes this a genuine 2D
    flux calculation. gamma=1.4 and p_inf=.2 match the selected M03 EOS.
    Both CFL values are finite witnesses, not a general admissibility bound.
    """
    if len(cells) != 2 or cells[0] < 16 or cells[1] < 4 or cfl not in (.25, .4):
        raise ValueError("Euler correction witness requires nx>=16, ny>=4, CFL .25 or .4")
    gamma, pressure_offset, floor = 1.4, .2, 1.e-10
    x = (np.arange(cells[0])+.5)/cells[0]
    y = (np.arange(cells[1])+.5)/cells[1]
    rho = np.broadcast_to(np.where(x < .5, 1., .015), (cells[1], cells[0])).copy()
    rho *= 1+.04*np.sin(2*np.pi*y[:, None])
    pressure = np.broadcast_to(np.where(x < .5, 1., .03), rho.shape)
    vx = np.broadcast_to(np.where(x < .5, .5, 0.), rho.shape)
    vy = np.broadcast_to(.08*np.sin(2*np.pi*y[:, None]), rho.shape)
    energy = (pressure+gamma*pressure_offset)/(gamma-1)+.5*rho*(vx*vx+vy*vy)
    initial = np.stack((rho, rho*vx, rho*vy, energy))
    spacing = (1/cells[0], 1/cells[1])
    sound = np.sqrt(gamma*(pressure+pressure_offset)/rho)
    dt = float(cfl/np.max((abs(vx)+sound)/spacing[0]+(abs(vy)+sound)/spacing[1]))
    reference = reference_step(initial, dt=dt, spacing=spacing)

    frame, layout = _domain("T12_Euler_periodic", cells)
    model = pops.Model("T12_authored_Euler", frame=frame)
    # Nondimensional cell averages make the candidate's midpoint projection
    # explicit. The retained physical balance is the full four-equation Euler
    # system, not storage for an unregistered numerical evolution.
    U = model.state("conserved", components=("rho", "mx", "my", "energy"),
                    sampling="cell_average", units=(PhysicalDimension(),)*4)
    r, mx, my, total_energy = U
    ux, uy = mx/r, my/r
    pressure_expression = ((gamma-1)*(total_energy-(mx*mx+my*my)/(2*r))
                           -gamma*pressure_offset)
    sound_expression = pm.sqrt(gamma*(pressure_expression+pressure_offset)/r)
    xaxis, yaxis = frame.axes
    physical_flux = model.flux("Euler_flux", state=U, frame=frame,
        components={xaxis: (mx, mx*ux+pressure_expression, my*ux,
                            (total_energy+pressure_expression)*ux),
                    yaxis: (my, mx*uy, my*uy+pressure_expression,
                            (total_energy+pressure_expression)*uy)},
        waves={xaxis: (ux-sound_expression, ux, ux, ux+sound_expression),
               yaxis: (uy-sound_expression, uy, uy, uy+sound_expression)})
    balance = model.rate("Euler_balance", equation=pm.ddt(U) == -pm.div(physical_flux))
    case = pops.Case("T12_Euler_shared_face_correction")
    block = case.block("fluid", model, states=(U,))
    use = Use(balance.occurrences[0], "mesh_face", "accepted_step", "cell_average")
    high_flux = authored_flux(corrected=False)
    correction = authored_flux(corrected=True, admissibility_floor=floor)
    plan = DiscretizationPlan()
    # Storage follows the accesses retained by the actual equations, including
    # transverse samples at upper faces; radius alone does not give this bound.
    ghost_depth = max(*high_flux.ghost_extent, *correction.ghost_extent)
    plan.rates.add(balance, ProgramRealization(uses=(use,), ghost_depth=ghost_depth))
    case.numerics(plan, block=block)
    p = pops.Program("T12_Euler_predict_detect_replace")
    q = p.state(block[U])
    old, duration = q.n, p.requested_dt()
    high = p.stencil_flux_divergence("Euler_high_flux", old,
                                    flux=high_flux, parameters=(duration,))
    provisional = p.value("Euler_provisional_state", old+p.dt*high, at=q.next.point)
    sampled = p.cell_field(provisional, quadrature=CellMidpoint(), name="Euler_candidate_samples")
    accepted = Realize("Euler_face_replacement", owns=(use,),
        body=lambda b, old, samples, h: b.stencil_flux_divergence(
            "Euler_selected_flux", old, flux=correction, fields=(samples,), parameters=(h,)),
        bindings={"old": old, "samples": sampled, "h": duration}, outputs={use:()}).evaluate(p)[use]
    p.commit(q.next, p.value("Euler_accepted_state", old+p.dt*accepted, at=q.next.point))
    p.step_strategy(FixedDt(dt))
    case.program(p)

    def check(runtime):
        actual = state_array(runtime, "fluid", initial.shape)
        if actual is None:
            return non_owner_metrics()
        pressure_actual = reference_pressure(actual)
        error = float(np.max(abs(actual-reference["corrected"])))
        conserved = np.mean(actual-initial, axis=(1,2))
        unilateral = np.mean(reference["unilateral"]-initial, axis=(1,2))
        one_gap = float(np.max(abs(reference["one_incident"]-reference["corrected"])))
        y_gap = float(np.max(abs(reference["without_y_delivery"]-reference["corrected"])))
        provisional_pressure = reference_pressure(reference["provisional"])
        half_rho = min(float(qhalf[0].min()) for qhalf in reference["half_states"])
        half_pressure = min(float(reference_pressure(qhalf).min()) for qhalf in reference["half_states"])
        changed = float(np.max(abs(actual-reference["provisional"])))
        metrics = dict(reference_error=error, conservative_defects=conserved.tolist(),
            maximum_conservative_defect=float(np.max(abs(conserved))),
            minimum_density=float(actual[0].min()), minimum_pressure=float(pressure_actual.min()),
            reference_provisional_minimum_pressure=float(provisional_pressure.min()),
            reference_troubled_cells=int(reference["bad"].sum()),
            reference_replaced_faces=[int(mask.sum()) for mask in reference["replaced"]],
            reference_half_minimum_density=half_rho, reference_half_minimum_pressure=half_pressure,
            unilateral_conservative_defects=unilateral.tolist(), missing_incident_gap=one_gap,
            omitted_y_flux_gap=y_gap, difference_from_uncorrected=changed, metrics_owner=True)
        metrics["passed"] = bool(np.isfinite(actual).all() and error < 5.e-12
            and np.max(abs(conserved)) < 3.e-13 and actual[0].min() > floor
            and pressure_actual.min() > floor and provisional_pressure.min() < -1.e-3
            and half_rho > floor and half_pressure > floor and reference["bad"].any()
            and np.max(abs(unilateral)) > 1.e-3 and one_gap > 1.e-7 and y_gap > 1.e-5
            and changed > 1.e-3)
        return metrics

    return CaseSpec(case, layout, _initial(case, block[U], initial), {"fluid": block[U]}, dt, dt, check,
        "T12 Euler: four conserved components on genuinely varying periodic 2D cells, "
        "stiffened gas gamma=1.4 p_inf=.2, transverse Richtmyer predictor and Rusanov correction; "
        "one shared face flux selected from both incident provisional density/pressure states, "
        "only selected face ledger accepted; one-pass finite reception, general positivity, "
        "iterative MOOD, AMR and other EOS unqualified")
