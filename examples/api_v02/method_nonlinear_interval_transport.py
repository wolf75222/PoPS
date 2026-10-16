"""T09: self-consistent isothermal D1Q3 characteristic interval in Python.

This selected kinetic model has three discrete velocities and theta=1/4. The
nonlinear equilibrium preserves density and momentum. Its spatial and temporal
Taylor coefficients use Euler compatibility; face Duhamel and volume collision
share the actual requested interval. This is not a continuous Maxwellian or a
claim of universal AP behaviour or arbitrary-order ADER.
"""
from __future__ import annotations

import numpy as np
import pops
from pops import math as pm
from pops.numerics import DiscretizationPlan, ProgramRealization
from pops.numerics.stencil_flux import StencilFlux
from pops.realization import Realize, Use
from pops.time import ComputedDt, FixedDt, RejectAttempt
from pops.time.expressions import ProgramScalar

from .common import CaseSpec, non_owner_metrics, state_array
from .models_M01_M14 import _domain, _initial

VELOCITIES = (-1., 0., 1.)


def equilibrium(rho, momentum, theta):
    pressure = theta*rho+momentum*momentum/rho
    return ((pressure-momentum)/2, rho-pressure, (pressure+momentum)/2)


def equilibrium_direction(rho, momentum, rho_direction, momentum_direction, theta):
    velocity = momentum/rho
    pressure_direction = (theta-velocity*velocity)*rho_direction+2*velocity*momentum_direction
    return ((pressure_direction-momentum_direction)/2, rho_direction-pressure_direction,
            (pressure_direction+momentum_direction)/2)


def build_nonlinear_interval_transport(*, cells=(24, 4), dt=.002, steps=3, retry=False):
    if len(cells) != 2 or any(type(n) is not int or n < 4 for n in cells):
        raise ValueError("D1Q3 reception requires a two-dimensional periodic layout with at least four cells per axis")
    if not 0 < dt <= .004 or type(steps) is not int or steps < 1 or type(retry) is not bool:
        raise ValueError("reception requires 0<dt<=.004, positive steps, and an exact retry Bool")
    theta, tau = .25, .03
    frame, layout = _domain("T09_nonlinear_isothermal_interval", cells)
    model = pops.Model("isothermal_D1Q3", frame=frame)
    U = model.state("f", components=("minus", "rest", "plus"))
    transport = model.flux("free_flight", state=U, frame=frame,
        components={frame.axes[0]: tuple(v*U[c] for c, v in enumerate(VELOCITIES)),
                    frame.axes[1]: (0., 0., 0.)},
        waves={frame.axes[0]: VELOCITIES, frame.axes[1]: (0., 0., 0.)})
    geq = equilibrium(U[0]+U[1]+U[2], U[2]-U[0], theta)
    collision = model.source("self_consistent_BGK", on=U,
        value=tuple((geq[c]-U[c])/tau for c in range(3)))
    equation = model.rate("kinetic_balance", equation=pm.ddt(U) == -pm.div(transport)+collision)
    case = pops.Case("T09_nonlinear_Duhamel"); block = case.block("method", model)

    def face(sample, spacing, axis, h):
        if axis == 1:
            return (0., 0., 0.), 0.
        dx = spacing[0]
        first, slope = [], []
        for c, velocity in enumerate(VELOCITIES):
            left_slope = (sample((0, 0), c)-sample((-2, 0), c))/(2*dx)
            right_slope = (sample((1, 0), c)-sample((-1, 0), c))/(2*dx)
            left = sample((-1, 0), c)+dx/2*left_slope
            right = sample((0, 0), c)-dx/2*right_slope
            first.append(left if velocity > 0 else right if velocity < 0 else (left+right)/2)
            slope.append(left_slope if velocity > 0 else right_slope if velocity < 0 else (left_slope+right_slope)/2)
        rho, momentum = sum(first), first[2]-first[0]
        rho_x, momentum_x = sum(slope), slope[2]-slope[0]
        velocity = momentum/rho
        rho_t = -momentum_x
        momentum_t = -(theta-velocity*velocity)*rho_x-2*velocity*momentum_x
        g = equilibrium(rho, momentum, theta)
        gx = equilibrium_direction(rho, momentum, rho_x, momentum_x, theta)
        gt = equilibrium_direction(rho, momentum, rho_t, momentum_t, theta)
        e, one_minus_e = pm.exp(-h/tau), -pm.expm1(-h/tau)
        A = tau*one_minus_e
        B = tau*tau*one_minus_e-tau*h*e
        C = h-A
        D = tau*h-2*tau*tau*one_minus_e+tau*h*e
        E = h*h/2-tau*h+tau*tau*one_minus_e
        return tuple(v*(A*first[c]-v*B*slope[c]+C*g[c]-v*D*gx[c]+E*gt[c])/h
                     for c, v in enumerate(VELOCITIES)), 1/dx

    flux = StencilFlux(dimension=2, components=3, radius=2, parameters=("duration",), body=face)
    p = pops.Program("nonlinear_characteristic_interval"); q = p.state(block[U]); h = p.requested_dt()
    face_use = Use(equation.occurrences[0], "mesh_face", "whole_interval", "inventory", "transport")
    volume_use = Use(equation.occurrences[1], "cell", "whole_interval", "inventory", "collision")
    plan = DiscretizationPlan()
    plan.rates.add(equation, ProgramRealization(uses=(face_use, volume_use), ghost_depth=max(flux.ghost_extent)))
    case.numerics(plan, block=block)

    def body(builder, old, duration):
        face_rate = builder.stencil_flux_divergence("nonlinear_characteristic_average", old,
            flux=flux, parameters=(duration,))
        # This affine State keeps the spatial contribution and its ledger explicit.
        transported = builder.value("transported_moments", old+duration*face_rate, at=old.point)
        gn = equilibrium(old[0]+old[1]+old[2], old[2]-old[0], theta)
        gy = equilibrium(transported[0]+transported[1]+transported[2], transported[2]-transported[0], theta)
        z = ProgramScalar(duration)/(2*tau)
        volume = builder.rate("self_consistent_trapezoidal_collision",
            tuple((gn[c]-old[c]+gy[c]-transported[c])/(2*tau*(1+z)) for c in range(3)), of=transported)
        return {"face": face_rate, "collision": volume}

    realized = Realize("joint_nonlinear_interval", owns=(face_use, volume_use), reads=(volume_use,),
        body=body, bindings={"old": q.n, "duration": h},
        outputs={face_use: ("face",), volume_use: ("collision",)}).evaluate(p)
    candidate = p.value("conservative_nonlinear_interval",
        q.n+p.dt*realized[face_use]+p.dt*realized[volume_use], at=q.next.point)
    if retry:
        candidate = p.guard("rebuild_nonlinear_interval_after_retry", candidate,
            h <= dt*(1+1.e-12), action=RejectAttempt())
        p.reached_duration(h)
    p.commit(q.next, candidate)
    p.step_strategy(ComputedDt(2*dt, max_rejections=2, endpoint_ulps=4) if retry else FixedDt(dt))
    case.program(p)

    from . import nonlinear_interval_transport_oracle as reference
    initial = reference.initial_values(cells, theta)
    expected = reference.reference(initial, dt, 1/cells[0], steps, tau, theta)
    mutants = {"missing_temporal_equilibrium": {"temporal_equilibrium": False},
               "frozen_interval": {"flux_duration": 2*dt},
               "frozen_transported_equilibrium": {"transported_equilibrium": False}}
    separations = {name: float(np.max(abs(expected-reference.reference(
        initial, dt, 1/cells[0], steps, tau, theta, **options)))) for name, options in mutants.items()}
    rho0, momentum0 = reference.moments(initial)

    def check(runtime):
        actual = state_array(runtime, "method", initial.shape)
        if actual is None:
            return non_owner_metrics()
        rho, momentum = reference.moments(actual)
        error = float(np.max(abs(actual-expected)))
        mass = float(abs(rho.sum()-rho0.sum())/np.prod(cells))
        momentum_defect = float(abs(momentum.sum()-momentum0.sum())/np.prod(cells))
        return dict(passed=bool(error < 4.e-13 and mass < 4.e-14 and momentum_defect < 4.e-14
                               and min(separations.values()) > 1.e-8 and np.min(actual) > 0),
            metrics_owner=True, full_state_quadrature_error=error, mass_defect=mass,
            momentum_defect=momentum_defect, minimum=float(np.min(actual)), **separations)

    return CaseSpec(case, layout, _initial(case, block[U], initial), {"method": block[U]}, steps*dt, dt, check,
        "T09 nonlinear isothermal D1Q3: moment-preserving equilibrium; Euler-compatible space/time equilibrium gradients; "
        "exact linearized Duhamel face integral; self-consistent trapezoidal volume collision; native SSA interval and retry; "
        "continuous Maxwellians, universal AP and arbitrary-order ADER unqualified")
