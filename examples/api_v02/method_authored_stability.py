"""A signed explicit midpoint composition with an executed energy admission.

The predicate is an authored numerical acceptance rule, not an SSP theorem.
The full periodic upwind operator and the signed stage algebra are Python DAGs.
"""
import numpy as np
from pops import math as pm

from pops.numerics.stencil_flux import StencilFlux
from pops.time import ComputedDt, FixedDt, GuardRole, RejectAttempt
from pops.time.expressions import ProgramScalar
from .common import CaseSpec, non_owner_metrics, state_array
from .methods_T01_T12 import _storage
from .models_M01_M14 import _initial


def build_authored_stability(*, retry=False, cells=(24, 4), dt=.005, steps=3):
    case, layout, _, block, state, p, q, initial = _storage(
        "signed_midpoint_authored_energy", ("q",), (1.,), cells=cells)
    x = (np.arange(cells[0])+.5)/cells[0]
    initial[0] = 1+.1*np.sin(2*np.pi*x)
    flux = StencilFlux(dimension=2, components=1, radius=1,
        body=lambda sample, spacing, axis:
            ((sample((-1, 0), 0),), 1/spacing[0]) if axis == 0 else ((0.,), 0.))
    first = p.stencil_flux_divergence("first_rate", q.n, flux=flux)
    midpoint = p.value("midpoint", q.n+.5*p.dt*first, at=q.n.point)
    second = p.stencil_flux_divergence("midpoint_rate", midpoint, flux=flux)
    candidate = p.value("signed_final", -q.n+2*midpoint+p.dt*(second-first), at=q.next.point)
    energy_defect = p.dot(candidate, candidate) - (1+1.e-12)*p.dot(q.n, q.n)
    energy_admission = energy_defect <= 0
    if retry:
        duration = p.requested_dt()
        combined_defect = p.scalar_expression("energy_and_duration_defect", pm.maximum(
            ProgramScalar(energy_defect), ProgramScalar(duration)/dt-1-1.e-12))
        energy_admission = combined_defect <= 0
        p.reached_duration(duration)
    guarded = p.guard("authored_midpoint_energy", candidate, energy_admission,
                      role=GuardRole.NUMERICAL_STABILITY, action=RejectAttempt())
    p.commit(q.next, guarded)
    p.step_strategy(ComputedDt(2*dt, max_rejections=2, endpoint_ulps=4) if retry else FixedDt(dt))
    case.program(p)
    expected = initial.copy()
    # Independent array matrix action; it does not call any authored PoPS helper.
    def action(u):
        return cells[0]*(np.roll(u, 1, axis=-1)-u)
    for _ in range(steps):
        expected = expected+dt*action(expected+.5*dt*action(expected))
    def check(runtime):
        actual = state_array(runtime, "method", initial.shape)
        if actual is None:
            return non_owner_metrics()
        error = float(np.max(abs(actual-expected)))
        mass = float(abs(actual.sum()-initial.sum())/np.prod(cells))
        energy = float(np.sum(actual*actual)-np.sum(initial*initial))
        return dict(passed=error < 2.e-13 and mass < 2.e-14 and energy <= 1.e-12,
                    metrics_owner=True, full_state_error=error, mass_defect=mass, energy_change=energy)
    return CaseSpec(case, layout, _initial(case, block[state], initial), {"method": block[state]},
                    steps*dt, dt, check, "signed midpoint; compiled authored energy admission; no SSP claim")
