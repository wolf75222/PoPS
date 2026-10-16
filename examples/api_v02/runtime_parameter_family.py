"""Finite families of runtime parameters are algebra, not a model-count ceiling."""
from __future__ import annotations

import numpy as np
import pops
from pops import math as pm
from pops.numerics import DiscretizationPlan, StateStorage
from pops.params import RuntimeParam
from pops.time import FixedDt

from .common import CaseSpec, non_owner_metrics, state_array
from .models_M01_M14 import _domain, _initial


def build_runtime_parameter_family(*, count=65, cells=(8, 4), dt=.01):
    if type(count) is not int or count <= 0:
        raise ValueError("parameter family needs a positive finite cardinality")
    frame, layout = _domain("runtime_parameter_family", cells)
    model = pops.Model("runtime_parameter_family", frame=frame)
    state = model.state("u", components=("q",))
    coefficients = [model.param(RuntimeParam("p_%03d" % i, default=float(i+1)))
                    for i in range(count)]
    growth = sum((model.value(parameter)/(count*(i+1))
                  for i, parameter in enumerate(coefficients)), 0.)
    source = model.source("growth", on=state, value=(growth*state[0],))
    rate = model.rate("evolution", equation=pm.ddt(state) == source)
    case = pops.Case("runtime_parameter_family")
    block = case.block("method", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, StateStorage(ghost_depth=1))
    case.numerics(plan, block=block)
    program = pops.Program("runtime_parameter_family_euler")
    q = program.state(block[state])
    candidate = program.value("advance", q.n+program.dt*rate(q.n), at=q.next.point)
    program.commit(q.next, candidate)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    x = (np.arange(cells[0])+.5)/cells[0]
    initial = np.broadcast_to(1+.2*np.cos(2*np.pi*x), (1, cells[1], cells[0])).copy()

    def invariants(runtime):
        actual = state_array(runtime, "method", initial.shape)
        if actual is None:
            return non_owner_metrics()
        expected = initial*(1+dt)**runtime.macro_step()
        error = float(np.max(np.abs(actual-expected)))
        return dict(passed=error < 3.e-13, metrics_owner=True, state_error=error)

    return CaseSpec(case, layout, _initial(case, block[state], initial),
                    {"method": block[state]}, 2*dt, dt, invariants,
                    "65+ independent runtime coefficients executed through native StateStorage")
