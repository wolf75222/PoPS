"""Authored face equations inside native branches and bounded iterations.

The NumPy recurrence is a receiver only. It never supplies the executed state.
"""
from __future__ import annotations

import numpy as np
from pops.numerics import DiscretizationPlan, StateStorage
from pops.numerics.stencil_flux import StencilFlux
from pops.time import FixedDt, RejectAttempt

from .common import CaseSpec, non_owner_metrics, state_array
from .methods_T01_T12 import _storage
from .models_M01_M14 import _initial


def face_equation(velocity, *, invalid=False):
    def body(sample, spacing, axis):
        if axis != 0:
            return (0.,), 0.
        upstream = sample((-1, 0) if velocity > 0 else (0, 0), 0)
        return (velocity * upstream,), (-1. if invalid else abs(velocity) / spacing[0])
    return StencilFlux(dimension=2, components=1, radius=1, body=body)


def reference_step(initial, *, dt, dx, velocity, iterations):
    current = np.asarray(initial, dtype=float).copy()
    faces = []
    for _ in range(iterations):
        upstream = np.roll(current, 1, axis=-1) if velocity > 0 else current
        evaluated = velocity * upstream
        faces.append(evaluated.copy())
        current -= (dt / iterations / dx) * (
            np.roll(evaluated, -1, axis=-1) - evaluated)
    return current, faces


def build_scoped_stencil(*, mode="branch", positive=True, dead_invalid=False,
                         nonaffine=False, reject_second=False, cells=(16, 4), dt=.01):
    if mode not in ("branch", "fold"):
        raise ValueError("mode must be branch or fold")
    case, layout, _model, block, state, p, q, initial = _storage(
        "scoped-face-equations", ("q",), (1.,), cells=cells)
    plan = DiscretizationPlan()
    plan.states.add(state, StateStorage(ghost_depth=1))
    case.numerics(plan, block=block)
    old = p.value("initial_iterate", 1 * q.n, at=q.next.point)
    count = 2 if mode == "fold" else 1
    velocity = 1. if positive else -.6

    def update(builder, current, speed, *, invalid=False):
        rate = builder.stencil_flux_divergence(
            "scoped_face_evaluation", current, flux=face_equation(speed, invalid=invalid))
        candidate = builder.value(
            "scoped_update", current + (p.dt / count) * rate, at=current.point)
        if nonaffine:
            return builder.value("nonaffine_update", (candidate[0] ** 2,), at=current.point)
        return candidate

    if mode == "branch":
        # This predicate is evaluated on the native state, not as a Python if.
        condition = p.max(old) > (0. if positive else 3.)
        candidate = p.branch(condition,
            lambda b: update(b, old, 1., invalid=dead_invalid and not positive),
            lambda b: update(b, old, -.6, invalid=dead_invalid and positive))
    else:
        candidate = p.fold(old, 2, lambda b, carried, index: update(b, carried, velocity))
    if reject_second:
        interval = p.interval(duration=p.requested_dt())
        candidate = p.guard("reject_second_scoped_step", candidate,
            interval.begin < p.float64(dt / 2), action=RejectAttempt())
    p.commit(q.next, candidate)
    p.step_strategy(FixedDt(dt))
    case.program(p)
    x, y = np.meshgrid((np.arange(cells[0]) + .5) / cells[0],
                       (np.arange(cells[1]) + .5) / cells[1])
    initial[0] = 1 + .2 * np.cos(2 * np.pi * x) + .1 * np.sin(2 * np.pi * y)
    expected, _faces = reference_step(initial, dt=dt, dx=1 / cells[0],
                                     velocity=velocity, iterations=count)

    def check(runtime):
        actual = state_array(runtime, "method", initial.shape)
        if actual is None:
            return non_owner_metrics()
        error = float(np.max(abs(actual - expected)))
        return dict(passed=error < 3.e-14, metrics_owner=True, state_error=error,
                    no_op_error=float(np.max(abs(actual - initial))))

    return CaseSpec(case, layout, _initial(case, block[state], initial),
                    {"method": block[state]}, dt, dt, check,
                    "Uniform scoped face expression: " + mode)
