"""The explicit Runge-Kutta methods of the surface (spec 0.3, 2.6.9; SU5, SU10, A4).

Each method is a function decorated with ``@pops.method``: PoPS supplies ``step``, and ``rate`` is
an option, given by keyword at the call, ``ssprk2(rate=model)``. In the body a rate times a
duration is an increment and a state plus an increment is a state (SU10), so each formula is
written as on paper; the increment of the candidate and its exchanges are derived (T34). The
bodies are those of the gallery (``lib_methods.py`` of the surface drafts), and the trace of N8
gives each the program that the same method written for N8 gives (SU-T3).
"""
from __future__ import annotations

import pops.surface as pops


@pops.method
def forward_euler(step, rate):
    """Forward Euler: u_new = u + h L(u, t)."""
    return pops.Candidate(step.state + step.dt * rate(step.state, step.time))


@pops.method
def ssprk2(step, rate):
    """Heun's method, SSPRK2: u1 = u + h L(u, t); u_new = (u + u1 + h L(u1, t + h)) / 2."""
    u = step.state
    t = step.time
    h = step.dt
    first = u + h * rate(u, t)
    return pops.Candidate((u + first + h * rate(first, t + h)) / 2)


@pops.method
def ssprk3(step, rate):
    """SSPRK3 of Shu and Osher: u1 = u + h L(u, t); u2 = (3 u + u1 + h L(u1, t + h)) / 4;
    u_new = (u + 2 u2 + 2 h L(u2, t + h / 2)) / 3."""
    u = step.state
    t = step.time
    h = step.dt
    first = u + h * rate(u, t)
    second = (3 * u + first + h * rate(first, t + h)) / 4
    return pops.Candidate((u + 2 * (second + h * rate(second, t + h / 2))) / 3)


@pops.method
def rk4(step, rate):
    """Classical RK4: k1 = L(u, t), k2 = L(u + h k1 / 2, t + h / 2), k3 = L(u + h k2 / 2,
    t + h / 2), k4 = L(u + h k3, t + h); u_new = u + h (k1 + 2 k2 + 2 k3 + k4) / 6."""
    u = step.state
    t = step.time
    h = step.dt
    k1 = rate(u, t)
    k2 = rate(u + h / 2 * k1, t + h / 2)
    k3 = rate(u + h / 2 * k2, t + h / 2)
    k4 = rate(u + h * k3, t + h)
    return pops.Candidate(u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))


__all__ = ["forward_euler", "rk4", "ssprk2", "ssprk3"]
