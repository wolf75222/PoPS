"""A two-point diffusion written as a ``StencilFlux`` whose coefficient is a runtime parameter.

The heat equation du/dt = div(nu grad u) on a periodic square, its face flux -nu (u_R - u_L) / h
and its incident frequency (nu / h) / h written in the body of a ``StencilFlux`` that reads ``nu``,
a ``RuntimeParam`` of the model of its block (N3 lot 1). The model declares the diffusive flux,
so ``nu`` is read by a formula of the model and has a slot in the block's RuntimeParams table.
"""
from __future__ import annotations

import pops
from pops import math as pm
from pops.numerics import DiscretizationPlan, StateStorage
from pops.numerics.stencil_flux import StencilFlux
from pops.params import Positive, RuntimeParam
from pops.time import FixedDt

from examples.api_v02.models_M01_M14 import _domain


def two_point_body(nu, dimension):
    def body(sample, spacing, axis):
        h = spacing[axis]
        left = tuple(-1 if d == axis else 0 for d in range(dimension))
        right = (0,) * dimension
        gradient = (sample(right, 0) - sample(left, 0)) / h
        return (-(nu * gradient),), (nu / h) / h

    return body


def heat_case(name="stencil_runtime_parameter", *, cells=(16, 16), nu_default=0.01, dt=1.e-3,
              declare_flux=True, other_model_parameter=False, stencils=1):
    """The case, its layout, its block, its state and the parameter handle ``block[nu]``."""
    frame, layout = _domain(name, cells)
    model = pops.Model("heat", frame=frame)
    nu_handle = model.param(RuntimeParam("nu", default=nu_default, domain=Positive()))
    state = model.state("state", components=("u",))
    nu = model.value(nu_handle)
    if declare_flux:
        model.diffusive_flux("heat_flux", state=state, value=nu * pm.grad(state[0]))
    if other_model_parameter:
        other = pops.Model("other", frame=frame)
        nu = other.value(other.param(RuntimeParam("nu", default=nu_default)))
    case = pops.Case(name)
    block = case.block("scalar", model)
    plan = DiscretizationPlan()
    plan.states.add(state, StateStorage(ghost_depth=1))
    case.numerics(plan, block=block)
    flux = StencilFlux(dimension=2, components=1, radius=1, body=two_point_body(nu, 2))
    program = pops.Program("heat_step")
    u = program.state(block[state])
    rates = [program.stencil_flux_divergence("two_point_%d" % index, u.n, flux=flux)
             for index in range(stencils)]
    total = rates[0]
    for rate in rates[1:]:
        total = total + rate
    candidate = program.value("candidate", u.n + program.dt * total / stencils,
                              at=u.next.point)
    program.commit(u.next, candidate)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    return case, layout, block, state, block[nu_handle]


__all__ = ["heat_case", "two_point_body"]
