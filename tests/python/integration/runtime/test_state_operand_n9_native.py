"""The State operand of a ``matrix_free_operator`` (IR26), measured on build_m05 and build_m23: the
test of the core ledger entry ``state-operand-of-a-matrix-free-operator`` (N9 « La référence sous
les solves de relation », §3.9 of the lot note, commit 4).

The stiff faces of the two structures are not linear in the state: M05 divides by rho and its
viscous work multiplies velocity by stress, M23's Hall term is J x B / rho. An implicit solve
applies their frozen-coefficient linearization: the same face, every factor that is not linear in
the iterate read from a frozen state through the StencilFlux's field input. Each linearization is
recorded on a State operand with ``LinearOperatorProperties.general()``, lowered inside a reference
region that evaluates it once (no solve), and compared with the explicit corpus face at the frozen
state, where the iterate is that state: div F(viscosity) - div F(0) for M05's
``navier_stokes_face``, div F(hall, ohmic) - div F(0, 0) for M23's ``hall_face``. The explicit
side is one forward Euler step of each face, so the difference carries the rounding of the state
over the step: 64 ulps of max|W| / dt bound it.
"""
from __future__ import annotations

import numpy as np
import pops
import pytest
from pops import math as pm
from pops.fields import CellMidpoint
from pops.linalg import LinearOperatorProperties, LinearProblem
from pops.model import PhysicalDimension
from pops.numerics import DiscretizationPlan, StateStorage
from pops.numerics.stencil_flux import StencilFlux
from pops.physics.roles import Custom
from pops.surface._solver import StateOperator
from pops.time import FixedDt
from pops.time.solve_outcome import FailRun
import pops.surface as surface
from examples.api_v02.models_M01_M14 import _domain, navier_stokes_face
from examples.api_v02.models_M22_M25_M28 import hall_face
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]

_CELLS = (8, 8)
_DT = 1.0 / 32          # the reference route: one evaluation, the step only clocks it
_DT_EXPLICIT = 1.0e-4   # the explicit routes: one forward Euler step under the face's guard


def _ns_viscous_frozen_face(*, gamma=1.4, viscosity=.01, prandtl=.72):
    """build_m05's viscous and conductive face, linear in the iterate: rho, the velocities of the
    work term and the temperature of mu are read from the frozen state (field 0)."""
    cv, cp = 1/(gamma-1), gamma/(gamma-1)

    def body(sample, h, axis):
        right = (0, 0)
        left = tuple(-int(d == axis) for d in range(2))
        frozen_cache, linear_cache = {}, {}

        def frozen(offset):
            if offset not in frozen_cache:
                rho, mx, my, energy = (sample(offset, i, field=0) for i in range(4))
                pressure = (gamma-1)*(energy-(mx*mx+my*my)/(2*rho))
                frozen_cache[offset] = rho, mx/rho, my/rho, pressure/rho
            return frozen_cache[offset]

        def linear(offset):
            # Velocity and temperature, linear in the iterate (m / rho_f, (gamma-1)(E - u_f.m/2)/rho_f).
            if offset not in linear_cache:
                rho, u, v, _ = frozen(offset)
                mx, my, energy = (sample(offset, i) for i in (1, 2, 3))
                linear_cache[offset] = (None, mx/rho, my/rho, (gamma-1)*(energy-(u*mx+v*my)/2)/rho)
            return linear_cache[offset]

        def derivative(component, direction):
            if direction == axis:
                return (linear(right)[component]-linear(left)[component])/h[direction]

            def shifted(offset, sign):
                return tuple(offset[d]+sign*int(d == direction) for d in range(2))
            return (linear(shifted(right, 1))[component]-linear(shifted(right, -1))[component]
                    + linear(shifted(left, 1))[component]-linear(shifted(left, -1))[component]
                    )/(4*h[direction])
        l, r = frozen(left), frozen(right)
        grad_u = tuple(tuple(derivative(i+1, j) for j in range(2)) for i in range(2))
        divergence = grad_u[0][0]+grad_u[1][1]
        mu = viscosity*((l[3]+r[3])/2)**.7
        conductivity = cp*mu/prandtl
        tau = tuple(mu*(grad_u[i][axis]+grad_u[axis][i]-(2/3)*int(i == axis)*divergence)
                    for i in range(2))
        work = sum((l[i+1]+r[i+1])*tau[i]/2 for i in range(2))
        flux = (0.*l[0], -tau[0], -tau[1], -work-conductivity*derivative(3, axis))
        rho_min = pm.minimum(l[0], r[0])
        frequency = (2*((4/3)*mu+conductivity/cv)/(rho_min*h[axis]**2)
                     + 2*mu/(rho_min*h[axis]*h[1-axis]))
        return flux, frequency
    return StencilFlux(dimension=2, components=4, radius=1, field_components=(4,), body=body)


def _hall_frozen_face(*, hall=.1, ohmic=.02):
    """build_m23's Hall and ohmic correction, linear in the iterate: rho and B of the face mean are
    read from the frozen state (field 0), the current J from the iterate."""
    def cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

    def body(sample, h, axis):
        left = tuple(-int(d == axis) for d in range(2))
        right = (0, 0)
        fl = tuple(sample(left, i, field=0) for i in range(8))
        fr = tuple(sample(right, i, field=0) for i in range(8))
        mean = tuple((a+b)/2 for a, b in zip(fl, fr))
        rho, b = mean[0], mean[5:8]
        r = tuple(sample(right, i) for i in range(8))
        l = tuple(sample(left, i) for i in range(8))

        def derivative(component, direction):
            if direction == 2:
                return 0.
            if direction == axis:
                return (r[component]-l[component])/h[direction]

            def shift(offset, sign):
                return tuple(offset[d]+sign*int(d == direction) for d in range(2))
            return (sample(shift(right, 1), component)-sample(shift(right, -1), component)
                    + sample(shift(left, 1), component)-sample(shift(left, -1), component)
                    )/(4*h[direction])
        current = (derivative(7, 1), -derivative(7, 0), derivative(6, 0)-derivative(5, 1))
        jxb = cross(current, b)
        electric = tuple(hall*jxb[i]/rho+ohmic*current[i] for i in range(3))
        poynting = cross(electric, b)[axis]
        normal = tuple(float(i == axis) for i in range(3))
        induction = cross(normal, electric)
        zero = 0.*rho
        frequency = 4*(abs(hall)*pm.sqrt(sum(x*x for x in b))/rho+ohmic)/h[axis]**2
        return (zero, zero, zero, zero, poynting, *induction), frequency
    return StencilFlux(dimension=2, components=8, radius=1, field_components=(8,), body=body)


@surface.solver
def _apply_once(problem, region):
    """One evaluation of the operator at the right-hand side, as the candidate (no solve)."""
    start = surface.record(x=problem.apply(problem.rhs), r=0.0, k=0)

    def stop(s):
        return s.k > 0

    def step(s):
        return surface.record(x=s.x, r=s.r, k=s.k + 1)
    result = region.iterate(start=start, step=step, stop=stop, max_steps=1)
    return problem.candidate(result.x, iterations=result.iterations, residual_norm=result.r)


_M23 = ("rho", "mx", "my", "mz", "E", "Bx", "By", "Bz")
_STRUCTURES = {
    # name: (block, components, roles, frozen face, explicit face, face without the stiff part)
    "M05": ("fluid", ("rho", "mx", "my", "E"), None, _ns_viscous_frozen_face,
            lambda: navier_stokes_face(viscosity=.01), lambda: navier_stokes_face(viscosity=0.)),
    "M23": ("magnetofluid", _M23, {n: Custom(n) for n in _M23}, _hall_frozen_face,
            lambda: hall_face(hall=.1, ohmic=.02), lambda: hall_face(hall=0., ohmic=0.)),
}


def _artifact(structure, route):
    block_name, names, roles, frozen_face, full_face, base_face = _STRUCTURES[structure]
    name = "%s_state_operand_%s" % (structure, route)
    frame, layout = _domain(name, _CELLS)
    model = pops.Model(name, frame=frame)
    # The frozen state enters the face as a cell field, sampled at the cell midpoints.
    state = model.state("q", components=names, roles=roles, sampling="cell_average",
                        units=(PhysicalDimension(),) * len(names))
    case = pops.Case(name)
    block = case.block(block_name, model)
    plan = DiscretizationPlan()
    plan.states.add(state, StateStorage(ghost_depth=2))
    case.numerics(plan, block=block)
    program = pops.Program(name + "_p")
    temporal = program.state(block[state])
    w = temporal.n
    if route == "reference":
        flux = frozen_face()

        def apply(p, v):
            frozen = p.cell_field(w, quadrature=CellMidpoint(), name="frozen")
            return p.stencil_flux_divergence("linearized", v, flux=flux, fields=(frozen,))
        problem = LinearProblem(StateOperator(structure.lower() + "_frozen", apply), w,
                                nullspace=None, properties=LinearOperatorProperties.general())
        outcome = program.solve(problem, name="once",
                                solver=_apply_once().with_request(rtol=0.0, atol=1e300))
        result = outcome.consume(action=FailRun())
        dt = _DT
    else:
        face = full_face() if route == "full" else base_face()
        result = w + program.dt * program.stencil_flux_divergence(route, w, flux=face)
        dt = _DT_EXPLICIT
    program.commit(temporal.next, program.value("accepted", result, at=temporal.next.point))
    program.step_strategy(FixedDt(dt))
    case.program(program)
    validated = pops.validate(case)
    if route == "reference":
        (operator,) = [v for v in program._values if v.op == "matrix_free_operator"]
        assert operator.attrs["apply_operand"] == "state"
    return block_name, dt, pops.compile(pops.resolve(validated, layout=layout))


def _evaluate(structure, route, w):
    block_name, dt, artifact = _artifact(structure, route)
    runtime = pops.bind(artifact, initial_state={block_name: w.copy()},
                        resources={"execution_context": artifact_execution_context(artifact)})
    pops.run(runtime, t_end=dt, max_steps=1, console=False)
    return np.asarray(runtime.state_global(block_name), dtype=np.float64).reshape(w.shape)


def _frozen_state(structure):
    x, y = np.meshgrid((np.arange(_CELLS[0])+.5)/_CELLS[0], (np.arange(_CELLS[1])+.5)/_CELLS[1])
    s, c, k = np.sin, np.cos, 2*np.pi
    if structure == "M05":
        rho = 1+.03*c(k*x)+.02*s(k*y)
        vx = .15+.03*s(k*x)+.02*s(k*y)
        vy = .02*c(k*x)+.03*c(k*y)
        temperature = 1+.08*c(k*x)+.04*s(k*(x+y))
        energy = rho*temperature/.4+.5*rho*(vx*vx+vy*vy)
        return np.asarray((rho, rho*vx, rho*vy, energy))
    rho = 1+.05*c(k*x)+.03*s(k*y)
    v = (.1+.02*s(k*y), .03*c(k*x), .02*s(k*(x+y)))
    b = (1+.04*c(k*y), .05*c(k*x)+.02*s(k*y), .04*s(k*x)+.03*c(k*y))
    energy = (1+.05*s(k*x))/.4+.5*rho*sum(a*a for a in v)+.5*sum(a*a for a in b)
    return np.asarray((rho, rho*v[0], rho*v[1], rho*v[2], energy, *b))


@pytest.mark.parametrize("structure, active", [("M05", (1, 2, 3)), ("M23", (4, 5, 6, 7))])
def test_the_frozen_coefficient_linearization_on_a_state_operand_is_the_explicit_face(
        structure, active, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    w = _frozen_state(structure)
    reference = _evaluate(structure, "reference", w)
    explicit = (_evaluate(structure, "full", w) - _evaluate(structure, "base", w)) / _DT_EXPLICIT
    inactive = [i for i in range(w.shape[0]) if i not in active]
    # The face leaves density (and M23's momentum) to the other terms: exactly zero on both sides.
    assert not np.any(reference[inactive]) and not np.any(explicit[inactive])
    # The stiff part is not vacuous on this state.
    assert all(np.abs(explicit[i]).max() > 1e-3 for i in active)
    np.testing.assert_allclose(reference, explicit, rtol=0.0,
                               atol=64 * np.finfo(float).eps * np.abs(w).max() / _DT_EXPLICIT)
