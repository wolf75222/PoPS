"""The exit test of the N9 lot « La référence sous les solves de relation » (ROM-90, §3.5).

L1-d of §17.1 on a box of rank two: ``ddt(u) == -div(a u) - div(-nu grad(u))``, ``upwind()`` for
the advection, ``two_point()`` for the diffusion, ``imex_euler`` with ``cg()`` of
``pops.lib.solve``, and ``nu = model.parameter("nu", default=1e-2, admissible=pops.Positive())``.
The relation solve runs the reference conjugate gradient as the region of its node, and its
operator is the two-point StencilFlux of the rate at the State, which reads nu from the block's
RuntimeParams table inside the region.

The case is compiled once and bound with two values of nu. The two runs differ, and each is the
run of the same case with nu a ``model.constant`` of that value, compiled on its own, within the
tolerance of the solve. The artifacts are bound through ``pops.bind`` of b6 with the execution
context of the loaded native runtime (``artifact_execution_context``), so that the module runs on a
serial build and on an MPI build alike; ``surface.bind`` and its parameters are lot G's. One step
of each is the dense solve of the same discrete operator,
(I - dt nu L_h) u1 = u0 + dt E_h(u0) with L_h the two-point Laplacian and E_h the upwind
divergence, within the request of the solve (§5 of the lot note, P2 made a test). A value outside
``pops.Positive()``, on which the claim of the operator rests (symmetric positive definite for every
admissible value), is refused at bind.
"""
from pathlib import Path

import numpy as np
import pytest

import pops
import pops.surface as surface
from pops.lib.numerics import two_point, upwind
from pops.lib.solve import cg
from pops.lib.time.methods import imex_euler
from pops.runtime._simulation import Simulation
from pops.surface.math import ddt, div, grad, vector
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

REPO_ROOT = Path(__file__).resolve().parents[4]
CELLS = (16, 16)
DT = 1e-3
UNTIL = 0.005                   # five accepted steps
RTOL, ATOL = 1e-10, 1e-12
VALUES = (1e-2, 5e-2)


def _case(nu):
    """L1-d with nu given by ``nu(model)``: a parameter or a constant of the model."""
    domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    model = surface.Model("advection_diffusion")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    coefficient = nu(model)
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=vector(a, 0.5 * a) * u)
    diffusion = model.diffusive_flux(of=u, value=-coefficient * grad(u))
    model.balance(ddt(u) == -div(advection) - div(diffusion))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: upwind(), diffusion: two_point()})
    imex = imex_euler(explicit=advection, implicit=diffusion)
    program = surface.Program(
        advance=surface.Advance(imex, dt=DT, attempts=2,
                                tolerance=surface.Tolerance(rtol=RTOL, atol=ATOL)),
        initial={u: u0}, solvers={imex: cg()})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=CELLS)})
    return case, u, u0, coefficient


def _initial():
    x = (np.arange(CELLS[0]) + 0.5) / CELLS[0]
    y = (np.arange(CELLS[1]) + 0.5) / CELLS[1]
    return (np.sin(2 * np.pi * x)[:, None] * np.cos(2 * np.pi * y)[None, :]
            + 0.3 * np.cos(6 * np.pi * x)[:, None] + 0.01 * np.arange(CELLS[1])[None, :])


@pytest.fixture(scope="module")
def compiled(tmp_path_factory, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    cache = tmp_path_factory.mktemp("pops-native-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        patch.chdir(REPO_ROOT)
        parameter = _case(lambda model: model.parameter("nu", default=1e-2,
                                                        admissible=surface.Positive()))
        constants = {value: _case(lambda model, value=value: model.constant("nu", value=value))
                     for value in VALUES}
        yield {"parameter": (surface.compile(parameter[0], target=surface.OpenMP()),
                             *parameter[1:]),
               "constants": {value: (surface.compile(case[0], target=surface.OpenMP()),
                                     *case[1:]) for value, case in constants.items()}}


def _simulation(artifact, params=None):
    """The artifact bound with ``u0`` and the values of its parameters (keyed by the surface's
    handles), as b6 binds it."""
    compiled = artifact.compiled
    elaborated = artifact.case.elaborated
    (binding,) = tuple(compiled.plan.initial_condition_plan.bindings)
    runtime = pops.bind(
        compiled, initial_values={binding.subject: np.ascontiguousarray(
            np.transpose(_initial())[np.newaxis, ...])},
        params={elaborated.block[elaborated.params[key]]: value
                for key, value in (params or {}).items()},
        resources={"execution_context": artifact_execution_context(compiled)})
    return Simulation(runtime)


def _run(artifact, until, params=None):
    """The state after a run to ``until``, in the order of the support's axes, and its steps."""
    result = _simulation(artifact, params).run(until)
    assert result.failure is None
    (state,) = result.state.values()
    return np.transpose(np.asarray(state, dtype=np.float64)[0]), result.steps


def test_one_artifact_solves_the_relation_with_each_value_bound_to_nu(compiled):
    artifact, _, _, nu = compiled["parameter"]
    runs = {}
    for value in VALUES:
        runs[value], steps = _run(artifact, UNTIL, params={nu: value})
        assert steps == 5
    # nu is read in the region of the solve: the two values give two runs.
    first, second = (runs[value] for value in VALUES)
    assert np.max(np.abs(first - second)) > 1e-3 * np.max(np.abs(first))
    for value in VALUES:
        expected, steps = _run(compiled["constants"][value][0], UNTIL)
        assert steps == 5
        difference = float(np.max(np.abs(runs[value] - expected)))
        print("nu = %r: max |parameter - constant| = %.3e" % (value, difference))
        assert difference <= 5 * (RTOL * np.max(np.abs(expected)) + ATOL)


def _dense_step(nu):
    """u1 of one imex_euler step by a dense solve: (I - dt nu L_h) u1 = u0 + dt E_h(u0), with
    E_h(u) = -(a (u_i - u_{i-1}) / hx + a/2 (u_j - u_{j-1}) / hy), the upwind divergence of
    vector(a, a/2) u, and L_h the periodic five-point Laplacian of the two-point flux."""
    nx, ny = CELLS
    hx, hy = 1.0 / nx, 1.0 / ny
    u0 = _initial()
    explicit = -((u0 - np.roll(u0, 1, axis=0)) / hx + 0.5 * (u0 - np.roll(u0, 1, axis=1)) / hy)
    size = nx * ny
    laplacian = np.zeros((size, size))
    for i in range(nx):
        for j in range(ny):
            row = i * ny + j
            for di, dj, weight in ((1, 0, 1 / hx**2), (-1, 0, 1 / hx**2), (0, 1, 1 / hy**2),
                                   (0, -1, 1 / hy**2)):
                laplacian[row, ((i + di) % nx) * ny + (j + dj) % ny] += weight
            laplacian[row, row] -= 2 / hx**2 + 2 / hy**2
    rhs = (u0 + DT * explicit).reshape(size)
    solution = np.linalg.solve(np.eye(size) - DT * nu * laplacian, rhs)
    return solution.reshape(CELLS), float(np.linalg.norm(rhs))


def test_one_step_is_the_dense_solve_of_the_discrete_operator(compiled):
    nu = compiled["parameter"][3]
    value = VALUES[1]
    stepped, steps = _run(compiled["parameter"][0], DT, params={nu: value})
    by_constant, constant_steps = _run(compiled["constants"][value][0], DT)
    assert steps == constant_steps == 1
    dense, norm = _dense_step(value)
    # The operator is SPD with a condition number near 1 (dt nu 8 / h^2 = 0.1), so the error of
    # the state is bounded by the residual that the request admits, rtol |b| + atol.
    bound = 4 * (RTOL * norm + ATOL)
    for state in (stepped, by_constant):
        print("one step, nu = %r: max |run - dense| = %.3e (bound %.3e)"
              % (value, float(np.max(np.abs(state - dense))), bound))
        assert np.linalg.norm(state - dense) <= bound
    # nu read from the table and nu folded into the body give the same step.
    assert np.max(np.abs(stepped - by_constant)) <= bound


def test_a_value_outside_the_admissible_set_of_nu_is_refused_at_bind(compiled):
    """The claim that makes the operator SPD for cg() holds for every value in pops.Positive()
    (§3.5): the bind refuses the others before any step."""
    artifact, nu = compiled["parameter"][0], compiled["parameter"][3]
    for value in (0.0, -1e-2):
        with pytest.raises(ValueError, match="runtime-param-domain: bind value .* for .*::parameter"
                                             "::nu::.* is outside domain {'kind': 'positive'}"):
            _simulation(artifact, params={nu: value})
