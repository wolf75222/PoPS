"""The relation route on the reference, native (N9 « La référence sous les solves de relation »,
ROM-90, §5 of the lot note).

L1-d of §17.1 on a periodic box of rank two, nu a ``model.constant``: ``imex_euler`` solves
(I - dt nu L_h) u1 = u0 + dt E_h(u0) through ``cg()`` of ``pops.lib.solve``, the reference
conjugate gradient run as the region of the solve's node, with the relation's own rate (the
two-point StencilFlux) as its operator.

- **The Trace witness.** The same IMEX step written by hand on b6: a Program with a field-level
  ``set_apply`` of x - dt nu ``P.laplacian``(x) solved by the native ``pops.solvers.CG``, run a
  step at a time through ``pops.step`` (it has no Advance). The two sides apply different
  discrete operators, the StencilFlux and kappa ``P.laplacian``, which agree to round-off (lot G's
  parity, ``test_surface_lot_g_run_native.py``), so over 50 steps the states agree within the
  requests of the solves, and the iteration counts of each solve, read through
  ``outcome.evidence``, are within 2% (SV13). A second case, whose atol is above rtol |b|, checks
  that the gate's absolute threshold means the same on both routes (§3.3). One step of each route
  is the dense solve of the same discrete operator within the request, which measures the gap
  between the two operators once.
- **The gate.** A solver of the case that returns a candidate it did not converge, with a
  residual of 0 claimed: the gate measures the true residual and refuses it, the status
  ``safeguard_failure``, and the attempt is rejected at the relation's retryable site.
- **The iteration limit at a retryable relation site.** ``cg(max_iterations=2)``: every attempt
  ends at the iteration limit, and the run ends with ``BudgetExhausted`` (attempts) whose cause is
  ``BudgetExhausted`` (``max_iterations``).
- **A right-hand side that is not finite**, on both routes: the run fails at the first attempt
  with ``invalid_evaluation action=fail_run``, with no attempt rejected; the witness's message
  names ``solve_linear``, the relation route's names the site.

The artifacts are bound through ``pops.bind`` of b6 with the execution context of the loaded
native runtime (``artifact_execution_context``), on a serial build and on an MPI build alike.
"""
from pathlib import Path

import numpy as np
import pytest

import pops
import pops.surface as surface
from pops import math as pm
from pops.domain import CartesianDomain
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.numerics import two_point, upwind
from pops.lib.solve import cg
from pops.lib.time.methods import imex_euler
from pops.linalg import LinearOperatorProperties, LinearProblem
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan
from pops.numerics.terms import Flux
from pops.params import ConstParam
from pops.projection import ConservativeCellAverage
from pops.runtime._simulation import Simulation
from pops.solvers import CG
from pops.surface.math import ddt, div, grad, vector
from pops.time import FailRun, FixedDt
from examples.api_v02 import b6_numerics
from tests.python.integration.runtime.test_reference_cg_trace_n9 import _as_real
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

REPO_ROOT = Path(__file__).resolve().parents[4]
DT = 1e-3
NU = 1e-2
STEPS = 50
MAXIT = 2000
# (rtol, atol): the relative threshold binds, then the absolute one (atol > rtol |b|, |b| ~ 64).
REQUESTS = {"relative": (1e-10, 1e-14), "absolute": (1e-10, 1e-6)}


def _initial(cells):
    x = (np.arange(cells) + 0.5) / cells
    return (np.sin(2 * np.pi * x)[:, None] * np.cos(2 * np.pi * x)[None, :]
            + 0.3 * np.cos(6 * np.pi * x)[:, None] + 0.01 * np.arange(cells)[None, :])


def _record_evidence(program, outcome):
    program.record_scalar("cg_iterations", _as_real(program, outcome.evidence.iterations))
    program.record_scalar("cg_residual", outcome.evidence.residual_norm)


@surface.method
def backward_euler(step, implicit):
    """Backward Euler of an implicit rate alone, X = u + h I(X, t + h): the right-hand side of
    its relation is the state, which no explicit rate reads first."""
    X = surface.unknown(step.state.space)
    return surface.Candidate(
        surface.solve(X == step.state + step.dt * implicit(X, step.time + step.dt), X))


def _relation(cells, rtol, atol, solver=None, attempts=2, advection=True):
    """L1-d through the surface, with the evidence of its relation solve recorded; without
    ``advection``, the diffusion alone by ``backward_euler``."""
    domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    model = surface.Model("advection_diffusion" if advection else "diffusion")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0) if advection else None
    nu = model.constant("nu", value=NU)
    u0 = model.data("u0", surface.Scalar(domain))
    transport = model.flux(of=u, value=vector(a, 0.5 * a) * u) if advection else None
    diffusion = model.diffusive_flux(of=u, value=-nu * grad(u))
    if advection:
        model.balance(ddt(u) == -div(transport) - div(diffusion))
        schemes = {transport: upwind(), diffusion: two_point()}
        method = imex_euler(explicit=transport, implicit=diffusion)
    else:
        model.balance(ddt(u) == -div(diffusion))
        schemes = {diffusion: two_point()}
        method = backward_euler(implicit=diffusion)
    numerics = surface.Numerics(represent={u: surface.CellAverage()}, schemes=schemes)
    program = surface.Program(
        advance=surface.Advance(method, dt=DT, attempts=attempts,
                                tolerance=surface.Tolerance(rtol=rtol, atol=atol)),
        initial={u: u0}, solvers={method: cg(max_iterations=MAXIT) if solver is None else solver})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=(cells, cells))})
    elaborated = case.elaborated
    (site,) = elaborated.advance.trace.solves
    _record_evidence(elaborated.case._time, site.outcome)
    return surface.compile(case, target=surface.OpenMP()).compiled


def _witness(cells, rtol, atol, explicit=True):
    """The same IMEX step written by hand on b6: the native CG on x - dt nu P.laplacian(x);
    without ``explicit``, the backward Euler step of the diffusion alone."""
    frame = CartesianDomain("box", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x, y = frame.axes
    model = pops.Model("advection_diffusion", frame=frame)
    a = model.value(model.param(ConstParam("a", 1.0)))
    u = model.state("u", components=("u",))
    flux_u = model.flux("flux_u", state=u, frame=frame,
                        components={x: (a * u[0],), y: (0.5 * a * u[0],)},
                        waves={x: (a,), y: (0.5 * a,)})
    rate = model.rate("u", equation=pm.ddt(u) == -pm.div(flux_u))
    plan = DiscretizationPlan()
    plan.rates.add(rate.select(flux_u), b6_numerics.rusanov(flux_u, u))
    case = pops.Case("witness")
    block = case.block("advection_diffusion", model)
    case.numerics(plan, block=block)
    case.initials.add(InitialCondition(state=block[u], value=BindArray(),
                                       projection=ConservativeCellAverage()))
    program = pops.Program("witness")
    temporal = program.state(block[u])
    current = temporal.n
    rhs = (program.value("rhs", current + program.dt * program.rhs(state=current, terms=[Flux()]),
                         at=current.point) if explicit else current)
    operator = program.matrix_free_operator("helmholtz", domain="state", range_="state", ncomp=1)
    program.set_apply(operator, lambda p, out, v: v - (DT * NU) * p.laplacian(
        p.scalar_field("lap", ncomp=1), v))
    problem = LinearProblem(operator, rhs, initial_guess=current, nullspace=None,
                            properties=LinearOperatorProperties.symmetric_positive_definite())
    outcome = program.solve(problem, solver=CG(max_iter=MAXIT, rel_tol=rtol, abs_tol=atol),
                            name="sol")
    solution = outcome.consume(action=FailRun())
    _record_evidence(program, outcome)
    program.commit(temporal.next, program.value("accepted", solution, at=temporal.next.point))
    program.step_strategy(FixedDt(DT))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(cells, cells),
                                   periodic=PeriodicAxes(frame.axes)))
    return pops.compile(pops.resolve(pops.validate(case), layout=layout))


def _bound(compiled, initial):
    (binding,) = tuple(compiled.plan.initial_condition_plan.bindings)
    return pops.bind(
        compiled, initial_values={binding.subject: np.ascontiguousarray(
            np.transpose(initial)[np.newaxis, ...])},
        resources={"execution_context": artifact_execution_context(compiled)})


def _steps(compiled, initial, steps, *, witness):
    """The evidence of each step's solve and the final state, a step at a time: the relation
    route through the Simulation of its compiled Advance, to each frontier k dt, and the witness,
    which has no Advance, through ``pops.step`` (its FixedDt chooses dt)."""
    runtime = _bound(compiled, initial)
    simulation = None if witness else Simulation(runtime)
    counts, residuals = [], []
    for step in range(1, steps + 1):
        if witness:
            pops.step(runtime, t_bound=1.0)
        else:
            result = simulation.run(step * DT)
            assert result.failure is None and result.steps == 1, result
        diagnostics = runtime.program_report().diagnostics
        counts.append(int(diagnostics["cg_iterations"]))
        residuals.append(float(diagnostics["cg_residual"]))
    if witness:
        state = np.asarray(runtime.state_global("advection_diffusion"), dtype=np.float64)
    else:
        (state,) = result.state.values()
    cells = initial.shape[0]
    return counts, residuals, np.transpose(np.asarray(state, dtype=np.float64)
                                           .reshape((1, cells, cells))[0])


@pytest.fixture(scope="module")
def compiled(tmp_path_factory, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    cache = tmp_path_factory.mktemp("pops-native-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        patch.chdir(REPO_ROOT)
        yield {name: (_relation(64, *request), _witness(64, *request))
               for name, request in REQUESTS.items()}


@pytest.mark.parametrize("name", sorted(REQUESTS))
def test_the_relation_route_is_in_the_trace_class_of_the_native_witness(compiled, name):
    relation, witness = compiled[name]
    rtol, atol = REQUESTS[name]
    initial = _initial(64)
    counts, residuals, state = _steps(relation, initial, STEPS, witness=False)
    native_counts, native_residuals, native_state = _steps(witness, initial, STEPS, witness=True)
    print("%s: iterations reference %s, native %s" % (name, counts[:5], native_counts[:5]))
    for reference, native in zip(counts, native_counts, strict=True):
        assert native > 0
        assert abs(reference - native) <= 0.02 * native
    # Each solve meets its request on both routes; the threshold of the request is
    # max(atol, rtol |b|) (§3.3), with |b| <= sqrt(64^2) max|u| + dt |E|, about 1.1 * 64 * 1.4.
    norm_b = 1.1 * 64 * np.max(np.abs(initial))
    threshold = max(atol, rtol * norm_b)
    assert max(residuals) <= threshold and max(native_residuals) <= threshold
    if name == "absolute":
        assert threshold == atol
    # The operators agree to round-off and the scheme does not amplify an error (the implicit
    # operator has eigenvalues >= 1, the upwind step is contractive at a dt/h = 0.064): the
    # states differ by at most the residuals that the requests admit, over the steps.
    difference = float(np.linalg.norm(state - native_state))
    print("%s: |reference - native| = %.3e after %d steps" % (name, difference, STEPS))
    assert difference <= 2 * STEPS * threshold


def _dense_step(cells):
    """u1 of one imex_euler step by a dense solve, as the exit test of the lot writes it
    (``test_relation_solve_parameter_n9_native.py``): (I - dt nu L_h) u1 = u0 + dt E_h(u0), E_h the
    upwind divergence of vector(a, a/2) u and L_h the periodic five-point Laplacian, which the
    two-point StencilFlux and ``P.laplacian`` both are on this grid."""
    h = 1.0 / cells
    u0 = _initial(cells)
    explicit = -((u0 - np.roll(u0, 1, axis=0)) / h + 0.5 * (u0 - np.roll(u0, 1, axis=1)) / h)
    size = cells * cells
    index = np.arange(size).reshape(cells, cells)
    matrix = (1.0 + DT * NU * 4.0 / h**2) * np.eye(size)
    for axis in (0, 1):
        for shift in (1, -1):
            matrix[index.ravel(), np.roll(index, shift, axis=axis).ravel()] -= DT * NU / h**2
    rhs = (u0 + DT * explicit).reshape(size)
    return np.linalg.solve(matrix, rhs).reshape(cells, cells), float(np.linalg.norm(rhs))


def test_one_step_of_each_route_is_the_dense_solve_of_the_same_discrete_operator(compiled):
    """The gap between the two discrete operators, measured once (§5): one step of each route from
    the same state against a dense solve. The operator is SPD with a condition number below 4/3
    (dt nu 8 / h^2 = 1/3), so the error of each state is bounded by the residual that the request
    admits, rtol |b| + atol."""
    relation, witness = compiled["relative"]
    rtol, atol = REQUESTS["relative"]
    dense, norm = _dense_step(64)
    bound = 4 * (rtol * norm + atol)
    states = {}
    for name, artifact in (("relation", relation), ("witness", witness)):
        _, _, states[name] = _steps(artifact, _initial(64), 1, witness=name == "witness")
        print("one step, %s: |state - dense| = %.3e (bound %.3e)"
              % (name, float(np.linalg.norm(states[name] - dense)), bound))
        assert np.linalg.norm(states[name] - dense) <= bound
    print("one step: |relation - witness| = %.3e"
          % float(np.linalg.norm(states["relation"] - states["witness"])))


def test_the_gate_refuses_a_candidate_that_the_body_accepts(native_cxx, kokkos_root, tmp_path,
                                                            monkeypatch):
    """A solver whose body returns the initial guess as its candidate, with a residual of 0
    claimed: the body accepts it, the gate measures the true residual and refuses it
    (safeguard_failure, §3.3), and the attempt is rejected at the relation's retryable site."""
    del native_cxx, kokkos_root
    monkeypatch.setenv("POPS_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("POPS_NATIVE_CACHE_DIR", str(tmp_path))
    monkeypatch.chdir(REPO_ROOT)

    @surface.solver
    def unconverged(problem, region):
        start = surface.record(x=problem.guess, r=0.0, k=0)
        result = region.iterate(start=start,
                                step=lambda s: surface.record(x=s.x, r=s.r, k=s.k + 1),
                                stop=lambda s: s.k > 0, max_steps=1)
        return problem.candidate(result.x, iterations=result.iterations, residual_norm=result.r)

    artifact = _relation(16, 1e-10, 1e-12, solver=unconverged(), attempts=2)
    simulation = Simulation(_bound(artifact, _initial(16)))
    from pops.runtime._simulation import BudgetExhausted

    with pytest.raises(BudgetExhausted) as caught:
        simulation.run(DT)
    message = str(caught.value)
    print("gate:", message[:600])
    # each attempt is rejected by the gate, which measured the true residual of the candidate
    assert caught.value.result.rejections == 2
    assert "solve status=safeguard_failure" in message, message
    assert "SV8 gate original_residual" in message and "realization reported solved" in message


def test_an_iteration_limit_at_a_retryable_relation_site_exhausts_the_attempts(
        native_cxx, kokkos_root, tmp_path, monkeypatch):
    del native_cxx, kokkos_root
    monkeypatch.setenv("POPS_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("POPS_NATIVE_CACHE_DIR", str(tmp_path))
    monkeypatch.chdir(REPO_ROOT)
    from pops.runtime._simulation import BudgetExhausted

    artifact = _relation(16, 1e-10, 1e-12, solver=cg(max_iterations=2), attempts=3)
    simulation = Simulation(_bound(artifact, _initial(16)))
    with pytest.raises(BudgetExhausted) as caught:
        simulation.run(DT)
    failure = caught.value.result.failure
    print("budget:", failure.payload)
    assert caught.value.result.rejections == 3
    assert (failure.variant, failure.origin) == (BudgetExhausted, "trial")
    assert (failure.payload["budget"], failure.payload["attempts"]) == ("max_attempts", 3)
    cause = failure.payload["cause"]
    assert cause.variant is BudgetExhausted, cause.describe()
    assert cause.payload["budget"] == "max_iterations", cause.describe()
    assert cause.payload["site"] == "imex_euler/solve", cause.describe()


def test_a_right_hand_side_that_is_not_finite_fails_the_run_on_both_routes(
        native_cxx, kokkos_root, tmp_path, monkeypatch):
    """§3.2 and the ruling on F2: the run fails at its first attempt with the RuntimeError
    ``<site> failed: invalid_evaluation action=fail_run``, no attempt rejected, on both
    realizations. The advection's face evaluation refuses a state that is not finite before any
    solve (« prepared ND hyperbolic face evaluation refused publication »), so the case is the
    diffusion alone by backward Euler, whose right-hand side is the state: on the surface with
    ``cg()``, and written by hand on b6 with the native CG, whose message names ``solve_linear``
    instead of the site."""
    del native_cxx, kokkos_root
    monkeypatch.setenv("POPS_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("POPS_NATIVE_CACHE_DIR", str(tmp_path))
    monkeypatch.chdir(REPO_ROOT)
    relation = _relation(16, 1e-10, 1e-14, advection=False)
    witness = _witness(16, 1e-10, 1e-14, explicit=False)
    initial = _initial(16)
    initial[3, 5] = np.nan
    messages = {}
    with pytest.raises(RuntimeError) as caught:
        Simulation(_bound(relation, initial)).run(DT)
    messages["relation"] = str(caught.value)
    with pytest.raises(RuntimeError) as caught:
        pops.step(_bound(witness, initial), t_bound=1.0)
    messages["witness"] = str(caught.value)
    for route in messages:
        print("%s: %s" % (route, messages[route][:400]))
    assert "solve_linear failed: invalid_evaluation action=fail_run" in messages["witness"]
    assert "backward_euler/solve failed: invalid_evaluation action=fail_run" in messages["relation"]
