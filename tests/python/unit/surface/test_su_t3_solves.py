"""SU-T3 (spec 0.3, 2.6.7) for the solves of a method, the lot « solves » of N15 (ROM-98).

The L1-d case of §17.1: ``imex_euler(explicit=advection, implicit=diffusion)`` with
``solvers={imex: cg()}`` and ``tolerance=pops.Tolerance(rtol=1e-10, atol=1e-12)``. Its ν is a
``model.constant`` here; a ``model.parameter`` ν is the exit test of the N9 lot « La référence
sous les solves de relation ». The case written by hand is the b6 model of
``test_su_t3_parity.py`` (the face of the b6 ``rusanov`` of ``examples/api_v02/b6_numerics.py``
and the two-point ``StencilFlux`` of N3 lot 1, D7) and the method written for N8 in the shape of
§5.5 (``unknowns``, ``Problem``, ``solve(request=p.request)``, ``require_success``), whose
implicit component states the claims of its two-point rate (symmetric, negative semidefinite,
linear and homogeneous in the state) and the realization ``{"imex_euler/*": cg()}``. The operator
of the solve is that rate at the State, solved by the reference ``cg()`` (N9). Both forms give
the same program IR, the same artifact without the source lines of its provenance, the same
semantic identity and the same generated C++. The solve starts from the state at the start of the
attempt (SV3) and its path is the key of the method, ``imex_euler`` (5.5, line 707).
"""
import json

import pytest

import pops
import pops.surface as surface
from pops import math as pm
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from examples.api_v02 import b6_numerics
from pops.lib.numerics import two_point, upwind
from pops.lib.solve import cg
from pops.lib.time.methods import imex_euler
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan
from pops.numerics.stencil_flux import StencilFlux
from pops.params import ConstParam
from pops.projection import ConservativeCellAverage
from pops.surface.math import ddt, div, grad
from pops.time._program.elaboration import Advance, Proposal, Retry, elaborate
from pops.time._program.relations import (
    LinearClaims, Problem, Request, require_success, solve, unknowns,
)
from pops.time._program.traced_method import Candidate, method

NU = 1e-2


def _surface_l1d():
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("advection_diffusion")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    nu = model.constant("nu", value=NU)
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=a * u)
    diffusion = model.diffusive_flux(of=u, value=-nu * grad(u))
    model.balance(ddt(u) == -div(advection) - div(diffusion))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: upwind(), diffusion: two_point()})
    imex = imex_euler(explicit=advection, implicit=diffusion)
    program = surface.Program(
        advance=surface.Advance(imex, dt=1e-3, attempts=2,
                                tolerance=surface.Tolerance(rtol=1e-10, atol=1e-12)),
        initial={u: u0}, solvers={imex: cg()})
    return surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=256)})


@method
def _n8_imex_euler(B, p, explicit, implicit):
    with unknowns(U=B.state.space) as z:
        s = solve(Problem(unknowns=z, equations=(
            z.U == B.state + p.dt * explicit(B.state, B.time)
            + p.dt * implicit(z.U, B.time + p.dt),)), request=p.request)
    return Candidate(require_success(s).values.U, B.time + p.dt)


_n8_imex_euler.name = "imex_euler"


class _TwoPointRate:
    """The two-point rate written by hand, with the claims of its rate read at the unknown (the
    operator of the solve evaluates the rate itself at its operand, N9 « La référence sous les
    solves de relation » §3.4)."""

    linear_claims = LinearClaims(symmetric=True, negative_semidefinite=True, homogeneous=True,
                                 proof="kappa = 0.01 > 0", description="kappa*Laplacian")

    def __init__(self, flux, name):
        self.flux = flux
        self.name = name

    def __call__(self, state):
        # -div_h of the face flux, named <name>_<k> in the k-th evaluation of the Program, top
        # level or inside a solve (N3 lot 1, D7)
        program = state.prog
        counts = program.__dict__.setdefault("_stencil_term_evaluations", {})
        evaluation = counts.get(self.name, 0)
        counts[self.name] = evaluation + 1
        return program.stencil_flux_divergence("%s_%d" % (self.name, evaluation), state,
                                               flux=self.flux)


def _two_point_body(nu):
    """The two-point face -nu (u_R - u_L) / h of ``two_point()`` and its frequency (nu / h) / h
    (D10), written by hand as the body of a ``StencilFlux``."""
    def body(sample, spacing, axis):
        h = spacing[axis]
        left, right = sample((-1,), 0), sample((0,), 0)  # sampled in the order of the scheme
        return ((-nu) * ((right - left) / h),), (nu / h) / h

    return body


def _hand_l1d():
    frame = CartesianDomain("box", lower=(0.0,), upper=(1.0,)).frame(Cartesian1D())
    (x,) = frame.axes
    model = pops.Model("advection_diffusion", frame=frame)
    a = model.value(model.param(ConstParam("a", 1.0)))
    nu = model.value(model.param(ConstParam("nu", NU)))
    u = model.state("u", components=("u",))
    flux_u = model.flux("flux_u", state=u, frame=frame, components={x: (a * u[0],)},
                        waves={x: (a,)})
    model.diffusive_flux("diffusive_flux_u", state=u, value=nu * pm.grad(u[0]))
    rate = model.rate("u", equation=pm.ddt(u) == -pm.div(flux_u))
    plan = DiscretizationPlan()
    advection = rate.select(flux_u)
    plan.rates.add(advection, b6_numerics.rusanov(flux_u, u))
    diffusion = StencilFlux(dimension=1, components=1, radius=1, body=_two_point_body(nu))
    case = pops.Case("advection_diffusion")
    block = case.block("advection_diffusion", model)
    case.numerics(plan, block=block)
    case.initials.add(InitialCondition(state=block[u], value=BindArray(),
                                       projection=ConservativeCellAverage()))
    elaboration = elaborate(
        Advance(_n8_imex_euler(explicit=advection,
                               implicit=_TwoPointRate(diffusion, "diffusive_flux_u")),
                retry=Retry(max_attempts=2)),
        case=case, state=block[u],
        proposal=Proposal(0.001, request=Request(rtol=1e-10, atol=1e-12)),
        realization={"imex_euler/*": cg()})
    layout = Uniform(CartesianGrid(frame=frame, cells=(256,), periodic=PeriodicAxes((x,))))
    return case, layout, elaboration


def _resolved(case, layout):
    return pops.resolve(pops.validate(case), layout=layout)


def _without_provenance(value):
    if isinstance(value, dict):
        return {key: _without_provenance(item) for key, item in value.items()
                if key != "provenance"}
    if isinstance(value, list):
        return [_without_provenance(item) for item in value]
    return value


def _cpp(resolved):
    return emit_cpp_program(resolved.time, model_graph=ProgramModelGraph.from_resolved_blocks(
        resolved.blocks))


@pytest.fixture(scope="module")
def l1d():
    case = _surface_l1d()
    hand, layout, elaboration = _hand_l1d()
    elaborated = case.elaborated
    return {"case": case, "elaborated": elaborated, "hand": hand, "hand_elaboration": elaboration,
            "resolved": _resolved(elaborated.case, elaborated.layout),
            "hand_resolved": _resolved(hand, layout)}


def test_l1d_gives_the_program_ir_of_the_hand_written_case(l1d):
    assert l1d["elaborated"].case._time._serialize(include_provenance=False) == \
        l1d["hand"]._time._serialize(include_provenance=False)


def test_l1d_gives_the_artifact_and_semantic_identity_of_the_hand_written_case(l1d):
    resolved, hand = l1d["resolved"], l1d["hand_resolved"]
    artifact = json.loads(resolved.snapshot._artifact_canonical_json)
    by_hand = json.loads(hand.snapshot._artifact_canonical_json)
    assert _without_provenance(artifact) == _without_provenance(by_hand)
    assert resolved.snapshot.semantic_identity == hand.snapshot.semantic_identity


def test_l1d_gives_the_cpp_of_the_hand_written_case(l1d):
    cpp = _cpp(l1d["resolved"])
    assert cpp == _cpp(l1d["hand_resolved"])
    # The reference CG of pops.lib.solve as the region of the solve (N9 « La référence sous les
    # solves de relation »): its operator is the two-point StencilFlux of the rate at the State,
    # not b6's Laplacian, no native Krylov route is emitted, and a failure rejects the attempt.
    (solve,) = [value for value in l1d["elaborated"].case._time._values
                if value.op == "solve_linear"]
    assert solve.attrs["realization"]["kind"] == "reference"
    assert (float.fromhex(solve.attrs["realization"]["rtol"]),
            float.fromhex(solve.attrs["realization"]["atol"])) == (1e-10, 1e-12)
    assert "pops::SolveOutcome kr%d = pops::SolveOutcome::collective_lane(" % solve.id in cpp
    assert "pops::cg_krylov_method" not in cpp and "solve_prepared_linear" not in cpp
    assert "ctx.laplacian(" not in cpp
    assert "pops::SolveAction::kRejectAttempt" in cpp


def test_l1d_solve_takes_the_key_of_the_method_and_fails_the_attempt(l1d):
    advance = l1d["elaborated"].advance
    (site,) = advance.trace.solves
    assert site.path == "imex_euler" and site.spd
    assert advance.control.max_attempts == 2
    rows = [json.loads(row) for row in advance.control.trial_failures]
    assert {"site": "imex_euler/solve", "guard": "solve",
            "variant": ["BudgetExhausted", "SolveFailed"],
            "retry_can_change": True, "classified": True, "lot": None} in rows
    # T34: the increment derives from the relation, h E(u) + h I(X), I at the stage of the solve.
    described = advance.trace.describe()
    assert described["increment"] == [{"evaluation": 0, "duration": "1*dt"},
                                      {"evaluation": 1, "duration": "1*dt"}]
    assert described["evaluations"][1] == {"evaluation": 1, "component": "implicit",
                                           "stage": 1}


def test_l1d_solve_starts_from_the_state_at_the_start_of_the_attempt(l1d):
    (site,) = l1d["elaborated"].advance.trace.solves
    guess = site.problem.initial_guess
    state = l1d["elaborated"].case._time._values[0]
    assert guess is state and state.op == "state"
    (solve,) = [value for value in l1d["elaborated"].case._time._values
                if value.op == "solve_linear"]
    assert solve.inputs[2] is state


def test_inspect_reports_the_solve_and_its_certificate(l1d):
    report = str(surface.inspect(l1d["case"]))
    assert "[program] pops.solve(...) in imex_euler" in report
    assert "path = imex_euler (derived)" in report
    assert "guess = the state at the start of the attempt (SV3) (derived)" in report
    assert "solver = cg(max_iterations=2000, preconditioner=None) (stated)" in report
    assert "SPD certificate = derived (IR16) from the claims of the rates of the components, " \
        "which the operator evaluates at the State" in report
    assert "kappa = nu = 0.01 >= 0" in report
    assert "tolerance = pops.Tolerance(rtol=1e-10, atol=1e-12) (stated)" in report


def test_a_relation_solve_enters_no_field_cache_and_is_never_kept_across_attempts(l1d, monkeypatch):
    """The relation solve is the third writer of ``trial_site`` (``relations.require_success``)
    beside the field requests of N8 lot 5 and the guards of lot 6. Its right-hand side is the
    ``linear_combine`` of its known part, never a ``field_problem_load``: the b6 key and the
    traced cache of the lot « reuse across stages » leave it out, and ``_attempt_reuse`` (T7
    across attempts, lot 6) skips it. Its site is retryable under ``cg()``; a fixed one, forced
    here, is skipped too, because the reuse across attempts is argued for field requests only."""
    from pops.codegen import program_emit_solve as emit_solve
    from pops.codegen import program_field_reuse as field_reuse

    resolved = l1d["hand_resolved"]
    program = resolved.time
    (solve,) = [value for value in program._values if value.op == "solve_linear"]
    site = emit_solve._trial_site(program, solve)
    assert site == {"site": "imex_euler/solve", "retry_can_change": True}
    assert solve.inputs[1].op == "linear_combine"
    assert not any(value.op == "field_problem_load" for value in program._values)
    # Neither cache: the b6 key and the time-independent request both ask for a field load.
    assert field_reuse._key(solve, {}, target="system") is None
    assert not field_reuse._time_independent_request(solve, target="system")
    assert not field_reuse.reuse_field_solve(solve, {}, [], target="system")
    # Under the reference realization (N9 « La référence sous les solves de relation ») the
    # relation solve is the region of its node, with no prepared Krylov solve: the emission never
    # reaches the attempt reuse, and a fixed relation site under a reference is refused at
    # elaboration (§3.7, test_relations_n8.py::
    # test_a_fixed_site_under_a_reference_realization_is_refused_by_name).
    assert solve.attrs["realization"]["kind"] == "reference"
    seen = []
    attempt_reuse = emit_solve._attempt_reuse

    def spy(*args, **kwargs):
        seen.append((kwargs["prepared"], attempt_reuse(*args, **kwargs)))
        return seen[-1][1]

    monkeypatch.setattr(emit_solve, "_attempt_reuse", spy)
    source = _cpp(resolved)
    assert seen == []
    assert "attempt_slot" not in source and "field.reuses/" not in source
    # A fixed relation site (none exists under cg(), see test_relations_n8.py) is not kept either.
    monkeypatch.setattr(emit_solve, "_trial_site",
                        lambda prog, value: dict(site, retry_can_change=False))
    assert attempt_reuse(program, solve, [], [], "sol", target="system", prepared=True) is None
