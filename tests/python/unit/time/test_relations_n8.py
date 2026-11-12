"""The relations and solves of a traced method (spec 0.3 §5.5, SV3, T38, IR16; N15 lot solves).

A method written for N8, without the surface: ``with unknowns(U=B.state.space) as z``,
``solve(Problem(unknowns=z, equations=(rel,)), request=p.request)`` and
``require_success(s).values.U``. A relation has no truth value, a traced value keeps its identity
hash, the request comes from the proposal unless the solve states one, the path is the name of the
method, a failed solve fails the attempt (T38), the operator's properties derive from the claims
of the components (IR16), and the solver refuses what they do not prove (``PropertyNotDerived``).
"""
import json

import pytest

import pops
from pops import math as pm
from pops._ir.refusals import PropertyNotDerived
from pops._ir.symbolic import SymbolicTruthValueError
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.lib.solve import cg
from pops.numerics import Diffusion, DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import ConstParam
from pops.time._program.elaboration import Advance, Proposal, Retry, elaborate
from pops.time._program.relations import (
    AmbiguousSolvePath, LinearClaims, MissingRequest, Problem, Relation, Request,
    UnknownSolvePath, require_success, solve, unknowns,
)
from pops.time._program.traced_method import (
    Candidate, ClosedBodyViolation, NotYetElaborated, method,
)

KAPPA = 0.01
REQUEST = Request(rtol=1e-10, atol=1e-12)


class _Linear:
    """A component with the linear map kappa*Laplacian and the claims ``claims``."""

    def __init__(self, selection, claims):
        self.selection = selection
        self.linear_claims = claims

    def __call__(self, state):
        return self.selection(state)

    def linear_apply(self, P, value):
        return KAPPA * P.laplacian(P.scalar_field("laplacian"), value)


_PROVED = LinearClaims(symmetric=True, negative_semidefinite=True, proof="kappa > 0",
                       description="kappa*Laplacian")
_UNPROVED = LinearClaims(symmetric=True, negative_semidefinite=False, proof="no sign",
                         description="kappa*Laplacian")


def _case():
    frame = CartesianDomain("box", lower=(0.0,), upper=(1.0,)).frame(Cartesian1D())
    (x,) = frame.axes
    model = pops.Model("transport", frame=frame)
    a = model.value(model.param(ConstParam("a", 1.0)))
    nu = model.value(model.param(ConstParam("nu", KAPPA)))
    u = model.state("u", components=("u",))
    flux = model.flux("flux_u", state=u, frame=frame, components={x: (a * u[0],)},
                      waves={x: (a,)})
    diffusive = model.diffusive_flux("diffusive_flux_u", state=u, value=nu * pm.grad(u[0]))
    rate = model.rate("u", equation=pm.ddt(u) == -pm.div(flux) + pm.div(diffusive))
    plan = DiscretizationPlan()
    advection = rate.select(flux)
    plan.rates.add(advection, FiniteVolume(
        flux=flux, variables=variables.Conservative(u),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    diffusion = rate.select(diffusive)
    plan.rates.add(diffusion, Diffusion(flux=diffusive))
    case = pops.Case("transport")
    block = case.block("transport", model)
    case.numerics(plan, block=block)
    return case, block[u], advection, diffusion


def _elaborate(body, *, claims=_PROVED, request=REQUEST, realization=None, attempts=2):
    case, state, advection, diffusion = _case()
    specialized = method(body)(explicit=advection, implicit=_Linear(diffusion, claims))
    realization = {"%s/*" % specialized.name: cg()} if realization is None else realization
    return elaborate(Advance(specialized, retry=Retry(max_attempts=attempts)), case=case,
                     state=state, proposal=Proposal(1e-3, request=request),
                     realization=realization)


def imex(B, p, explicit, implicit):
    with unknowns(U=B.state.space) as z:
        s = solve(Problem(unknowns=z, equations=(
            z.U == B.state + p.dt * explicit(B.state, B.time)
            + p.dt * implicit(z.U, B.time + p.dt),)), request=p.request)
    return Candidate(require_success(s).values.U, B.time + p.dt)


def test_a_relation_has_no_truth_value_and_traced_values_keep_their_identity_hash():
    seen = {}

    def body(B, p, explicit, implicit):
        with unknowns(U=B.state.space) as z:
            relation = z.U == B.state + p.dt * implicit(z.U, B.time + p.dt)
            seen["relation"] = relation
            with pytest.raises(SymbolicTruthValueError):
                bool(relation)
            # The identity hash finds a traced value in a dict or a set without a relation.
            table = {B.state: 1, z.U: 2}
            seen["lookups"] = (table[B.state], table[z.U], z.U in {z.U})
            with pytest.raises(TypeError, match="!= states no equation"):
                B.state != z.U  # noqa: B015
            with pytest.raises(TypeError, match="reads none"):
                B.state == B.state  # noqa: B015
            s = solve(Problem(unknowns=z, equations=(relation,)), request=p.request)
        return Candidate(require_success(s).values.U, B.time + p.dt)

    _elaborate(body)
    assert isinstance(seen["relation"], Relation)
    assert seen["lookups"] == (1, 2, True)


def test_the_solve_takes_the_request_of_the_proposal_and_the_path_of_the_method():
    elaboration = _elaborate(imex)
    (site,) = elaboration.trace.solves
    assert site.request is REQUEST and site.path == "imex" and site.spd
    rows = [json.loads(row) for row in elaboration.control.trial_failures]
    assert rows[-1] == {"site": "imex/solve", "guard": "solve",
                        "variant": ["BudgetExhausted", "SolveFailed"],
                        "retry_can_change": True, "classified": True, "lot": None}


def test_a_stated_request_and_name_override_the_proposal_and_the_path():
    own = Request(atol=1e-6)

    def body(B, p, explicit, implicit):
        with unknowns(U=B.state.space) as z:
            s = solve(Problem(unknowns=z, equations=(
                z.U == B.state + p.dt * implicit(z.U, B.time + p.dt),)),
                request=own, name="stage")
        return Candidate(require_success(s).values.U, B.time + p.dt)

    elaboration = _elaborate(body, request=None, realization={"body/stage": cg()})
    (site,) = elaboration.trace.solves
    assert site.request is own and site.path == "body/stage" and site.name == "stage"


def test_a_solve_without_request_is_refused():
    with pytest.raises(MissingRequest, match="has no request"):
        _elaborate(imex, request=None)


def test_a_failed_solve_fails_the_attempt_not_the_run():
    elaboration = _elaborate(imex)
    (outcome,) = [value for value in elaboration.program._values if value.op == "solve_outcome"]
    assert type(outcome.attrs["action"]).__name__ == "RejectAttempt"
    assert outcome.attrs["trial_site"] == {"site": "imex/solve", "retry_can_change": True}
    assert elaboration.control.max_attempts == 2


def test_the_increment_derives_from_the_relation():
    described = _elaborate(imex).trace.describe()
    assert described["increment"] == [{"evaluation": 0, "duration": "1*dt"},
                                      {"evaluation": 1, "duration": "1*dt"}]


def test_cg_refuses_an_operator_whose_claims_do_not_prove_it_spd():
    """R1: the core derives SPD from the claims of the components, never sets it blindly."""
    with pytest.raises(PropertyNotDerived, match="symmetric positive definite") as caught:
        _elaborate(imex, claims=_UNPROVED)
    assert caught.value.property == "symmetric_positive_definite"
    assert caught.value.phase == "resolve"


def test_an_anti_diffusive_relation_is_not_proved_spd():
    def body(B, p, explicit, implicit):
        with unknowns(U=B.state.space) as z:
            s = solve(Problem(unknowns=z, equations=(
                z.U == B.state - p.dt * implicit(z.U, B.time + p.dt),)), request=p.request)
        return Candidate(require_success(s).values.U, B.time + p.dt)

    with pytest.raises(PropertyNotDerived):
        _elaborate(body)


def test_a_realization_key_that_names_no_solve_is_refused_with_the_solves():
    with pytest.raises(UnknownSolvePath, match="'imex/other'.*'imex'"):
        _elaborate(imex, realization={"imex/*": cg(), "imex/other": cg()})


def test_a_solve_without_solver_names_the_selection_of_n9():
    with pytest.raises(NotYetElaborated, match="auto\\(\\) of SV11"):
        _elaborate(imex, realization={})


def test_two_unnamed_solves_are_ambiguous():
    def body(B, p, explicit, implicit):
        with unknowns(U=B.state.space) as z:
            first = solve(Problem(unknowns=z, equations=(
                z.U == B.state + p.dt * implicit(z.U, B.time + p.dt),)), request=p.request)
        require_success(first)
        with unknowns(V=B.state.space) as w:
            solve(Problem(unknowns=w, equations=(
                w.V == B.state + p.dt * implicit(w.V, B.time + p.dt),)), request=p.request)
        return Candidate(B.state, B.time + p.dt)

    with pytest.raises(AmbiguousSolvePath, match="takes name="):
        _elaborate(body)


def test_a_solve_that_no_require_success_reads_is_refused():
    def body(B, p, explicit, implicit):
        with unknowns(U=B.state.space) as z:
            solve(Problem(unknowns=z, equations=(
                z.U == B.state + p.dt * implicit(z.U, B.time + p.dt),)), request=p.request)
        return Candidate(B.state + p.dt * explicit(B.state, B.time), B.time + p.dt)

    with pytest.raises(TypeError, match="no require_success"):
        _elaborate(body)


def test_a_component_without_a_linear_map_names_its_lot():
    def body(B, p, explicit, implicit):
        with unknowns(U=B.state.space) as z:
            explicit(z.U, B.time + p.dt)
        return Candidate(B.state, B.time + p.dt)

    with pytest.raises(NotYetElaborated, match="states no linear map"):
        _elaborate(body)


def test_relations_outside_this_lot_name_their_lot():
    def several(B, p, explicit, implicit):
        unknowns(U=B.state.space, V=B.state.space)

    def nonlinear(B, p, explicit, implicit):
        with unknowns(U=B.state.space) as z:
            implicit(z.U + B.state, B.time)

    def weighted(B, p, explicit, implicit):
        with unknowns(U=B.state.space) as z:
            solve(Problem(unknowns=z, equations=(
                p.dt * z.U == B.state + p.dt * implicit(z.U, B.time + p.dt),)),
                request=p.request)

    for body, match in ((several, "one unknown"), (nonlinear, "combination of an unknown"),
                        (weighted, "weighs its unknown")):
        with pytest.raises(NotYetElaborated, match=match):
            _elaborate(body)


def test_a_relation_weighs_its_unknown_by_one():
    """A1 of the review of #776: the accepted commit attributes the transport of the solution to
    the right-hand side only at the weight 1, so another weight is refused by name in solve()."""
    def doubled(B, p, explicit, implicit):
        with unknowns(U=B.state.space) as z:
            s = solve(Problem(unknowns=z, equations=(
                2 * z.U == 2 * B.state + 2 * p.dt * implicit(z.U, B.time + p.dt),)),
                request=p.request)
        return Candidate(require_success(s).values.U, B.time + p.dt)

    with pytest.raises(NotYetElaborated, match="weighs its unknown by 2: this lot elaborates "
                                               "the weight 1"):
        _elaborate(doubled)


def test_a_relation_is_affine_in_its_unknown():
    def body(B, p, explicit, implicit):
        with unknowns(U=B.state.space) as z:
            z.U * B.state  # noqa: B018

    with pytest.raises(TypeError, match="affine in its unknowns"):
        _elaborate(body)


def test_unknowns_and_solve_are_written_in_a_method_body():
    with pytest.raises(ClosedBodyViolation):
        unknowns(U=None)


def test_a_request_states_a_positive_tolerance():
    with pytest.raises(ValueError, match="rtol=, atol= or both"):
        Request()
    with pytest.raises(ValueError, match="positive finite"):
        Request(rtol=-1.0)
    with pytest.raises(NotYetElaborated, match="observe"):
        Request(rtol=1e-6, observe=object())
    with pytest.raises(TypeError, match="takes a Request"):
        Proposal(1e-3, request=1e-6)


def test_a_relation_solve_under_cg_is_never_a_fixed_site():
    """``cg()``, the one solver of this lot with a native, needs an SPD operator (IR16): every
    ``b_k`` is a nonpositive multiple of ``p.dt``, so every image of the unknown depends on the
    proposal and the site is retryable (T38). A relation whose image has no ``p.dt`` is refused,
    and so is a relation without image. So ``require_success`` never writes a fixed trial site,
    which ``_attempt_reuse`` would otherwise reach (``test_reuse_across_stages_n8.py``)."""
    (site,) = _elaborate(imex).trace.solves
    assert not site.fixed

    def without_dt(B, p, explicit, implicit):
        with unknowns(U=B.state.space) as z:
            s = solve(Problem(unknowns=z, equations=(
                z.U == B.state - 0.5 * implicit(z.U, B.time),)), request=p.request)
        return Candidate(require_success(s).values.U, B.time + p.dt)

    def without_image(B, p, explicit, implicit):
        with unknowns(U=B.state.space) as z:
            s = solve(Problem(unknowns=z, equations=(z.U == B.state,)), request=p.request)
        return Candidate(require_success(s).values.U, B.time + p.dt)

    with pytest.raises(PropertyNotDerived, match="nonpositive multiple of p.dt"):
        _elaborate(without_dt)
    with pytest.raises(NotYetElaborated, match="reads no rate at its unknown"):
        _elaborate(without_image)
    assert cg().requires == ("symmetric_positive_definite",)
