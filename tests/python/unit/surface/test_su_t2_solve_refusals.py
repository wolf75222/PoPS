"""SU-T2 (spec 0.3, 2.6.7) for the solves of a method, the lot « solves » of N15 (ROM-98).

Each refusal names its rule and the construct to write, or the lot that elaborates what it
refuses: the tolerance of an advance whose method solves (SU9), a key of ``solvers=`` that names no
solve (``UnknownSolvePath``), a solve without solver (``auto()`` of SV11, N9), two unnamed solves
(``AmbiguousSolvePath``), an operator that is not proved symmetric positive definite for ``cg()``
(``PropertyNotDerived``, K4), and, with G10, an implicit term that is not a two-point diffusion,
a coefficient that varies with the state, a layout that is not uniform and periodic, and a
coefficient that is a parameter (the ``cg()`` bridge of MG9 has no coefficient bound at bind).
"""
import os

import pytest

import pops.surface as surface
from pops._ir.refusals import PropertyNotDerived
from pops.lib.numerics.schemes import two_point, upwind
from pops.lib.solve import cg
from pops.lib.time.methods import imex_euler
from pops.surface.math import ddt, div, grad

HERE = os.path.abspath(__file__)


def _line(marker):
    with open(HERE) as source:
        for number, text in enumerate(source, 1):
            if text.rstrip().endswith("# <- %s" % marker):
                return number
    raise AssertionError(marker)


def _model(nu=None, coefficient=None):
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("advection_diffusion")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    nu = model.constant("nu", value=1e-2) if nu is None else nu(model)
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=a * u)
    k = nu if coefficient is None else coefficient(nu, u)
    diffusion = model.diffusive_flux(of=u, value=-k * grad(u))
    model.balance(ddt(u) == -div(advection) - div(diffusion))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: upwind(), diffusion: two_point()})
    return domain, model, u, u0, advection, diffusion, numerics


TOLERANCE = surface.Tolerance(rtol=1e-10, atol=1e-12)


def _case(method=imex_euler, solvers=lambda imex: {imex: cg()}, tolerance=TOLERANCE, **model):
    domain, model_, u, u0, advection, diffusion, numerics = _model(**model)
    value = method(explicit=advection, implicit=diffusion)
    program = surface.Program(advance=surface.Advance(value, dt=1e-3, tolerance=tolerance),
                              initial={u: u0}, solvers=solvers(value))
    return surface.Case(model=model_, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=16)})


def test_an_advance_whose_method_solves_states_its_tolerance():
    with pytest.raises(surface.AmbiguousElaboration) as caught:
        _case(tolerance=None)
    message = str(caught.value)
    assert "the solve imex_euler of 'imex_euler' has no request" in message
    assert "a choice that PoPS never makes by a heuristic" in message
    assert "(D5d, SU9)" in message
    assert "Write pops.Advance(..., tolerance=pops.Tolerance(rtol=..., atol=...))" in message


def test_a_key_of_solvers_that_names_no_solve_is_refused_with_the_solves():
    from pops.lib.time.methods import forward_euler

    domain, model, u, u0, advection, diffusion, numerics = _model()
    euler = forward_euler(rate=model)
    program = surface.Program(  # <- no-solve
        advance=surface.Advance(euler, dt=1e-3), initial={u: u0}, solvers={euler: cg()})
    with pytest.raises(surface.UnknownSolvePath) as caught:
        surface.Case(model=model, numerics=numerics, program=program,
                     layout={domain: surface.Uniform(cells=16)})
    message = str(caught.value)
    assert "'forward_euler/*' name no solve of 'forward_euler', whose solves are none" in message
    assert (caught.value.site.file, caught.value.site.line) == (HERE, _line("no-solve"))


def test_a_named_solve_path_of_another_name_is_refused():
    with pytest.raises(surface.UnknownSolvePath, match="'imex_euler/stage2'.*'imex_euler'"):
        _case(solvers=lambda imex: {imex: cg(), imex.stage2: cg()})


def test_a_key_of_another_method_is_refused():
    from pops.lib.time.methods import ssprk2

    domain, model, u, u0, advection, diffusion, numerics = _model()
    other = ssprk2(rate=model)
    with pytest.raises(surface.UnknownSolvePath, match="names no solve: the advance runs"):
        _case(solvers=lambda imex: {imex: cg(), other: cg()})


def test_a_solve_without_solver_names_the_selection_of_n9():
    with pytest.raises(surface.NotYetElaborated, match="auto\\(\\) of SV11 comes with N9"):
        _case(solvers=lambda imex: None)


def test_solvers_are_keyed_by_handles_and_take_solvers_of_the_library():
    with pytest.raises(TypeError, match="keyed by handles, not by the string 'imex_euler'"):
        _case(solvers=lambda imex: {"imex_euler": cg()})
    with pytest.raises(TypeError, match="takes a solver of pops.lib.solve"):
        _case(solvers=lambda imex: {imex: "cg"})


def test_a_coefficient_that_is_a_parameter_names_the_bridge_that_reads_it():
    """§17.1 writes nu = model.parameter("nu", default=1e-2, admissible=pops.Positive()); the
    matrix-free apply of the cg() bridge has no coefficient bound at bind, so this lot refuses it
    until the reference CG of N9 replaces the bridge (MG9)."""
    with pytest.raises(surface.NotYetElaborated) as caught:
        _case(nu=lambda model: model.parameter("nu", default=1e-2,
                                               admissible=surface.Positive()))
    message = str(caught.value)
    assert "reads the parameter nu" in message
    assert "write model.constant('nu', value=...)" in message
    assert "the reference CG of N9 in pops.lib.solve with the operator generated by G10" in message


def test_a_coefficient_that_varies_with_the_state_is_refused():
    with pytest.raises(surface.NotYetElaborated, match="varies with the state"):
        _case(coefficient=lambda nu, u: nu * (1 + u * u))


def test_a_layout_that_is_not_uniform_and_periodic_is_refused():
    from pops.lib.numerics.schemes import _TwoPoint

    domain, model, u, u0, advection, diffusion, numerics = _model()
    view = surface.TermView(diffusion, None, None)
    for layout in ({"uniform": False, "cartesian": True, "periodic": True},
                   {"uniform": True, "cartesian": True, "periodic": False}):
        with pytest.raises(surface.NotYetElaborated, match="uniform, Cartesian and periodic"):
            _TwoPoint("two_point").linear_operator(view, weight=1, layout=layout)
    linear = _TwoPoint("two_point").linear_operator(
        view, weight=1, layout={"uniform": True, "cartesian": True, "periodic": True})
    assert linear.kappa == 1e-2 and linear.claims().negative_semidefinite


def test_an_anti_diffusive_relation_is_not_proved_spd_for_cg():
    @surface.method
    def backward(step, explicit, implicit):
        u, h = step.state, step.dt
        X = surface.unknown(step.state.space)
        return surface.Candidate(surface.solve(X == u + h * explicit(u, step.time)
                                               - h * implicit(X, step.time + h), X))

    with pytest.raises(PropertyNotDerived) as caught:
        _case(method=backward)
    assert "cg(max_iterations=2000, preconditioner=None) needs a symmetric positive definite " \
        "operator" in str(caught.value)
    assert caught.value.phase == "resolve" and caught.value.rule == "IR13"


def test_a_relation_of_another_weight_is_refused_by_name():
    """A1 of the review of #776: 2 X == 2 u + 2 h E(u) + 2 h I(X) on L1-d is refused when the
    method is traced, naming the weight, not by the codegen of the accepted transport."""
    @surface.method
    def doubled(step, explicit, implicit):
        u, t, h = step.state, step.time, step.dt
        X = surface.unknown(step.state.space)
        return surface.Candidate(surface.solve(  # <- doubled
            2 * X == 2 * u + 2 * h * explicit(u, t) + 2 * h * implicit(X, t + h), X))

    with pytest.raises(surface.NotYetElaborated) as caught:
        _case(method=doubled)
    assert "weighs its unknown by 2: this lot elaborates the weight 1" in str(caught.value)
    assert (caught.value.site.file, caught.value.site.line) == (HERE, _line("doubled"))


def test_two_unnamed_solves_of_a_method_are_ambiguous():
    @surface.method
    def twice(step, explicit, implicit):
        u, t, h = step.state, step.time, step.dt
        X = surface.unknown(step.state.space)
        first = surface.solve(X == u + h * implicit(X, t + h), X)
        Y = surface.unknown(step.state.space)
        return surface.Candidate(surface.solve(Y == first + h * explicit(first, t + h)  # <- twice
                                               + 0 * h * implicit(Y, t + h), Y))

    with pytest.raises(surface.AmbiguousSolvePath) as caught:
        _case(method=twice)
    assert "each solve of a method that holds several takes name=" in str(caught.value)
    assert (caught.value.site.file, caught.value.site.line) == (HERE, _line("twice"))


def test_named_solves_take_their_keys():
    @surface.method
    def twice(step, explicit, implicit):
        u, t, h = step.state, step.time, step.dt
        X = surface.unknown(step.state.space)
        first = surface.solve(X == u + h * implicit(X, t + h), X, name="first")
        Y = surface.unknown(step.state.space)
        return surface.Candidate(surface.solve(
            Y == first + h * explicit(first, t + h) + h * implicit(Y, t + h), Y, name="second",
            tolerance=surface.Tolerance(atol=1e-6)))

    case = _case(method=twice, solvers=lambda m: {m.first: cg(), m.second: cg(max_iterations=50)})
    first, second = case.elaborated.advance.trace.solves
    assert (first.path, second.path) == ("twice/first", "twice/second")
    assert second.request.atol == 1e-6 and second.solver.options["max_iterations"] == 50


def test_the_options_of_pops_solve_outside_this_lot_name_their_lot():
    for options, match in (({"default": cg()}, "default solver"), ({"guess": 0}, "predictor"),
                           ({"several": True}, "several relations")):
        @surface.method
        def body(step, explicit, implicit, options=options):
            X = surface.unknown(step.state.space)
            relation = X == step.state + step.dt * implicit(X, step.time + step.dt)
            if options.pop("several", False):
                return surface.Candidate(surface.solve([relation], [X]))
            return surface.Candidate(surface.solve(relation, X, **options))

        with pytest.raises(surface.NotYetElaborated, match=match):
            _case(method=body)


def test_a_relation_has_no_truth_value_in_a_method():
    @surface.method
    def body(step, explicit, implicit):
        X = surface.unknown(step.state.space)
        if X == step.state:  # <- truth
            pass
        return surface.Candidate(step.state)

    with pytest.raises(surface.SymbolicTruthValue) as caught:
        _case(method=body)
    assert (caught.value.site.file, caught.value.site.line) == (HERE, _line("truth"))


def test_a_tolerance_states_a_positive_rtol_or_atol():
    with pytest.raises(TypeError, match="rtol=, atol= or both"):
        surface.Tolerance()
    with pytest.raises(TypeError, match="positive number"):
        surface.Tolerance(rtol=0)
    with pytest.raises(surface.NotYetElaborated, match="observes and its norm"):
        surface.Tolerance(rtol=1e-6, norm=object())
    assert repr(surface.Tolerance(rtol=1e-6)) == "pops.Tolerance(rtol=1e-06)"


def test_cg_options_outside_this_lot_name_their_lot():
    from pops.lib import solve

    with pytest.raises(surface.NotYetElaborated, match="a preconditioned solver.*G10"):
        cg(preconditioner=object())
    # The reference checks its budget when its body runs; the bound native that a relation solve
    # runs checks it when the solve is traced.
    with pytest.raises(ValueError, match="max_iterations=n"):
        _case(solvers=lambda imex: {imex: cg(max_iterations=0)})
    for word in ("auto", "gmres", "bicgstab", "newton", "jacobi", "richardson", "dense_lu"):
        with pytest.raises(surface.NotYetElaborated, match="N9"):
            getattr(solve, word)()


def test_a_linear_map_without_claims_is_refused():
    """A linear map of a rate that states no claims cannot derive the properties of the operator
    (IR16): the rate sum refuses it, naming G10, rather than reading claims that are not there."""
    from pops.surface._elaborate import _RateSum

    class Unclaimed:
        def claims(self):
            return None

        def apply(self, P, value):
            return value

    rate = _RateSum([lambda state: state], [Unclaimed])
    with pytest.raises(surface.NotYetElaborated, match="states no claims.*G10"):
        rate.linear_claims  # noqa: B018
