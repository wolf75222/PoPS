"""SU-T2 (spec 0.3, 2.6.7) for the solves of a method, the lot « solves » of N15 (ROM-98).

Each refusal names its rule and the construct to write, or the lot that elaborates what it
refuses: the tolerance of an advance whose method solves (SU9), a key of ``solvers=`` that names no
solve (``UnknownSolvePath``), a solve without solver (``auto()`` of SV11, N9), two unnamed solves
(``AmbiguousSolvePath``), an operator that is not proved symmetric positive definite for ``cg()``
(``PropertyNotDerived``, K4), and, with G10, an implicit term that is not a two-point diffusion,
a coefficient that varies with the state and a layout that is not uniform and periodic. A
coefficient that is a parameter is claimed from its admissibility by the exit test of the N9 lot
« La référence sous les solves de relation », which runs every relation solve on the reference
``cg()``: a solver written in a case runs there as the library's does.
"""
import os

import pytest

import pops.surface as surface
from pops._ir.refusals import PropertyNotDerived
from pops.lib.numerics import two_point, upwind
from pops.lib.solve import cg, linear_claims
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


def _nu(admissible=surface.Positive()):
    return lambda model: model.parameter("nu", default=1e-2, admissible=admissible)


def test_a_coefficient_that_is_a_parameter_is_claimed_from_its_admissible_set():
    """§17.1 writes nu = model.parameter("nu", default=1e-2, admissible=pops.Positive()) (§3.5 of
    the N9 lot « La référence sous les solves de relation »): the reference cg() reads nu where the
    StencilFlux of two_point() reads it, inside the region of the solve, and the claim of the rate
    holds for every value that the bind accepts, which pops.inspect reports. The native run of the
    exit test is tests/python/integration/runtime/test_relation_solve_parameter_n9_native.py."""
    case = _case(nu=_nu())
    (solve,) = [value for value in case.elaborated.case._time._values
                if value.op == "solve_linear"]
    assert solve.attrs["realization"]["kind"] == "reference"
    report = str(surface.inspect(case))
    assert "implicit operator = implicit: the two_point() rate of -nu * grad(u), kappa = nu " \
        "(derived)" in report
    assert "kappa = nu >= 0 for every value of nu admissible in pops.Positive(), which the bind " \
        "checks, and the two-point flux of a uniform periodic grid is symmetric negative " \
        "semidefinite" in report


def test_a_parameter_with_no_admissible_set_gives_no_claim_and_cg_refuses_it():
    """§3.5: a nu with no admissible set gives the rate no sign, so cg() refuses the relation
    with PropertyNotDerived, naming the parameter and how to give it its sign."""
    with pytest.raises(PropertyNotDerived) as caught:
        _case(nu=_nu(admissible=None))
    message = str(caught.value)
    assert "cg(max_iterations=2000, preconditioner=None) needs a symmetric positive definite " \
        "operator" in message
    assert "the sign of kappa = nu is not derived from the admissible sets of nu " \
        "(model.parameter('nu', ..., admissible=pops.Positive()) gives a parameter its sign)" \
        in message


@pytest.mark.parametrize("coefficient, kappa, negative_semidefinite", [
    (lambda nu, u: nu, "nu", True),
    (lambda nu, u: 2 * nu, "2 * nu", True),
    (lambda nu, u: nu * nu, "nu * nu", True),
    (lambda nu, u: nu + 1, "nu + 1", False),
])
def test_the_claim_of_a_parameter_is_the_sign_that_its_admissible_set_derives(
        coefficient, kappa, negative_semidefinite):
    """A product of numbers and parameters admissible in pops.Positive() has their sign; a sum
    has none that this lot derives (R2, K7)."""
    domain, model, u, u0, advection, diffusion, numerics = _model(nu=_nu(), coefficient=coefficient)
    view = surface.TermView(diffusion, None, None)
    claims = linear_claims(view, scheme=numerics.scheme_of(diffusion), weight=1, layout={
        "uniform": True, "cartesian": True, "periodic": True})
    assert claims.symmetric and claims.homogeneous
    assert claims.negative_semidefinite is negative_semidefinite
    assert claims.description.endswith("kappa = %s" % kappa)
    assert ("for every value of nu admissible in pops.Positive()" in claims.proof) \
        is negative_semidefinite


def test_nu_has_its_slot_in_the_runtime_table_of_its_block():
    """§3.5: the StencilFlux of two_point() reads nu inside the region of the solve from its
    block's RuntimeParams table (ctx.program_params, N3 lot 1), where a formula of the model gives
    nu its slot: the surface declares b6's diffusive flux of the term. A later lot that stops
    declaring it fails here, before a native compile refuses the read."""
    import re

    import pops
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_emit_params import program_param_entries
    from pops.codegen.program_models import ProgramModelGraph

    elaborated = _case(nu=_nu()).elaborated
    resolved = pops.resolve(pops.validate(elaborated.case), layout=elaborated.layout)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    assert program_param_entries(resolved.time, graph) == [(0, "nu", 0, 1e-2)]
    cpp = emit_cpp_program(resolved.time, model_graph=graph)
    reads = re.findall(r"const pops::RuntimeParams (stencil_params_\d+) = ctx\.program_params\(0\);",
                       cpp)
    assert reads and set(re.findall(r"params\.get\((\d+)\)", cpp)) == {"0"}


def test_a_coefficient_that_varies_with_the_state_is_refused():
    with pytest.raises(surface.NotYetElaborated, match="varies with the state"):
        _case(coefficient=lambda nu, u: nu * (1 + u * u))


def test_a_layout_that_is_not_uniform_and_periodic_is_refused():
    """The claims of the rate sit beside cg() (N3 lot 1, D7) and read the scheme of the key."""
    domain, model, u, u0, advection, diffusion, numerics = _model()
    view = surface.TermView(diffusion, None, None)
    scheme = numerics.scheme_of(diffusion)
    for layout in ({"uniform": False, "cartesian": True, "periodic": True},
                   {"uniform": True, "cartesian": True, "periodic": False}):
        with pytest.raises(surface.NotYetElaborated, match="uniform, Cartesian and periodic"):
            linear_claims(view, scheme=scheme, weight=1, layout=layout)
    claims = linear_claims(view, scheme=scheme, weight=1, layout={
        "uniform": True, "cartesian": True, "periodic": True})
    assert claims.symmetric and claims.negative_semidefinite and claims.homogeneous
    assert claims.description.endswith("kappa = 0.01")


def test_the_claims_read_a_copy_of_two_point_and_refuse_another_face():
    """A copy of the body of two_point() written in a case is claimed as the word (CF4); a
    diffusive face of another normal form, here the two-point flux plus a constant, whose
    frequency is still derived (D10), is not stated linear and homogeneous in this lot (G10)."""
    import inspect

    import pops.lib.numerics.fluxes as fluxes

    source = inspect.getsource(fluxes)
    shifted = source.replace("return FaceFlux(dot(term.law(states=states, gradients=gradients), "
                             "face.normal))", "return FaceFlux(dot(term.law(states=states, "
                             "gradients=gradients), face.normal) + 1.0)")
    assert shifted != source
    layout = {"uniform": True, "cartesian": True, "periodic": True}
    for text, kappa in ((source, 1e-2), (shifted, None)):
        namespace = {}
        exec(compile(text, fluxes.__file__, "exec"), namespace)  # noqa: S102
        domain, model, u, u0, advection, diffusion, numerics = _model()
        written = surface.Numerics(represent={u: surface.CellAverage()},
                                   schemes={advection: upwind(),
                                            diffusion: namespace["two_point"]()})
        view = surface.TermView(diffusion, None, None)
        scheme = written.scheme_of(diffusion)
        if kappa is not None:
            assert linear_claims(view, scheme=scheme, weight=1, layout=layout).description \
                .endswith("kappa = %r" % kappa)
            continue
        with pytest.raises(surface.NotYetElaborated,
                           match="is not stated linear and homogeneous in the state.*G10"):
            linear_claims(view, scheme=scheme, weight=1, layout=layout)


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


def _case_solvers():
    """A @pops.solver written in a case, and a copy of the library's cg written in a case: each
    runs as the reference of a relation solve, as the library's cg() does (CF4; T1 of the review
    of #776, whose asymmetry left with the bridge, N9 « La référence sous les solves de
    relation »)."""
    import inspect

    import pops.lib.solve.krylov as krylov

    @surface.solver
    def mine(problem, region, max_iterations=10):
        start = surface.record(x=problem.guess, r=0.0, k=0)
        result = region.iterate(start=start, step=lambda s: surface.record(x=s.x, r=s.r, k=s.k + 1),
                                stop=lambda s: s.k > 0, max_steps=max_iterations)
        return problem.candidate(result.x, iterations=result.iterations, residual_norm=result.r)

    namespace = {"__name__": "a_case"}
    exec(compile(inspect.getsource(krylov), "a_case.py", "exec"), namespace)  # noqa: S102
    copy = namespace["cg"]
    assert copy is not krylov.cg
    return {"mine": (mine(), "mine(max_iterations=10)"),
            "copy-of-cg": (copy(), "cg(max_iterations=2000, preconditioner=None)")}


@pytest.mark.parametrize("which", ["mine", "copy-of-cg"])
def test_a_solver_of_a_case_runs_as_the_reference_of_a_relation_solve(which):
    """A @pops.solver written in a case is the realization of a relation solve, its body the
    region of the solve's node, as for problem/region callers."""
    solver, spelled = _case_solvers()[which]

    @surface.method
    def imex(step, explicit, implicit):
        u, t, h = step.state, step.time, step.dt
        X = surface.unknown(step.state.space)
        return surface.Candidate(surface.solve(
            X == u + h * explicit(u, t) + h * implicit(X, t + h), X))

    case = _case(method=imex, solvers=lambda m: {m: solver})
    (site,) = case.elaborated.advance.trace.solves
    assert repr(site.solver) == spelled
    (node,) = [value for value in case.elaborated.case._time._values
               if value.op == "solve_linear"]
    assert node.attrs["realization"]["kind"] == "reference"


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


def _caps(case):
    """The iteration cap of the region of each solve node, by the node's name."""
    def walk(block):
        for value in block:
            yield value
            for key in ("body_block", "cond_block", "true_block", "false_block"):
                yield from walk(value.attrs.get(key) or ())

    caps = {}
    for node in case.elaborated.case._time._values:
        if node.op == "solve_linear":
            (iterate,) = [value for value in walk(node.attrs["body_block"])
                          if value.op == "structured_iterate"]
            caps[node.name] = iterate.inputs[1].attrs["value"]
    return caps


@pytest.mark.parametrize("order", ["first-then-second", "second-then-first"])
def test_each_named_solve_runs_the_solver_of_its_own_key(order):
    """O1 of the delta review of #778 (N9 « La référence sous les solves de relation »): the
    elaborator took the linear map of the first solver of solvers= for every solve. The operator
    of a relation solve is now the rate at the State, and each solve runs the solver keyed by its
    own site, whatever the order of solvers=."""
    @surface.method
    def twice(step, explicit, implicit):
        u, t, h = step.state, step.time, step.dt
        X = surface.unknown(step.state.space)
        first = surface.solve(X == u + h * implicit(X, t + h), X, name="first")
        Y = surface.unknown(step.state.space)
        return surface.Candidate(surface.solve(
            Y == first + h * explicit(first, t + h) + h * implicit(Y, t + h), Y, name="second"))

    def solvers(m):
        keyed = [(m.first, cg(max_iterations=7)), (m.second, cg(max_iterations=9))]
        return dict(keyed if order == "first-then-second" else keyed[::-1])

    case = _case(method=twice, solvers=solvers)
    assert _caps(case) == {"twice_first": 7, "twice_second": 9}


def test_a_rate_option_with_a_term_that_is_not_linear_and_homogeneous_names_the_term():
    """§3.4 of the N9 lot: the operator evaluates the whole rate of the component at its operand,
    so every term states its claim; the transport under upwind() does not, and is named."""
    @surface.method
    def backward(step, implicit):
        X = surface.unknown(step.state.space)
        return surface.Candidate(surface.solve(
            X == step.state + step.dt * implicit(X, step.time + step.dt), X))

    domain, model, u, u0, advection, diffusion, numerics = _model()
    value = backward(implicit=model)
    program = surface.Program(advance=surface.Advance(value, dt=1e-3, tolerance=TOLERANCE),
                              initial={u: u0}, solvers={value: cg()})
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.Case(model=model, numerics=numerics, program=program,
                     layout={domain: surface.Uniform(cells=16)})
    message = str(caught.value)
    assert "the term %s under upwind() read at the unknown of a solve" % advection in message
    assert "is not stated linear and homogeneous in the state" in message


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
    # The reference checks its budget when its body runs, which a relation solve does when the
    # method is traced.
    with pytest.raises(ValueError, match="max_iterations=n"):
        _case(solvers=lambda imex: {imex: cg(max_iterations=0)})
    for word in ("auto", "gmres", "bicgstab", "newton", "jacobi", "richardson", "dense_lu"):
        with pytest.raises(surface.NotYetElaborated, match="N9"):
            getattr(solve, word)()


def test_a_linear_map_without_claims_is_refused():
    """A rate that states no claims cannot derive the properties of the operator (IR16): the rate
    sum refuses it, naming G10, rather than reading claims that are not there."""
    from pops.surface._elaborate import _RateSum

    class Unclaimed:
        pass

    rate = _RateSum([lambda state: state], [Unclaimed])
    with pytest.raises(surface.NotYetElaborated, match="states no claims.*G10"):
        rate.linear_claims  # noqa: B018
