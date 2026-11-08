"""SU-T2 (spec 0.3, 2.6.7): the refusals of the user surface, lot 1 of N15 (ROM-98).

Each refusal of SU3 to SU14 and of the tables of 2.6.2 to 2.6.4 that level 1 meets has a test that
triggers it and checks its message: the file and the line of the user's statement, the rule in one
sentence with its ids, and the construct to write (SU11, §14.1). Each NumPy and Python habit of SU4
and SU11 has one such test. A habit that SD82 leaves to Python (a tuple where a list is meant, a
call without its options) is a Python ``TypeError`` raised at the user's line, whose message names
the construct to write.

The statement that a refusal names ends with a marker comment, ``# <- name``; the tests find its
line by the marker. SU7 (``OutOfScope``) has no test: no word of this lot is scoped to an entity.
The tests read the surface as ``surface``, as SU-T3 does: until the cut-over of §21.7 (SD86), the
root names Periodic and Scalar of ``pops`` are attributes that b6 retired, which
``test_no_legacy_runtime_routes`` refuses in a test.
"""
import os
import re

import numpy
import pytest

import pops.surface as surface
from pops.lib.numerics.schemes import two_point, upwind
from pops.lib.time.methods import ssprk2
from pops.surface.math import all_of, ddt, div, exp, grad, sqrt, where

HERE = os.path.abspath(__file__)


def _line(marker):
    lines = [number for number, text in enumerate(open(HERE, encoding="utf-8"), start=1)
             if text.rstrip().endswith("# <- " + marker)]
    assert len(lines) == 1, "marker %r is not on exactly one line" % marker
    return lines[0]


def _refused(caught, kind, marker, rules, write):
    """The refusal is a ``kind`` at the marked line, with its rule ids and the construct."""
    error = caught.value
    assert type(error) is kind
    line = _line(marker)
    assert (error.site.file, error.site.line) == (HERE, line)
    message = str(error)
    assert message.startswith("%s:%d" % (HERE, line))
    assert ": %s: " % kind.__name__ in message
    assert "(%s)" % ", ".join(rules) in message
    assert "Write %s" % write in message
    return message


def _spelled(construct):
    """A construct of the surface as its messages spell it, ``pops.<name>`` (SD86)."""
    return "pops.%s" % construct.__name__


def _python_error(caught, kind, marker, words):
    """A Python error raised by the statement at the marked line, whose message names ``words``."""
    assert type(caught.value) is kind
    lines = [entry.lineno + 1 for entry in caught.traceback
             if os.path.abspath(str(entry.path)) == HERE]
    assert lines[-1] == _line(marker)
    for word in words:
        assert word in str(caught.value)


def _model():
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("advection_diffusion")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    nu = model.parameter("nu", default=1e-2, admissible=surface.Positive())
    u0 = model.data("u0", surface.Scalar(domain))
    return domain, model, u, a, nu, u0


def _transport():
    domain, model, u, a, nu, u0 = _model()
    advection = model.flux(of=u, value=a * u)
    diffusion = model.diffusive_flux(of=u, value=-nu * grad(u))
    model.balance(ddt(u) == -div(advection + diffusion))
    return domain, model, u, u0, advection, diffusion


def _numerics(u, advection, diffusion):
    return surface.Numerics(represent={u: surface.CellAverage()},
                            schemes={advection: upwind(), diffusion: two_point()})


def _layout(domain):
    return {domain: surface.Uniform(cells=16)}


# --- SU3: one way per task; the spellings of earlier texts -----------------------------------

def test_su3_an_earlier_word_of_pops_math_is_refused_on_import():
    with pytest.raises(surface.SymbolNotInSurface) as caught:
        from pops.surface.math import maximum  # noqa: F401  # <- su3-maximum
    _refused(caught, surface.SymbolNotInSurface, "su3-maximum", ("SU3", "SU13", "§21.7"),
             "largest([a, b, ...])")
    assert isinstance(caught.value, ImportError)


def test_su3_an_earlier_name_of_pops_is_refused_with_its_row_of_21_7():
    with pytest.raises(surface.SymbolNotInSurface) as caught:
        surface.Rectangle  # noqa: B018  # <- su3-rectangle
    _refused(caught, surface.SymbolNotInSurface, "su3-rectangle", ("SU3", "SU13", "§21.7"),
             "pops.Box(x=(0, 1), y=(0, 1)), one keyword per axis")
    with pytest.raises(surface.SymbolNotInSurface) as caught:
        surface.validate  # noqa: B018  # <- su3-validate
    _refused(caught, surface.SymbolNotInSurface, "su3-validate", ("SU3", "SU13", "§21.7"),
             "pops.inspect(case), which validates and resolves without compiling")


def test_su3_the_frame_of_a_model_is_derived_not_stated():
    with pytest.raises(surface.SymbolNotInSurface) as caught:
        surface.Model("advection_diffusion", frame=None)  # <- su3-frame
    _refused(caught, surface.SymbolNotInSurface, "su3-frame", ("SU3", "SU13", "§21.7"),
             'pops.Model("name"): the frame of a model is that of the supports of its quantities')


def test_su3_the_parameters_of_a_model_have_one_spelling():
    _, model, *_ = _model()
    with pytest.raises(surface.SymbolNotInSurface) as caught:
        model.param  # noqa: B018  # <- su3-param
    _refused(caught, surface.SymbolNotInSurface, "su3-param", ("SU3", "SU13", "§21.7"),
             'model.constant("a", value=v), folded at compile time')


def test_su3_the_keywords_of_earlier_texts_are_refused():
    domain, model, u, u0, advection, diffusion = _transport()
    with pytest.raises(surface.SymbolNotInSurface) as caught:
        surface.Advance(ssprk2(rate=model), dt=1e-3, retry=2)  # <- su3-retry
    _refused(caught, surface.SymbolNotInSurface, "su3-retry", ("SU3", "SU13", "§21.7"),
             "pops.Advance(method, attempts=n)")
    advance = surface.Advance(ssprk2(rate=model), dt=1e-3)
    with pytest.raises(surface.SymbolNotInSurface) as caught:
        surface.Program(advance=advance, initial={u: u0}, initialize={})  # <- su3-initialize
    _refused(caught, surface.SymbolNotInSurface, "su3-initialize", ("SU3", "SU13", "§21.7"),
             "pops.Program(advance=A, initial={u: u0}, solvers={key: solver})")
    with pytest.raises(surface.SymbolNotInSurface) as caught:
        surface.Numerics(realizations=())  # <- su3-realizations
    _refused(caught, surface.SymbolNotInSurface, "su3-realizations", ("SU3", "SU13", "§21.7"),
             "pops.Numerics(schemes={F: s}), the key being the term that the scheme covers")
    program = surface.Program(advance=advance, initial={u: u0})
    numerics = _numerics(u, advection, diffusion)
    with pytest.raises(surface.SymbolNotInSurface) as caught:
        parts = dict(model=model, numerics=numerics, program=program, layout=_layout(domain))
        surface.Case(**parts, data={})  # <- su3-data
    _refused(caught, surface.SymbolNotInSurface, "su3-data", ("SU3", "SU13", "§21.7"),
             'model.data("d", space) and pops.bind(artifact, data={d: array})')


def test_su3_stated_gradients_may_not_contradict_the_derivation():
    _, model, u, _, nu, _ = _model()
    w = model.quantity("w", surface.Scalar(u.support))
    with pytest.raises(surface.ContradictsDerivation) as caught:
        model.diffusive_flux(of=u, value=-nu * grad(u), gradients=[w])  # <- su3-gradients
    message = _refused(caught, surface.ContradictsDerivation, "su3-gradients", ("SU3",),
                       "gradients=[u], or leave it out to derive it")
    assert "gradients=[w] omits u, whose gradient the value -nu * grad(u) reads" in message


# --- SU4: the operators keep their meaning; NumPy does not read PoPS values ------------------

def test_su4_operators_outside_the_algebra_name_the_word_to_write():
    _, _, u, a, _, _ = _model()
    cases = []
    with pytest.raises(surface.NotAPopsOperation) as caught:
        u @ a  # <- su4-matmul
    cases.append((caught, "su4-matmul", "dot(a, b) or outer(a, b)"))
    with pytest.raises(surface.NotAPopsOperation) as caught:
        u != a  # noqa: B015  # <- su4-ne
    cases.append((caught, "su4-ne", "none_of([equal(a, b)])"))
    with pytest.raises(surface.NotAPopsOperation) as caught:
        (u > 0) & (u < 1)  # <- su4-and
    cases.append((caught, "su4-and", "all_of([p, q])"))
    with pytest.raises(surface.NotAPopsOperation) as caught:
        (u > 0) | (u < 1)  # <- su4-or
    cases.append((caught, "su4-or", "any_of([p, q])"))
    with pytest.raises(surface.NotAPopsOperation) as caught:
        ~(u > 0)  # noqa: B018  # <- su4-invert
    cases.append((caught, "su4-invert", "none_of([p])"))
    for caught, marker, write in cases:
        _refused(caught, surface.NotAPopsOperation, marker, ("SU4", "SU5"), write)


def test_su4_a_numpy_function_of_a_pops_value_names_the_word_of_pops_math():
    _, _, u, _, _, _ = _model()
    with pytest.raises(surface.NotAPopsOperation) as caught:
        numpy.exp(u)  # <- su4-numpy-exp
    _refused(caught, surface.NotAPopsOperation, "su4-numpy-exp", ("SU4", "SU5"), "exp(e)")
    with pytest.raises(surface.NotAPopsOperation) as caught:
        numpy.maximum(u, 0.0)  # <- su4-numpy-maximum
    _refused(caught, surface.NotAPopsOperation, "su4-numpy-maximum", ("SU4", "SU5"),
             "largest([a, b])")


def test_su4_where_chooses_by_a_predicate_not_by_a_relation():
    _, _, u, _, _, _ = _model()
    with pytest.raises(surface.NotAPopsOperation) as caught:
        where(u == 0.5, 1.0, 0.0)  # <- su4-where
    _refused(caught, surface.NotAPopsOperation, "su4-where", ("SU4", "SU5"),
             "equal(u, 0.5) or abs(u - 0.5) <= tolerance")


def test_su4_a_model_element_is_not_callable_outside_a_method():
    domain, model, u, u0, advection, diffusion = _transport()
    with pytest.raises(surface.NotAPopsOperation) as caught:
        advection(u, 0.0)  # <- su4-call-term
    _refused(caught, surface.NotAPopsOperation, "su4-call-term", ("SU4", "SU5"),
             "ssprk2(rate=[F]) and rate(u, t) in the method")
    with pytest.raises(surface.NotAPopsOperation) as caught:
        model.balances[0](u)  # <- su4-call-balance
    _refused(caught, surface.NotAPopsOperation, "su4-call-balance", ("SU4", "SU5"),
             "ssprk2(rate=model) and rate(u, t) in the method")


def test_su4_membership_compares_with_the_relation_and_is_refused():
    """SD79: ``in`` on a list compares by ``==``, which builds a relation, before identity."""
    _, _, u, a, _, _ = _model()
    assert u in [u, a]
    with pytest.raises(surface.SymbolicTruthValue) as caught:
        a in [u, a]  # noqa: B015  # <- su4-in
    _refused(caught, surface.SymbolicTruthValue, "su4-in", ("E1", "SU11"), "where(p, a, b)")
    assert a in {u: 1, a: 2}


# --- SU5: decorated words are called; plain functions are passed --------------------------------

def test_su5_a_scheme_word_is_always_called():
    domain, model, u, u0, advection, diffusion = _transport()
    schemes = {advection: upwind, diffusion: two_point()}
    with pytest.raises(TypeError) as caught:
        surface.Numerics(represent={u: surface.CellAverage()}, schemes=schemes)  # <- su5-scheme
    _python_error(caught, TypeError, "su5-scheme", ("always called", "write upwind()"))


def test_su5_a_method_is_called_with_its_options():
    with pytest.raises(TypeError) as caught:
        surface.Advance(ssprk2, dt=1e-3)  # <- su5-method
    _python_error(caught, TypeError, "su5-method", ("ssprk2(rate=...)", "SU5"))


def test_su5_a_method_takes_step_and_no_other_entity():
    def body(step, cell):
        return surface.Candidate(step.state)

    with pytest.raises(TypeError) as caught:
        surface.method(body)  # <- su5-entity
    _python_error(caught, TypeError, "su5-entity", ("PoPS supplies step to a method, not cell",))


def test_su5_each_rate_call_states_the_state_and_then_its_time():
    @surface.method
    def euler_without_time(step, rate):
        return surface.Candidate(step.state + step.dt * rate(step.state))  # <- su5-rate-time

    domain, model, u, u0, advection, diffusion = _transport()
    program = surface.Program(advance=surface.Advance(euler_without_time(rate=model), dt=1e-3),
                              initial={u: u0})
    with pytest.raises(TypeError) as caught:
        surface.Case(model=model, numerics=_numerics(u, advection, diffusion), program=program,
                     layout=_layout(domain))
    _python_error(caught, TypeError, "su5-rate-time", ("rate(step.state, step.time)", "T4"))


# --- SU6: one level per construct ------------------------------------------------------------

def test_su6_a_scheme_is_not_a_value_of_the_model():
    _, model, u, _, _, _ = _model()
    with pytest.raises(surface.LevelMismatch) as caught:
        model.flux(of=u, value=upwind())  # <- su6-model
    _refused(caught, surface.LevelMismatch, "su6-model", ("SU6",),
             "the value at the model level, and the choice in pops.Numerics or pops.Program")


def test_su6_a_scheme_is_not_a_time_method():
    with pytest.raises(surface.LevelMismatch) as caught:
        surface.Advance(upwind(), dt=1e-3)  # <- su6-advance
    _refused(caught, surface.LevelMismatch, "su6-advance", ("SU6",),
             "pops.Advance(ssprk2(rate=model), dt=...)")


def test_su6_a_time_method_is_not_a_scheme():
    domain, model, u, u0, advection, diffusion = _transport()
    schemes = {advection: ssprk2(rate=model), diffusion: two_point()}
    with pytest.raises(surface.LevelMismatch) as caught:
        surface.Numerics(represent={u: surface.CellAverage()}, schemes=schemes)  # <- su6-numerics
    _refused(caught, surface.LevelMismatch, "su6-numerics", ("SU6",),
             "the method in pops.Advance(...), and a scheme of the numerics under the key")


# --- SU8: names, handles and keywords ----------------------------------------------------------

def test_su8_an_axis_may_not_take_the_name_of_a_reading_of_a_support():
    with pytest.raises(surface.ReservedName) as caught:
        surface.Box(axes=(0, 1))  # <- su8-axes
    message = _refused(caught, surface.ReservedName, "su8-axes", ("SU8",),
                       "another axis name, such as pops.Box(x=(0, 1))")
    assert "axes, factors, position, point, piece, boundary, identify, copy" in message


def test_su8_tables_are_keyed_by_handles():
    with pytest.raises(TypeError) as caught:
        surface.Numerics(represent={"u": surface.CellAverage()})  # <- su8-string-key
    _python_error(caught, TypeError, "su8-string-key", ("keyed by handles", "'u'", "SU8"))


def test_su8_a_decorated_word_takes_its_options_by_keyword():
    _, model, *_ = _model()
    with pytest.raises(TypeError) as caught:
        ssprk2(model)  # <- su8-positional
    _python_error(caught, TypeError, "su8-positional", ("ssprk2(rate=...)", "SU8"))


def test_su8_a_declaration_takes_its_name_first_as_an_identifier():
    domain, model, *_ = _model()
    with pytest.raises(TypeError) as caught:
        model.quantity("u v", surface.Scalar(domain))  # <- su8-name
    _python_error(caught, TypeError, "su8-name", ("identifier string", "SU8"))


# --- SU9: what is omitted is derived, defaulted or refused ------------------------------------

def test_su9_the_step_rule_is_never_chosen():
    _, model, *_ = _model()
    with pytest.raises(surface.AmbiguousElaboration) as caught:
        surface.Advance(ssprk2(rate=model))  # <- su9-dt
    _refused(caught, surface.AmbiguousElaboration, "su9-dt", ("D5d", "SU9"),
             "pops.Advance(ssprk2(rate=pops.Model('advection_diffusion')), dt=...)")


def test_su9_the_representation_of_an_evolved_quantity_is_stated():
    domain, model, u, u0, advection, diffusion = _transport()
    schemes = {advection: upwind(), diffusion: two_point()}
    numerics = surface.Numerics(schemes=schemes)  # <- su9-represent
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3), initial={u: u0})
    layout = _layout(domain)
    with pytest.raises(surface.AmbiguousElaboration) as caught:
        surface.Case(model=model, numerics=numerics, program=program, layout=layout)  # <- su9-case
    message = _refused(caught, surface.AmbiguousElaboration, "su9-represent", ("D5d", "SU9"),
                       "pops.Numerics(represent={u: pops.CellAverage()})")
    assert message.endswith("The case is traced at %s:%d." % (HERE, _line("su9-case")))


def test_su9_each_term_has_a_scheme():
    domain, model, u, u0, advection, diffusion = _transport()
    represent = {u: surface.CellAverage()}
    numerics = surface.Numerics(represent=represent, schemes={advection: upwind()})  # <- su9-scheme
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3), initial={u: u0})
    with pytest.raises(surface.CoverageNotExact) as caught:
        surface.Case(model=model, numerics=numerics, program=program, layout=_layout(domain))
    message = _refused(caught, surface.CoverageNotExact, "su9-scheme", ("D5", "T5", "CC14"),
                       "pops.Numerics(schemes={-nu * grad(u): <a scheme word>()})")
    assert "the term -nu * grad(u) of the balance of u, declared at" in message


def test_su9_an_evolved_quantity_has_an_initial_value():
    domain, model, u, u0, advection, diffusion = _transport()
    advance = surface.Advance(ssprk2(rate=model), dt=1e-3)
    program = surface.Program(advance=advance, initial={})  # <- su9-initial
    with pytest.raises(surface.AmbiguousElaboration) as caught:
        surface.Case(model=model, numerics=_numerics(u, advection, diffusion), program=program,
                     layout=_layout(domain))
    _refused(caught, surface.AmbiguousElaboration, "su9-initial", ("D5d", "SU9"),
             "pops.Program(initial={u: d})")


def test_su9_a_boundary_that_no_relation_closes_is_refused():
    domain = surface.Box(x=(0, 1))  # <- su9-boundary
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=2.0 * u)
    model.balance(ddt(u) == -div(advection))
    numerics = surface.Numerics(represent={u: surface.CellAverage()}, schemes={advection: upwind()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3), initial={u: u0})
    with pytest.raises(surface.MissingBoundaryRealization) as caught:
        surface.Case(model=model, numerics=numerics, program=program, layout=_layout(domain))
    message = _refused(caught, surface.MissingBoundaryRealization, "su9-boundary", ("SU9", "D9"),
                       "%s(lower, upper) for an axis whose ends are identified"
                       % _spelled(surface.Periodic))
    assert "x_min, x_max" in message


# --- SU10: the step algebra of a method ----------------------------------------------------------

def _traced(method_value_of):
    domain, model, u, u0, advection, diffusion = _transport()
    program = surface.Program(advance=surface.Advance(method_value_of(model), dt=1e-3),
                              initial={u: u0})
    return surface.Case(model=model, numerics=_numerics(u, advection, diffusion), program=program,
                        layout=_layout(domain))


def test_su10_a_rate_is_added_to_a_state_only_times_a_duration():
    @surface.method
    def without_duration(step, rate):
        return surface.Candidate(step.state + rate(step.state, step.time))  # <- su10-duration

    with pytest.raises(surface.IncrementTypeMismatch) as caught:
        _traced(lambda model: without_duration(rate=model))
    message = _refused(caught, surface.IncrementTypeMismatch, "su10-duration", ("SU10", "T34"),
                       "a rate times a duration, step.dt * rate(u, t)")
    assert "a rate times a duration of step.dt" in message


def test_su10_the_weights_of_the_states_of_a_candidate_sum_to_one():
    @surface.method
    def doubled(step, rate):
        return surface.Candidate(2 * step.state)  # <- su10-weights

    with pytest.raises(surface.IncrementTypeMismatch) as caught:
        _traced(lambda model: doubled(rate=model))
    message = _refused(caught, surface.IncrementTypeMismatch, "su10-weights", ("SU10", "T34"),
                       "a rate times a duration")
    assert "weights sum to 2" in message


def test_su10_a_closed_form_has_no_derived_increment():
    @surface.method
    def closed_form(step, rate):
        return surface.Candidate(step.state * (1 + step.dt))  # <- su10-closed-form

    with pytest.raises(surface.IncrementNotDerivable) as caught:
        _traced(lambda model: closed_form(rate=model))
    _refused(caught, surface.IncrementNotDerivable, "su10-closed-form", ("SU10", "T34"),
             "a candidate built from rates, or pops.exact_flow(method, of=term) for a closed form")


def test_su10_a_body_reads_no_value_of_another_trace():
    kept = []

    @surface.method
    def keeper(step, rate):
        kept.append(step.state)
        return surface.Candidate(step.state + step.dt * rate(step.state, step.time))

    @surface.method
    def thief(step, rate):
        return surface.Candidate(kept[0] + step.dt * rate(step.state, step.time))  # <- su10-closed

    _traced(lambda model: keeper(rate=model))
    with pytest.raises(surface.ClosedBodyViolation) as caught:
        _traced(lambda model: thief(rate=model))
    _refused(caught, surface.ClosedBodyViolation, "su10-closed", ("T33",),
             "a body that reads only step, its options and immutable values")


# --- SU11: no Python truth value, no Python number --------------------------------------------

def test_su11_an_expression_has_no_truth_value():
    _, _, u, _, _, _ = _model()
    with pytest.raises(surface.SymbolicTruthValue) as caught:
        if u > 0:  # <- su11-if
            pass
    _refused(caught, surface.SymbolicTruthValue, "su11-if", ("E1", "SU11"), "where(p, a, b)")
    with pytest.raises(surface.SymbolicTruthValue) as caught:
        (u > 0) and (u < 1)  # noqa: B018  # <- su11-and
    _refused(caught, surface.SymbolicTruthValue, "su11-and", ("E1", "SU11"), "where(p, a, b)")
    with pytest.raises(surface.SymbolicTruthValue) as caught:
        0 < u < 1  # noqa: B015  # <- su11-chained
    _refused(caught, surface.SymbolicTruthValue, "su11-chained", ("E1", "SU11"), "where(p, a, b)")
    with pytest.raises(surface.SymbolicTruthValue) as caught:
        max(u, 0.0)  # <- su11-max
    _refused(caught, surface.SymbolicTruthValue, "su11-max", ("E1", "SU11"),
             "where(p, a, b) between values, all_of([...]), any_of([...]) or none_of([...]) "
             "between predicates, largest([...]) or smallest([...]) in place of max and min")


def test_su11_an_expression_has_no_python_number_and_no_array():
    _, _, u, _, _, _ = _model()
    with pytest.raises(surface.SymbolicTruthValue) as caught:
        float(u)  # <- su11-float
    _refused(caught, surface.SymbolicTruthValue, "su11-float", ("E1", "SU11"),
             "the expression itself; PoPS evaluates it")
    with pytest.raises(surface.SymbolicTruthValue) as caught:
        numpy.asarray(u)  # <- su11-asarray
    _refused(caught, surface.SymbolicTruthValue, "su11-asarray", ("E1", "SU11"),
             "the expression itself, with the words of pops.math")


def test_su11_a_state_in_a_method_has_no_truth_value():
    @surface.method
    def branching(step, rate):
        if step.state:  # <- su11-method
            pass
        return surface.Candidate(step.state + step.dt * rate(step.state, step.time))

    with pytest.raises(surface.SymbolicTruthValue) as caught:
        _traced(lambda model: branching(rate=model))
    _refused(caught, surface.SymbolicTruthValue, "su11-method", ("E1", "SU11"),
             "where(p, a, b) between values; the tableau of a method is Python data")


# --- SU13: imports ---------------------------------------------------------------------------

def test_su13_an_unknown_word_is_refused_on_import_with_the_words():
    with pytest.raises(surface.SymbolNotInSurface) as caught:
        from pops.surface.math import spline  # noqa: F401  # <- su13-unknown
    message = _refused(caught, surface.SymbolNotInSurface, "su13-unknown", ("SU3", "SU13", "§21.7"),
                       "one of these words, or Python's own numbers and operators")
    assert "ddt, grad, div" in message


def test_su13_the_star_import_of_pops_math_gives_its_words_only():
    namespace = {}
    exec("from pops.surface.math import *", namespace)
    words = {name for name in namespace if name != "__builtins__"}
    assert {"ddt", "grad", "div", "exp", "pi", "where", "all_of"} <= words
    assert not any(name.startswith("_") for name in words)


def test_su13_a_word_of_a_later_lot_is_importable_and_refuses_its_use():
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.Tolerance(rtol=1e-10, atol=1e-12)  # <- su13-later
    message = str(caught.value)
    assert (caught.value.site.file, caught.value.site.line) == (HERE, _line("su13-later"))
    assert "pops.Tolerance has no elaborated form yet; a later lot of N15 with the solves of N9" \
        in message


def test_su13_curl_and_outflow_are_words_of_pops_math_of_later_lots():
    """``curl`` and ``outflow`` of 2.6.3 are importable from ``pops.math`` (CF2 admits their
    definition there, R20) and refuse their use with the lot that elaborates them."""
    namespace = {}
    exec("from pops.surface.math import *", namespace)
    assert {"curl", "outflow"} <= set(namespace)
    _, _, u, _, _, _ = _model()
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.math.curl(u)
    assert "the word curl of pops.math has no elaborated form yet; N11" in str(caught.value)
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.math.outflow(u)
    assert "the word outflow of pops.math has no elaborated form yet; a later lot of N15 with " \
        "the boundary relations of G04" in str(caught.value)


def test_su13_laplacian_is_a_spelling_that_pops_math_refuses():
    with pytest.raises(surface.SymbolNotInSurface) as caught:
        surface.math.laplacian  # noqa: B018  # <- su13-laplacian
    _refused(caught, surface.SymbolNotInSurface, "su13-laplacian", ("SU3", "SU13", "§21.7"),
             "div(grad(e))")


# --- SU14: tuples and lists ----------------------------------------------------------------------

def test_su14_several_predicates_are_a_list():
    _, _, u, _, _, _ = _model()
    with pytest.raises(TypeError) as caught:
        all_of((u > 0, u < 1))  # <- su14-all-of
    _python_error(caught, TypeError, "su14-all-of", ("all_of([...])", "SU14"))


def test_su14_several_terms_are_a_list():
    domain, model, u, u0, advection, diffusion = _transport()
    with pytest.raises(TypeError) as caught:
        ssprk2(rate=(advection, diffusion))  # <- su14-terms
    _python_error(caught, TypeError, "su14-terms", ("rate=[...]", "SU14"))


# --- the refusals of the tables of 2.6.2 to 2.6.4 ----------------------------------------------

def test_a_term_is_counted_once_in_its_balance():
    _, model, u, a, _, _ = _model()
    advection = model.flux(of=u, value=a * u)
    with pytest.raises(surface.TermCountedTwice) as caught:
        model.balance(ddt(u) == -div(advection) - div(advection))  # <- r7-twice
    _refused(caught, surface.TermCountedTwice, "r7-twice", ("R7",), "each term once")


def test_a_diffusive_flux_is_the_physical_flux_with_its_sign():
    _, model, u, _, nu, _ = _model()
    with pytest.raises(surface.AntiDiffusive) as caught:
        model.diffusive_flux(of=u, value=nu * grad(u))  # <- r2-sign
    message = _refused(caught, surface.AntiDiffusive, "r2-sign", ("R2", "§2.6.2"),
                       "the physical flux with its sign, as Fick writes it")
    assert "its k = -nu is negative" in message


def test_a_diffusive_flux_enters_its_balance_under_minus_div():
    _, model, u, _, nu, _ = _model()
    diffusion = model.diffusive_flux(of=u, value=-nu * grad(u))
    with pytest.raises(surface.AntiDiffusive) as caught:
        model.balance(ddt(u) == div(diffusion))  # <- r2-plus-div
    _refused(caught, surface.AntiDiffusive, "r2-plus-div", ("R2", "§2.6.2"),
             "ddt(u) == -div(... + -nu * grad(u))")


def test_a_diffusive_flux_reads_a_gradient():
    _, model, u, _, nu, _ = _model()
    with pytest.raises(surface.NotDiffusive) as caught:
        model.diffusive_flux(of=u, value=-nu * u)  # <- r2-no-gradient
    _refused(caught, surface.NotDiffusive, "r2-no-gradient", ("R2", "§2.6.2"),
             "model.diffusive_flux(of=u, value=-k * grad(u))")


def test_a_quantity_has_one_balance():
    _, model, u, a, _, _ = _model()
    advection = model.flux(of=u, value=a * u)
    model.balance(ddt(u) == -div(advection))
    with pytest.raises(surface.AmbiguousElaboration) as caught:
        model.balance(ddt(u) == -div(advection))  # <- su9-second-balance
    _refused(caught, surface.AmbiguousElaboration, "su9-second-balance", ("D5d", "SU9"),
             "one balance per quantity")


def test_upwind_refuses_a_flux_that_is_not_linear_in_its_quantity():
    domain, model, u, _, _, u0 = _model()
    burgers = model.flux(of=u, value=u**2 / 2)
    model.balance(ddt(u) == -div(burgers))
    represent = {u: surface.CellAverage()}
    numerics = surface.Numerics(represent=represent, schemes={burgers: upwind()})  # <- d5-upwind
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3), initial={u: u0})
    with pytest.raises(surface.RealizationMismatch) as caught:
        surface.Case(model=model, numerics=numerics, program=program, layout=_layout(domain))
    message = _refused(caught, surface.RealizationMismatch, "d5-upwind", ("D5", "§2.6.4"),
                       "rusanov() or hll(), which take the wave speeds of the flux")
    assert " (in pops/lib/numerics/schemes.py:" in message
    assert "the flux of u is u**2 / 2" in message


def test_a_scheme_covers_the_keys_of_its_kind():
    domain, model, u, u0, advection, diffusion = _transport()
    schemes = {advection: two_point(), diffusion: two_point()}  # <- d5-kind
    with pytest.raises(surface.RealizationMismatch) as caught:
        surface.Numerics(represent={u: surface.CellAverage()}, schemes=schemes)
    _refused(caught, surface.RealizationMismatch, "d5-kind", ("D5", "§2.6.4"),
             "a scheme of the kind of the key")


def test_a_rate_option_covers_each_term_once():
    domain, model, u, u0, advection, diffusion = _transport()
    partial = ssprk2(rate=[advection])  # <- t5-partial
    program = surface.Program(advance=surface.Advance(partial, dt=1e-3), initial={u: u0})
    with pytest.raises(surface.CoverageNotExact) as caught:
        surface.Case(model=model, numerics=_numerics(u, advection, diffusion), program=program,
                     layout=_layout(domain))
    _refused(caught, surface.CoverageNotExact, "t5-partial", ("D5", "T5", "CC14"),
             "ssprk2(rate=model), or a rate option that names it")


def test_a_declared_term_is_in_a_balance():
    domain, model, u, a, nu, u0 = _model()
    advection = model.flux(of=u, value=a * u)
    unused = model.flux(of=u, value=2 * a * u)  # <- t5-unused
    model.balance(ddt(u) == -div(advection))
    numerics = surface.Numerics(represent={u: surface.CellAverage()}, schemes={advection: upwind()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3), initial={u: u0})
    with pytest.raises(surface.CoverageNotExact) as caught:
        surface.Case(model=model, numerics=numerics, program=program, layout=_layout(domain))
    _refused(caught, surface.CoverageNotExact, "t5-unused", ("D5", "T5", "CC14"),
             "the term in the balance of its quantity, or no such term")
    assert unused is model.terms[1]


def test_a_flux_on_a_support_of_rank_two_is_a_vector():
    plane = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    model = surface.Model("drift")
    q = model.quantity("q", surface.Scalar(plane))
    with pytest.raises(surface.SupportMismatch) as caught:
        model.flux(of=q, value=2.0 * q)  # <- e2-vector
    _refused(caught, surface.SupportMismatch, "e2-vector", ("E2",),
             "model.flux(of=q, value=vector(..., ...))")


def test_an_initial_datum_lives_on_the_support_of_its_quantity():
    domain, model, u, u0, advection, diffusion = _transport()
    elsewhere = model.data("v0", surface.Scalar(surface.Box(x=surface.Periodic(0, 2))))
    advance = surface.Advance(ssprk2(rate=model), dt=1e-3)
    with pytest.raises(surface.SupportMismatch) as caught:
        surface.Program(advance=advance, initial={u: elsewhere})  # <- e2-datum
    _refused(caught, surface.SupportMismatch, "e2-datum", ("E2",), "a datum of the support of u")


def test_a_construct_of_a_later_lot_names_the_lot_that_elaborates_it():
    _, model, *_ = _model()

    def stable_step(cell):
        return 1e-3

    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.Advance(ssprk2(rate=model), dt=stable_step)  # <- later-dt
    assert (caught.value.site.file, caught.value.site.line) == (HERE, _line("later-dt"))
    assert "a later lot of N8 (the step rules of 5.11, PF8) elaborates it" in str(caught.value)
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.compile(None)  # <- later-compile
    assert (caught.value.site.file, caught.value.site.line) == (HERE, _line("later-compile"))
    assert "the run of N8 (Simulation.run, RunResult and Output, CC14)" in str(caught.value)


# --- the refusals that the review of lot 1 asked for (#745) -------------------------------------

def _later(caught, marker, words):
    """A NotYetElaborated at the marked line, whose message names ``words``."""
    assert type(caught.value) is surface.NotYetElaborated
    assert (caught.value.site.file, caught.value.site.line) == (HERE, _line(marker))
    for word in words:
        assert word in str(caught.value)


def test_su10_a_weight_of_pops_math_beyond_the_rationals_is_not_elaborated_yet():
    weight = 1 - 1 / sqrt(2)

    @surface.method
    def blend(step, rate):
        stage = step.state + step.dt * rate(step.state, step.time)
        return surface.Candidate(weight * step.state + (1 - weight) * stage)  # <- su10-weight

    @surface.method
    def blend_right(step, rate):
        stage = step.state + step.dt * rate(step.state, step.time)
        return surface.Candidate(step.state * weight + stage * (1 - weight))  # <- su10-weight-r

    for method_value, marker in ((blend, "su10-weight"), (blend_right, "su10-weight-r")):
        with pytest.raises(surface.NotYetElaborated) as caught:
            _traced(lambda model, value=method_value: value(rate=model))
        _later(caught, marker, ("a weight or a duration of SU10 that is an expression of "
                                "pops.math", "a later lot of N8 with N11 (open point 16, SD95)"))


def test_su10_a_duration_beyond_the_polynomials_of_the_step_is_not_elaborated_yet():
    @surface.method
    def damped(step, rate):
        damping = exp(-step.dt)  # <- su10-exp
        return surface.Candidate(step.state + damping * rate(step.state, step.time))

    with pytest.raises(surface.NotYetElaborated) as caught:
        _traced(lambda model: damped(rate=model))
    _later(caught, "su10-exp", ("here exp with a duration of the method", "N8 with N11"))


def test_a_relayed_refusal_of_n8_speaks_of_step():
    """A refusal of N8 that names B and p reaches the user with ``step`` (delta review of #745,
    O-b). No construct of level 1 reaches such a refusal of N8 yet, since pops.Candidate always
    ends at step.time + step.dt, so the relay is called on the refusal that N8 writes."""
    from types import SimpleNamespace

    from pops.surface._method import translate
    from pops.time._program import traced_method as n8

    error = n8.NotYetElaborated(
        "the method 'm' reaches B.time + 1/2*p.dt: an achieved time other than B.time + p.dt "
        "(T26, T36) is elaborated in Phase 2 of N8; B.state stays")
    relayed = translate(error, SimpleNamespace(candidate_site=None, site=None))
    assert type(relayed) is surface.NotYetElaborated
    message = str(relayed)
    assert "reaches step.time + 1/2*step.dt: an achieved time other than step.time + step.dt " \
        "(T26, T36)" in message
    assert "step.state stays" in message
    assert not re.search(r"\bB\.|\bp\.dt", message)


def test_a_refusal_of_n8_keeps_its_own_sentence():
    @surface.method
    def compared(step, rate):
        if step.dt < 1:  # <- su4-n8-compare
            pass
        return surface.Candidate(step.state + step.dt * rate(step.state, step.time))

    with pytest.raises(surface.NotYetElaborated) as caught:
        _traced(lambda model: compared(rate=model))
    _later(caught, "su4-n8-compare", ())
    assert ": NotYetElaborated: a comparison of traced values builds a predicate (SU4), which a " \
        "later lot of N8 elaborates, with Python's max(...) and min(...) of traced values." \
        in str(caught.value)
    assert "has no elaborated form yet" not in str(caught.value)


def test_a_weight_other_than_one_before_div_is_not_elaborated_yet():
    domain, model, u, a, nu, u0 = _model()
    advection = model.flux(of=u, value=a * u)
    diffusion = model.diffusive_flux(of=u, value=-nu * grad(u))
    model.balance(ddt(u) == -2 * div(advection) - div(diffusion))  # <- r7-weight
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.Case(model=model, numerics=_numerics(u, advection, diffusion), program=program,
                     layout=_layout(domain))
    _later(caught, "r7-weight", ("the weight -2 of div(", "a weight other than 1 or -1"))


def test_two_point_takes_a_coefficient_that_does_not_vary_with_the_state():
    domain, model, u, a, nu, u0 = _model()
    advection = model.flux(of=u, value=a * u)
    diffusion = model.diffusive_flux(of=u, value=-u * grad(u))
    model.balance(ddt(u) == -div(advection + diffusion))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: upwind(), diffusion: two_point()})  # <- d5-k
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    with pytest.raises(surface.NotYetElaborated) as caught:
        surface.Case(model=model, numerics=numerics, program=program, layout=_layout(domain))
    _later(caught, "d5-k", ("two_point() under", "varies with the state", "face means of 2.6.4"))


def test_upwind_refuses_an_affine_flux():
    domain, model, u, a, nu, u0 = _model()
    advection = model.flux(of=u, value=a * u + a)
    model.balance(ddt(u) == -div(advection))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: upwind()})  # <- d5-affine
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    with pytest.raises(surface.RealizationMismatch) as caught:
        surface.Case(model=model, numerics=numerics, program=program, layout=_layout(domain))
    _refused(caught, surface.RealizationMismatch, "d5-affine", ("D5", "§2.6.4"),
             "rusanov() or hll(), which take the wave speeds of the flux")


@pytest.mark.parametrize("spelling, write", [
    ("Realization", "solvers={key: solver} in pops.Program, keyed by handles"),
    ("from_data", "initial={u: pops.project(e, quadrature=...)}, or initial={u: u0} for a datum"),
    ("integrate_rates", "the state arithmetic, step.state + step.dt * rate(step.state, "
                        "step.time)"),
    ("apply_increment", "S + J, a state plus an increment"),
    ("accept_all", "accept=[check1, check2]; a list of checks accepts when all of them do"),
    ("step_bound", "the bound as dt="),
    ("cfl_guard", "dt=cfl_limit(advection, cfl=..., max_dt=...) in pops.Advance"),
])
def test_su13_a_spelling_of_level_1_in_an_earlier_text_names_the_construct(spelling, write):
    with pytest.raises(surface.SymbolNotInSurface) as caught:
        getattr(surface, spelling)  # <- su13-earlier-level1
    _refused(caught, surface.SymbolNotInSurface, "su13-earlier-level1", ("SU3", "SU13", "§21.7"),
             write)
    assert "pops.%s is a spelling of an earlier text" % spelling in str(caught.value)


def test_su13_pops_and_pops_math_have_their_words_only():
    math = surface.math
    later = {name for name in dir(surface) if not name.startswith("_")
             and getattr(getattr(surface, name), "__module__", None) == surface.__name__}
    assert {name for name in dir(math) if not name.startswith("_")} == set(math.__all__)
    assert {name for name in dir(surface) if not name.startswith("_")} - later \
        == set(surface.__all__)
    for leaked in ("Expr", "Call", "Any", "annotations", "SpellingRefusal", "operand"):
        with pytest.raises(surface.SymbolNotInSurface):
            getattr(math, leaked)
