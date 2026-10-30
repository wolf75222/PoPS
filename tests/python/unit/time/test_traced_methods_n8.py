"""N8 lot 1: traced methods elaborated by the b6 builder (spec 0.3, sections 5.2 to 5.4).

The seven profiles whose stages read one rate of their model (M01, M02, M03, M04, M07, M24, M25)
now write their step as the traced method ``shu_osher_ssprk2`` of
``examples/api_v02/models_M01_M14.py``, in a Program that keeps the b6 name of the profile. Their
b6 builder ``_ssprk2`` stays as the parity reference (MG1). Both routes must elaborate the same
program: equal serialized IR and equal generated C++ once the presentation names (the values and
stages, CC2) are renamed by SSA position, and once the History metadata capacity that ``_ssprk2``
writes for a Program without History is set aside (B3: a traced case writes no resource). The C++
of M25 (17 components) takes about two minutes per route to generate, so this source-only module
compares its IR, and its native run is compared bit for bit outside CI (handoff of N8 lot 1). The
refusals of E1, T33, T34 and SU10 are tested on M01.
"""
import base64
import json
import re
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

import pops
import examples.api_v02.models_M01_M14 as early
import examples.api_v02.models_M22_M25_M28 as late
from pops._ir.symbolic import SymbolicTruthValueError
from pops._structural_data import canonical_json
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.time import FixedDt, TimePoint
from pops.time._program.elaboration import Advance, Proposal, Retry, elaborate
from pops.time._program.global_history_storage import prepare_issuance, publish_issuance
from pops.time._program.traced_method import (
    _GUARDED_TABLES, Candidate, ClosedBodyViolation, IncrementNotDerivable, IncrementTypeMismatch,
    NotYetElaborated, method,
)

# profile -> (module that calls _traced_ssprk2, builder, name of the program on both routes)
PROFILES = {
    "M01": (early, early.build_m01, "M01_SSPRK2"),
    "M02": (early, early.build_m02, "M02_SSPRK2"),
    "M03": (early, early.build_m03, "M03_SSPRK2"),
    "M04": (early, early.build_m04, "M04_combined_SSPRK2"),
    "M07": (early, early.build_m07, "M07_SSPRK2"),
    "M24": (late, late.build_m24, "M24_SSPRK2"),
    "M25": (late, late.build_m25, "M25_SSPRK2"),
}
CXX_PROFILES = ("M01", "M02", "M03", "M04", "M07", "M24")
B6_HISTORY_METADATA = (8 * 1024 * 1024, 16 * 1024 * 1024)
B6_HISTORY_LINE = "  ctx.configure_history_field_metadata(UINT64_C(8388608), UINT64_C(16777216));"


def _b6_route(case, block, state, flow, dt, *, name):
    return early._ssprk2(case, block, state, lambda p, u, point, label: flow(u), dt, name=name)


def _spec(module, build, monkeypatch, route=None):
    with monkeypatch.context() as patch:
        if route is not None:
            patch.setattr(module, "_traced_ssprk2", route)
        return build()


def _source(spec):
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    assert resolved.time is spec.case._time
    return emit_cpp_program(resolved.time, model_graph=ProgramModelGraph.from_resolved_blocks(
        resolved.blocks))


def _presentation_names(program):
    """Each presentation name of ``program``, renamed by the SSA position that first carries it."""
    names = {program.name: "<program>"}
    for value in program._values:
        names.setdefault(value.name, "<value %d>" % value.id)
        point_name = getattr(value.point, "name", None)
        if point_name is not None:
            names.setdefault(point_name, "<stage of value %d>" % value.id)
    return names


def _normalized(text, names):
    text = re.sub(r"#authoring=\d+", "#authoring=N", text)
    text = re.sub(r"// pops-source-map@1 (\w+) (\S+)", lambda m: "// pops-source-map@1 %s %s" % (
        m.group(1), canonical_json(json.loads(base64.b64decode(m.group(2))))), text)
    for name in sorted(names, key=len, reverse=True):
        text = re.sub(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(name), names[name], text)
    hashes = {}
    # Program hashes and clock identities digest the names: rename them by first appearance.
    return re.sub(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])",
                  lambda m: hashes.setdefault(m.group(0), "<digest %d>" % len(hashes)), text)


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_traced_profile_authors_the_b6_program(profile, monkeypatch):
    module, build, name = PROFILES[profile]
    b6 = _spec(module, build, monkeypatch, _b6_route).case._time
    traced = _spec(module, build, monkeypatch).case._time
    assert b6.name == traced.name == name
    assert [(v.op, v.vtype) for v in b6._values] == [(v.op, v.vtype) for v in traced._values]
    # _ssprk2 authors a History metadata capacity for a Program without History rings.
    assert not b6._histories and not traced._histories
    assert b6.history_field_metadata_budget == B6_HISTORY_METADATA
    assert traced.history_field_metadata_budget is None
    b6_ir = b6._serialize(include_provenance=False)
    b6_ir["history_field_metadata_budget"] = None
    assert _normalized(canonical_json(b6_ir), _presentation_names(b6)) == _normalized(
        canonical_json(traced._serialize(include_provenance=False)), _presentation_names(traced))
    assert b6.to_graph().graph_hash != traced.to_graph().graph_hash  # names are hashed (C47)


@pytest.mark.parametrize("profile", CXX_PROFILES)
def test_traced_profile_generates_the_b6_cpp(profile, monkeypatch):
    module, build, _ = PROFILES[profile]
    b6_spec = _spec(module, build, monkeypatch, _b6_route)
    traced_spec = _spec(module, build, monkeypatch)
    b6_lines = _normalized(_source(b6_spec), _presentation_names(b6_spec.case._time)).splitlines()
    # The one line that the History capacity of _ssprk2 adds; the Program has no History ring.
    assert b6_lines.count(B6_HISTORY_LINE) == 1
    b6_lines.remove(B6_HISTORY_LINE)
    assert b6_lines == _normalized(_source(traced_spec),
                                   _presentation_names(traced_spec.case._time)).splitlines()


def test_traced_profile_records_its_closed_form(monkeypatch):
    elaborations = []

    def record(case, block, state, flow, dt, *, name):
        elaborations.append(early.elaborate(
            early.Advance(early.shu_osher_ssprk2(flow=flow)), case=case, state=block[state],
            proposal=early.Proposal(dt), name=name))
    monkeypatch.setattr(early, "_traced_ssprk2", record)
    early.build_m01()
    (elaboration,) = elaborations
    assert elaboration.program.name == "M01_SSPRK2"
    assert elaboration.program._step_strategy.dt == elaboration.proposal.dt
    assert elaboration.advance.retry.max_attempts == 1 and elaboration.advance.accept is None
    assert elaboration.trace.describe() == {
        "method": "shu_osher_ssprk2",
        "components": ["flow"],
        "stages": [{"stage": 0, "c": "0", "materializes": None},
                   {"stage": 1, "c": "1", "materializes": "shu_osher_ssprk2_state1"}],
        "evaluations": [{"evaluation": 0, "component": "flow", "stage": 0},
                        {"evaluation": 1, "component": "flow", "stage": 1}],
        "increment": [{"evaluation": 0, "duration": "1/2*dt"},
                      {"evaluation": 1, "duration": "1/2*dt"}],
        "tolerated_weights": [],
    }


# --- refusals, on the M01 case -----------------------------------------------------------------

@pytest.fixture
def m01(monkeypatch):
    """The M01 case, its block, state, rate and step, without a program attached."""
    parts = {}

    def capture(case, block, state, flow, dt, *, name):
        parts.update(case=case, block=block, state=state, flow=flow, dt=dt)
    monkeypatch.setattr(early, "_traced_ssprk2", capture)
    early.build_m01()
    return parts


def _elaborate(parts, specialized, proposal=None, **advance):
    return elaborate(Advance(specialized, **advance), case=parts["case"],
                     state=parts["block"][parts["state"]],
                     proposal=Proposal(parts["dt"]) if proposal is None else proposal)


def _stage_formula(update):
    """A two-stage method whose candidate is update(B, p, k0, U1, k1)."""
    @method
    def two_stage(B, p, flow):
        k0 = flow(B.state, B.time)
        U1 = B.state + p.dt*k0
        k1 = flow(U1, B.time + p.dt)
        return Candidate(update(B, p, k0, U1, k1), B.time + p.dt)
    return two_stage


@method
def forward(B, p, flow):
    return Candidate(B.state + p.dt*flow(B.state, B.time), B.time + p.dt)


def test_a_traced_method_is_traced_once_per_elaboration(m01):
    calls = []

    @method
    def counted(B, p, flow):
        calls.append(None)
        return Candidate(B.state + p.dt*flow(B.state, B.time), B.time + p.dt)
    elaboration = _elaborate(m01, counted(m01["flow"]))
    assert len(calls) == 1
    assert m01["case"]._time is elaboration.program and elaboration.program.name == "counted"
    assert elaboration.trace.increment == ((0, {1: Fraction(1)}),)


@pytest.mark.parametrize("update, increment, tolerated, sums", [
    (lambda B, p, k0, U1, k1: Fraction(1, 3)*B.state + Fraction(2, 3)*(U1 + p.dt*k1),
     ((0, {1: Fraction(2, 3)}), (1, {1: Fraction(2, 3)})), (), []),
    # The binary64 values of 1/3 and 2/3 sum to 1 - 2**-54: within 1e-12, listed (SU10).
    (lambda B, p, k0, U1, k1: (1/3)*B.state + (2/3)*(U1 + p.dt*k1),
     ((0, {1: Fraction(2/3)}), (1, {1: Fraction(2/3)})),
     (("the candidate of 'two_stage'", 1 - Fraction(1, 2**54)),),
     [{"where": "the candidate of 'two_stage'", "deviation": "-5.551115123125783e-17"}]),
    # 0.1 + 0.2 + 0.7 == 1.0 in Python; their binary64 values sum to 1 - 2**-55.
    (lambda B, p, k0, U1, k1: .1*B.state + .2*B.state + .7*U1,
     ((0, {1: Fraction(.7)}),),
     (("the candidate of 'two_stage'", 1 - Fraction(1, 2**55)),),
     [{"where": "the candidate of 'two_stage'", "deviation": "-2.7755575615628914e-17"}]),
], ids=["fractions-exact", "binary64-within-tolerance", "binary64-rounded-to-one"])
def test_weights_are_exact_or_binary64_within_tolerance(m01, update, increment, tolerated, sums):
    trace = _elaborate(m01, _stage_formula(update)(m01["flow"])).trace
    assert trace.increment == increment and trace.tolerated == tolerated
    assert trace.describe()["tolerated_weights"] == sums


@pytest.mark.parametrize("update, match", [
    (lambda B, p, k0, U1, k1: .5*B.state + .4999*(U1 + p.dt*k1),
     "sum to 0.9999, .*within 1e-12"),
    (lambda B, p, k0, U1, k1: .5*B.state + .5*(U1 + p.dt*k1) - .5*B.state, "sum to 0.5"),
    (lambda B, p, k0, U1, k1: 2*B.state + p.dt*k1, "sum to 2, neither one nor zero"),
    (lambda B, p, k0, U1, k1: U1 - B.state, "is a difference"),
    (lambda B, p, k0, U1, k1: (.1 + .2)*B.state - .3*B.state + p.dt*k1, "is a difference"),
    (lambda B, p, k0, U1, k1: B.state + k1, "coefficient 1, which is not a duration"),
    (lambda B, p, k0, U1, k1: B.state + (1 + p.dt)*k1, "coefficient 1 \\+ 1\\*dt"),
], ids=["binary64-beyond-tolerance", "weights-sum-to-one-half", "weights-sum-to-two",
        "difference", "binary64-difference", "rate-without-duration", "rate-with-a-constant"])
def test_candidate_outside_the_step_algebra_is_an_increment_type_mismatch(m01, update, match):
    with pytest.raises(IncrementTypeMismatch, match=match):
        _elaborate(m01, _stage_formula(update)(m01["flow"]))
    assert m01["case"]._time is None


def test_durations_are_polynomials_of_the_step_that_vanish_with_it(m01):
    trace = _elaborate(m01, _stage_formula(lambda B, p, k0, U1, k1: B.state + p.dt*p.dt*k1
                                           + p.dt/2*k0)(m01["flow"])).trace
    assert trace.increment == ((1, {2: Fraction(1)}), (0, {1: Fraction(1, 2)}))
    assert trace.describe()["increment"] == [{"evaluation": 1, "duration": "1*dt^2"},
                                             {"evaluation": 0, "duration": "1/2*dt"}]


def test_a_closed_form_candidate_has_no_derivable_increment(m01):
    @method
    def decay(B, p, flow):
        return Candidate((1 - p.dt)*B.state, B.time + p.dt)
    with pytest.raises(IncrementNotDerivable, match="exact_flow.*increment="):
        _elaborate(m01, decay(m01["flow"]))
    assert m01["case"]._time is None


@pytest.mark.parametrize("argument, match", [
    (lambda B, p, k0: 2*B.state, "state of evaluation 1 of 'flow' combines states whose weights "
                                 "sum to 2"),
    (lambda B, p, k0: B.state + k0, "adds the rate flow#0 to a state with the coefficient 1"),
    (lambda B, p, k0: p.dt*k0, "is a difference"),
    (lambda B, p, k0: (1 - p.dt)*B.state, "depends on the step"),
], ids=["weights-sum-to-two", "rate-without-duration", "increment-alone", "closed-form"])
def test_the_state_of_a_rate_is_a_state(m01, argument, match):
    @method
    def staged(B, p, flow):
        k0 = flow(B.state, B.time)
        k1 = flow(argument(B, p, k0), B.time + p.dt)
        return Candidate(B.state + p.dt*k1, B.time + p.dt)
    with pytest.raises(IncrementTypeMismatch, match=match):
        _elaborate(m01, staged(m01["flow"]))

    @method
    def taylor(B, p, flow):
        k0 = flow(B.state, B.time)
        return Candidate(B.state + p.dt*flow(B.state + p.dt*p.dt*k0, B.time + p.dt), B.time + p.dt)
    assert _elaborate(m01, taylor(m01["flow"])).trace.increment == ((1, {1: Fraction(1)}),)


def test_ambient_step_size_and_foreign_values_are_closed_body_violations(m01):
    ambient = pops.Program("ambient")
    with pytest.raises(ClosedBodyViolation, match="step size"):
        _elaborate(m01, _stage_formula(
            lambda B, p, k0, U1, k1: .5*B.state + .5*(U1 + ambient.dt*k1))(m01["flow"]))
    foreign = pops.Program("foreign")
    captured = foreign.state(m01["block"][m01["state"]]).n
    with pytest.raises(ClosedBodyViolation, match="value of a Program"):
        _elaborate(m01, _stage_formula(lambda B, p, k0, U1, k1: B.state + captured)(m01["flow"]))

    @method
    def clocked(B, p, flow):
        return Candidate(B.state + p.dt*flow(B.state, pops.time.TimePoint(ambient.clock, 0)),
                         B.time + p.dt)
    with pytest.raises(ClosedBodyViolation, match="clock or a time point"):
        _elaborate(m01, clocked(m01["flow"]))


@pytest.mark.parametrize("component", [
    lambda rate: (lambda u: u.prog.dt*rate(u)),
    lambda rate: (lambda u: rate(u)*u.prog.dt),
    lambda rate: (lambda u: u.prog.value("scaled", u.prog.dt*rate(u), at=u.point)),
], ids=["dt-times-rate", "rate-times-dt", "materialized"])
def test_a_component_that_reads_the_step_size_is_a_closed_body_violation(m01, component):
    with pytest.raises(ClosedBodyViolation, match="reads the step size of its Program"):
        _elaborate(m01, forward(component(m01["flow"])))


@pytest.mark.parametrize("publish, table", [
    (lambda prog: prog.integral_state("stash", initial=0.), "integral states"),
    (lambda prog: prog.cadence(stride=2), "cadence"),
    (lambda prog: prog.cell_local_time(tick_denominator=4), "cell local time"),
    (lambda prog: prog.step_strategy(FixedDt(1.)), "step strategy"),
], ids=["integral-state", "cadence", "cell-local-time", "step-strategy"])
def test_a_component_that_publishes_is_a_closed_body_violation(m01, publish, table):
    def publishing(u):
        publish(u.prog)  # a change of the Program outside the Candidate
        return m01["flow"](u)
    with pytest.raises(ClosedBodyViolation, match="changed the %s of its Program" % table):
        _elaborate(m01, forward(publishing))


@pytest.mark.parametrize("component", [
    lambda rate: (lambda u: u),
    lambda rate: (lambda u: u + rate(u)),
    lambda rate: (lambda u: 2*u - u),
], ids=["state", "state-plus-rate", "combination-of-states"])
def test_a_component_returns_a_rate(m01, component):
    with pytest.raises(IncrementTypeMismatch, match="type state, not a rate"):
        _elaborate(m01, forward(component(m01["flow"])))


@pytest.mark.parametrize("restage, match", [
    (lambda u: u.prog.value("later", u, at=u.prog.stage("later_stage", c=Fraction(1, 2))),
     "evaluated later, rhs2 at another time"),
    (lambda u: u.prog.value("end", u, at=TimePoint(u.prog.clock, 1)),
     "evaluated end, rhs2 at another time"),
    (lambda u: u.prog.value("previous", u, at=TimePoint(u.prog.clock, -1)),
     "evaluated previous, rhs2 at another time"),
], ids=["stage-in-component", "end-of-the-step", "previous-step"])
def test_a_component_evaluates_at_the_time_it_is_given(m01, restage, match):
    with pytest.raises(ClosedBodyViolation, match=match):
        _elaborate(m01, forward(lambda u: m01["flow"](restage(u))))


def test_a_component_may_read_its_state_again_at_its_own_time(m01):
    again = forward(lambda u: m01["flow"](u.prog.value("start", u, at=TimePoint(u.prog.clock, 0))))
    assert _elaborate(m01, again).trace.increment == ((0, {1: Fraction(1)}),)


@pytest.mark.parametrize("reuse, match", [
    (lambda rate, u, first: first[1], "returned a rate that this evaluation did not compute"),
    (lambda rate, u, first: rate(u.prog.value("mixed", .5*u + .5*first[0], at=u.point)),
     "read fluid, which it was not given and did not compute"),
], ids=["rate-of-an-earlier-evaluation", "state-of-an-earlier-evaluation"])
def test_a_component_evaluates_only_the_state_it_is_given(m01, reuse, match):
    first = []  # the state and the rate of the first evaluation

    def keeping(u):
        if not first:
            first.extend((u, m01["flow"](u)))
            return first[1]
        return reuse(m01["flow"], u, first)
    heun = _stage_formula(lambda B, p, k0, U1, k1: B.state + p.dt/2*k0 + p.dt/2*k1)
    with pytest.raises(ClosedBodyViolation, match=match):
        _elaborate(m01, heun(keeping))


@pytest.mark.parametrize("body, match", [
    (lambda rate: (lambda P, x: P.value("y", x + P.dt*rate(x), at=x.point)),
     "reads the step size of its Program"),
    (lambda rate: (lambda P, x: P.value("later", x, at=TimePoint(P.clock, 1))),
     "evaluated later at another time"),
], ids=["step-size", "end-of-the-step"])
def test_a_control_flow_body_of_a_component_is_checked_as_the_component(m01, body, match):
    # b6 records the values of a range, while_ or subcycle body outside Program._values.
    with pytest.raises(ClosedBodyViolation, match=match):
        _elaborate(m01, forward(lambda u: m01["flow"](u.prog.range(u, 1, body(m01["flow"])))))


def test_a_control_flow_body_reads_only_the_state_given_to_its_component(m01):
    first = []  # the state of the first evaluation

    def keeping(u):
        if not first:
            first.append(u)
            return m01["flow"](u)
        return m01["flow"](u.prog.range(
            u, 1, lambda P, x: P.value("mixed", .5*x + .5*first[0], at=x.point)))
    heun = _stage_formula(lambda B, p, k0, U1, k1: B.state + p.dt/2*k0 + p.dt/2*k1)
    with pytest.raises(ClosedBodyViolation, match="read fluid, which it was not given"):
        _elaborate(m01, heun(keeping))
    # A body that stays at the state it is given is a rate of that state.
    honest = forward(lambda u: m01["flow"](u.prog.range(
        u, 2, lambda P, x: P.value("same", x, at=x.point))))
    assert _elaborate(m01, honest).trace.increment == ((0, {1: Fraction(1)}),)


def test_the_guarded_tables_are_tables_of_the_builder(m01):
    """The guarded tables are named by strings: a table that b6 renames would leave the guard."""
    program = _elaborate(m01, forward(m01["flow"])).program
    # b6 creates the issuance table at the first global field history that it issues.
    assert [name for name in _GUARDED_TABLES if not hasattr(program, name)] == [
        "_global_field_history_issuance"]
    publish_issuance(program, "n8", prepare_issuance(program, "n8", {}))
    assert all(hasattr(program, name) for name in _GUARDED_TABLES)


def test_a_traced_value_does_not_outlive_its_trace(m01):
    kept = []

    @method
    def keeping(B, p, flow):
        kept.append(B.state)
        return Candidate(B.state + p.dt*flow(B.state, B.time), B.time + p.dt)
    _elaborate(m01, keeping(m01["flow"]))

    @method
    def reusing(B, p, flow):
        return Candidate(B.state + p.dt*flow(kept[0], B.time), B.time + p.dt)
    with pytest.raises(ClosedBodyViolation, match="trace has ended"):
        _elaborate(m01, reusing(m01["flow"]))


@pytest.mark.parametrize("use, error", [
    (lambda B, p, k0: B.state + p.dt*k0 if k0 else B.state, SymbolicTruthValueError),
    (lambda B, p, k0: float(p.dt), SymbolicTruthValueError),
    (lambda B, p, k0: int(p.dt), SymbolicTruthValueError),
    (lambda B, p, k0: np.asarray(B.state), SymbolicTruthValueError),
    (lambda B, p, k0: p.dt < 1, NotYetElaborated),
    (lambda B, p, k0: max(B.state, k0), NotYetElaborated),
], ids=["if", "float", "int", "asarray", "comparison", "max"])
def test_a_traced_value_has_no_truth_value_and_no_number(m01, use, error):
    @method
    def deciding(B, p, flow):
        k0 = flow(B.state, B.time)
        use(B, p, k0)
        return Candidate(B.state + p.dt*k0, B.time + p.dt)
    with pytest.raises(error, match="E1|predicate"):
        _elaborate(m01, deciding(m01["flow"]))


def test_stage_formulas_stay_affine(m01):
    with pytest.raises(TypeError, match="affine"):
        _elaborate(m01, _stage_formula(lambda B, p, k0, U1, k1: B.state + p.dt*(k0*k1))(m01["flow"]))
    with pytest.raises(TypeError, match="not a field"):
        _elaborate(m01, _stage_formula(lambda B, p, k0, U1, k1: 1. + B.state)(m01["flow"]))


@pytest.mark.parametrize("dt", [Fraction(1, 320), Decimal("0.003125")], ids=["fraction", "decimal"])
def test_an_exact_proposal_runs_its_nearest_binary64_step(m01, dt):
    elaboration = _elaborate(m01, forward(m01["flow"]), proposal=Proposal(dt))
    assert elaboration.proposal.dt == dt and type(elaboration.proposal.dt) is type(dt)
    assert elaboration.program._step_strategy.dt == float(dt)


@pytest.mark.parametrize("construct, match", [
    (lambda parts: _elaborate(parts, _stage_formula(lambda B, p, k0, U1, k1: U1)(parts["flow"]),
                              accept=object()), "assessment"),
    (lambda parts: _elaborate(parts, _stage_formula(lambda B, p, k0, U1, k1: U1)(parts["flow"]),
                              retry=Retry(max_attempts=2)), "retry budgets"),
    (lambda parts: _elaborate(parts, _stage_formula(lambda B, p, k0, U1, k1: U1)(parts["flow"]),
                              clock=object()), "clocks"),
    (lambda parts: method(lambda B, p: None, memory=object()), "method memory"),
    (lambda parts: Proposal(parts["dt"], tolerance=1.e-6), "choices"),
], ids=["accept", "retry", "clock", "memory", "choices"])
def test_constructs_of_later_lots_are_refused_by_name(m01, construct, match):
    with pytest.raises(NotYetElaborated, match=match):
        construct(m01)


def test_relaxed_endpoints_and_term_selection_are_refused_by_name(m01):
    @method
    def half(B, p, flow):
        return Candidate(B.state + p.dt/2*flow(B.state, B.time), B.time + p.dt/2)
    with pytest.raises(NotYetElaborated, match="T26"):
        _elaborate(m01, half(m01["flow"]))

    @method
    def selecting(B, p, flow):
        return Candidate(B.state + p.dt*flow(B.state, B.time, terms=()), B.time + p.dt)
    with pytest.raises(NotYetElaborated, match="T5"):
        _elaborate(m01, selecting(m01["flow"]))


def test_specialization_names_every_component(m01):
    two_stage = _stage_formula(lambda B, p, k0, U1, k1: U1)
    with pytest.raises(TypeError, match="missing components flow"):
        two_stage()
    with pytest.raises(TypeError, match="unknown or repeated component"):
        two_stage(m01["flow"], flow=m01["flow"])
    with pytest.raises(TypeError, match="specialized"):
        Advance(two_stage)
    with pytest.raises(TypeError, match="Candidate"):
        _elaborate(m01, method(lambda B, p, flow: B.state)(m01["flow"]))
