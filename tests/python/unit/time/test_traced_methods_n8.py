"""N8 lot 1: traced methods elaborated by the b6 builder (spec 0.3, sections 5.2 to 5.4).

The seven profiles whose stages read one rate of their model (M01, M02, M03, M04, M07, M24, M25)
now write their step as the traced method ``ssprk2`` of ``examples/api_v02/models_M01_M14.py``.
Their b6 builder ``_ssprk2`` stays as the parity reference (MG1). Both routes must elaborate the
same program: equal serialized IR and equal generated C++ once the presentation names (the
program, its values and its stages, CC2) are renamed by SSA position, and once the History
metadata capacity that ``_ssprk2`` writes for a Program without History is set aside (B3: a traced
case writes no resource). The C++ of M25 (17 components) takes about two minutes per route to
generate, so this source-only module compares its IR, and its native run is compared bit for bit
outside CI (handoff of N8 lot 1). The refusals of T33 and T34 are tested on M01.
"""
import base64
import json
import re
from fractions import Fraction

import pytest

import pops
import examples.api_v02.models_M01_M14 as early
import examples.api_v02.models_M22_M25_M28 as late
from pops._ir.symbolic import SymbolicTruthValueError
from pops._structural_data import canonical_json
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.time._program.elaboration import Advance, Proposal, Retry, elaborate
from pops.time._program.traced_method import (
    Candidate, ClosedBodyViolation, IncrementNotDerivable, NotYetElaborated, method,
)

# profile -> (module that calls _traced_ssprk2, builder, name of the b6 program)
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


def _b6_route(name):
    def route(case, block, state, flow, dt):
        return early._ssprk2(case, block, state, lambda p, u, point, label: flow(u), dt, name=name)
    return route


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
    module, build, b6_name = PROFILES[profile]
    b6 = _spec(module, build, monkeypatch, _b6_route(b6_name)).case._time
    traced = _spec(module, build, monkeypatch).case._time
    assert b6.name == b6_name and traced.name == "ssprk2"
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
    module, build, b6_name = PROFILES[profile]
    b6_spec = _spec(module, build, monkeypatch, _b6_route(b6_name))
    traced_spec = _spec(module, build, monkeypatch)
    b6_lines = _normalized(_source(b6_spec), _presentation_names(b6_spec.case._time)).splitlines()
    # The one line that the History capacity of _ssprk2 adds; the Program has no History ring.
    assert b6_lines.count(B6_HISTORY_LINE) == 1
    b6_lines.remove(B6_HISTORY_LINE)
    assert b6_lines == _normalized(_source(traced_spec),
                                   _presentation_names(traced_spec.case._time)).splitlines()


def test_traced_profile_records_its_closed_form(monkeypatch):
    elaborations = []

    def record(case, block, state, flow, dt):
        elaborations.append(early.elaborate(early.Advance(early.ssprk2(flow=flow)), case=case,
                                            state=block[state], proposal=early.Proposal(dt)))
    monkeypatch.setattr(early, "_traced_ssprk2", record)
    early.build_m01()
    (elaboration,) = elaborations
    assert elaboration.program.name == "ssprk2"
    assert elaboration.program._step_strategy.dt == elaboration.proposal.dt
    assert elaboration.advance.retry.max_attempts == 1 and elaboration.advance.accept is None
    assert elaboration.trace.describe() == {
        "method": "ssprk2",
        "components": ["flow"],
        "stages": [{"stage": 0, "c": "0", "materializes": None},
                   {"stage": 1, "c": "1", "materializes": "ssprk2_state1"}],
        "evaluations": [{"evaluation": 0, "component": "flow", "stage": 0},
                        {"evaluation": 1, "component": "flow", "stage": 1}],
        "increment": [{"evaluation": 0, "duration": "1/2*dt"},
                      {"evaluation": 1, "duration": "1/2*dt"}],
    }


# --- refusals, on the M01 case -----------------------------------------------------------------

@pytest.fixture
def m01(monkeypatch):
    """The M01 case, its block, state, rate and step, without a program attached."""
    parts = {}

    def capture(case, block, state, flow, dt):
        parts.update(case=case, block=block, state=state, flow=flow, dt=dt)
    monkeypatch.setattr(early, "_traced_ssprk2", capture)
    early.build_m01()
    return parts


def _elaborate(parts, specialized, **advance):
    return elaborate(Advance(specialized, **advance), case=parts["case"],
                     state=parts["block"][parts["state"]], proposal=Proposal(parts["dt"]))


def _stage_formula(update):
    """A two-stage method whose candidate is update(B, p, k0, U1, k1)."""
    @method
    def two_stage(B, p, flow):
        k0 = flow(B.state, B.time)
        U1 = B.state + p.dt*k0
        k1 = flow(U1, B.time + p.dt)
        return Candidate(update(B, p, k0, U1, k1), B.time + p.dt)
    return two_stage


def test_a_traced_method_is_traced_once_per_elaboration(m01):
    calls = []

    @method
    def forward(B, p, flow):
        calls.append(None)
        return Candidate(B.state + p.dt*flow(B.state, B.time), B.time + p.dt)
    elaboration = _elaborate(m01, forward(m01["flow"]))
    assert len(calls) == 1
    assert m01["case"]._time is elaboration.program
    assert elaboration.trace.increment == ((0, Fraction(1)),)


def test_exact_coefficients_decide_the_increment(m01):
    exact = _stage_formula(lambda B, p, k0, U1, k1:
                           Fraction(1, 3)*B.state + Fraction(2, 3)*(U1 + p.dt*k1))
    trace = _elaborate(m01, exact(flow=m01["flow"])).trace
    assert trace.increment == ((0, Fraction(2, 3)), (1, Fraction(2, 3)))
    # The binary64 values of 1/3 and 2/3 sum to 1 - 2**-54: no increment derives from them.
    rounded = _stage_formula(lambda B, p, k0, U1, k1: (1/3)*B.state + (2/3)*(U1 + p.dt*k1))
    with pytest.raises(IncrementNotDerivable, match="exact rationals"):
        _elaborate(m01, rounded(flow=m01["flow"]))


@pytest.mark.parametrize("update", [
    lambda B, p, k0, U1, k1: .5*B.state + .5*(U1 + p.dt*k1) - .5*B.state,
    lambda B, p, k0, U1, k1: B.state + k1,
    lambda B, p, k0, U1, k1: B.state + p.dt*p.dt*k1,
], ids=["state-coefficient", "rate-without-duration", "squared-duration"])
def test_candidate_without_a_derivable_increment_is_refused(m01, update):
    with pytest.raises(IncrementNotDerivable):
        _elaborate(m01, _stage_formula(update)(flow=m01["flow"]))
    assert m01["case"]._time is None


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


def test_a_component_that_publishes_is_a_closed_body_violation(m01):
    def publishing(u):
        u.prog.integral_state("stash", initial=0.)  # a published quantity outside the Candidate
        return m01["flow"](u)

    @method
    def forward(B, p, flow):
        return Candidate(B.state + p.dt*flow(B.state, B.time), B.time + p.dt)
    with pytest.raises(ClosedBodyViolation, match="component"):
        _elaborate(m01, forward(publishing))


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


def test_python_branching_on_a_traced_value_is_refused(m01):
    @method
    def branching(B, p, flow):
        k0 = flow(B.state, B.time)
        return Candidate(B.state + p.dt*k0 if k0 else B.state, B.time + p.dt)
    with pytest.raises(SymbolicTruthValueError):
        _elaborate(m01, branching(m01["flow"]))


def test_stage_formulas_stay_affine(m01):
    with pytest.raises(TypeError, match="affine"):
        _elaborate(m01, _stage_formula(lambda B, p, k0, U1, k1: B.state + p.dt*(k0*k1))(m01["flow"]))
    with pytest.raises(TypeError, match="not a field"):
        _elaborate(m01, _stage_formula(lambda B, p, k0, U1, k1: 1. + B.state)(m01["flow"]))


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
