"""N8 lot 6: admissibility guards of a traced method (spec 0.3 IR4, S5, T38, T39).

M15, M16 and M17 write their explicit Euler step between two admissibility guards,
``admissible_euler`` of ``examples/api_v02/models_M15_orders.py`` (M17: ``twice_admissible_euler``,
two predicates in turn):

    U = cond(domain(B.state), lambda: B.state, lambda: fail(NotAdmissible))
    C = U + p.dt * flow(U, B.time)
    return Candidate(cond(domain(C), lambda: C, lambda: fail(NotAdmissible)), B.time + p.dt)

Each builder passes its b6 Program, kept verbatim, to ``traced_admissible_euler``; patching that
function selects the b6 route (MG1).

The IR parity rule (design note of lot 6, section 4): set aside, after checking each, the action
of each guard (``FailRun`` at b6, ``RejectAttempt`` with its ``trial_site`` on the traced route) and
the guard names; take out lot 2's ``finite()``; rename the presentation names by SSA position. For
M15, whose b6 candidate is the component-wise pointwise expression ``U[k] + dt*K[k]``, set aside
its form after checking it: the traced route writes the same affine combination as one
``linear_combine``. M17's b6 candidate is already that combination. The remaining Programs are
equal. M16 adds the Lorentz source of the profile to the transport rate: its candidate would be the
affine combination ``U + dt*K + dt*S``, which b6 writes ``U + dt*(K + S)``, and the source is not
a rate that a component returns: M16 keeps its b6 Program (pilot ruling Q1, 11/11).
"""
import json
import re

import pytest

import pops
import examples.api_v02.models_M15_M18 as moments
import examples.api_v02.models_M15_orders as orders
from pops._structural_data import canonical_json
from pops.runtime._simulation import NotAdmissible, SolveFailed
from pops.time import FailRun
from pops.time._program.control import AdvanceControl
from pops.time._program.elaboration import Advance, Proposal, elaborate
from pops.time._program.guards import Failure, cond, fail
from pops.time._program.traced_method import (
    Candidate, ClosedBodyViolation, NotYetElaborated, method,
)
from pops.time.solve_outcome import RejectAttempt
from tests.python.unit.time.test_field_requests_n8 import (
    CONTROL_LINES, INVENTORY, _digests, _named, _presentation_names, _renumbered, _source, _spans,
)

# profile -> (module that builds it, builder, program name, sites in order)
PROFILES = {
    "M15": (orders, orders.build_m15_orders, "M15_S04_Euler",
            ("admissible_euler/body#0/domain", "admissible_euler/body#1/domain")),
    "M17": (moments, moments.build_m17, "FanLi_raw_path_Euler",
            tuple("twice_admissible_euler/body#%d/%s" % (k, name)
                  for k, name in enumerate(("positive", "domain", "positive", "domain")))),
}
FINITE_OPS = ("reduce", "scalar_isfinite", "acceptance_guard")


def _b6_route(case, block, state, dt, *, name, b6, **components):
    return b6()


def _spec(build, monkeypatch, route=None):
    with monkeypatch.context() as patch:
        if route is not None:
            patch.setattr(orders, "traced_admissible_euler", route)
        return build()


def _fixed(site):
    """T38: a guard of B.state, before the rate, is fixed; a guard of the candidate is not."""
    method_name, body, _ = site.split("/")
    guards = 4 if method_name == "twice_admissible_euler" else 2
    return int(body[len("body#"):]) < guards // 2


# --- the IR parity rule ------------------------------------------------------------------------

def _guards(ir, action, sites):
    """Set aside the action and the name of each acceptance guard, after checking them: b6's are
    ``FailRun`` without a site, the traced route's ``RejectAttempt`` at ``sites`` in order."""
    nodes = [node for node in ir["nodes"] if node["op"] == "acceptance_guard"]
    assert len(nodes) == len(sites or nodes)
    for index, node in enumerate(nodes):
        attrs = node["attrs"]
        assert attrs.pop("action") == action
        site = attrs.pop("trial_site", None)
        if sites is None:
            assert site is None
        else:
            assert site == {"site": sites[index], "retry_can_change": not _fixed(sites[index]),
                            "variant": "NotAdmissible"}
            assert attrs["guard"] == sites[index]
        attrs["guard"] = "<guard %d>" % index
    rows = ir["step_transaction"]["guards"]
    assert [row["name"] for row in rows] == [node["name"] for node in rows]
    for index, row in enumerate(rows):
        assert row.pop("action") == action and row["role"] == "invariant"
        if sites is not None:
            assert row["name"] == sites[index]
        row["name"] = "<guard %d>" % index


def _traced_without_finite(ir):
    """Take lot 2's finite() assessment out of the traced ``ir``: its last three nodes."""
    nodes = ir["nodes"]
    norm, finite, guard = nodes[-3:]
    assert [node["op"] for node in (norm, finite, guard)] == list(FINITE_OPS)
    assert guard["attrs"]["guard"] == "finite" and guard["inputs"][1] == finite["id"]
    (commit,) = ir["commits"]
    assert commit["value"] == guard["id"]
    commit["value"] = guard["inputs"][0]
    assert ir["step_transaction"]["guards"][-1]["name"] == "finite"
    del ir["step_transaction"]["guards"][-1]
    return {norm["id"], finite["id"], guard["id"]}


def _euler_candidate(b6_ir, traced_ir):
    """M15: b6's candidate ``U[k] + dt*K[k]``, component by component, and the traced
    ``linear_combine`` of U and K with the weights 1 and dt: set aside the form, after checking
    that both read the same two values and compute the same combination."""
    (b6_node,) = [node for node in b6_ir["nodes"] if node["name"] == "candidate"
                  and node["op"] == "pointwise_expression"]
    (node,) = [node for node in traced_ir["nodes"] if node["op"] == "linear_combine"]
    assert b6_node["op"] == "pointwise_expression" and b6_node["inputs"] == node["inputs"]
    one = [[1, {"kind": "integer", "value": "1"}]]
    rows = []
    for k in range(len(b6_node["attrs"]["expressions"])):
        rows += [["input", 0, k], ["coefficient", one], ["input", 1, k],
                 ["mul", 5 * k + 1, 5 * k + 2], ["add", 5 * k, 5 * k + 3]]
    assert b6_node["attrs"]["expression_nodes"] == rows
    assert b6_node["attrs"]["expressions"] == [5 * k + 4 for k in
                                               range(len(b6_node["attrs"]["expressions"]))]
    assert node["attrs"]["coeffs"] == [[[0, {"kind": "integer", "value": "1"}]], one]
    b6_node["op"], b6_node["attrs"] = node["op"], node["attrs"]


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_guarded_profile_authors_the_b6_program(profile, monkeypatch):
    module, build, name, sites = PROFILES[profile]
    b6 = _spec(build, monkeypatch, _b6_route).case._time
    traced = _spec(build, monkeypatch).case._time
    assert b6.name == traced.name == name
    b6_ir = json.loads(canonical_json(b6._serialize(include_provenance=False)))
    traced_ir = json.loads(canonical_json(traced._serialize(include_provenance=False)))
    dropped = _traced_without_finite(traced_ir)
    b6_new = _renumbered(b6_ir, set())
    traced_new = _renumbered(traced_ir, dropped)
    control = traced_ir.pop("advance_control")
    assert [row["site"] for row in control["trial_failures"]] == list(sites)
    for ir in (b6_ir, traced_ir):
        ir.pop("version")
    _guards(b6_ir, FailRun().to_data(), None)
    _guards(traced_ir, RejectAttempt().to_data(), sites)
    if profile == "M15":
        _euler_candidate(b6_ir, traced_ir)
    assert _normalized(b6_ir, b6, b6_new) == _normalized(traced_ir, traced, traced_new)


def _normalized(ir, program, new):
    return _digests(_named(canonical_json(ir), _presentation_names(program, new)))


def test_m16_stays_on_its_b6_program():
    """M16 adds its Lorentz source to the candidate, which a component cannot return as a rate:
    it keeps its b6 Program until a model source-term route (design note, revision 2, Q1)."""
    program = moments.build_m16().case._time
    assert getattr(program, "_advance_control", None) is None
    assert [type(guard.action) for guard in program._acceptance_guards] == [FailRun, FailRun]


# --- the C++ parity rule -----------------------------------------------------------------------

def _guard_lines(lines, sites, fixed):
    """Set aside the throw of each guard: b6's ``FailRun`` error, the traced rejection of the
    attempt with the phase and the site of the guard (A2)."""
    out, seen = [], 0
    for line in lines:
        b6 = re.match(r'(\s*)throw std::runtime_error\("acceptance guard \'[^\']+\' failed"\);$',
                      line)
        traced = re.match(r'(\s*)throw pops::runtime::program::StepAttemptRejected\(pops::'
                          r'SolveStatus::kInvalidEvaluation, "(trial|trial_failure)", '
                          r'"([^"]+) failed: NotAdmissible"\);$', line)
        if b6 or traced:
            if traced:
                indent, phase, site = traced.groups()
                assert site == sites[seen]
                assert phase == ("trial_failure" if fixed[seen] else "trial")
            out.append((b6 or traced).group(1) + "<guard %d>" % seen)
            seen += 1
            continue
        out.append(line)
    assert seen == len(sites)
    return out


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_guarded_profile_generates_the_b6_cpp(profile, monkeypatch):
    """The generated C++ of each profile is b6's, line for line, once the guard throws, lot 2's
    finite() and the control lines are set aside. M15 builds its candidate componentwise in b6
    and as a linear combination on the traced route (the IR test checks that both compute the
    same sum): the source-map span of that one node is set aside in both (review of the note,
    round 2, optional point)."""
    _, build, _, sites = PROFILES[profile]
    b6_spec = _spec(build, monkeypatch, _b6_route)
    traced_spec = _spec(build, monkeypatch)
    b6, traced = b6_spec.case._time, traced_spec.case._time
    tail = traced._values[-3:]
    assert [value.op for value in tail] == list(FINITE_OPS)
    tail_ids = {value.id for value in tail}
    candidate = set()
    if profile == "M15":
        (b6_node,) = [v for v in b6._values if v.op == "pointwise_expression"
                      and v.name == "candidate"]
        (node,) = [v for v in traced._values if v.op == "linear_combine"]
        assert node.id == b6_node.id and node.vtype == b6_node.vtype
        candidate = {node.id}
    assert [(v.id, v.op if v.id not in candidate else None, v.vtype) for v in b6._values] == [
        (v.id, v.op if v.id not in candidate else None, v.vtype)
        for v in traced._values if v.id not in tail_ids]
    new = {value.id: value.id for value in traced._values if value.id not in tail_ids}
    names = _presentation_names(traced, new)
    names.update({site: "<guard %d>" % k for k, site in enumerate(sites)})
    b6_names = _presentation_names(b6, new)
    b6_names.update({guard.name: "<guard %d>" % k for k, guard in
                     enumerate(b6._acceptance_guards)})
    b6_lines = _named(_source(b6_spec), b6_names).splitlines()
    traced_lines = _named(_source(traced_spec), names).splitlines()
    traced_lines, dropped = _spans(traced_lines, tail_ids)
    assert any('"assessment", "Advance assessment \'finite\' rejected the candidate"' in line
               for line in dropped)
    (row,) = [i for i, line in enumerate(traced_lines) if line.startswith(INVENTORY)]
    data = json.loads(traced_lines[row][len(INVENTORY):])
    assert [entry["op"] for entry in data["nodes"][-3:]] == list(FINITE_OPS)
    data["nodes"] = data["nodes"][:-3]
    traced_lines[row] = INVENTORY + canonical_json(data)
    if candidate:
        b6_lines, b6_kernel = _spans(b6_lines, candidate)
        traced_lines, kernel = _spans(traced_lines, candidate)
        assert b6_kernel and kernel
        b6_lines = _inventory_without(b6_lines, candidate)
        traced_lines = _inventory_without(traced_lines, candidate)
    start = traced_lines.index(CONTROL_LINES[0])
    assert traced_lines[start:start + 3] == CONTROL_LINES
    del traced_lines[start:start + 3]
    renamed = ["<guard %d>" % k for k in range(len(sites))]
    fixed = [_fixed(site) for site in sites]
    b6_lines = _guard_lines(b6_lines, renamed, fixed)
    traced_lines = _guard_lines(traced_lines, renamed, fixed)
    assert _digests("\n".join(b6_lines)).splitlines() == _digests(
        "\n".join(traced_lines)).splitlines()


def _inventory_without(lines, ids):
    """The inventory line of ``lines`` without the rows of the nodes ``ids``."""
    (row,) = [i for i, line in enumerate(lines) if line.startswith(INVENTORY)]
    data = json.loads(lines[row][len(INVENTORY):])
    kept = [entry for entry in data["nodes"] if entry.get("id", entry.get("ssa")) not in ids]
    assert len(kept) == len(data["nodes"]) - len(ids)
    data["nodes"] = kept
    return lines[:row] + [INVENTORY + canonical_json(data)] + lines[row + 1:]


# --- classification (T38) and the compiled control -----------------------------------------------

@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_guards_of_the_accepted_state_are_fixed_sites(profile, monkeypatch):
    _, build, _, sites = PROFILES[profile]
    elaborated = _spec(build, monkeypatch).case._time
    control = elaborated._advance_control
    rows = control.to_data()["trial_failures"]
    assert rows == [{"site": site, "guard": "admissibility", "variant": ["NotAdmissible"],
                     "retry_can_change": not _fixed(site), "classified": True, "lot": None}
                    for site in sites]
    assert AdvanceControl.from_data(control.to_data()) == control
    assert [guard.name for guard in elaborated._acceptance_guards] == list(sites) + ["finite"]


def test_the_generated_guard_names_its_site_and_phase(monkeypatch):
    spec = _spec(orders.build_m15_orders, monkeypatch)
    source = _source(spec)
    assert ('throw pops::runtime::program::StepAttemptRejected(pops::SolveStatus::'
            'kInvalidEvaluation, "trial_failure", "admissible_euler/body#0/domain failed: '
            'NotAdmissible");') in source
    assert ('throw pops::runtime::program::StepAttemptRejected(pops::SolveStatus::'
            'kInvalidEvaluation, "trial", "admissible_euler/body#1/domain failed: '
            'NotAdmissible");') in source


# --- the elaborated forms and their refusals ----------------------------------------------------

def _parts(monkeypatch):
    """The case, block, state and components of M15, taken from its builder."""
    captured = {}

    def route(case, block, state, dt, *, name, b6, **components):
        captured.update(case=case, block=block, state=state, dt=dt, **components)
    with monkeypatch.context() as patch:
        patch.setattr(orders, "traced_admissible_euler", route)
        orders.build_m15_orders()
    return captured


def _elaborate(parts, body, **components):
    components = components or {"flow": parts["flow"], "domain": parts["domain"]}
    return elaborate(Advance(body(**components)), case=parts["case"],
                     state=parts["block"][parts["state"]], proposal=Proposal(parts["dt"]))


def test_fail_elaborates_not_admissible_without_a_payload():
    failure = fail(NotAdmissible)
    assert isinstance(failure, Failure) and failure.variant == "NotAdmissible"
    with pytest.raises(NotYetElaborated, match="fail\\(SolveFailed\\).*later lot of N8"):
        fail(SolveFailed)
    with pytest.raises(NotYetElaborated, match="payload"):
        fail(NotAdmissible, payload=1.0)
    with pytest.raises(TypeError, match="closed sum of T39"):
        fail(ValueError)
    with pytest.raises(TypeError, match="closed sum of T39"):
        fail("NotAdmissible")


def test_cond_elaborates_the_guard_of_a_field_by_a_predicate(monkeypatch):
    parts = _parts(monkeypatch)

    @method
    def other_then(B, p, flow, domain):
        U = cond(domain(B.state), lambda: B.state + p.dt * flow(B.state, B.time),
                 lambda: fail(NotAdmissible))
        return Candidate(U, B.time + p.dt)

    @method
    def other_otherwise(B, p, flow, domain):
        U = cond(domain(B.state), lambda: B.state, lambda: B.state)
        return Candidate(U + p.dt * flow(U, B.time), B.time + p.dt)

    @method
    def not_a_predicate(B, p, flow, domain):
        U = cond(True, lambda: B.state, lambda: fail(NotAdmissible))
        return Candidate(U + p.dt * flow(U, B.time), B.time + p.dt)

    @method
    def eager_arms(B, p, flow, domain):
        U = cond(domain(B.state), B.state, fail(NotAdmissible))
        return Candidate(U + p.dt * flow(U, B.time), B.time + p.dt)

    @method
    def truth_value(B, p, flow, domain):
        if domain(B.state):
            pass
        return Candidate(B.state + p.dt * flow(B.state, B.time), B.time + p.dt)

    with pytest.raises(NotYetElaborated, match="then-arm returns U"):
        _elaborate(parts, other_then)
    with pytest.raises(NotYetElaborated, match="otherwise-arm that is fail\\(NotAdmissible\\)"):
        _elaborate(parts, other_otherwise)
    with pytest.raises(NotYetElaborated, match="guard of a field by a predicate component"):
        _elaborate(parts, not_a_predicate)
    with pytest.raises(TypeError, match="functions of no argument"):
        _elaborate(parts, eager_arms)
    with pytest.raises(TypeError, match="no truth value"):
        _elaborate(parts, truth_value)


def test_a_predicate_is_consumed_by_exactly_one_cond(monkeypatch):
    # Review of the note, round 2 (optional point): a predicate read that no cond consumes
    # would guard nothing, silently.
    parts = _parts(monkeypatch)

    @method
    def dangling(B, p, flow, domain):
        domain(B.state)
        return Candidate(B.state + p.dt * flow(B.state, B.time), B.time + p.dt)

    @method
    def twice(B, p, flow, domain):
        holds = domain(B.state)
        U = cond(holds, lambda: B.state, lambda: fail(NotAdmissible))
        V = cond(holds, lambda: B.state, lambda: fail(NotAdmissible))
        return Candidate(U + p.dt * flow(V, B.time), B.time + p.dt)

    with pytest.raises(TypeError, match="domain\\(U\\) in 'dangling' is a predicate that no "
                                        "cond\\(...\\) consumes"):
        _elaborate(parts, dangling)
    with pytest.raises(TypeError, match="consumed by one cond only"):
        _elaborate(parts, twice)


def test_a_guarded_combination_is_read_by_a_rate_a_guard_or_the_candidate(monkeypatch):
    parts = _parts(monkeypatch)

    @method
    def guarded_then_combined(B, p, flow, domain):
        C = B.state + p.dt * flow(B.state, B.time)
        G = cond(domain(C), lambda: C, lambda: fail(NotAdmissible))
        return Candidate(G + p.dt * flow(B.state, B.time), B.time + p.dt)

    with pytest.raises(NotYetElaborated, match="guarded field that is not materialized"):
        _elaborate(parts, guarded_then_combined)


def test_a_predicate_returns_a_bool_and_reads_only_its_value(monkeypatch):
    parts = _parts(monkeypatch)
    domain = parts["domain"]

    def not_bool(value):
        return value.prog.min(value.prog.value(value.name + "_m", value)) + 1.0

    def reads_dt(value):
        return domain(value.prog.value(value.name + "_dt", value + value.prog.dt * value,
                                      at=value.point))

    def publishes(value):
        value.prog.guard("inner", value, domain(value), action=FailRun())
        return domain(value)

    for predicate, error, match in (
            (not_bool, TypeError, "must return a scalar Bool"),
            (reads_dt, ClosedBodyViolation, "reads the step size"),
            (publishes, ClosedBodyViolation, "changed the acceptance guards")):
        with pytest.raises(error, match=match):
            _elaborate(parts, orders.admissible_euler, flow=parts["flow"], domain=predicate)


def test_a_predicate_inside_a_component_is_not_elaborated(monkeypatch):
    parts = _parts(monkeypatch)
    held = {}

    def flow(value):
        held["domain"](held["state"])
        return parts["flow"](value)

    @method
    def leak(B, p, flow, domain):
        held.update(domain=domain, state=B.state)
        return Candidate(B.state + p.dt * flow(B.state, B.time), B.time + p.dt)

    with pytest.raises(NotYetElaborated, match="inside the component 'flow'"):
        _elaborate(parts, leak, flow=flow, domain=parts["domain"])


def test_a_predicate_of_another_trace_is_refused(monkeypatch):
    parts = _parts(monkeypatch)
    held = {}

    @method
    def first(B, p, flow, domain):
        held["state"] = B.state
        return Candidate(B.state + p.dt * flow(B.state, B.time), B.time + p.dt)

    @method
    def second(B, p, flow, domain):
        U = cond(domain(held["state"]), lambda: held["state"], lambda: fail(NotAdmissible))
        return Candidate(B.state + p.dt * flow(B.state, B.time), B.time + p.dt)

    _elaborate(parts, first)
    with pytest.raises(ClosedBodyViolation, match="another method trace"):
        _elaborate(parts, second)
