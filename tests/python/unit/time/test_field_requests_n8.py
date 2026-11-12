"""N8 lot 5: field requests and Auxiliary method memory (spec 0.3 T6, T7, T27, T38; CN4, C51).

The five profiles whose stages solve a field (M08, M09, M10, M11, M13) write their step as
``recorded_ssprk2`` of ``examples/api_v02/models_M01_M14.py``: a component requests its field with
``field.at(U)``, and the Candidate returns the stage-1 state and observations as an Auxiliary
memory, lowered to History rings of depth one. Each builder passes its b6 evaluator, kept
verbatim, to ``_traced_recorded_ssprk2``; patching that function selects the b6 route, so the
reference of the parity tests is the b6 code itself (MG1).

The IR parity rule (design note of lot 5, section 5): take out of both routes the History stores
and the nodes that only stores read, transitively (on b6, M08's and M09's twin observation at each
stage); renumber the rest by position and rename the presentation names; take out lot 2's
``finite()``; set aside, after checking each, the consume action of each request outcome (``FailRun``
at b6, ``RejectAttempt`` on the traced route) and its ``trial_site``. The remaining Programs are
equal. The C++ parity takes the b6 code with its stores elided (and, for M08 and M09, its reads
memoized), which the IR parity shows to be the b6 Program without its store-only nodes, so that
the SSA positions, which the generated identifiers carry, agree; the set-aside is then line by
line (A2). Since N8 lot 6 it also sets aside the reuse of each stage-0 solve, a fixed site, across
the attempts of one step (T7), after checking its exact shape.
"""
import base64
import json
import re
from types import SimpleNamespace

import pytest

import pops
import examples.api_v02.models_M01_M14 as early
import examples.api_v02.models_M13 as late
from pops._structural_data import canonical_json
from pops.codegen.program_codegen import _refuse_advance_memory_route, emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.fields._program_problem import FieldSolution
from pops.fields.problem import FieldProblem
from pops.problem.handles import FieldHandle
from pops.time import FailRun
from pops.time._program import field_requests
from pops.time._program.control import ADVANCE_CONTROL_TRIAL_VARIANTS_SCHEMA, AdvanceControl
from pops.time._program.elaboration import Advance, Proposal, elaborate
from pops.time._program.memory import Auxiliary, State, declared_memory
from pops.time._program.outputs import Output, Product
from pops.time._program.traced_method import (
    Candidate, ClosedBodyViolation, NotYetElaborated, method,
)
from pops.time.solve_outcome import RejectAttempt

# profile -> (module that calls _traced_recorded_ssprk2, builder, program name, memory members)
PROFILES = {
    "M08": (early, early.build_m08, "M08_SSPRK2", ("stage1_state", "stage1_potential")),
    "M09": (early, early.build_m09, "M09_SSPRK2", ("stage1_state", "stage1_potential")),
    "M10": (early, early.build_m10, "M10_SSPRK2", ("stage1_state", "stage1_potential")),
    "M11": (early, early.build_m11, "M11_SSPRK2", ("stage1_state", "stage1_potential")),
    "M13": (late, late.build_m13, "M13_SSPRK2",
            ("stage1_state", "stage1_potential", "stage1_photo")),
}
# The profiles whose b6 evaluator observes the potential twice: once for its store, once for
# the gradient that it publishes (fields/_program_problem.py, FieldSolution.gradient).
TWIN_READS = ("M08", "M09")
METHOD = "shu_osher_ssprk2_recorded"
FINITE_OPS = ("reduce", "scalar_isfinite", "acceptance_guard")
SOURCE_MAP = re.compile(r"// pops-source-map@1 (\w+) (\S+)")
INVENTORY = "// pops-source-map@1 inventory "
UINT64_DIGEST = re.compile(r"\{UINT64_C\(\d+\), UINT64_C\(\d+\), UINT64_C\(\d+\), UINT64_C\(\d+\)\}")
CONTROL_LINES = [
    "// Compiled control of the elaborated Advance (spec 0.3 section 5.2): the attempts of",
    "// one step (Retry.max_attempts, T38). System::run_advance repeats a rejected proposal.",
    'extern "C" int pops_program_advance_max_attempts() { return 1; }',
]


def _b6_route(case, block, state, flow, dt, *, name, memory, record, b6):
    return early._ssprk2(case, block, state, b6, dt, name=name)


def _spec(module, build, monkeypatch, route=None):
    with monkeypatch.context() as patch:
        if route is not None:
            patch.setattr(module, "_traced_recorded_ssprk2", route)
        return build()


def _site(evaluation, problem="live_Poisson"):
    return "%s/flow#%d/%s/solve" % (METHOD, evaluation, problem)


# --- the IR parity rule ------------------------------------------------------------------------

def _store_closure(ir):
    """The History stores of ``ir`` and the nodes that only stores read, transitively."""
    readers = {}
    for node in ir["nodes"]:
        for source in node["inputs"]:
            readers.setdefault(source, set()).add(node["id"])
    dropped = {node["id"] for node in ir["nodes"] if node["op"] == "store_history"}
    grown = True
    while grown:
        grown = False
        for node in ir["nodes"]:
            if node["id"] not in dropped and readers.get(node["id"]) \
                    and readers[node["id"]] <= dropped:
                dropped.add(node["id"])
                grown = True
    return dropped


def _renumbered(ir, dropped):
    """``ir`` without the nodes ``dropped``, its SSA ids renumbered by position; the map."""
    ids = []
    for node in ir["nodes"]:
        ids.append(node["id"])
        ids.extend(inner["id"] for inner in node["attrs"].get("apply_block", ()))
    new = {old: position for position, old in enumerate(sorted(set(ids) - set(dropped)))}

    def renumber(node):
        node["id"] = new[node["id"]]
        node["inputs"] = [new[source] for source in node["inputs"]]
        attrs = node["attrs"]
        for inner in attrs.get("apply_block", ()):
            renumber(inner)
        for key in ("apply_in", "apply_out", "apply_result"):
            if key in attrs:
                attrs[key] = new[attrs[key]]
        context = node.get("field_context")
        for read in (context or {}).get("reads", [context] if context else []):
            read["stage_sources"] = [[source, new[value]]
                                     for source, value in read["stage_sources"]]

    ir["nodes"] = [node for node in ir["nodes"] if node["id"] not in dropped]
    for node in ir["nodes"]:
        renumber(node)
    for commit in ir["commits"]:
        commit["value"] = new[commit["value"]]
    return new


def _consume_actions(ir, action, sites):
    """Set aside the consume action of every request outcome, nested solve graphs included,
    after checking it is ``action``; the top-level outcomes carry ``sites`` in order (None: no
    ``trial_site``)."""
    top = [node["attrs"].get("trial_site") for node in ir["nodes"] if node["op"] == "solve_outcome"]
    assert top == (sites if sites is not None else [None] * len(top))

    def walk(item):
        if isinstance(item, dict):
            if item.get("op") == "solve_outcome":
                attrs = item["attrs"]
                assert attrs["action"] == action
                assert attrs.pop("trial_site", None) in (sites or [None])
                attrs["action"] = "<consume action>"
            for value in item.values():
                walk(value)
        elif isinstance(item, list):
            for value in item:
                walk(value)
    walk(ir)


def _presentation_names(program, new):
    names = {program.name: "<program>"}
    for value in program._values:
        if value.id not in new:
            continue
        names.setdefault(value.name, "<value %d>" % new[value.id])
        point = getattr(value.point, "name", None)
        if point is not None:
            names.setdefault(point, "<stage of value %d>" % new[value.id])
    return names


def _normalized(text, names):
    return _digests(_named(text, names))


def _named(text, names):
    """``text`` with its presentation names renamed by SSA position and its source maps
    decoded."""
    text = re.sub(r"#authoring=\d+", "#authoring=N", text)
    text = SOURCE_MAP.sub(lambda m: "// pops-source-map@1 %s %s" % (
        m.group(1), canonical_json(json.loads(base64.b64decode(m.group(2))))), text)
    for name in sorted(names, key=len, reverse=True):
        text = re.sub(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(name), names[name], text)
    return text


def _digests(text):
    """``text`` with the digests, which hash the names, renamed by first appearance."""
    digests = {}
    text = re.sub(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])",
                  lambda m: digests.setdefault(m.group(0), "<digest %d>" % len(digests)), text)
    quads = {}
    return UINT64_DIGEST.sub(lambda m: quads.setdefault(m.group(0), "<uint64 digest %d>"
                                                         % len(quads)), text)


def _sites(profile):
    """The trial sites of the requests of ``profile``: stage 0 fixed, stage 1 retryable (T38)."""
    problems = {"M11": ("M11_Poisson",), "M13": ("streamer_Poisson", "photon_Helmholtz")}.get(
        profile, ("live_Poisson",))
    return [{"site": _site(stage, problem), "retry_can_change": bool(stage)}
            for stage in (0, 1) for problem in problems]


def _traced_without_memory(traced, ir):
    """Take the finite() assessment of lot 2 and the memory stores out of the traced ``ir``,
    checking that the stores are the last nodes, after the assessment, and read the candidate's
    stage-1 state and request reads."""
    nodes = ir["nodes"]
    stores = [node for node in nodes if node["op"] == "store_history"]
    assert nodes[-len(stores):] == stores
    assert [node["op"] for node in nodes[-len(stores) - 3:-len(stores)]] == list(FINITE_OPS)
    norm, finite, guard = nodes[-len(stores) - 3:-len(stores)]
    (commit,) = ir["commits"]
    assert commit["value"] == guard["id"] and guard["inputs"][1] == finite["id"]
    commit["value"] = guard["inputs"][0]
    ir["step_transaction"]["guards"] = []
    return {norm["id"], finite["id"], guard["id"]} | {store["id"] for store in stores}


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_field_profile_authors_the_b6_program(profile, monkeypatch):
    module, build, name, members = PROFILES[profile]
    b6 = _spec(module, build, monkeypatch, _b6_route).case._time
    traced = _spec(module, build, monkeypatch).case._time
    assert b6.name == traced.name == name
    # The traced route has exactly the stage-1 rings, one per member, of depth one.
    assert sorted(traced._histories) == sorted(members)
    assert sorted(b6._histories) == sorted(members + tuple(
        member.replace("stage1", "stage0") for member in members))
    b6_ir = json.loads(canonical_json(b6._serialize(include_provenance=False)))
    traced_ir = json.loads(canonical_json(traced._serialize(include_provenance=False)))
    # The failure sites of the requests list their variants: control schema 5, IR 35 (N9 « La
    # référence sous les solves de relation » §3.6; schema 3 and IR 32 until then).
    assert traced_ir["version"] == 35 and b6_ir["version"] < 30
    b6_rows = {row["name"]: row for row in b6_ir["histories"]}
    assert [row for row in traced_ir["histories"]] == [b6_rows[member] for member in
                                                      sorted(members)]
    b6_persistence = {row["name"]: row for row in b6_ir["history_persistence"]}
    assert traced_ir["history_persistence"] == [b6_persistence[member] for member in
                                                sorted(members)]
    # Each member's store reads the node that b6's same-named store reads, renumbered.
    b6_stores = {node["name"]: node["inputs"] for node in b6_ir["nodes"]
                 if node["op"] == "store_history"}
    traced_stores = {node["name"]: node["inputs"] for node in traced_ir["nodes"]
                     if node["op"] == "store_history"}
    assert sorted(traced_stores) == sorted(members)
    b6_dropped = _store_closure(b6_ir)
    traced_dropped = _traced_without_memory(traced, traced_ir)
    assert traced_dropped == _store_closure(traced_ir) | {
        node["id"] for node in traced_ir["nodes"] if node["op"] in FINITE_OPS}
    b6_rows_by_id = {node["id"]: node for node in b6_ir["nodes"]}
    twins = {}
    for member in members:
        (source,) = b6_stores[member]
        if source in b6_dropped:
            # M08 and M09: b6's twin observation, the same node as the read the gradient takes.
            assert profile in TWIN_READS and member == "stage1_potential"
            (twin,) = [node for node in b6_ir["nodes"] if node["id"] == source + 1]
            assert twin["op"] == "field_component" and twin["inputs"] == \
                b6_rows_by_id[source]["inputs"]
            twins[member] = source + 1
    b6_new = _renumbered(b6_ir, b6_dropped)
    traced_new = _renumbered(traced_ir, traced_dropped)
    for member in members:
        (b6_source,) = b6_stores[member]
        (traced_source,) = traced_stores[member]
        assert traced_new[traced_source] == b6_new[twins.get(member, b6_source)]
    for ir in (b6_ir, traced_ir):
        for key in ("histories", "history_persistence", "history_field_metadata_budget",
                    "version"):
            ir.pop(key, None)
    control = traced_ir.pop("advance_control")
    assert control["schema"] == ADVANCE_CONTROL_TRIAL_VARIANTS_SCHEMA
    assert [row["ring"] for row in control["memory"]] == list(members)
    _consume_actions(b6_ir, FailRun().to_data(), None)
    _consume_actions(traced_ir, RejectAttempt().to_data(), _sites(profile))
    assert _normalized(canonical_json(b6_ir), _presentation_names(b6, b6_new)) == _normalized(
        canonical_json(traced_ir), _presentation_names(traced, traced_new))


# --- the C++ parity rule -----------------------------------------------------------------------

def _source(spec):
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    return emit_cpp_program(resolved.time, model_graph=ProgramModelGraph.from_resolved_blocks(
        resolved.blocks))


def _b6_without_stores(module, build, monkeypatch, profile):
    """The b6 code with its stores elided and, for M08 and M09, its reads memoized: the b6
    Program without its store-only nodes, as the IR parity shows."""
    with monkeypatch.context() as patch:
        patch.setattr(module, "_traced_recorded_ssprk2", _b6_route)
        patch.setattr(pops.Program, "store_history", lambda self, *args, **kwargs: None)
        if profile in TWIN_READS:
            read = FieldSolution.__getitem__
            memo = {}

            def memoized(solution, unknown):
                key = (id(solution), unknown.qualified_id)
                if key not in memo:
                    memo[key] = (solution, read(solution, unknown))
                return memo[key][1]
            patch.setattr(FieldSolution, "__getitem__", memoized)
        return build()


def _spans(lines, ssa_ids):
    """``lines`` without the source-map spans of ``ssa_ids``, and the lines taken out."""
    kept, dropped, inside = [], [], None
    for line in lines:
        span = re.match(r"\s*// pops-source-map@1 (begin|end) (\{.*\})$", line)
        ssa = json.loads(span.group(2))["ssa"] if span else None
        if inside is None and span and span.group(1) == "begin" and ssa in ssa_ids:
            inside = ssa
        if inside is not None:
            dropped.append(line.strip())
            if span and span.group(1) == "end" and ssa == inside:
                inside = None
            continue
        kept.append(line)
    assert inside is None
    return kept, dropped


def _set_aside_requests(lines, profile, *, traced):
    """Set aside, line by line, what the consume action of each request changes (A2): the
    ``KrylovFailureActions`` of its controls, its ``consume(...)`` line, the branch that throws
    ``StepAttemptRejected`` with the phase and site, and the site name in its run error."""
    action = "kRejectAttempt" if traced else "kFailRun"
    actions = "pops::KrylovFailureActions{%s}" % ", ".join(["pops::SolveAction::%s" % action] * 3)
    sites = iter(_sites(profile))
    out, index, seen = [], 0, 0
    while index < len(lines):
        line = lines[index]
        if "pops::KrylovFailureActions{" in line:
            assert actions in line
            line = line.replace(actions, "<failure actions>")
        if "apply_measured_original_residual_gate(" in line:
            # N9's residual gate (SV8) takes the same consume action as the request.
            gate = ', pops::SolveAction::%s, "solve_linear");' % action
            assert line.endswith(gate)
            line = line[:-len(gate)] + ', <gate action>, "solve_linear");'
        consume = re.match(r"(\s*)const pops::SolveReport (kr\d+)_report = \2\.consume\(", line)
        if consume:
            indent, kr = consume.groups()
            seen += 1
            if traced:
                site = next(sites)
                phase = "trial" if site["retry_can_change"] else "trial_failure"
                assert lines[index + 1] == indent + "if (%s_report.action == pops::SolveAction::"\
                    "kRejectAttempt) {" % kr
                assert lines[index + 2] == indent + (
                    '  throw pops::runtime::program::StepAttemptRejected(%s_report.status, "%s", '
                    'std::string("%s failed: ") + %s_report.reason);' % (
                        kr, phase, site["site"], kr))
                assert lines[index + 3] == indent + "}"
                error = lines[index + 4]
                assert error.startswith(indent + 'throw std::runtime_error(std::string("%s failed: '
                                        '") + ' % site["site"])
                error = error.replace(site["site"], "<site>")
                index += 5
            else:
                assert line == indent + "const pops::SolveReport %s_report = %s.consume(pops::"\
                    "SolveConsumption::kFailRun);" % (kr, kr)
                error = lines[index + 1].replace('"solve_linear failed: ', '"<site> failed: ')
                index += 2
            out.extend([indent + "<consume %s>" % kr, error])
            continue
        out.append(line)
        index += 1
    assert seen == len(_sites(profile))
    return out


def _attempt_reuse(lines, fixed):
    """Set aside the reuse of a fixed-site solve across the attempts of one step (N8 lot 6, T7),
    after checking its exact shape: the slot of the prelude, the flag, the guard of the initial
    guess, the block around the solve that records the slot, and the counter of a reuse. What
    remains is b6's solve, line for line. ``fixed`` is the number of fixed sites."""
    out, index, slots, blocks = [], 0, set(), 0
    while index < len(lines):
        line = lines[index]
        slot = re.fullmatch(r"\s*auto attempt_slot(\d+) = std::make_shared<std::remove_cvref_t<"
                            r"decltype\(ctx\)>::AttemptReuseSlot>\(\);", line)
        if slot:
            slots.add(slot.group(1))
            index += 1
            continue
        flag = re.fullmatch(r"(\s*)const bool attempt_reused(\d+) = ctx\.attempt_slot_reused\("
                            r"\*attempt_slot\2\);", line)
        if flag:
            indent, solve = flag.groups()
            assert solve in slots
            guess = lines[index + 1]
            prefix = indent + "if (!attempt_reused%s) " % solve
            assert guess.startswith(prefix + "pops::PureFieldAlgebra::zero_valid(")
            out.append(indent + guess[len(prefix):])
            index += 2
            while lines[index] != indent + "if (!attempt_reused%s) {" % solve:
                out.append(lines[index])
                index += 1
            index += 1
            while lines[index] != indent + "  ctx.complete_attempt_slot(*attempt_slot%s);" % solve:
                assert lines[index].startswith(indent + "  ")
                out.append(lines[index][2:] if lines[index].startswith(indent + "  ") else
                           lines[index])
                index += 1
            assert lines[index + 1] == indent + "}"
            count = re.fullmatch(r"(\s*)if \(attempt_reused%s\) \+\+(.+)\[1\]; else "
                                 r"\+\+\2\[0\];" % solve, lines[index + 2])
            assert count and count.group(1) == indent
            out.append(indent + "++%s[0];" % count.group(2))
            index += 3
            blocks += 1
            continue
        assert "attempt_reused" not in line and "attempt_slot" not in line
        out.append(line)
        index += 1
    assert blocks == len(slots) == fixed
    return out


def _history_lines(lines, members):
    """Set aside the History lines of the traced memory: the registration of each ring, the
    checkpoint table of the state rings and the rotation of the accepted step."""
    kept, dropped = [], []
    table = False
    for line in lines:
        if line.startswith('extern "C" int pops_program_checkpoint_history_count()'):
            # The state rings are checkpointed: stage1_state; b6's elided Program holds none.
            assert line.endswith("{ return %d; }" % sum(m == "stage1_state" for m in members))
            kept.append(line.replace("{ return 1; }", "{ return 0; }"))
            continue
        if line.startswith('extern "C"') and "checkpoint_history" in line:
            table = True
        elif table and line.startswith("}"):
            table = False
        if table and line.strip().startswith("case 0:"):
            dropped.append(line)
            continue
        if line.strip().startswith("ctx.register_history(") or \
                line.strip().startswith("ctx.rotate_histories("):
            assert any('"%s"' % member in line for member in members) or "rotate" in line
            dropped.append(line)
            continue
        kept.append(line)
    assert sum("register_history" in line for line in dropped) == len(members)
    assert sum("rotate_histories" in line for line in dropped) == 1
    return kept


def _stencil_capacity(lines, program):
    """``lines`` with the two stencil expression capacities that History and the IR size set
    taken out, checked first: the bases, one per stencil evaluation and per retained History
    slot (``program_stencil_budget.py``), and the byte length of the serialized IR."""
    capacity = len(canonical_json(program._serialize(include_provenance=False)).encode("utf-8"))
    evaluations = sum(value.op == "stencil_flux_divergence" for value in program._values)
    bases = evaluations*(1 + sum(depth + 1 for depth in program._histories.values()))
    pattern = re.compile(r"ctx\.configure_stencil_expressions\(UINT64_C\((\d+)\), UINT64_C\(1\), "
                         r"UINT64_C\((\d+)\)\);")

    def replace(match):
        assert (int(match.group(1)), int(match.group(2))) == (bases, capacity)
        return "ctx.configure_stencil_expressions(<bases>, UINT64_C(1), <IR bytes>);"
    return [pattern.sub(replace, line) for line in lines]


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_field_profile_generates_the_b6_cpp(profile, monkeypatch):
    module, build, _, members = PROFILES[profile]
    b6_spec = _b6_without_stores(module, build, monkeypatch, profile)
    traced_spec = _spec(module, build, monkeypatch)
    b6, traced = b6_spec.case._time, traced_spec.case._time
    assert not b6._histories
    # The elided b6 Program is the traced one before its assessment and memory stores.
    tail = [value for value in traced._values if value.op in FINITE_OPS + ("store_history",)]
    tail_ids = {value.id for value in tail}
    assert [(v.id, v.op, v.vtype) for v in b6._values] == [
        (v.id, v.op, v.vtype) for v in traced._values if v.id not in tail_ids]
    # The member names stay: they are the rings that the set-aside checks by name.
    new = {value.id: value.id for value in traced._values if value.id not in tail_ids}
    b6_lines = _stencil_capacity(_named(_source(b6_spec), _presentation_names(
        b6, new)).splitlines(), b6)
    traced_lines = _stencil_capacity(_named(_source(traced_spec), _presentation_names(
        traced, new)).splitlines(), traced)
    # Lot 2's finite() and the memory stores, by their source-map spans and inventory rows.
    traced_lines, dropped = _spans(traced_lines, tail_ids)
    assert sum("ctx.store_history(" in line for line in dropped) == len(members)
    assert any('"assessment", "Advance assessment \'finite\' rejected the candidate"' in line
               for line in dropped)
    (row,) = [i for i, line in enumerate(traced_lines) if line.startswith(INVENTORY)]
    data = json.loads(traced_lines[row][len(INVENTORY):])
    assert [entry["op"] for entry in data["nodes"][-len(tail):]] == [v.op for v in tail]
    data["nodes"] = data["nodes"][:-len(tail)]
    traced_lines[row] = INVENTORY + canonical_json(data)
    start = traced_lines.index(CONTROL_LINES[0])
    assert traced_lines[start:start + 3] == CONTROL_LINES
    del traced_lines[start:start + 3]
    traced_lines = _history_lines(traced_lines, members)
    traced_lines = _attempt_reuse(traced_lines, sum(
        not site["retry_can_change"] for site in _sites(profile)))
    b6_lines = _set_aside_requests(b6_lines, profile, traced=False)
    traced_lines = _set_aside_requests(traced_lines, profile, traced=True)
    assert _digests("\n".join(b6_lines)).splitlines() == _digests(
        "\n".join(traced_lines)).splitlines()


# --- classification (T38) ----------------------------------------------------------------------

@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_the_stage_zero_request_is_a_fixed_site(profile, monkeypatch):
    module, build, _, _ = PROFILES[profile]
    elaborated = _spec(module, build, monkeypatch).case._time
    control = elaborated._advance_control
    outcomes = [dict(value.attrs["trial_site"]) for value in elaborated._values
                if value.op == "solve_outcome"]
    assert outcomes == [{"site": site["site"], "retry_can_change": site["retry_can_change"]}
                        for site in _sites(profile)]
    solves = [row for row in control.to_data()["trial_failures"] if row["guard"] == "solve"]
    assert [(row["site"], row["retry_can_change"], row["variant"], row["classified"])
            for row in solves] == [(site["site"], site["retry_can_change"],
                                    ["BudgetExhausted", "SolveFailed"], True)
                                   for site in _sites(profile)]
    guards = [row for row in control.to_data()["trial_failures"] if row["guard"] != "solve"]
    assert guards and all(row["guard"] == "field_evaluation" and row["variant"] == []
                          and not row["classified"]
                          and row["lot"] == "N8 field_evaluation guards" for row in guards)
    assert AdvanceControl.from_data(control.to_data()) == control


def test_the_memory_rows_carry_type_identities_and_no_reader(monkeypatch):
    elaborated = _spec(early, early.build_m10, monkeypatch).case._time
    rows = elaborated._advance_control.to_data()["memory"]
    assert [(row["path"], row["role"], row["writer"], row["readers"], row["ring"])
            for row in rows] == [
        ("%s/stage1_state" % METHOD, "auxiliary", METHOD, [], "stage1_state"),
        ("%s/stage1_potential" % METHOD, "auxiliary", METHOD, [], "stage1_potential")]
    state, potential = (row["type"] for row in rows)
    assert state["kind"] == "state" and state["space"]["name"] == "density"
    assert potential["kind"] == "field_observation"
    assert potential["authority"]["unknown"]["local_id"] == "solved_potential"
    # M10 declares the unknown's space; the row carries it, and never a node id.
    assert potential["space"]["name"] == "potential"
    assert "id" not in json.dumps(rows).replace('"local_id"', "").replace('"qualified_id"', "")\
        .replace("_id", "")


# --- the surface of the lot: the cases of M08 and M10 ------------------------------------------------

def _parts(monkeypatch, build=early.build_m08):
    """The case of a live-field profile (M08 by default) and its pieces, before any Program is
    elaborated."""
    seen = {}
    potential = early._potential_memory

    def capture(case, block, state, flow, dt, *, name, memory, record, b6):
        seen.update(case=case, block=block, state=state, flow=flow, dt=dt, memory=memory,
                    record=record)

    def memory(field, phi):
        seen.update(field=field, phi=phi)
        return potential(field, phi)
    with monkeypatch.context() as patch:
        patch.setattr(early, "_traced_recorded_ssprk2", capture)
        patch.setattr(early, "_potential_memory", memory)
        build()
    return SimpleNamespace(**seen)


def _elaborate(parts, body, flow=None, outputs=()):
    return elaborate(Advance(body(flow=flow or parts.flow)), case=parts.case,
                     state=parts.block[parts.state], proposal=Proposal(parts.dt), name="probe",
                     outputs=outputs)


def _ops(program, op):
    return [value for value in program._values if value.op == op]


def test_requests_with_one_key_share_one_solve(monkeypatch):
    parts = _parts(monkeypatch, early.build_m10)
    phi = parts.field[parts.phi]

    def flow(state):
        request = parts.field.at(state)
        assert parts.field.at(state) is request and request[phi] is request[phi]
        return parts.flow(state)

    @method
    def one_stage(B, p, flow):
        return Candidate(B.state + p.dt*flow(B.state, B.time), B.time + p.dt)
    elaboration = _elaborate(parts, one_stage, flow)
    assert len(_ops(elaboration.program, "solve_linear")) == 1
    assert len(_ops(elaboration.program, "field_component")) == 1


def test_one_state_read_at_two_times_is_solved_once(monkeypatch):
    """b6 ties an observation to the stage value that its solve read: the copy of B.state at
    the second stage is another stage value, so the IR keeps one request and one solve outcome
    per stage. The request is time-independent: the emission solves it once and aliases the
    second (lot « reuse across stages », ``test_reuse_across_stages_n8``)."""
    parts = _parts(monkeypatch, early.build_m10)

    @method
    def twice(B, p, flow):
        k0 = flow(B.state, B.time)
        k1 = flow(B.state, B.time + p.dt)
        return Candidate(B.state + .5*p.dt*(k0 + k1), B.time + p.dt)
    elaboration = _elaborate(parts, twice)
    assert len(_ops(elaboration.program, "solve_linear")) == 2
    assert len(_ops(elaboration.program, "solve_outcome")) == 2
    assert [(request.site, request.fixed) for request in elaboration.trace.requests] == [
        ("twice/flow#0/live_Poisson", True), ("twice/flow#1/live_Poisson", True)]

    def route(case, block, state, flow, dt, *, name, memory, record, b6):
        return elaborate(Advance(twice(flow=flow)), case=case, state=block[state],
                         proposal=Proposal(dt), name=name)
    source = _source(_spec(early, early.build_m10, monkeypatch, route))
    assert source.count("ctx.solve_prepared_linear(") == 1
    assert [line.strip() for line in source.splitlines()
            if "field_counters_0)[1]" in line and "++" in line][-1] == "++(*field_counters_0)[1];"

def test_two_keys_give_two_solves_and_a_body_read_reuses_the_component(monkeypatch):
    spec = _spec(early, early.build_m08, monkeypatch)
    program = spec.case._time
    assert len(_ops(program, "solve_linear")) == 2
    # The record's request at U1 adds no node: the potential store reads the component's read.
    (store,) = [value for value in program._values if value.op == "store_history"
                and value.name == "stage1_potential"]
    (read,) = store.inputs
    readers = [value for value in program._values
               if read.id in [source.id for source in value.inputs]]
    assert read.op == "field_component" and [value.op for value in readers] == [
        "field_gradient", "store_history"]


def test_a_body_request_is_named_by_the_body(monkeypatch):
    parts = _parts(monkeypatch)
    phi = parts.field[parts.phi]

    @method(memory=Product(start=phi), role=Auxiliary())
    def early_request(B, p, flow):
        start = parts.field.at(B.state)[phi]
        k0 = flow(B.state, B.time)  # reuses the body's request: same field, same state
        return Candidate(B.state + p.dt*k0, B.time + p.dt, memory=Product(start=start))
    elaboration = _elaborate(parts, early_request)
    assert len(_ops(elaboration.program, "solve_linear")) == 1
    (request,) = elaboration.trace.requests
    assert request.site == "early_request/body#0/live_Poisson" and request.fixed
    rows = elaboration.control.to_data()["trial_failures"]
    assert [row["site"] for row in rows if row["guard"] == "solve"] == [
        "early_request/body#0/live_Poisson/solve"]


def test_two_handles_of_one_registered_field_share_the_key(monkeypatch):
    parts = _parts(monkeypatch)
    registry = parts.field._field_registry
    twin = FieldHandle(parts.field.local_id, owner=registry.owner_path, field_registry=registry)
    assert twin is not parts.field and twin == parts.field

    def flow(state):
        first = parts.field.at(state)
        assert twin.at(state) is first
        return parts.flow(state)

    @method
    def one_stage(B, p, flow):
        return Candidate(B.state + p.dt*flow(B.state, B.time), B.time + p.dt)
    program = _elaborate(parts, one_stage, flow).program
    assert len(_ops(program, "solve_linear")) == 1


# --- refusals ----------------------------------------------------------------------------------

def _forward_euler(parts):
    @method
    def forward_euler(B, p, flow):
        return Candidate(B.state + p.dt*flow(B.state, B.time), B.time + p.dt)
    return forward_euler


def test_a_request_outside_a_method_is_refused(monkeypatch):
    parts = _parts(monkeypatch)
    with pytest.raises(ClosedBodyViolation, match=r"outside a method.*\(T6\)"):
        parts.field.at(parts.block[parts.state])


def test_a_request_in_an_output_body_is_not_elaborated(monkeypatch):
    parts = _parts(monkeypatch)
    phi = parts.field[parts.phi]
    output = Output(lambda B, interval: Product(u=parts.field.at(B.state)[phi]))
    with pytest.raises(NotYetElaborated, match="in an Output body"):
        _elaborate(parts, _forward_euler(parts), outputs=(output,))


def test_a_request_at_an_unmaterialized_state_is_not_elaborated(monkeypatch):
    parts = _parts(monkeypatch)

    @method
    def ahead(B, p, flow):
        k0 = flow(B.state, B.time)
        parts.field.at(B.state + p.dt*k0)
        return Candidate(B.state + p.dt*k0, B.time + p.dt)
    with pytest.raises(NotYetElaborated, match="no evaluation materialized"):
        _elaborate(parts, ahead)


def test_a_request_at_a_value_of_another_stage_is_refused(monkeypatch):
    parts = _parts(monkeypatch)
    given = []

    def flow(state):
        given.append(state)
        parts.field.at(given[0])
        return parts.flow(state)

    @method
    def two_stages(B, p, flow):
        k0 = flow(B.state, B.time)
        k1 = flow(B.state + p.dt*k0, B.time + p.dt)
        return Candidate(B.state + .5*p.dt*(k0 + k1), B.time + p.dt)
    with pytest.raises(ClosedBodyViolation, match="value it was not given"):
        _elaborate(parts, two_stages, flow)


def test_a_request_at_a_value_the_component_computed_is_not_elaborated(monkeypatch):
    parts = _parts(monkeypatch)

    def flow(state):
        rate = parts.flow(state)
        parts.field.at(rate)
        return rate
    with pytest.raises(NotYetElaborated, match="at a state that it computed"):
        _elaborate(parts, _forward_euler(parts), flow)


def test_a_request_at_a_state_of_another_trace_is_refused(monkeypatch):
    parts = _parts(monkeypatch)
    kept = []

    @method
    def keep(B, p, flow):
        kept.append(B.state)
        return Candidate(B.state + p.dt*flow(B.state, B.time), B.time + p.dt)
    _elaborate(parts, keep)
    other = _parts(monkeypatch)

    @method
    def foreign(B, p, flow):
        other.field.at(kept[0])
        return Candidate(B.state + p.dt*flow(B.state, B.time), B.time + p.dt)
    with pytest.raises(ClosedBodyViolation, match="another method trace"):
        _elaborate(other, foreign)


@pytest.mark.parametrize("dependencies", [(), "twice"])
def test_a_problem_that_reads_no_state_or_another_is_not_elaborated(dependencies, monkeypatch):
    parts = _parts(monkeypatch)
    real = FieldProblem.dependencies

    def patched(problem):
        found = real(problem)
        return () if dependencies == () else tuple(found) * 2
    monkeypatch.setattr(FieldProblem, "dependencies", patched)
    with pytest.raises(NotYetElaborated, match="dependency set is its field and its state"):
        _elaborate(parts, _forward_euler(parts))


def test_a_nonlinear_route_is_not_elaborated(monkeypatch):
    parts = _parts(monkeypatch)
    monkeypatch.setattr(field_requests, "_LINEAR_METHODS", ())
    with pytest.raises(NotYetElaborated, match="general linear field route"):
        _elaborate(parts, _forward_euler(parts))


def test_a_manual_solve_in_a_component_is_refused(monkeypatch):
    parts = _parts(monkeypatch)
    phi = parts.field[parts.phi]

    def flow(state):
        solution = state.prog.solve(parts.field, values={parts.block[parts.state]: state},
                                    at=state.point).consume(action=FailRun())
        parts.field.observe(solution)[phi]
        return parts.flow(state)
    with pytest.raises(ClosedBodyViolation, match=r"requests a field with field.at\(U\) \(T6\)"):
        _elaborate(parts, _forward_euler(parts), flow)


@pytest.mark.parametrize("record, error, match", [
    (lambda parts: lambda U: Product(stage1_state=U), TypeError, "declares the memory"),
    (lambda parts: lambda U: Product(stage1_potential=parts.field.at(U)[parts.field[parts.phi]],
                                     stage1_state=U), TypeError, "declares the memory"),
    (lambda parts: lambda U: Product(stage1_state=parts.field.at(U)[parts.field[parts.phi]],
                                     stage1_potential=parts.field.at(U)[parts.field[parts.phi]]),
     TypeError, "is a State"),
    (lambda parts: lambda U: Product(stage1_state=U, stage1_potential=U), ClosedBodyViolation,
     "not a read of a field request"),
    (lambda parts: lambda U: Product(stage1_state=.5*U + .5*U,
                                     stage1_potential=parts.field.at(U)[parts.field[parts.phi]]),
     NotYetElaborated, "no evaluation materialized"),
])
def test_a_memory_that_is_not_its_declaration_is_refused(record, error, match, monkeypatch):
    parts = _parts(monkeypatch)
    with pytest.raises(error, match=match):
        _elaborate(parts, early.recorded_ssprk2(parts.memory, record(parts)))


def test_a_memory_member_of_another_trace_is_refused(monkeypatch):
    parts = _parts(monkeypatch)
    kept = []

    @method
    def keep(B, p, flow):
        kept.append(B.state)
        return Candidate(B.state + p.dt*flow(B.state, B.time), B.time + p.dt)
    _elaborate(parts, keep)
    other = _parts(monkeypatch)
    phi = other.field[other.phi]

    def record(U):
        return Product(stage1_state=kept[0], stage1_potential=other.field.at(U)[phi])
    with pytest.raises(ClosedBodyViolation, match="another method trace"):
        _elaborate(other, early.recorded_ssprk2(other.memory, record))


def test_a_field_member_observing_another_unknown_type_is_refused(monkeypatch):
    with pytest.raises(TypeError, match="the qualified field unknown"):
        declared_memory(Product(stage1_state=State, stage1_potential=float), Auxiliary())


@pytest.mark.parametrize("role, match", [
    (None, r"Scientific memory .*gap C51"),
    (object(), r"role Auxiliary\(\) of a method memory only"),
])
def test_roles_other_than_auxiliary_are_not_elaborated(role, match):
    with pytest.raises(NotYetElaborated, match=match):
        method(memory=Product(stage1_state=State), role=role)(lambda B, p: None)


def test_the_arguments_of_auxiliary_are_not_elaborated():
    with pytest.raises(NotYetElaborated, match="gap C46"):
        Auxiliary(persisted=True)


def test_memory_policies_and_reads_stay_refused(monkeypatch):
    with pytest.raises(NotYetElaborated, match="policies"):
        method(memory=Product(stage1_state=State), role=Auxiliary(),
               policies={"stage1_state": "keep"})(lambda B, p: None)
    parts = _parts(monkeypatch)

    @method(memory=Product(stage1_state=State), role=Auxiliary())
    def reads(B, p, flow):
        assert B.memory is None
        return Candidate(B.state + p.dt*flow(B.state, B.time), B.time + p.dt)
    with pytest.raises(NotYetElaborated):
        _elaborate(parts, reads)


def test_a_ring_that_the_program_holds_is_refused(monkeypatch):
    parts = _parts(monkeypatch)
    from pops.time._program.api import Program
    init = Program.__init__

    def holding(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self._histories["stage1_state"] = None
    monkeypatch.setattr(Program, "__init__", holding)
    with pytest.raises(ValueError, match="already holds"):
        _elaborate(parts, early.recorded_ssprk2(parts.memory, parts.record))


def test_the_amr_route_refuses_method_memory(monkeypatch):
    program = _spec(early, early.build_m08, monkeypatch).case._time
    assert _refuse_advance_memory_route(program, "system") == ""
    with pytest.raises(NotYetElaborated, match="method memory of an elaborated Advance runs on "
                                               "the Uniform route"):
        _refuse_advance_memory_route(program, "amr_system")
