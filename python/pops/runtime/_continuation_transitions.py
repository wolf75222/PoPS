"""Derived continuation obligations and committed lifecycle receipts.

The resolved Program, field providers and layout own the choices. This table is a checked
projection of those authorities, never a second user-authored policy registry.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from pops.identity import make_identity

_EVENTS = ("initialization", "restart", "projection", "regrid", "rollback")
_ACTIONS = frozenset(("preserve", "transfer", "reconstruct", "solve", "invalidate"))


@dataclass(frozen=True, slots=True)
class ContinuationTransitionPlan:
    _json: str

    def __post_init__(self):
        data = json.loads(self._json)
        if set(data) != {"schema_version", "kind", "target", "evidence_stage", "objects"} \
                or data["schema_version"] != 1 or data["kind"] != "pops.continuation-transitions" \
                or data["target"] not in ("system", "amr_system"):
            raise ValueError("invalid resolved continuation policy schema")
        if len({row["identity"] for row in data["objects"]}) != len(data["objects"]):
            raise ValueError("duplicate retained object continuation policy")
        for event in _EVENTS:
            self.require(event)

    def to_data(self) -> dict[str, Any]:
        return json.loads(self._json)

    @property
    def identity(self):
        return make_identity("continuation-transition-plan", self.to_data())

    def require(self, event: str) -> tuple[dict[str, Any], ...]:
        if event not in _EVENTS:
            raise ValueError("unknown continuation transition %r" % event)
        rows = self.to_data()["objects"]
        for row in rows:
            if set(row["transitions"]) != set(_EVENTS):
                raise ValueError("required transition policy omitted for %s" % row["identity"])
            selected = row["transitions"][event]
            if selected["action"] not in _ACTIONS or not selected["authority"]:
                raise ValueError("unsupported continuation action for %s" % row["identity"])
        return tuple(rows)


def derive_continuation_transitions(plan: Any) -> ContinuationTransitionPlan:
    rows = []

    def add(kind, identity, name, actions, authority, validity):
        rows.append({
            "kind": kind, "identity": identity, "name": name, "validity": validity,
            "transitions": {event: {"action": action, "authority": authority[event]}
                            for event, action in zip(_EVENTS, actions, strict=True)},
        })

    def providers(initial, restart, topology, rollback="native.accepted_snapshot"):
        return dict(zip(_EVENTS, (initial, restart, topology, topology, rollback), strict=True))

    for block in plan.blocks:
        for state in block.state_identities:
            add("state", state, block.name, ("transfer", "preserve", "transfer", "transfer", "preserve"),
                providers("resolved.initial_condition_plan", "native.checkpoint_state",
                          "resolved.amr_transfer"), "accepted continuation until replacement")
        operations = block.resolved_operations
        evidence = {} if operations is None else operations.to_data().get("provider_evidence", {})
        for component in evidence.get("auxiliary", {}).get("entries", ()):
            producer = component["provider"]["producer"]
            if producer == "field_output":
                continue  # The solved field observation owns this same provider storage.
            identity = make_identity("retained-auxiliary-component", component["key"]).token
            add("auxiliary", identity, block.name,
                ("transfer" if producer == "runtime_input" else "invalidate",
                 "preserve", "invalidate", "invalidate", "preserve"),
                providers("resolved.auxiliary_provider", "native.restore_auxiliary_checkpoint_accepted_state",
                          "native.invalidate_auxiliary_after_topology_regrid"),
                component["key"])
    fields = dict(plan.field_plans)
    seen_fields = set()
    for name, field in fields.items():
        native_options = field.native_install_data()
        identity = native_options["provider_slot"]
        if identity in seen_fields:
            continue
        seen_fields.add(identity)
        bootstrap = getattr(plan, "bootstrap_plan", None)
        initial_action = "solve" if any(
            action.operation == "recompute" and action.evidence.get("field_name") == name
            for action in (() if bootstrap is None else bootstrap.actions)) else "invalidate"
        for kind in ("field_value", "field_observation"):
            add(kind, kind + ":" + identity, identity,
                (initial_action, "preserve", "solve", "solve", "preserve"),
                providers("resolved.field_provider", "native.checkpoint_field_provider",
                          "native.rematerialize_fields_after_topology_change"),
                "exact solved dependency, observation and topology identities")
        add("solver_cache", "solver_cache:" + identity, identity,
            ("invalidate", "invalidate", "invalidate", "invalidate", "preserve"),
            providers("native.field_solver", "native.field_solver", "native.field_solver"),
            "reusable only under the exact discrete operator identity")
    temporal = plan.time.temporal_manifest()
    for history in temporal["histories"]:
        name = history["name"]
        add("history", "history:" + name, name,
            ("invalidate", "reconstruct", "reconstruct", "reconstruct", "preserve"),
            providers("program.declared_first_history_store", "native.restore_history_provenance",
                      "native.prepare_regridded_program_histories"), history["validity"])
        if history["interpolation"].get("kind") == "linear":
            add("dense_output", "dense_output:" + name, name,
                ("invalidate", "reconstruct", "reconstruct", "reconstruct", "preserve"),
                providers("program.declared_first_history_store", "native.restore_history_provenance",
                          "native.interpolate_history_linear"), history["validity"])
    add("controller", "controller:accepted", "accepted controller and queued events",
        ("reconstruct", "preserve", "preserve", "preserve", "preserve"),
        providers("TemporalRestartState.configure_program", "TemporalRestartState.from_checkpoint",
                  "TemporalRestartState.accepted_boundary"), "next accepted attempt on the declared clocks")
    add("accepted_exchanges", "accepted_exchanges:program", "accepted Program exchange mailbox",
        ("invalidate", "preserve", "preserve", "preserve", "preserve"),
        providers("native.begin_step_transaction", "native.restore_checkpoint_program_exchanges",
                  "native.accepted_snapshot"), "last accepted window; cleared at next enclosing attempt")
    for schedule in temporal["schedules"]:
        if schedule.get("cache_required") is True:
            if plan.target == "amr_system":
                raise NotImplementedError(
                    "checkpointed AMR scheduler cache provider is unavailable for continuation")
            name = str(schedule["node_id"])
            add("scheduler_cache", "scheduler_cache:" + name, name,
                ("invalidate", "preserve", "invalidate", "invalidate", "preserve"),
                providers("native.register_cache", "native.restore_program_cache", "native.cache_invalidation"),
                "declared cache window and exact dependency identity")
    payload = {"schema_version": 1, "kind": "pops.continuation-transitions", "target": plan.target,
               "evidence_stage": "resolved", "objects": sorted(rows, key=lambda row: row["identity"])}
    if len({row["identity"] for row in rows}) != len(rows):
        raise ValueError("continuation obligations contain duplicate retained objects")
    return ContinuationTransitionPlan(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def require_resolved_continuation(plan):
    actual = getattr(plan, "continuation_transitions", None)
    if type(actual) is not ContinuationTransitionPlan:
        raise ValueError("resolved plan omits required continuation transition policies")
    if hasattr(plan, "time"):
        expected = derive_continuation_transitions(plan)
        if actual.to_data() != expected.to_data():
            raise ValueError("continuation policies differ from resolved retained-object authorities")
    else:
        # CompiledPlanRecord authenticates the detached immutable projection in its own identity.
        plan.verify()
    return actual


def prepare_bind_continuation(owner, install_plan, *, program=None, block_names=None, field_names=None):
    plan = require_resolved_continuation(install_plan.artifact.plan)
    if block_names is not None:
        allowed_blocks = set(block_names)
        temporal = program.temporal_manifest()
        histories = {row["name"] for row in temporal["histories"]}
        schedules = {str(row["node_id"]) for row in temporal["schedules"]}
        fields = set(field_names or ())
        source_fields = dict(install_plan.artifact.plan.field_plans)
        fields.update(field.native_install_data()["provider_slot"] for name, field in source_fields.items()
                      if name in fields)
        data = plan.to_data()
        data["objects"] = [row for row in data["objects"] if (
            row["name"] in allowed_blocks if row["kind"] in ("state", "auxiliary") else
            row["name"] in histories if row["kind"] in ("history", "dense_output") else
            row["name"] in fields if row["kind"] in ("field_value", "field_observation", "solver_cache") else
            row["name"] in schedules if row["kind"] == "scheduler_cache" else True)]
        plan = ContinuationTransitionPlan(json.dumps(data, sort_keys=True, separators=(",", ":")))
    plan.require("initialization")
    owner._continuation_transition_plan = plan


def prepare_receipt(owner, event, *, evidence=None):
    plan = getattr(owner, "_continuation_transition_plan", None)
    if type(plan) is not ContinuationTransitionPlan:
        raise RuntimeError("lifecycle transition lacks its resolved continuation obligations")
    rows = plan.require(event)
    return {"schema_version": 1, "transition": event, "plan_identity": plan.identity.token,
            "objects": [{"identity": row["identity"], "kind": row["kind"],
                         "action": row["transitions"][event]["action"],
                         "authority": row["transitions"][event]["authority"],
                         "validity": row["validity"]} for row in rows],
            "evidence": dict(evidence or {})}


def completed_initialization_receipt(owner, snapshot):
    receipt = prepare_receipt(owner, "initialization", evidence={"bind_identity": snapshot.bind_identity.token})
    epoch = getattr(owner._s, "checkpoint_topology_epoch", None)
    if callable(epoch):
        receipt["evidence"]["topology_epoch"] = int(epoch())
    bootstrap = getattr(owner, "_bootstrap_execution", None)
    if bootstrap is not None:
        receipt["evidence"]["bootstrap_execution"] = bootstrap.to_data()
    if any(row["action"] == "solve" for row in receipt["objects"]):
        if bootstrap is None or not any(row.evidence.get("operation") == "recompute"
                                        for row in bootstrap.receipts):
            raise RuntimeError("initial continuation solve lacks its consumed bootstrap receipt")
    return receipt

def completed_restart_receipt(owner):
    receipt = prepare_receipt(owner, "restart", evidence={
        "restart_identity": owner._last_restart_identity.token,
        "accepted_time": float(owner._s.time()).hex(), "macro_step": int(owner._s.macro_step())})
    replay = getattr(owner, "_last_restart_report", None)
    replay_rows = {} if replay is None else {row["name"]: row for row in replay.histories}
    for row in receipt["objects"]:
        if row["kind"] in ("history", "dense_output"):
            name = row["identity"].split(":", 1)[1]
            restored = replay_rows.get(name)
            if restored is None:
                raise RuntimeError("restart receipt lacks required history evidence for %s" % name)
            row["action"] = "reconstruct" if restored["recomputed_slots"] else "preserve"
            row["restored_slots"] = restored["stored_slots"] + restored["recomputed_slots"]
    epoch = getattr(owner._s, "checkpoint_topology_epoch", None)
    if callable(epoch):
        receipt["evidence"]["topology_epoch"] = int(epoch())
    regrid = getattr(owner, "_last_restart_regrid_receipt", None)
    if regrid is not None and regrid["changed"]:
        receipt["evidence"]["regrid_on_restart"] = regrid
        receipt = _with_native_outcomes(owner, receipt, owner._s._continuation_transition_rows())
    return receipt


def committed_continuation_report(owner):
    """Combine only committed native topology decisions with the existing lifecycle receipt."""
    import copy
    if "_checkpoint_restart_python_snapshot" in owner.__dict__:
        raise RuntimeError("continuation receipt is unavailable during a provisional restart")
    depth = getattr(owner._s, "_step_transaction_depth", None)
    if callable(depth) and int(depth()) != 0:
        raise RuntimeError("continuation receipt is unavailable during a provisional attempt")
    prior = copy.deepcopy(getattr(owner, "_last_continuation_transition_report", None))
    native = getattr(owner._s, "_continuation_transition_rows", None)
    rows = () if not callable(native) else native()
    if not rows:
        return prior
    epoch = int(rows[0][5])
    generation = int(rows[0][6])
    # A just-committed restart receipt is the current event; native topology receipts describe
    # only publications performed since that receipt (including RegridOnRestart).
    if prior is not None:
        restored_epoch = prior["evidence"].get("topology_epoch")
        if restored_epoch is not None and epoch <= restored_epoch:
            return prior
    receipt = prepare_receipt(owner, "regrid", evidence={
        "topology_epoch": epoch, "materialization_generation": generation})
    return _with_native_outcomes(owner, receipt, rows)


def _with_native_outcomes(owner, receipt, rows):
    if not rows:
        raise RuntimeError("topology transition lacks its native continuation receipt")
    epoch, generation = int(rows[0][5]), int(rows[0][6])
    actual = {}
    for kind, name, action, validity, authority, row_epoch, row_generation in rows:
        if int(row_epoch) != epoch or int(row_generation) != generation or action not in _ACTIONS:
            raise RuntimeError("native continuation receipt has a mixed generation or unknown action")
        actual.setdefault((kind, name), []).append({
            "action": action, "validity": validity, "authority": authority})
    definitions = {row["identity"]: row for row in owner._continuation_transition_plan.require("regrid")}
    for row in receipt["objects"]:
        definition = definitions[row["identity"]]
        kind = "history" if row["kind"] == "dense_output" else row["kind"]
        decisions = actual.get((kind, definition["name"]))
        if decisions:
            row["outcomes"] = decisions
            if len({decision["action"] for decision in decisions}) == 1:
                row["action"] = decisions[0]["action"]
            else:
                row.pop("action", None)
        elif row["kind"] in ("state", "auxiliary", "field_value", "field_observation", "solver_cache",
                              "scheduler_cache", "controller", "accepted_exchanges"):
            raise RuntimeError("native regrid receipt omits required object %s" % row["identity"])
        elif row["kind"] in ("history", "dense_output"):
            # Only the directly replaced child levels have remap entries; ancestor rings are retained.
            row["action"] = "preserve"
    return receipt
