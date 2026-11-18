"""Identity and source projection of compiled, already-global admissibility guards (T39)."""
from __future__ import annotations

from typing import Any

from pops._structural_data import canonical_json_sha256
from pops.time._step.transaction import GuardFailureSite, GuardRole
from pops.time.solve_outcome import FailRun, RejectAttempt

SCHEMA = "pops.global-guard-failure@1"
_ACCEPTED_PREDICATE_OPS = frozenset({
    "compare", "reduce", "pointwise_expression", "scalar_isfinite",
    "bool_and", "bool_or", "bool_not", "const_scalar",
})


def _predicate_of_accepted_state(state: Any, predicate: Any) -> bool:
    """Prove the supported scalar/field expression reads only this accepted State."""
    if state.op != "state":
        return False
    reads_state = False
    pending, seen = [predicate], set()
    while pending:
        node = pending.pop()
        if node is state:
            reads_state = True
            continue
        if node.id in seen:
            continue
        seen.add(node.id)
        if node.op not in _ACCEPTED_PREDICATE_OPS or node.point != state.point:
            return False
        pending.extend(node.inputs)
    return reads_state


def guard_failure_site(program: Any, node: Any, program_hash: str) -> GuardFailureSite | None:
    if node.op != "acceptance_guard" or node.region != 0:
        return None
    state, predicate = node.inputs
    action = node.attrs["action"]
    trial = node.attrs.get("trial_site")
    if trial is None:
        if type(action) is not FailRun or "assessment" in node.attrs \
                or "numerical_stability" in node.attrs \
                or not _predicate_of_accepted_state(state, predicate):
            return None
        if not any(record.name == node.attrs["guard"] and record.role is GuardRole.INVARIANT
                   and type(record.action) is FailRun for record in program._acceptance_guards):
            return None
        retry_can_change, disposition = False, "fail_run"
        origin = node.provenance.primary
    else:
        if type(action) is not RejectAttempt or "assessment" in node.attrs \
                or trial["variant"] != "NotAdmissible":
            return None
        retry_can_change, disposition = trial["retry_can_change"], "reject_attempt"
        # A lowered trial guard protects this exact authored value. Its own creation
        # may occur inside resolve/compile after the authoring stack has unwound.
        origin = state.provenance.primary
    if type(retry_can_change) is not bool or not 0 <= predicate.id < 1 << 63:
        raise ValueError("global guard failure site has invalid typed metadata")
    token = canonical_json_sha256({
        "schema": SCHEMA, "program": program_hash,
        "site": node.id, "predicate": predicate.id,
    })
    return GuardFailureSite(token, node.attrs["guard"], predicate.id, disposition,
                            retry_can_change, origin)


def guard_failure_sites(program: Any, program_hash: str) -> tuple[GuardFailureSite, ...]:
    from .temporal_manifest import _walk

    if program is None:
        return ()
    return tuple(site for node in _walk(program._values)
                 if (site := guard_failure_site(program, node, program_hash)) is not None)
