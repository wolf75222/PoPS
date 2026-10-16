"""Explicit, compiled numerical admission; this is not a stability theorem."""
from collections.abc import Mapping


def depends_on_candidate(condition, candidate):
    """Follow live SSA data edges and region exits, excluding support-only edges."""
    from pops.time.values import ProgramValue
    pending, seen = [condition], set()
    def leaves(item):
        if isinstance(item, ProgramValue):
            yield item
        elif isinstance(item, Mapping):
            for child in item.values():
                yield from leaves(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                yield from leaves(child)
    while pending:
        node = pending.pop()
        if node is candidate:
            return True
        if node.id in seen:
            continue
        seen.add(node.id)
        support = node.attrs.get("support_only_inputs", ())
        pending.extend(child for index, child in enumerate(node.inputs) if index not in support)
        # Region result identities are data uses; their dead body operations are not.
        for key in ("body", "true_result", "false_result", "result"):
            pending.extend(leaves(node.attrs.get(key)))
    return False


def require_candidate_dependency(condition, candidate):
    if not depends_on_candidate(condition, candidate):
        raise ValueError("numerical stability guard must depend on its exact candidate")


def is_stability_admission_guard(program, value):
    if value.op != "acceptance_guard":
        return False
    from pops.time._step.transaction import GuardRole
    from pops.time.solve_outcome import RejectAttempt, FailRun
    name = value.attrs.get("guard")
    records = [record for record in getattr(program, "_acceptance_guards", ()) if record.name == name]
    declared = len(records) == 1 and records[0].role is GuardRole.NUMERICAL_STABILITY
    tagged = value.attrs.get("numerical_stability") == "pops.authored-stability-admission@1"
    if declared != tagged:
        raise ValueError("numerical stability guard lost its declared admission authority")
    if not declared:
        return False
    if (len(value.inputs) != 2 or value.inputs[0].vtype != "state"
            or value.inputs[1].vtype != "bool"
            or not isinstance(value.attrs.get("action"), (RejectAttempt, FailRun))
            or records[0].action.to_data() != value.attrs["action"].to_data()):
        raise ValueError("numerical stability guard has an invalid candidate or failure action")
    require_candidate_dependency(value.inputs[1], value.inputs[0])
    return True
