"""Detached inspection payloads for Problem."""
from __future__ import annotations

from typing import Any

from pops.problem._registry_freeze import inspection_copy


def inspect_payload(problem: Any) -> dict[str, Any]:
    info = {
        "name": problem._name,
        "category": problem.category,
        "native_id": problem.native_id,
        "options": problem.options(),
        "requirements": problem.requirements().to_dict(),
        "capabilities": problem.capabilities().to_dict(),
    }
    info["layout"] = problem._layout.inspect() if problem._layout is not None else None
    info["blocks"] = problem._block_registry.inspect()
    info["fields"] = problem._field_registry.inspect(problem.resolve)
    info["params"] = problem._param_registry.inspect()
    runtime = problem._runtime_registry.inspect()
    info["aux"] = runtime["aux"]
    info["outputs"] = runtime["outputs"]
    info["diagnostics"] = runtime["diagnostics"]
    info["schedules"] = runtime["schedules"]
    info["constraints"] = problem._constraint_registry.inspect()
    info["numerics"] = {
        block: plan.inspect() for block, plan in sorted(problem._numerics_assignments.items())
    }
    info["time"] = problem._time_registry.inspect()["program"]
    return inspection_copy(info)


def serialization_payload(problem: Any) -> dict[str, Any]:
    info = inspect_payload(problem)
    info["handles"] = {
        "blocks": [
            problem.resolve(handle).canonical_identity()
            for handle in problem.blocks().values()
        ],
        "fields": [
            problem.resolve(handle).canonical_identity()
            for handle in problem.fields().values()
        ],
    }
    return info


__all__ = ["inspect_payload", "serialization_payload"]
