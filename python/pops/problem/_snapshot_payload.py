"""Raw structural Problem payload used exclusively by :mod:`pops.problem._snapshot`.

The public ``Problem.to_dict()`` is an inspection view and intentionally shortens several objects
to display names.  A compile-cache identity cannot use that lossy view: two models may share a name
while carrying different coefficients.  This module reads the typed registries and returns detached
container shells whose leaves remain the real descriptors; ``ProblemSnapshot`` then applies its
strict structural protocol to every leaf.
"""
from __future__ import annotations

from typing import Any


def problem_snapshot_payload(problem: Any) -> dict[str, Any]:
    """Return all cache-relevant Problem declarations without display-name reduction."""
    blocks = {
        name: {
            "model": spec["model"],
            "spatial": spec["spatial"],
            "time": _time_snapshot_value(spec["time"]),
            "diagnostics": spec["diagnostics"],
        }
        for name, spec in problem._block_registry.items()
    }
    fields = {
        name: descriptor
        for name, descriptor in problem._field_registry.items()
    }
    params = {
        name: dict(declaration)
        for name, declaration in problem._param_registry.items()
    }
    runtime = problem._runtime_registry
    return {
        "name": problem.name,
        "category": problem.category,
        "native_id": problem.native_id,
        "owner_path": problem.owner_path,
        "options": problem.options(),
        "layout": problem.layout,
        "blocks": blocks,
        "fields": fields,
        "params": params,
        "param_declarations": problem._param_registry.declarations(),
        "aux": runtime.aux,
        "outputs": runtime.outputs,
        "diagnostics": runtime.diagnostics,
        "runtime_policies": runtime.policies,
        "constraints": problem._constraint_registry.refinement,
        "time": _time_snapshot_value(problem._time_registry.program),
        "handles": {
            "blocks": [handle.canonical_identity() for handle in problem.blocks().values()],
            "fields": [handle.canonical_identity() for handle in problem.fields().values()],
        },
    }


def _time_snapshot_value(program: Any) -> Any:
    """Use the Program's structural report (including its IR hash), or the value itself."""
    if program is None:
        return None
    inspect = getattr(program, "inspect", None)
    return inspect() if callable(inspect) else program


__all__ = ["problem_snapshot_payload"]
