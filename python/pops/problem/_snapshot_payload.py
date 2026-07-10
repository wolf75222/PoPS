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
            "handle": problem._block_registry.canonical_block(
                problem._block_registry.handle(name)),
            "model_owner_path": spec["model"].owner_path.canonical(),
            "model": _model_snapshot_value(spec["model"]),
            "spatial": spec["spatial"],
            "time": _time_snapshot_value(spec["time"]),
            "diagnostics": spec["diagnostics"],
        }
        for name, spec in problem._block_registry.items()
    }
    fields = dict(problem._field_registry.resolved_items(problem.resolve))
    params = {
        name: dict(declaration)
        for name, declaration in problem._param_registry.items()
    }
    runtime = problem._runtime_registry
    return {
        "name": problem.name,
        "category": problem.category,
        "native_id": problem.native_id,
        "owner_path": problem.owner_path.canonical(),
        "options": problem.options(),
        "layout": problem.layout,
        "blocks": blocks,
        "fields": fields,
        "params": params,
        "param_declarations": problem._param_registry.declarations(),
        "aux": runtime.aux,
        "outputs": runtime.outputs,
        "diagnostics": runtime.diagnostics,
        "schedules": runtime.schedules,
        "constraints": problem._constraint_registry.refinement,
        "time": _time_snapshot_value(problem._time_registry.program),
        "handles": {
            "blocks": [problem._block_registry.canonical_block(handle)
                       for handle in problem.blocks().values()],
            "fields": [problem._field_registry.canonicalize(handle)
                       for handle in problem.fields().values()],
        },
    }


def _time_snapshot_value(program: Any) -> Any:
    """Use the Program's structural report (including its IR hash), or the value itself."""
    if program is None:
        return None
    inspect = getattr(program, "inspect", None)
    return inspect() if callable(inspect) else program


def _model_snapshot_value(model: Any) -> Any:
    """Project a model definition through its canonical manifest authority.

    Blackboard and PDE facades contain live authoring handles in their internal family registries;
    traversing those private dicts would either leak authority tokens or make a reused definition
    look ambiguous between its instances.  Their operator-first Module manifest is the canonical
    scientific-definition boundary.  A model without that protocol remains structural so an
    explicit foreign descriptor can still explain why snapshotting is unsupported.
    """
    manifest_owner = model
    manifest = getattr(manifest_owner, "manifest", None)
    if not callable(manifest):
        module = getattr(model, "module", None)
        if module is not None:
            manifest_owner = module
            manifest = getattr(module, "manifest", None)
    if not callable(manifest):
        return model

    built = manifest()
    to_dict = getattr(built, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("model manifest() must return an object exposing to_dict()")
    result = {
        "type": "%s.%s" % (type(model).__module__, type(model).__qualname__),
        "manifest": to_dict(),
    }
    scientific_hash = _scientific_model_hash(model, manifest_owner)
    if scientific_hash is not None:
        result["scientific_hash"] = scientific_hash
    return result


def _scientific_model_hash(model: Any, manifest_owner: Any) -> Any:
    for candidate, name in (
        (model, "module_hash"),
        (model, "_model_hash"),
        (getattr(model, "_dsl", None), "_model_hash"),
        (manifest_owner, "module_hash"),
    ):
        method = getattr(candidate, name, None) if candidate is not None else None
        if callable(method):
            value = method()
            if not isinstance(value, str) or not value:
                raise TypeError("%s() must return a non-empty string" % name)
            return value
    return None


__all__ = ["problem_snapshot_payload"]
