"""Raw structural Problem payload used exclusively by :mod:`pops.problem._snapshot`.

The public ``Problem.to_dict()`` is an inspection view and intentionally shortens several objects
to display names.  A compile-cache identity cannot use that lossy view: two models may share a name
while carrying different coefficients.  This module reads the typed registries and returns detached
container shells whose leaves remain the real descriptors; ``ProblemSnapshot`` then applies its
strict structural protocol to every leaf.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def problem_snapshot_payload(problem: Any) -> dict[str, Any]:
    """Return all cache-relevant Problem declarations without display-name reduction."""
    return _problem_snapshot_payload(problem, artifact=False)


def problem_snapshot_artifact_payload(problem: Any) -> dict[str, Any]:
    """Return the compile-relevant projection of one Problem.

    Parameter declarations are projected through their authoritative ``artifact_data()`` methods.
    This is intentionally a second construction from typed registries, not a recursive deletion of
    keys from :func:`problem_snapshot_payload`.
    """
    return _problem_snapshot_payload(problem, artifact=True)


def _problem_snapshot_payload(problem: Any, *, artifact: bool) -> dict[str, Any]:
    blocks = {
        name: {
            "handle": problem._block_registry.canonical_block(
                problem._block_registry.handle(name)),
            "model_owner_path": spec["model"].owner_path.canonical(),
            "model": _model_snapshot_value(spec["model"], artifact=artifact),
            "spatial": spec["spatial"],
            "time": _time_snapshot_value(spec["time"]),
            "diagnostics": spec["diagnostics"],
        }
        for name, spec in problem._block_registry.items()
    }
    fields = dict(problem._field_registry.resolved_items(problem.resolve))
    params = _problem_param_rows(problem, artifact=artifact)
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
            "params": list(problem._param_registry.handles()),
        },
    }


def _problem_param_rows(problem: Any, *, artifact: bool) -> dict[str, Any]:
    """Project the canonical case-owned parameter authority with qualified handles."""
    registry = problem._param_registry
    rows = {}
    for name, declaration in sorted(registry.items()):
        handle = registry.canonicalize(registry.handle(declaration))
        declaration_data = (
            _validated_parameter_artifact_data(
                declaration, where="Problem parameter %r" % name)
            if artifact else declaration.bind_data()
        )
        if not isinstance(declaration_data, Mapping):
            raise TypeError("Problem parameter %r projection must be a mapping" % name)
        row = dict(declaration_data)
        if row.get("name") != name:
            raise ValueError("Problem parameter %r projection changed its registry name" % name)
        row["qid"] = handle.qualified_id
        row["handle"] = handle.canonical_identity()
        rows[name] = row
    return rows


def _validated_parameter_artifact_data(declaration: Any, *, where: str) -> dict[str, Any]:
    """Authenticate one declaration-owned compile projection against its full bind projection."""
    bind_data = declaration.bind_data()
    artifact_data = declaration.artifact_data()
    if not isinstance(bind_data, Mapping) or not isinstance(artifact_data, Mapping):
        raise TypeError("%s bind_data()/artifact_data() must return mappings" % where)
    bind_row = dict(bind_data)
    artifact_row = dict(artifact_data)
    kind = bind_row.get("kind")
    removed = {"provenance"}
    if kind == "runtime":
        removed.add("default")
    if set(bind_row) != set(artifact_row) | removed:
        raise ValueError(
            "%s artifact_data() may omit exactly %s (got full=%s artifact=%s)"
            % (where, sorted(removed), sorted(bind_row), sorted(artifact_row)))
    for key in artifact_row:
        if key not in bind_row or artifact_row[key] != bind_row[key]:
            raise ValueError("%s artifact_data() changed bind metadata %r" % (where, key))
    if artifact_row.get("name") != getattr(declaration, "name", None):
        raise ValueError("%s artifact_data() changed the parameter name" % where)
    return artifact_row


def _time_snapshot_value(program: Any) -> Any:
    """Use the Program's structural report (including its IR hash), or the value itself."""
    if program is None:
        return None
    inspect = getattr(program, "inspect", None)
    return inspect() if callable(inspect) else program


def _model_snapshot_value(model: Any, *, artifact: bool = False) -> Any:
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
    manifest_data = to_dict()
    if artifact:
        manifest_data = _artifact_manifest_data(manifest_owner, manifest_data)
    result = {
        "type": "%s.%s" % (type(model).__module__, type(model).__qualname__),
        "manifest": manifest_data,
    }
    scientific_hash = _scientific_model_hash(model, manifest_owner, artifact=artifact)
    if scientific_hash is not None:
        result["scientific_hash"] = scientific_hash
    return result


def _artifact_manifest_data(manifest_owner: Any, manifest_data: Any) -> dict[str, Any]:
    """Replace only authenticated parameter declaration rows by their artifact projections."""
    if not isinstance(manifest_data, Mapping):
        raise TypeError("model manifest to_dict() must return a mapping")
    result = dict(manifest_data)
    params = result.get("params")
    if not isinstance(params, Mapping):
        raise TypeError("model manifest params must be a mapping")
    if not params:
        return result
    declarations_provider = getattr(manifest_owner, "params", None)
    if not callable(declarations_provider):
        raise TypeError(
            "a model manifest with parameters must expose its authoritative params() registry")
    declarations = declarations_provider()
    if not isinstance(declarations, Mapping) or set(declarations) != set(params):
        raise ValueError("model manifest params do not match its authoritative params() registry")
    projected = {}
    for name, declaration in sorted(declarations.items()):
        row = params[name]
        if not isinstance(row, Mapping) or not {"qid", "handle"}.issubset(row):
            raise TypeError("model manifest parameter %r lacks its qualified identity" % name)
        bind_data = getattr(declaration, "bind_data", None)
        if not callable(bind_data):
            raise TypeError(
                "model manifest parameter %r must be a canonical ParameterDeclaration" % name)
        bind_row = dict(bind_data())
        if {key: value for key, value in row.items() if key not in {"qid", "handle"}} != bind_row:
            raise ValueError(
                "model manifest parameter %r disagrees with its declaration authority" % name)
        projected[name] = {
            **_validated_parameter_artifact_data(
                declaration, where="model manifest parameter %r" % name),
            "qid": row["qid"],
            "handle": row["handle"],
        }
    result["params"] = projected
    return result


def _scientific_model_hash(model: Any, manifest_owner: Any, *, artifact: bool = False) -> Any:
    candidates = (
        (model, "module_hash"),
        (model, "_model_hash"),
        (getattr(model, "_dsl", None), "_model_hash"),
        (manifest_owner, "module_hash"),
    )
    if artifact:
        # Only hashes that declare compile semantics may enter the artifact projection. The private
        # facade _model_hash historically included runtime defaults, whereas Module.module_hash now
        # consumes ParameterDeclaration.artifact_data().
        candidates = (
            (manifest_owner, "module_hash"),
            (model, "artifact_hash"),
            (model, "module_hash"),
        )
    for candidate, name in candidates:
        method = getattr(candidate, name, None) if candidate is not None else None
        if callable(method):
            value = method()
            if not isinstance(value, str) or not value:
                raise TypeError("%s() must return a non-empty string" % name)
            return value
    return None


__all__ = ["problem_snapshot_artifact_payload", "problem_snapshot_payload"]
