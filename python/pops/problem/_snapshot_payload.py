"""Raw structural Case payload used exclusively by :mod:`pops.problem._snapshot`.

The public ``Case.to_dict()`` is an inspection view and intentionally shortens several objects
to display names.  A compile-cache identity cannot use that lossy view: two models may share a name
while carrying different coefficients.  This module reads the typed registries and returns detached
container shells whose leaves remain the real descriptors; ``AuthoringSnapshot`` then applies its
strict structural protocol to every leaf.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def problem_snapshot_payload(problem: Any) -> dict[str, Any]:
    """Return all cache-relevant Case declarations without display-name reduction."""
    return _problem_snapshot_payload(problem, artifact=False)


def problem_snapshot_artifact_payload(problem: Any) -> dict[str, Any]:
    """Return the compile-relevant projection of one Case.

    Parameter declarations are projected through their authoritative ``artifact_data()`` methods.
    This is intentionally a second construction from typed registries, not a recursive deletion of
    keys from :func:`problem_snapshot_payload`.
    """
    return _problem_snapshot_payload(problem, artifact=True)


def problem_semantic_payload(problem: Any, *, layout: Any, time: Any) -> dict[str, Any]:
    """Return the closed scientific projection used by ``semantic_identity``.

    Presentation/report policies, runtime values, lowering routes and platform facts never enter
    this payload. Every accepted leaf comes from a typed owner or an explicit descriptor protocol;
    there is no object walk or lossy fallback.
    """
    from pops.identity.semantic import model_semantic_data, program_semantic_data, semantic_value

    blocks = {}
    for name, spec in sorted(problem._block_registry.items()):
        block = problem._block_registry.canonical_block(problem._block_registry.handle(name))
        row = {
            "handle": block.canonical_identity(),
            "model": model_semantic_data(spec["model"]),
            "states": [
                problem._block_registry.canonicalize(state, block=block).canonical_identity()
                for state in spec["states"]
            ],
            "spatial": _spatial_semantic_data(spec["spatial"]),
            "numerics": (
                None if name not in problem._numerics_assignments
                else problem._resolved_numerics_for(name).to_data()
            ),
        }
        if spec["time"] is not None:
            row["time"] = program_semantic_data(spec["time"])
        blocks[name] = row

    resolved_fields = dict(problem._field_registry.resolved_items(problem.resolve))
    fields = {
        name: {
            "handle": problem._field_registry.canonicalize(handle).canonical_identity(),
            "operator": _descriptor_semantic_data(
                resolved_fields[name].operator, where="field %s operator" % name),
            "discretization": _descriptor_semantic_data(
                resolved_fields[name].discretization,
                where="field %s discretization" % name),
        }
        for name, handle in sorted(problem.fields().items())
    }
    effective_time = time if time is not None else problem._time_registry.program
    payload = {
        "owner": problem.owner_path.canonical().to_data(),
        "blocks": blocks,
        "fields": fields,
        "parameters": _semantic_parameter_rows(problem),
        "initials": [
            initial.canonical_identity()
            for initial in problem._initial_registry.resolved()
        ],
        "layout": _layout_semantic_data(layout),
        "time": None if effective_time is None else program_semantic_data(effective_time),
    }
    return semantic_value(payload, where="Case semantic payload")


def _problem_snapshot_payload(problem: Any, *, artifact: bool) -> dict[str, Any]:
    blocks = {
        name: {
            "handle": problem._block_registry.canonical_block(
                problem._block_registry.handle(name)),
            "model_owner_path": spec["model"].owner_path.canonical(),
            "model": _model_snapshot_value(spec["model"], artifact=artifact),
            "states": tuple(
                problem._block_registry.canonicalize(
                    state, block=problem._block_registry.handle(name))
                for state in spec["states"]),
            "spatial": spec["spatial"],
            "time": _time_snapshot_value(spec["time"], artifact=artifact),
            "diagnostics": spec["diagnostics"],
        }
        for name, spec in problem._block_registry.items()
    }
    fields = dict(problem._field_registry.resolved_items(problem.resolve))
    params = _problem_param_rows(problem, artifact=artifact)
    return {
        "name": problem.name,
        "category": problem.category,
        "native_id": problem.native_id,
        "owner_path": problem.owner_path.canonical(),
        "options": problem.options(),
        "blocks": blocks,
        "fields": fields,
        "params": params,
        "initials": tuple(problem._initial_registry),
        "consumers": (
            None if problem._consumer_graph is None
            else problem._consumer_graph.authoring_data(problem.resolve)
        ),
        "numerics": {
            name: problem._resolved_numerics_for(name).to_data()
            for name in sorted(problem._numerics_assignments)
        },
        "time": _time_snapshot_value(problem._time_registry.program, artifact=artifact),
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
                declaration, where="Case parameter %r" % name)
            if artifact else declaration.bind_data()
        )
        if not isinstance(declaration_data, Mapping):
            raise TypeError("Case parameter %r projection must be a mapping" % name)
        row = dict(declaration_data)
        if row.get("name") != name:
            raise ValueError("Case parameter %r projection changed its registry name" % name)
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


def _time_snapshot_value(program: Any, *, artifact: bool = False) -> Any:
    """Project a Program from its IR, excluding lifecycle-only inspection state.

    ``Case.freeze`` first captures the snapshot and then seals the Program.  Inspection reports
    include that lifecycle state, so using ``inspect()`` made the same Case hash differently
    immediately after a successful freeze.  The serialized IR and its hash are the complete compile
    input and remain identical across the mutable-to-frozen transition.
    """
    if program is None:
        return None
    serialize = getattr(program, "_serialize", None)
    if callable(serialize):
        ir_hash = getattr(program, "_ir_hash", None)
        return {
            "type": "%s.%s" % (type(program).__module__, type(program).__qualname__),
            "ir": serialize(include_provenance=not artifact),
            "hash": ir_hash() if callable(ir_hash) else None,
        }
    return program


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


def _semantic_parameter_rows(problem: Any) -> dict[str, Any]:
    """Case parameters without defaults, runtime values, provenance or lowering policy."""
    rows = {}
    for name, declaration in sorted(problem._param_registry.items()):
        data = declaration.bind_data()
        if not isinstance(data, Mapping):
            raise TypeError("Case parameter %r bind_data() must return a mapping" % name)
        required = {"kind", "domain", "unit", "storage"}
        if not required.issubset(data):
            raise TypeError("Case parameter %r lacks semantic declaration metadata" % name)
        rows[name] = {key: data[key] for key in sorted(required)}
    return rows


def _descriptor_semantic_data(value: Any, *, where: str) -> Any:
    """Project the public Descriptor protocol; arbitrary structural objects are refused."""
    if value is None:
        return None
    from pops.descriptors import Descriptor

    if not isinstance(value, Descriptor):
        raise TypeError("%s must be a typed pops Descriptor, got %s" % (
            where, type(value).__name__))
    semantic_data = getattr(value, "semantic_data", None)
    if callable(semantic_data):
        return _semantic_option_data(semantic_data(), where=where)
    options = value.options()
    if not isinstance(options, Mapping):
        raise TypeError("%s options() must return a mapping" % where)
    return _semantic_option_data({
        "category": value.category,
        "name": value.name,
        "options": dict(options),
    }, where=where)


def _semantic_option_data(value: Any, *, where: str) -> Any:
    """Canonicalize typed Descriptor options through the open semantic-value protocol."""
    from decimal import Decimal
    from fractions import Fraction
    from pops.descriptors import Descriptor
    from pops.identity.semantic import semantic_value
    from pops.model import Handle
    from pops.params import ParameterDeclaration

    if isinstance(value, ParameterDeclaration):
        data = value.bind_data()
        keys = {"kind", "domain", "unit", "storage"}
        return {key: _semantic_option_data(data[key], where="%s.%s" % (where, key))
                for key in sorted(keys)}
    if isinstance(value, Handle):
        return value.canonical_identity()
    if isinstance(value, Descriptor):
        return _descriptor_semantic_data(value, where=where)
    semantic_data = getattr(value, "__pops_semantic_data__", None)
    if callable(semantic_data):
        data = semantic_data()
        if not isinstance(data, Mapping):
            raise TypeError(
                "%s __pops_semantic_data__() must return a mapping" % where)
        return _semantic_option_data(data, where=where)
    if isinstance(value, Mapping):
        return {key: _semantic_option_data(item, where="%s.%s" % (where, key))
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_semantic_option_data(item, where=where) for item in value]
    if isinstance(value, (Decimal, Fraction)):
        from pops._ir.literals import scalar_literal
        return scalar_literal(value).to_data()
    return semantic_value(value, where=where)


def _spatial_semantic_data(value: Any) -> Any:
    """Project the public finite-volume spatial selection without native route metadata."""
    if value is None:
        return None
    from pops.runtime._bricks_scheme import Spatial

    if not isinstance(value, Spatial):
        return _descriptor_semantic_data(value, where="block spatial")
    return _semantic_option_data(value.to_data(), where="block spatial")


def _layout_semantic_data(layout: Any) -> Any:
    """Project a layout through one small descriptor protocol, never concrete class dispatch."""
    if layout is None:
        return None
    from pops.mesh import LayoutPlan

    if isinstance(layout, LayoutPlan):
        return {
            "kind": "layout_plan",
            "owner": layout.owner.to_data(),
            "layouts": [row.to_data() for row in layout.layouts],
            "assignments": [row.to_data() for row in layout.assignments],
            "mappings": [row.to_data() for row in layout.mappings],
        }
    return _descriptor_semantic_data(layout, where="layout")


__all__ = [
    "problem_semantic_payload", "problem_snapshot_artifact_payload", "problem_snapshot_payload",
]
