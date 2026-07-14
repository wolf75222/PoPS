"""Scientific identities, independent from lowering and presentation.

This module deliberately accepts only authenticated PoPS semantic authorities.  It does not walk
arbitrary Python objects: adding a new semantic family requires adding an explicit projection at
its owning boundary.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .digest import Identity, make_identity
from .encoding import canonical_bytes
from pops._generated_release_contract import SEMANTIC_IR_VERSION as SEMANTIC_SCHEMA_VERSION


def semantic_identity(payload: Any) -> Identity:
    """Return the versioned identity of one already-projected scientific payload."""
    return make_identity(
        "semantic", semantic_value(payload, where="semantic payload"),
        schema_version=SEMANTIC_SCHEMA_VERSION,
    )


def semantic_identity_of(*, snapshot: Any = None, model: Any = None,
                         program: Any = None) -> Identity:
    """Identity one and only one supported semantic authority.

    ``snapshot`` is the complete problem authority. ``model`` and ``program`` are useful at the
    lower-level compiler seams where a complete Problem is intentionally unavailable.
    """
    supplied = [snapshot is not None, model is not None, program is not None]
    if sum(supplied) != 1:
        raise TypeError("semantic_identity_of requires exactly one of snapshot, model, or program")
    if snapshot is not None:
        identity = getattr(snapshot, "semantic_identity", None)
        if not isinstance(identity, Identity) or identity.domain != "semantic":
            raise TypeError("snapshot must expose an authenticated semantic Identity")
        return identity
    if model is not None:
        return semantic_identity(model_semantic_data(model))
    return semantic_identity(program_semantic_data(program))


def model_semantic_data(model: Any) -> dict[str, Any]:
    """Project an operator-first model through its authenticated ModuleManifest."""
    protocol = getattr(model, "_semantic_data", None)
    if callable(protocol):
        payload = protocol()
        if not isinstance(payload, Mapping) or "kind" not in payload:
            raise TypeError("model _semantic_data() must return a mapping with a kind")
        return semantic_value(payload, where="model semantic protocol")

    from pops.model.module import Module

    module = model if isinstance(model, Module) else getattr(model, "module", None)
    if not isinstance(module, Module):
        raise TypeError("semantic model identity requires a pops.model.Module authority")
    manifest = module.manifest().to_dict()
    required = {
        "schema_version", "name", "owner_path", "state_spaces", "field_spaces", "params",
        "params_utilization", "aux", "provider_pack", "has_eigenvalues", "operators",
        "operator_aliases",
        "capabilities", "native_routes", "native_catalog", "abi_requirements",
    }
    if set(manifest) != required:
        raise TypeError("ModuleManifest semantic projection received an unsupported schema")

    operators = []
    for row in manifest["operators"]:
        expected = {
            "id", "name", "kind", "qid", "handle", "signature", "inputs", "output",
            "capabilities", "requirements", "lowering_route", "provenance",
        }
        optional_digests = {key for key in ("content_hash", "body_hash") if key in row}
        if set(row) != expected | optional_digests:
            raise TypeError("operator semantic projection received unsupported keys for %r"
                            % row.get("name"))
        projected = {
            "id": row["id"],
            "name": row["name"],
            "kind": row["kind"],
            "handle": row["handle"],
            "signature": row["signature"],
            "capabilities": row["capabilities"],
            "requirements": row["requirements"],
        }
        for key in sorted(optional_digests):
            projected[key] = row[key]
        operators.append(projected)

    return semantic_value({
        "owner": manifest["owner_path"],
        "spaces": {
            "state": _space_rows(manifest["state_spaces"], state=True),
            "field": _space_rows(manifest["field_spaces"], state=False),
            "aux": _aux_rows(manifest["aux"]),
        },
        "parameters": _parameter_rows(manifest["params"]),
        "providers": manifest["provider_pack"],
        "operators": operators,
        "operator_aliases": manifest["operator_aliases"],
        "has_eigenvalues": manifest["has_eigenvalues"],
        "component_digests": {"module": module.module_hash()},
    }, where="model semantic payload")


def program_semantic_data(program: Any) -> dict[str, Any]:
    """Normalize Program IR while dropping presentation-only program and node names."""
    from pops.time._program.api import Program

    if not isinstance(program, Program):
        raise TypeError("semantic program identity requires a pops.time.Program")
    serialized = program._serialize(include_provenance=False)
    expected = {"name", "version", "clock", "nodes", "commits", "block_order"}
    optional = {"histories", "history_persistence", "dt_bound", "step_transaction"}
    if not expected.issubset(serialized) or not set(serialized).issubset(expected | optional):
        raise TypeError("Program semantic projection received an unsupported IR schema")
    program_clock_owner = serialized["clock"].get("owner")
    result = {
        "version": serialized["version"],
        "clock": serialized["clock"],
        "nodes": [_semantic_node(row) for row in serialized["nodes"]],
        "commits": serialized["commits"],
        "block_order": serialized["block_order"],
    }
    for key in ("histories", "history_persistence", "step_transaction"):
        if key in serialized:
            result[key] = serialized[key]
    if "dt_bound" in serialized:
        bound = serialized["dt_bound"]
        if not isinstance(bound, Mapping) or set(bound) != {"nodes", "result"}:
            raise TypeError("Program dt_bound semantic data has an unsupported schema")
        result["dt_bound"] = {
            "nodes": [_semantic_node(row) for row in bound["nodes"]],
            "result": bound["result"],
        }
    return semantic_value(
        _drop_program_presentation(result, program_clock_owner),
        where="Program semantic payload",
    )


def _drop_program_presentation(value: Any, program_clock_owner: Any) -> Any:
    """Remove labels that identify an authoring presentation, not a method.

    Program-owned clocks carry the Program's display name in their consumer
    ``OwnerPath`` and ``StagePoint.name`` is only a label for reports.  Clock
    names, partition names and exact coordinates remain semantic.  Foreign
    clock owners are preserved because they are genuine cross-authority
    references.
    """
    if isinstance(value, Mapping):
        normalized = dict(value)
        if (
            set(normalized) == {"schema_version", "name", "owner"}
            and normalized.get("schema_version") == 1
            and normalized.get("owner") == program_clock_owner
        ):
            normalized["owner"] = None
        if (
            set(normalized) == {"schema_version", "name", "partitions"}
            and normalized.get("schema_version") == 1
            and isinstance(normalized.get("partitions"), Mapping)
        ):
            normalized.pop("name")
        return {
            key: _drop_program_presentation(item, program_clock_owner)
            for key, item in normalized.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _drop_program_presentation(item, program_clock_owner)
            for item in value
        ]
    return value


def semantic_value(value: Any, *, where: str) -> Any:
    """Normalize the closed value language accepted by semantic identities."""
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return value
    if isinstance(value, float):
        return {"binary64": value.hex()}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("%s mapping keys must be non-empty strings" % where)
        return {key: semantic_value(item, where="%s.%s" % (where, key))
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [semantic_value(item, where="%s[%d]" % (where, index))
                for index, item in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        items = [semantic_value(item, where="%s{item}" % where) for item in value]
        return sorted(items, key=lambda item: (len(canonical_bytes(item)), canonical_bytes(item)))
    raise TypeError("%s contains unsupported semantic value %s" % (where, type(value).__name__))


def _semantic_node(row: Any) -> dict[str, Any]:
    expected = {
        "id", "name", "vtype", "op", "block", "state", "point", "inputs", "attrs",
    }
    optional = {"space", "field_context"}
    if not isinstance(row, Mapping) or not expected.issubset(row) \
            or not set(row).issubset(expected | optional):
        raise TypeError("Program node semantic data has an unsupported schema")
    result = {key: row[key] for key in (
        "id", "vtype", "op", "block", "state", "point", "inputs", "attrs")}
    for key in ("space", "field_context"):
        if key in row:
            result[key] = row[key]
    for key in ("cond_block", "body_block", "apply_block", "residual_block"):
        block = result["attrs"].get(key)
        if block is not None:
            attrs = dict(result["attrs"])
            attrs[key] = [_semantic_node(node) for node in block]
            result["attrs"] = attrs
    return result


def _space_rows(rows: Any, *, state: bool) -> dict[str, Any]:
    keys = {"components", "layout", "representation", "centering", "units", "frame", "clock"}
    if state:
        keys |= {"roles", "storage"}
    return {name: {key: row[key] for key in sorted(keys)} for name, row in rows.items()}


def _aux_rows(rows: Any) -> dict[str, Any]:
    keys = {"aux_kind", "representation", "centering", "unit", "frame", "clock"}
    return {name: {key: row[key] for key in sorted(keys)} for name, row in rows.items()}


def _parameter_rows(rows: Any) -> dict[str, Any]:
    keys = {"kind", "domain", "unit", "storage"}
    return {name: {key: row[key] for key in sorted(keys)} for name, row in rows.items()}


__all__ = [
    "SEMANTIC_SCHEMA_VERSION", "model_semantic_data", "program_semantic_data",
    "semantic_identity", "semantic_identity_of", "semantic_value",
]
