"""Authenticated immutable operator rows used by ModuleManifest."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .handles import Handle, OperatorHandle
from .manifest_data import (
    freeze_json as _freeze_json,
    require_manifest_id,
    thaw_json as _thaw_json,
)
from .manifest_support import space_name as _space_name
from .ownership import OwnerKind, OwnerPath


def require_exact_keys(value: Any, expected: set[str], *, where: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError("%s must be a mapping" % where)
    actual = set(value)
    if actual != expected:
        raise TypeError(
            "%s must contain exactly %s (got %s)"
            % (where, sorted(expected), sorted(repr(key) for key in actual))
        )
    return value


def canonical_owner(value: Any, *, where: str) -> OwnerPath:
    owner = OwnerPath.from_data(value) if isinstance(value, Mapping) else OwnerPath.coerce(value)
    if not owner.is_canonical:
        raise ValueError("%s must be a canonical OwnerPath" % where)
    if len(owner.nodes) != 1 or owner.kind is not OwnerKind.MODEL_DEFINITION:
        raise ValueError("%s must identify one root model definition" % where)
    return owner


def validate_handle_identity(
    data: Any,
    qid: Any,
    *,
    owner: OwnerPath,
    local_id: str,
    kind: str,
    where: str,
    operator_target: str | None = None,
) -> Handle:
    handle = Handle.from_canonical_identity(data)
    if handle.owner_path != owner:
        raise ValueError("%s handle is owned by %s, expected %s" % (where, handle.owner_path, owner))
    if handle.local_id != local_id or handle.kind != kind:
        raise ValueError("%s handle identity does not match declaration %s:%s" % (where, kind, local_id))
    if qid != handle.qualified_id:
        raise ValueError("%s qid does not match its canonical handle" % where)
    if operator_target is not None:
        if not isinstance(handle, OperatorHandle):
            raise TypeError("%s handle must be an OperatorHandle" % where)
        if handle.registered_operator_name != operator_target:
            raise ValueError(
                "%s alias target %r does not match handle target %r"
                % (where, operator_target, handle.registered_operator_name)
            )
    elif isinstance(handle, OperatorHandle):
        raise TypeError("%s declaration handle must not be an OperatorHandle" % where)
    return handle


class OperatorManifestEntry:
    """One authenticated operator row in stable registration order."""

    __slots__ = (
        "id",
        "name",
        "kind",
        "qid",
        "handle",
        "signature",
        "inputs",
        "output",
        "capabilities",
        "requirements",
        "lowering_route",
    )

    def __init__(self, operator: Any, operator_id: Any, handle: Any) -> None:
        signature = operator.signature
        require_manifest_id(operator_id)
        if not isinstance(handle, OperatorHandle):
            raise TypeError("operator manifest handle must be an OperatorHandle")
        if not handle.is_resolved:
            raise ValueError("operator manifest handle must have canonical ownership")
        if (
            handle.local_id != operator.name
            or handle.kind != operator.kind
            or handle.registered_operator_name != operator.name
        ):
            raise ValueError(
                "operator manifest handle does not authenticate registry operator %r" % operator.name
            )
        object.__setattr__(self, "id", operator_id)
        object.__setattr__(self, "name", operator.name)
        object.__setattr__(self, "kind", operator.kind)
        object.__setattr__(self, "qid", handle.qualified_id)
        object.__setattr__(
            self,
            "handle",
            _freeze_json(handle.canonical_identity(), where="operator %s handle" % operator.name),
        )
        object.__setattr__(
            self,
            "signature",
            _freeze_json(signature.to_data(), where="operator %s signature" % operator.name),
        )
        object.__setattr__(self, "inputs", tuple(_space_name(item) for item in signature.inputs))
        object.__setattr__(self, "output", _space_name(signature.output))
        object.__setattr__(
            self,
            "capabilities",
            _freeze_json(operator.capabilities, where="operator %s capabilities" % operator.name),
        )
        object.__setattr__(
            self,
            "requirements",
            _freeze_json(operator.requirements, where="operator %s requirements" % operator.name),
        )
        object.__setattr__(
            self,
            "lowering_route",
            _freeze_json(operator.lowering or {}, where="operator %s lowering" % operator.name),
        )

    def __setattr__(self, name: Any, value: Any) -> None:
        raise AttributeError("OperatorManifestEntry is immutable")

    def __delattr__(self, name: Any) -> None:
        raise AttributeError("OperatorManifestEntry is immutable")

    def to_dict(self) -> Any:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "qid": self.qid,
            "handle": _thaw_json(self.handle),
            "signature": _thaw_json(self.signature),
            "inputs": list(self.inputs),
            "output": self.output,
            "capabilities": _thaw_json(self.capabilities),
            "requirements": _thaw_json(self.requirements),
            "lowering_route": _thaw_json(self.lowering_route),
        }

    @classmethod
    def from_dict(cls, data: Any, *, owner: Any) -> OperatorManifestEntry:
        expected = {
            "id",
            "name",
            "kind",
            "qid",
            "handle",
            "signature",
            "inputs",
            "output",
            "capabilities",
            "requirements",
            "lowering_route",
        }
        row = require_exact_keys(data, expected, where="operator manifest row")
        require_manifest_id(row["id"])
        if not isinstance(row["name"], str) or not row["name"]:
            raise ValueError("operator manifest name must be a non-empty string")
        if not isinstance(row["kind"], str) or not row["kind"]:
            raise ValueError("operator manifest kind must be a non-empty string")
        owner = canonical_owner(owner, where="operator manifest owner")
        validate_handle_identity(
            row["handle"],
            row["qid"],
            owner=owner,
            local_id=row["name"],
            kind=row["kind"],
            operator_target=row["name"],
            where="operator %s" % row["name"],
        )
        if not isinstance(row["inputs"], list) or any(
            not isinstance(value, str) or not value for value in row["inputs"]
        ):
            raise TypeError("operator manifest inputs must be a list of non-empty strings")
        if not isinstance(row["output"], str) or not row["output"]:
            raise TypeError("operator manifest output must be a non-empty string")
        result = object.__new__(cls)
        for name in ("id", "name", "kind", "qid", "output"):
            object.__setattr__(result, name, row[name])
        object.__setattr__(
            result,
            "handle",
            _freeze_json(row["handle"], where="operator %s handle" % row["name"]),
        )
        object.__setattr__(
            result,
            "signature",
            _freeze_json(row["signature"], where="operator %s signature" % row["name"]),
        )
        object.__setattr__(result, "inputs", tuple(row["inputs"]))
        for name in ("capabilities", "requirements", "lowering_route"):
            object.__setattr__(
                result,
                name,
                _freeze_json(row[name], where="operator %s %s" % (row["name"], name)),
            )
        if result.to_dict() != dict(row):
            raise ValueError("operator manifest row is not canonical")
        return result

    def __repr__(self) -> str:
        return "OperatorManifestEntry(id=%d, name=%r, kind=%r)" % (
            self.id,
            self.name,
            self.kind,
        )


class OperatorRegistryManifest:
    """Ordered operator rows and authenticated alias identities."""

    __slots__ = ("_entries", "_aliases", "_owner_path")

    def __init__(self, entries: Any, aliases: Any = None, *, owner: Any) -> None:
        owner = canonical_owner(owner, where="operator registry manifest owner")
        frozen = tuple(entries)
        if any(not isinstance(entry, OperatorManifestEntry) for entry in frozen):
            raise TypeError("OperatorRegistryManifest entries must be OperatorManifestEntry values")
        if [entry.id for entry in frozen] != list(range(len(frozen))):
            raise ValueError("operator manifest ids must be contiguous registration-order integers")
        if len({entry.name for entry in frozen}) != len(frozen):
            raise ValueError("operator manifest names must be unique")
        for entry in frozen:
            validate_handle_identity(
                _thaw_json(entry.handle),
                entry.qid,
                owner=owner,
                local_id=entry.name,
                kind=entry.kind,
                operator_target=entry.name,
                where="operator %s" % entry.name,
            )
        alias_rows = {} if aliases is None else aliases
        normalized = self._validate_aliases(alias_rows, entries=frozen, owner=owner)
        object.__setattr__(self, "_entries", frozen)
        object.__setattr__(
            self, "_aliases", _freeze_json(normalized, where="operator registry aliases")
        )
        object.__setattr__(self, "_owner_path", owner)

    def __setattr__(self, name: Any, value: Any) -> None:
        raise AttributeError("OperatorRegistryManifest is immutable")

    def __delattr__(self, name: Any) -> None:
        raise AttributeError("OperatorRegistryManifest is immutable")

    def __iter__(self) -> Any:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def owner_path(self) -> OwnerPath:
        return self._owner_path

    def names(self) -> Any:
        return [entry.name for entry in self._entries]

    def describe(self, name: Any) -> Any:
        for entry in self._entries:
            if entry.name == name:
                return entry
        raise KeyError(
            "operator %r is not in this module's operator registry (registered: %s)"
            % (name, ", ".join(self.names()) or "<none>")
        )

    def to_dict(self) -> Any:
        return [entry.to_dict() for entry in self._entries]

    def aliases(self) -> Any:
        return _thaw_json(self._aliases)

    @classmethod
    def from_dict(cls, entries: Any, aliases: Any, *, owner: Any) -> OperatorRegistryManifest:
        if not isinstance(entries, list):
            raise TypeError("module manifest operators must be a list")
        owner = canonical_owner(owner, where="operator registry manifest owner")
        return cls(
            [OperatorManifestEntry.from_dict(row, owner=owner) for row in entries],
            aliases=aliases,
            owner=owner,
        )

    @staticmethod
    def _validate_aliases(aliases: Any, *, entries: Any, owner: OwnerPath) -> dict[str, Any]:
        if not isinstance(aliases, Mapping):
            raise TypeError("operator registry aliases must be a mapping")
        if any(not isinstance(alias, str) or not alias for alias in aliases):
            raise ValueError("operator alias names must be non-empty strings")
        by_name = {entry.name: entry for entry in entries}
        expected = {"name", "target", "qid", "handle", "target_qid", "target_handle"}
        normalized = {}
        for alias in sorted(aliases):
            row = require_exact_keys(aliases[alias], expected, where="operator alias %s" % alias)
            if row["name"] != alias:
                raise ValueError("operator alias map key %r does not match row name" % alias)
            target = row["target"]
            if not isinstance(target, str) or not target:
                raise ValueError("operator alias %r target must be a non-empty string" % alias)
            if target not in by_name:
                raise ValueError("operator alias %r targets unknown operator %r" % (alias, target))
            target_entry = by_name[target]
            validate_handle_identity(
                row["handle"],
                row["qid"],
                owner=owner,
                local_id=alias,
                kind=target_entry.kind,
                operator_target=target,
                where="operator alias %s" % alias,
            )
            validate_handle_identity(
                row["target_handle"],
                row["target_qid"],
                owner=owner,
                local_id=target,
                kind=target_entry.kind,
                operator_target=target,
                where="operator alias %s target" % alias,
            )
            if row["target_handle"] != _thaw_json(target_entry.handle):
                raise ValueError(
                    "operator alias %r target identity does not match operator %r" % (alias, target)
                )
            normalized[alias] = dict(row)
        return normalized

    @property
    def hash(self) -> str:
        blob = json.dumps(
            {"entries": self.to_dict(), "aliases": self.aliases()},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def __repr__(self) -> str:
        return "OperatorRegistryManifest(%s)" % ", ".join(self.names())


__all__ = [
    "OperatorManifestEntry",
    "OperatorRegistryManifest",
    "canonical_owner",
    "require_exact_keys",
    "validate_handle_identity",
]
