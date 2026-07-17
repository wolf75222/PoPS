"""Authenticated immutable module manifest payload."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from ._operator_manifest import (
    OperatorRegistryManifest,
    canonical_owner,
    require_exact_keys,
    validate_handle_identity,
)
from .handles import Handle, OperatorHandle
from .manifest_data import (
    freeze_json as _freeze_json,
    require_manifest_name,
    strict_json_loads as _strict_json_loads,
    thaw_json as _thaw_json,
)
from .manifest_support import params_utilization as _params_utilization
from .ownership import OwnerPath
from .provider_pack import ProviderPack


SCHEMA_VERSION = 8

_WAVE_SPEED_PROVIDERS = frozenset({"explicit_pair", "jacobian", "pressure_derived"})

_PARAM_ROW_KEYS = {
    "schema_version",
    "name",
    "kind",
    "dtype",
    "unit",
    "domain",
    "default",
    "storage",
    "provenance",
    "expression",
    "depends_on",
    "phase",
    "invalidation",
    "qid",
    "handle",
}

_DECLARATION_ROW_KEYS = {
    "state": ({
        "components", "roles", "layout", "storage", "representation", "centering",
        "units", "frame", "clock", "qid", "handle",
    },),
    "field": ({
        "components", "layout", "representation", "centering", "units", "frame", "clock",
        "qid", "handle",
    },),
    "parameter": (_PARAM_ROW_KEYS,),
    "aux": ({
        "aux_kind", "representation", "centering", "unit", "frame", "clock", "qid", "handle",
    },),
}


def _validate_declaration_rows(
    rows: Any,
    *,
    owner: OwnerPath,
    kind: str,
    where: str,
) -> None:
    if not isinstance(rows, Mapping):
        raise TypeError("%s must be a mapping" % where)
    allowed_shapes = _DECLARATION_ROW_KEYS[kind]
    for name, row in rows.items():
        if not isinstance(name, str) or not name:
            raise ValueError("%s names must be non-empty strings" % where)
        if not isinstance(row, Mapping) or set(row) not in allowed_shapes:
            got = (
                sorted(repr(key) for key in row)
                if isinstance(row, Mapping)
                else type(row).__name__
            )
            raise TypeError("%s %r has unsupported keys %s" % (where, name, got))
        handle = validate_handle_identity(
            row["handle"],
            row["qid"],
            owner=owner,
            local_id=name,
            kind=kind,
            where="%s %s" % (where, name),
        )
        if kind == "parameter":
            from .handles import ParamHandle
            from pops.params import validate_parameter_data

            if not isinstance(handle, ParamHandle):
                raise TypeError("%s %s handle must be a ParamHandle" % (where, name))
            if row["name"] != name:
                raise ValueError("%s %s row name does not match its registry key" % (where, name))
            if handle.param_kind != row["kind"]:
                raise ValueError(
                    "%s %s ParamHandle kind does not match declaration kind" % (where, name)
                )
            declaration_data = {
                key: value for key, value in row.items() if key not in {"qid", "handle"}
            }
            validate_parameter_data(declaration_data)


def _validate_operator_bindings(
    rows: Any,
    *,
    owner: OwnerPath,
    operators: OperatorRegistryManifest,
) -> list[dict[str, Any]]:
    """Authenticate the generic scientific-handle -> operator projection registry."""
    if not isinstance(rows, (list, tuple)):
        raise TypeError("ModuleManifest operator_bindings must be a list")
    expected = {"subject_qid", "subject_handle", "target_qid", "target_handle"}
    normalized = []
    seen = set()
    for position, value in enumerate(rows):
        row = require_exact_keys(
            value, expected, where="operator binding row %d" % position
        )
        subject = Handle.from_canonical_identity(row["subject_handle"])
        if isinstance(subject, OperatorHandle):
            raise TypeError("operator binding subject must not be an OperatorHandle")
        validate_handle_identity(
            row["subject_handle"],
            row["subject_qid"],
            owner=owner,
            local_id=subject.local_id,
            kind=subject.kind,
            where="operator binding subject %s:%s" % (subject.kind, subject.local_id),
        )
        target = Handle.from_canonical_identity(row["target_handle"])
        if not isinstance(target, OperatorHandle):
            raise TypeError("operator binding target must be an OperatorHandle")
        entry = operators.describe(target.registered_operator_name)
        validate_handle_identity(
            row["target_handle"],
            row["target_qid"],
            owner=owner,
            local_id=target.registered_operator_name,
            kind=entry.kind,
            operator_target=target.registered_operator_name,
            where="operator binding target %s" % target.registered_operator_name,
        )
        if row["target_handle"] != entry.to_dict()["handle"]:
            raise ValueError(
                "operator binding target identity does not match operator %r"
                % target.registered_operator_name
            )
        subject_key = (subject.kind, subject.local_id, subject.schema_version)
        if subject_key in seen:
            raise ValueError(
                "duplicate operator binding subject %s:%s"
                % (subject.kind, subject.local_id)
            )
        seen.add(subject_key)
        normalized.append((subject_key, target.registered_operator_name, dict(row)))
    return [
        row for _, _, row in sorted(normalized, key=lambda item: (item[0], item[1]))
    ]


class ModuleManifest:
    """Self-describing, canonical manifest of a model Module."""

    __slots__ = (
        "schema_version",
        "name",
        "owner_path",
        "state_spaces",
        "field_spaces",
        "params",
        "aux",
        "provider_pack",
        "has_eigenvalues",
        "wave_speed_provider",
        "operators",
        "operator_bindings",
        "capabilities",
        "native_routes",
        "native_catalog",
        "abi_requirements",
        "params_utilization",
    )

    def __init__(
        self,
        *,
        name: Any,
        owner_path: Any,
        state_spaces: Any,
        field_spaces: Any,
        params: Any,
        aux: Any,
        provider_pack: Any,
        has_eigenvalues: Any,
        wave_speed_provider: Any,
        operators: Any,
        operator_bindings: Any = None,
        capabilities: Any,
        native_routes: Any,
        native_catalog: Any,
        abi_requirements: Any,
        params_utilization: Any = None,
    ) -> None:
        if not isinstance(operators, OperatorRegistryManifest):
            raise TypeError("ModuleManifest operators must be an OperatorRegistryManifest")
        require_manifest_name(name)
        owner = canonical_owner(owner_path, where="ModuleManifest owner_path")
        if owner.name != name:
            raise ValueError(
                "ModuleManifest name %r does not match owner_path name %r" % (name, owner.name)
            )
        if operators.owner_path != owner:
            raise ValueError("ModuleManifest operator registry has a different owner_path")
        checked_bindings = _validate_operator_bindings(
            [] if operator_bindings is None else operator_bindings,
            owner=owner,
            operators=operators,
        )
        _validate_declaration_rows(
            state_spaces, owner=owner, kind="state", where="module state_spaces"
        )
        _validate_declaration_rows(
            field_spaces, owner=owner, kind="field", where="module field_spaces"
        )
        _validate_declaration_rows(params, owner=owner, kind="parameter", where="module params")
        _validate_declaration_rows(aux, owner=owner, kind="aux", where="module aux")
        provider_pack = ProviderPack.from_data(provider_pack).to_data()
        if wave_speed_provider is not None and wave_speed_provider not in _WAVE_SPEED_PROVIDERS:
            raise ValueError(
                "ModuleManifest wave_speed_provider %r must be None or one of %s"
                % (wave_speed_provider, ", ".join(sorted(_WAVE_SPEED_PROVIDERS)))
            )
        object.__setattr__(self, "schema_version", SCHEMA_VERSION)
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self, "owner_path", _freeze_json(owner.to_data(), where="module owner_path")
        )
        for attr, value in (
            ("state_spaces", state_spaces),
            ("field_spaces", field_spaces),
            ("params", params),
            ("aux", aux),
            ("provider_pack", provider_pack),
            ("has_eigenvalues", has_eigenvalues),
            ("wave_speed_provider", wave_speed_provider),
            ("capabilities", capabilities),
            ("native_routes", native_routes),
            ("native_catalog", native_catalog),
            ("abi_requirements", abi_requirements),
        ):
            object.__setattr__(self, attr, _freeze_json(value, where="module %s" % attr))
        object.__setattr__(self, "operators", operators)
        object.__setattr__(
            self,
            "operator_bindings",
            _freeze_json(checked_bindings, where="module operator_bindings"),
        )
        object.__setattr__(
            self,
            "params_utilization",
            _freeze_json(
                params_utilization or _params_utilization(self.params),
                where="module params_utilization",
            ),
        )

    def __setattr__(self, name: Any, value: Any) -> None:
        raise AttributeError("ModuleManifest is immutable")

    def __delattr__(self, name: Any) -> None:
        raise AttributeError("ModuleManifest is immutable")

    def with_abi_key(self, abi_key: Any) -> ModuleManifest:
        requirements = _thaw_json(self.abi_requirements)
        requirements["abi_key"] = abi_key
        return ModuleManifest(
            name=self.name,
            owner_path=_thaw_json(self.owner_path),
            state_spaces=_thaw_json(self.state_spaces),
            field_spaces=_thaw_json(self.field_spaces),
            params=_thaw_json(self.params),
            aux=_thaw_json(self.aux),
            provider_pack=_thaw_json(self.provider_pack),
            has_eigenvalues=_thaw_json(self.has_eigenvalues),
            wave_speed_provider=_thaw_json(self.wave_speed_provider),
            operators=self.operators,
            operator_bindings=_thaw_json(self.operator_bindings),
            capabilities=_thaw_json(self.capabilities),
            native_routes=_thaw_json(self.native_routes),
            native_catalog=_thaw_json(self.native_catalog),
            abi_requirements=requirements,
            params_utilization=_thaw_json(self.params_utilization),
        )

    def to_dict(self) -> Any:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "owner_path": _thaw_json(self.owner_path),
            "state_spaces": _thaw_json(self.state_spaces),
            "field_spaces": _thaw_json(self.field_spaces),
            "params": _thaw_json(self.params),
            "params_utilization": _thaw_json(self.params_utilization),
            "aux": _thaw_json(self.aux),
            "provider_pack": _thaw_json(self.provider_pack),
            "has_eigenvalues": _thaw_json(self.has_eigenvalues),
            "wave_speed_provider": _thaw_json(self.wave_speed_provider),
            "operators": self.operators.to_dict(),
            "operator_aliases": self.operators.aliases(),
            "operator_bindings": _thaw_json(self.operator_bindings),
            "capabilities": _thaw_json(self.capabilities),
            "native_routes": _thaw_json(self.native_routes),
            "native_catalog": _thaw_json(self.native_catalog),
            "abi_requirements": _thaw_json(self.abi_requirements),
        }

    @classmethod
    def from_dict(cls, data: Any) -> ModuleManifest:
        expected = {
            "schema_version",
            "name",
            "owner_path",
            "state_spaces",
            "field_spaces",
            "params",
            "params_utilization",
            "aux",
            "provider_pack",
            "has_eigenvalues",
            "wave_speed_provider",
            "operators",
            "operator_aliases",
            "operator_bindings",
            "capabilities",
            "native_routes",
            "native_catalog",
            "abi_requirements",
        }
        row = require_exact_keys(data, expected, where="ModuleManifest")
        version = row["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise TypeError("ModuleManifest schema_version must be an integer")
        if version != SCHEMA_VERSION:
            raise ValueError(
                "unsupported ModuleManifest schema_version %r (expected %d)"
                % (version, SCHEMA_VERSION)
            )
        owner = canonical_owner(row["owner_path"], where="ModuleManifest owner_path")
        operators = OperatorRegistryManifest.from_dict(
            row["operators"], row["operator_aliases"], owner=owner
        )
        result = cls(
            name=row["name"],
            owner_path=owner,
            state_spaces=row["state_spaces"],
            field_spaces=row["field_spaces"],
            params=row["params"],
            aux=row["aux"],
            provider_pack=row["provider_pack"],
            has_eigenvalues=row["has_eigenvalues"],
            wave_speed_provider=row["wave_speed_provider"],
            operators=operators,
            operator_bindings=row["operator_bindings"],
            capabilities=row["capabilities"],
            native_routes=row["native_routes"],
            native_catalog=row["native_catalog"],
            abi_requirements=row["abi_requirements"],
            params_utilization=row["params_utilization"],
        )
        if result.to_dict() != dict(row):
            raise ValueError("ModuleManifest is not in canonical form")
        return result

    @classmethod
    def from_json(cls, text: Any) -> ModuleManifest:
        return cls.from_dict(_strict_json_loads(text))

    def to_json(self, path: Any = None, *, indent: int = 2) -> Any:
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    @property
    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def __repr__(self) -> str:
        return "ModuleManifest(name=%r, operators=[%s])" % (
            self.name,
            ", ".join(self.operators.names()),
        )


__all__ = ["ModuleManifest", "SCHEMA_VERSION"]
