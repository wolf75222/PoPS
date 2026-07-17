"""Closed enums and strict data helpers for parameter declarations."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Any

from pops.math import Bool, Integer, Real


PARAM_DECLARATION_SCHEMA_VERSION = 1
PARAM_DECLARATION_DATA_KEYS = frozenset({
    "schema_version", "name", "kind", "dtype", "unit", "domain", "default",
    "storage", "provenance", "expression", "depends_on", "phase", "invalidation",
})


class ParamKind(Enum):
    Runtime = "runtime"
    Const = "const"
    Derived = "derived"


class ParamStorage(Enum):
    Inline = "inline"
    RuntimeSlot = "runtime_slot"
    DerivedCache = "derived_cache"


class ParamPhase(Enum):
    Compile = "compile"
    Bind = "bind"
    Runtime = "runtime"
    PerBlock = "per_block"
    PerLevel = "per_level"


class ParamInvalidation(Enum):
    Never = "never"
    PerBind = "per_bind"
    OnDependencies = "on_dependencies"
    PerBlock = "per_block"
    PerLevel = "per_level"


class ParamDefaultState(Enum):
    Missing = "missing"
    Value = "value"
    Derived = "derived"


class _MissingDefault:
    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"

    def __reduce__(self) -> Any:
        return (_missing_default, ())


def _missing_default() -> Any:
    return MISSING


MISSING = _MissingDefault()


def strict_name(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("parameter name must be a non-empty string")
    return value


def dtype_name(dtype: Any) -> str:
    if dtype is Real:
        return "Real"
    if dtype is Integer:
        return "Integer"
    if dtype is Bool:
        return "Bool"
    raise TypeError(
        "parameter dtype must be pops.math.Real, Integer or Bool, not %r" % (dtype,)
    )


def checked_unit(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError("parameter unit must be a non-empty string or None")
    return value


def freeze_json(value: Any, *, where: str) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError("%s keys must be non-empty strings" % where)
            result[key] = freeze_json(item, where="%s.%s" % (where, key))
        return MappingProxyType(result)
    if isinstance(value, (tuple, list)):
        return tuple(freeze_json(item, where=where) for item in value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("%s contains a non-finite float" % where)
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    hook = getattr(value, "to_data", None)
    if callable(hook):
        return freeze_json(hook(), where=where)
    raise TypeError("%s contains non-serializable %s" % (where, type(value).__name__))


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


class ParamProvenance:
    """Stable, reportable origin of one parameter declaration."""

    __slots__ = ("source", "metadata")

    def __init__(self, source: Any, *, metadata: Any = None) -> None:
        if not isinstance(source, str) or not source:
            raise ValueError("ParamProvenance source must be a non-empty string")
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self, "metadata", freeze_json(metadata or {}, where="parameter provenance")
        )

    def to_data(self) -> dict[str, Any]:
        return {"source": self.source, "metadata": thaw_json(self.metadata)}

    @classmethod
    def from_data(cls, data: Any) -> ParamProvenance:
        if not isinstance(data, Mapping) or set(data) != {"source", "metadata"}:
            raise TypeError("ParamProvenance data requires exactly source and metadata")
        return cls(data["source"], metadata=data["metadata"])

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, ParamProvenance) and self.to_data() == other.to_data()

    def __hash__(self) -> int:
        return hash(json.dumps(self.to_data(), sort_keys=True, separators=(",", ":")))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("ParamProvenance is immutable")


def check_provenance(value: Any) -> ParamProvenance | None:
    if value is None:
        return None
    if not isinstance(value, ParamProvenance):
        raise TypeError("parameter provenance must be ParamProvenance or None")
    return value


def value_data(value: Any, *, dtype: Any, unit: str | None, where: str) -> dict[str, Any]:
    target = dtype_name(dtype)
    if dtype is Bool:
        if not isinstance(value, bool):
            raise TypeError("%s requires a bool value" % where)
        if unit is not None:
            raise TypeError("%s cannot attach a physical unit to Bool" % where)
        return {"kind": "boolean", "value": value, "target": target}
    if dtype is Integer and (isinstance(value, bool) or not isinstance(value, int)):
        raise TypeError("%s requires an int value (bool is not an Integer)" % where)
    if isinstance(value, bool):
        raise TypeError("%s requires a numeric value, not bool" % where)
    from pops._ir.literals import scalar_literal

    try:
        return scalar_literal(value, unit=unit, target=target).to_data()
    except (TypeError, ValueError) as exc:
        raise type(exc)("%s: %s" % (where, exc)) from None


def validate_parameter_data(data: Any) -> dict[str, Any]:
    """Authenticate one strict, JSON-ready declaration row without rehydrating its Expr."""
    if not isinstance(data, Mapping) or set(data) != PARAM_DECLARATION_DATA_KEYS:
        raise TypeError(
            "parameter declaration data requires exactly %s"
            % sorted(PARAM_DECLARATION_DATA_KEYS)
        )
    row = thaw_json(freeze_json(data, where="parameter declaration"))
    if row["schema_version"] != PARAM_DECLARATION_SCHEMA_VERSION:
        raise ValueError(
            "unsupported parameter declaration schema_version %r" % row["schema_version"]
        )
    strict_name(row["name"])
    try:
        kind = ParamKind(row["kind"])
        storage = ParamStorage(row["storage"])
        phase = ParamPhase(row["phase"])
        invalidation = ParamInvalidation(row["invalidation"])
    except ValueError as exc:
        raise ValueError("invalid closed parameter enum: %s" % exc) from None
    if row["dtype"] not in {"Real", "Integer", "Bool"}:
        raise ValueError("parameter dtype must be Real, Integer or Bool")
    checked_unit(row["unit"])
    if row["domain"] is not None:
        from .constraints import Constraint

        domain = Constraint.from_data(row["domain"])
        if domain.to_data() != row["domain"]:
            raise ValueError("parameter domain is not canonical")
    if row["provenance"] is not None:
        provenance = ParamProvenance.from_data(row["provenance"])
        if provenance.to_data() != row["provenance"]:
            raise ValueError("parameter provenance is not canonical")
    if not isinstance(row["default"], Mapping) or "state" not in row["default"]:
        raise TypeError("parameter default must be a state mapping")
    state = row["default"]["state"]
    expected_default_keys = {"state", "value"} if state == "value" else {"state"}
    if set(row["default"]) != expected_default_keys:
        raise TypeError("parameter default state %r has invalid keys" % state)
    if state not in {member.value for member in ParamDefaultState}:
        raise ValueError("unknown parameter default state %r" % state)
    if state == "value" and not isinstance(row["default"]["value"], Mapping):
        raise TypeError("parameter default value must be a canonical scalar mapping")
    if not isinstance(row["depends_on"], list):
        raise TypeError("parameter depends_on must be a list")
    for dependency in row["depends_on"]:
        if not isinstance(dependency, Mapping) or set(dependency) != {"name", "param_kind"}:
            raise TypeError("parameter dependency rows require name and param_kind")
        strict_name(dependency["name"])
        try:
            ParamKind(dependency["param_kind"])
        except ValueError:
            raise ValueError("parameter dependency has unknown param_kind") from None
    if kind is ParamKind.Runtime:
        if storage is not ParamStorage.RuntimeSlot or row["expression"] is not None \
                or row["depends_on"] or state not in {"missing", "value"}:
            raise ValueError("RuntimeParam data has inconsistent storage/expression/default")
        if phase is not ParamPhase.Bind or invalidation is not ParamInvalidation.PerBind:
            raise ValueError("RuntimeParam requires phase=bind and invalidation=per_bind")
    elif kind is ParamKind.Const:
        if storage is not ParamStorage.Inline or row["expression"] is not None \
                or row["depends_on"] or state != "value":
            raise ValueError("ConstParam data has inconsistent storage/expression/default")
        if phase is not ParamPhase.Compile or invalidation is not ParamInvalidation.Never:
            raise ValueError("ConstParam requires phase=compile and invalidation=never")
    else:
        if storage not in {ParamStorage.Inline, ParamStorage.DerivedCache} \
                or row["expression"] is None or not row["depends_on"] or state != "derived":
            raise ValueError("DerivedParam data has inconsistent storage/expression/default")
        contracts = {
            ParamPhase.Compile: (ParamStorage.Inline, {ParamInvalidation.Never}),
            ParamPhase.Bind: (
                ParamStorage.DerivedCache,
                {ParamInvalidation.OnDependencies, ParamInvalidation.PerBind},
            ),
            ParamPhase.Runtime: (
                ParamStorage.DerivedCache, {ParamInvalidation.OnDependencies},
            ),
            ParamPhase.PerBlock: (
                ParamStorage.DerivedCache,
                {ParamInvalidation.OnDependencies, ParamInvalidation.PerBlock},
            ),
            ParamPhase.PerLevel: (
                ParamStorage.DerivedCache,
                {ParamInvalidation.OnDependencies, ParamInvalidation.PerLevel},
            ),
        }
        expected_storage, allowed_invalidation = contracts[phase]
        if storage is not expected_storage or invalidation not in allowed_invalidation:
            raise ValueError(
                "DerivedParam phase=%s requires storage=%s and invalidation in %s"
                % (phase.value, expected_storage.value,
                   sorted(item.value for item in allowed_invalidation))
            )
    return row


__all__ = [
    "MISSING", "PARAM_DECLARATION_DATA_KEYS", "PARAM_DECLARATION_SCHEMA_VERSION", "ParamDefaultState",
    "ParamInvalidation", "ParamKind", "ParamPhase", "ParamProvenance", "ParamStorage",
    "check_provenance", "checked_unit", "dtype_name", "freeze_json", "strict_name",
    "thaw_json", "validate_parameter_data", "value_data",
]
