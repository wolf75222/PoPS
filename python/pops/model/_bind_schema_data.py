"""Strict wire-data validation shared by model BindSlot and BindSchema."""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from fractions import Fraction
from typing import Any

from pops.model.handles import ParamHandle
from pops.params._declaration_data import value_data
from pops.params.constraints import Constraint


DECLARATION_KEYS = {
    "schema_version", "name", "kind", "dtype", "unit", "domain", "default",
    "storage", "provenance", "expression", "depends_on", "phase", "invalidation",
}
SLOT_KEYS = {"ordinal", "qid", "handle", "declaration"}
KINDS = frozenset({"runtime", "const", "derived"})
DTYPES = frozenset({"Real", "Integer", "Bool"})
STORAGE = frozenset({"inline", "runtime_slot", "derived_cache"})
PHASES = frozenset({"compile", "bind", "runtime", "per_block", "per_level"})
INVALIDATION = frozenset(
    {"never", "per_bind", "on_dependencies", "per_block", "per_level"}
)
DEFAULT_STATES = frozenset({"missing", "value", "derived"})


def exact_mapping(value: Any, keys: set[str], *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("%s must be a mapping" % where)
    if set(value) != keys:
        missing = sorted(keys - set(value))
        unknown = sorted(set(value) - keys)
        raise TypeError(
            "%s requires exactly %s (missing=%s, unknown=%s)"
            % (where, sorted(keys), missing, unknown)
        )
    return value


def enum_value(value: Any, allowed: frozenset[str], *, where: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError("%s must be one of %s (got %r)" % (where, sorted(allowed), value))
    return value


def literal_value(data: Any, *, where: str) -> Any:
    """Decode the closed scalar-literal rows accepted as runtime defaults."""
    if not isinstance(data, Mapping) or not isinstance(data.get("kind"), str):
        raise TypeError("%s must be canonical scalar-literal data" % where)
    kind = data["kind"]
    common = {"kind", "unit", "target"}
    if kind == "boolean":
        if set(data) not in ({"kind", "value"}, {"kind", "value", "target"}):
            raise TypeError("%s has an invalid Boolean literal shape" % where)
        if not isinstance(data["value"], bool):
            raise TypeError("%s Boolean literal value must be bool" % where)
        return data["value"]
    if kind == "integer":
        if not {"kind", "value"} <= set(data) or set(data) - common - {"value"}:
            raise TypeError("%s has an invalid integer literal shape" % where)
        if not isinstance(data["value"], str):
            raise TypeError("%s integer literal value must be a decimal string" % where)
        return int(data["value"])
    if kind == "rational":
        required = {"kind", "numerator", "denominator"}
        if not required <= set(data) or set(data) - common - required:
            raise TypeError("%s has an invalid rational literal shape" % where)
        if not isinstance(data["numerator"], str) or not isinstance(data["denominator"], str):
            raise TypeError("%s rational numerator/denominator must be decimal strings" % where)
        return Fraction(int(data["numerator"]), int(data["denominator"]))
    if kind == "decimal":
        if not {"kind", "value"} <= set(data) or set(data) - common - {"value"}:
            raise TypeError("%s has an invalid decimal literal shape" % where)
        if not isinstance(data["value"], str):
            raise TypeError("%s decimal literal value must be a string" % where)
        value = Decimal(data["value"])
        if not value.is_finite():
            raise ValueError("%s decimal literal must be finite" % where)
        return value
    if kind == "binary64":
        if not {"kind", "value"} <= set(data) or set(data) - common - {"value"}:
            raise TypeError("%s has an invalid binary64 literal shape" % where)
        if not isinstance(data["value"], str):
            raise TypeError("%s binary64 literal value must be a float.hex string" % where)
        return float.fromhex(data["value"])
    raise TypeError(
        "%s uses non-materializable scalar literal kind %r; runtime defaults must be numeric"
        % (where, kind)
    )


def dtype_object(name: str) -> Any:
    from pops.math import Bool, Integer, Real

    return {"Real": Real, "Integer": Integer, "Bool": Bool}[name]


def validate_default(row: Mapping[str, Any], *, declaration: Mapping[str, Any]) -> Any:
    where = "parameter %r default" % declaration["name"]
    if not isinstance(row, Mapping) or "state" not in row:
        raise TypeError("%s must contain an explicit state" % where)
    state = enum_value(row["state"], DEFAULT_STATES, where="%s state" % where)
    expected = {"state", "value"} if state == "value" else {"state"}
    if set(row) != expected:
        raise TypeError("%s state %r requires exactly %s" % (where, state, sorted(expected)))
    if state != "value":
        return None
    value = literal_value(row["value"], where=where)
    canonical_value = value_data(
        value, dtype=dtype_object(declaration["dtype"]), unit=declaration["unit"], where=where,
    )
    if canonical_value != dict(row["value"]):
        raise ValueError("%s scalar literal is not in canonical form" % where)
    domain_data = declaration["domain"]
    if domain_data is not None:
        domain = Constraint.from_data(domain_data)
        if domain.to_data() != dict(domain_data):
            raise ValueError("%s domain is not in canonical form" % where)
        domain.check(value, who=where)
    return value


def validate_declaration(data: Any, *, handle: ParamHandle) -> dict[str, Any]:
    row = dict(exact_mapping(data, DECLARATION_KEYS, where="BindSlot declaration"))
    version = row["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("BindSlot declaration schema_version must be 1")
    if not isinstance(row["name"], str) or not row["name"]:
        raise ValueError("BindSlot declaration name must be a non-empty string")
    kind = enum_value(row["kind"], KINDS, where="BindSlot declaration kind")
    dtype = enum_value(row["dtype"], DTYPES, where="BindSlot declaration dtype")
    storage = enum_value(row["storage"], STORAGE, where="BindSlot declaration storage")
    phase = enum_value(row["phase"], PHASES, where="BindSlot declaration phase")
    invalidation = enum_value(
        row["invalidation"], INVALIDATION, where="BindSlot declaration invalidation"
    )
    if row["name"] != handle.local_id or kind != handle.param_kind:
        raise ValueError("BindSlot declaration does not match its ParamHandle identity")
    unit = row["unit"]
    if unit is not None and (not isinstance(unit, str) or not unit):
        raise TypeError("BindSlot declaration unit must be a non-empty string or None")
    if dtype == "Bool" and unit is not None:
        raise TypeError("Bool parameters cannot carry a physical unit")
    if row["domain"] is not None:
        domain = Constraint.from_data(row["domain"])
        if domain.to_data() != dict(row["domain"]):
            raise ValueError("BindSlot declaration domain is not canonical")
    if row["provenance"] is not None:
        from pops.params import ParamProvenance

        provenance = ParamProvenance.from_data(row["provenance"])
        if provenance.to_data() != dict(row["provenance"]):
            raise ValueError("BindSlot declaration provenance is not canonical")
    dependencies = row["depends_on"]
    if not isinstance(dependencies, (list, tuple)):
        raise TypeError("BindSlot declaration depends_on must be a list")
    seen_dependencies = set()
    for dependency in dependencies:
        dep = exact_mapping(dependency, {"name", "param_kind"}, where="DerivedParam dependency")
        if not isinstance(dep["name"], str) or not dep["name"]:
            raise ValueError("DerivedParam dependency name must be non-empty")
        enum_value(dep["param_kind"], KINDS, where="DerivedParam dependency kind")
        key = (dep["name"], dep["param_kind"])
        if key in seen_dependencies:
            raise ValueError("DerivedParam depends_on contains duplicate dependency %r" % (key,))
        seen_dependencies.add(key)
    default_state = row["default"].get("state") if isinstance(row["default"], Mapping) else None
    validate_default(row["default"], declaration=row)

    if kind == "runtime":
        if storage != "runtime_slot" or phase != "bind" or invalidation != "per_bind":
            raise ValueError(
                "RuntimeParam requires storage=runtime_slot, phase=bind, invalidation=per_bind"
            )
        if default_state not in ("missing", "value"):
            raise ValueError("RuntimeParam default state must be missing or value")
        if row["expression"] is not None or dependencies:
            raise ValueError("RuntimeParam cannot carry an expression or dependencies")
    elif kind == "const":
        if storage != "inline" or phase != "compile" or invalidation != "never":
            raise ValueError("ConstParam requires storage=inline, phase=compile, invalidation=never")
        if default_state != "value":
            raise ValueError("ConstParam requires an explicit value")
        if row["expression"] is not None or dependencies:
            raise ValueError("ConstParam cannot carry an expression or dependencies")
    else:
        if storage not in ("inline", "derived_cache"):
            raise ValueError("DerivedParam storage must be inline or derived_cache")
        if default_state != "derived" or row["expression"] is None or not dependencies:
            raise ValueError("DerivedParam requires default=derived, an expression and dependencies")
    return row


__all__ = [
    "SLOT_KEYS", "dtype_object", "exact_mapping", "literal_value", "validate_declaration",
    "validate_default",
]
