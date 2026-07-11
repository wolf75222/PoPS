"""Immutable owner-qualified slot of a model BindSchema."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pops.model.handles import Handle, ParamHandle
from pops.model.ownership import OwnerKind
from pops.params._declaration_data import freeze_json, thaw_json, value_data
from pops.params.constraints import Constraint

from ._bind_expression import eval_expression_key, expression_reference_keys
from ._bind_schema_data import (
    SLOT_KEYS,
    dtype_object,
    exact_mapping,
    validate_declaration,
    validate_default,
)


class BindSlot:
    """One immutable parameter declaration projected into one owner-qualified block."""

    __slots__ = ("ordinal", "handle", "_declaration")

    def __init__(self, ordinal: Any, handle: Any, declaration: Any) -> None:
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("BindSlot ordinal must be an integer >= 0")
        if not isinstance(handle, ParamHandle):
            raise TypeError("BindSlot handle must be a ParamHandle")
        if not handle.is_resolved:
            raise ValueError("BindSlot ParamHandle must have canonical ownership")
        root_kind = handle.owner_path.nodes[0].kind
        if not handle.is_instance and root_kind not in (OwnerKind.SHARED, OwnerKind.CASE):
            raise ValueError(
                "BindSlot ParamHandle must be block-qualified, Case-owned or explicitly shared"
            )
        validated = validate_declaration(declaration, handle=handle)
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "handle", handle)
        object.__setattr__(
            self, "_declaration", freeze_json(validated, where="BindSlot declaration")
        )

    @property
    def qid(self) -> str:
        return self.handle.qualified_id

    @property
    def declaration(self) -> Mapping[str, Any]:
        return self._declaration

    @property
    def kind(self) -> str:
        return self._declaration["kind"]

    @property
    def dtype(self) -> str:
        return self._declaration["dtype"]

    @property
    def required(self) -> bool:
        return self.kind == "runtime" and self._declaration["default"]["state"] == "missing"

    @property
    def has_default(self) -> bool:
        return self._declaration["default"]["state"] == "value"

    def default_value(self) -> Any:
        if not self.has_default:
            raise ValueError("BindSlot %s has no explicit default" % self.qid)
        return validate_default(self._declaration["default"], declaration=self._declaration)

    def validate_value(self, value: Any) -> Any:
        if self.kind != "runtime":
            raise TypeError(
                "BindSlot %s is %s and cannot accept a bind-time value" % (self.qid, self.kind)
            )
        return self._validate_typed_value(value)

    def _validate_typed_value(self, value: Any) -> Any:
        value_data(
            value, dtype=dtype_object(self.dtype), unit=self._declaration["unit"],
            where="bind value for %s" % self.qid,
        )
        domain_data = self._declaration["domain"]
        if domain_data is not None:
            domain = Constraint.from_data(domain_data)
            try:
                domain.check(value, who=self.qid)
            except (TypeError, ValueError):
                raise ValueError(
                    "bind value %r for %s is outside domain %s" % (value, self.qid, domain_data)
                ) from None
        return value

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        if self.kind != "derived":
            raise TypeError("only DerivedParam slots have an expression to evaluate")
        expression = self._declaration["expression"]
        if not isinstance(expression, Mapping) or set(expression) != {"protocol", "value"}:
            raise TypeError("DerivedParam %s has invalid expression metadata" % self.qid)
        if expression["protocol"] != "pops.expr.key.v1":
            raise NotImplementedError(
                "DerivedParam %s expression protocol %r is unsupported at Bind phase"
                % (self.qid, expression["protocol"])
            )
        value = eval_expression_key(
            expression["value"], env, where="DerivedParam %s" % self.qid
        )
        return self._validate_typed_value(value)

    def expression_references(self) -> frozenset[tuple[str, str]]:
        if self.kind != "derived":
            return frozenset()
        expression = self._declaration["expression"]
        if not isinstance(expression, Mapping) or set(expression) != {"protocol", "value"}:
            raise TypeError("DerivedParam %s has invalid expression metadata" % self.qid)
        if expression["protocol"] != "pops.expr.key.v1":
            raise NotImplementedError(
                "DerivedParam %s expression protocol %r cannot authenticate depends_on"
                % (self.qid, expression["protocol"])
            )
        return expression_reference_keys(
            expression["value"], where="DerivedParam %s" % self.qid
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal, "qid": self.qid,
            "handle": self.handle.canonical_identity(),
            "declaration": thaw_json(self._declaration),
        }

    @classmethod
    def from_dict(cls, data: Any) -> BindSlot:
        row = exact_mapping(data, SLOT_KEYS, where="BindSlot")
        handle = Handle.from_canonical_identity(row["handle"])
        if not isinstance(handle, ParamHandle):
            raise TypeError("BindSlot handle payload must reconstruct a ParamHandle")
        if row["qid"] != handle.qualified_id:
            raise ValueError("BindSlot qid does not match its handle payload")
        result = cls(row["ordinal"], handle, row["declaration"])
        if result.to_dict() != dict(row):
            raise ValueError("BindSlot payload is not canonical")
        return result

    def artifact_data(self) -> dict[str, Any]:
        declaration = thaw_json(self._declaration)
        declaration.pop("provenance")
        if self.kind == "runtime":
            declaration.pop("default")
        return {
            "ordinal": self.ordinal, "qid": self.qid,
            "handle": self.handle.canonical_identity(), "declaration": declaration,
        }

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("BindSlot is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("BindSlot is immutable")

    def __repr__(self) -> str:
        return "BindSlot(ordinal=%d, qid=%r, kind=%r)" % (
            self.ordinal, self.qid, self.kind,
        )


__all__ = ["BindSlot"]
