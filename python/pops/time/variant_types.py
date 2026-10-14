"""Structural finite value types shared by graph functions and native tagged sums.

Types carry the exact authority of field/index/array templates.  A serialized field authority is
not reconstructed by guessing: use ``type_of`` on a validated value in its owning Program.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, init=False, eq=False)
class ValueType:
    """Immutable exact Program value signature; no implicit scientific conversion."""

    signature: Mapping
    __pops_ir_immutable__ = True

    def __init__(self, signature):
        if not isinstance(signature, Mapping) or not isinstance(signature.get("vtype"), str):
            raise TypeError("ValueType requires a structural value signature")
        object.__setattr__(self, "signature", _freeze(signature))

    def to_data(self):
        from pops.time.canonical_data import _json_ready
        return _json_ready(self.signature)

    @classmethod
    def from_data(cls, data):
        """Read scalar/product/sum types without fabricating dependent authorities."""
        if not isinstance(data, Mapping):
            raise TypeError("value type data must be a mapping")
        kind = data.get("vtype")
        if kind in ("bool", "int64", "scalar"):
            allowed = {"vtype", "block", "space", "state_ref", "field_context", "point"}
            if set(data) - allowed or any(v is not None for k, v in data.items() if k != "vtype"):
                raise ValueError("dependent type data requires authenticated authority reconstruction")
            return {"bool": Bool, "int64": Int64, "scalar": Float64}[kind]
        if kind in ("product", "variant") and set(data) == {"vtype", "members"}:
            members = data["members"]
            if len({key for key, _ in members}) != len(members):
                raise ValueError("duplicate structural type member")
            constructor = Product if kind == "product" else Variant
            return constructor(**{key: cls.from_data(child) for key, child in members})
        raise ValueError("dependent type data requires authenticated authority reconstruction")

    def matches(self, value):
        from pops.time.value_signatures import same_signature
        return same_signature(instantiate(self, value.point).signature, type_of(value).signature)

    def __eq__(self, other):
        return isinstance(other, ValueType) and self.to_data() == other.to_data()


def _scalar(kind):
    return ValueType(dict(vtype=kind, block=None, space=None, state_ref=None,
                          field_context=None, point=None))


Bool, Int64, Float64 = _scalar("bool"), _scalar("int64"), _scalar("scalar")


def type_of(value):
    from pops.time.values import ProgramValue
    from pops.time.value_signatures import value_signature
    if not isinstance(value, ProgramValue):
        raise TypeError("type_of requires a ProgramValue")
    return ValueType(value_signature(value))


def as_type(value):
    return value if isinstance(value, ValueType) else type_of(value)


def instantiate(value_type, point):
    """Bind only undeclared scalar evaluation points; exact field/index authorities stay exact."""
    def bind(signature):
        result = dict(signature)
        if result["vtype"] in ("scalar", "bool", "int64") and result.get("point") is None:
            result["point"] = point
        if result["vtype"] in ("product", "variant"):
            result["members"] = tuple((key, bind(child)) for key, child in result["members"])
        return result
    return ValueType(bind(as_type(value_type).signature))


def _members(kind, fields):
    if any(not isinstance(key, str) or not key for key in fields):
        raise TypeError("type member names must be non-empty strings")
    if kind == "variant" and not fields:
        raise ValueError("Variant requires at least one tag")
    items = sorted(fields.items()) if kind == "variant" else fields.items()
    return ValueType({"vtype": kind,
                      "members": tuple((key, as_type(value).signature) for key, value in items)})


def Product(**fields):
    return _members("product", fields)


def Variant(**tags):
    return _members("variant", tags)


def Result(success, error):
    return Variant(ok=success, err=error)


RuntimeFailure = Product(code=Int64, origin=Int64)


def Failure(**tags):
    """Closed application error union including checked native execution failures."""
    if "runtime" in tags:
        raise ValueError("Failure reserves runtime for RuntimeFailure")
    return Variant(runtime=RuntimeFailure, **tags)


__all__ = ["ValueType", "Bool", "Int64", "Float64", "Product", "Variant", "Result",
           "RuntimeFailure", "Failure", "as_type", "type_of", "instantiate"]
