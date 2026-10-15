"""Structural finite value types shared by graph functions and native tagged sums.

Types carry the exact authority of field/index/array templates.  A serialized field authority is
not reconstructed by guessing: use ``type_of`` on a validated value in its owning Program.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from pops.time.value_signatures import _validate_signature_shape
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pops.time.value_signatures import ValueSignature


def _freeze(value):
    from pops._structural_data import transform
    def describe(node):
        if isinstance(node, Mapping):
            keys = tuple(node)
            return tuple(node.values()), lambda children: MappingProxyType(dict(zip(keys, children)))
        if isinstance(node, (list, tuple)):
            return node, tuple
        return None, node
    return transform(value, describe)


@dataclass(frozen=True, init=False, eq=False)
class ValueType:
    """Immutable exact Program value signature; no implicit scientific conversion."""

    signature: ValueSignature
    __pops_ir_immutable__ = True

    def __init__(self, signature):
        if not isinstance(signature, Mapping) or not isinstance(signature.get("vtype"), str):
            raise TypeError("ValueType requires a structural value signature")
        _validate_signature_shape(signature)
        object.__setattr__(self, "signature", cast("ValueSignature", _freeze(signature)))

    def to_data(self):
        from pops.time.canonical_data import _json_ready
        return _json_ready(self.signature)

    @classmethod
    def from_data(cls, data):
        """Read scalar/product/sum types without fabricating dependent authorities."""
        from pops._structural_data import transform
        def describe(node):
            if not isinstance(node, Mapping):
                raise TypeError("value type data must be a mapping")
            kind = node.get("vtype")
            if kind in ("bool", "int64", "scalar"):
                allowed = {"vtype", "block", "space", "state_ref", "field_context", "point"}
                if set(node) - allowed or any(v is not None for k, v in node.items() if k != "vtype"):
                    raise ValueError("dependent type data requires authenticated authority reconstruction")
                return None, {"bool": Bool, "int64": Int64, "scalar": Float64}[kind].signature
            if kind in ("product", "variant") and set(node) == {"vtype", "members"}:
                members = node["members"]
                names = tuple(key for key, _ in members)
                if any(not isinstance(key, str) or not key for key in names):
                    raise TypeError("type member names must be non-empty strings")
                if len(set(names)) != len(names):
                    raise ValueError("duplicate structural type member")
                if kind == "variant" and not names:
                    raise ValueError("Variant requires at least one tag")
                rows = sorted(members) if kind == "variant" else members
                keys = tuple(key for key, _ in rows)
                return tuple(child for _, child in rows), lambda children: {
                    "vtype": kind, "members": tuple(zip(keys, children))}
            raise ValueError("dependent type data requires authenticated authority reconstruction")
        return cls(transform(data, describe))

    def matches(self, value):
        from pops.time.value_signatures import same_signature
        return same_signature(instantiate(self, value.point).signature, type_of(value).signature)

    def __eq__(self, other):
        from pops.time.value_signatures import same_signature
        return isinstance(other, ValueType) and same_signature(self.signature, other.signature)


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
    from pops._structural_data import transform
    def describe(signature):
        result = dict(signature)
        if result["vtype"] in ("scalar", "bool", "int64") and result.get("point") is None:
            result["point"] = point
        if result["vtype"] in ("product", "variant"):
            members = result["members"]
            keys = tuple(key for key, _ in members)
            def build(children):
                result["members"] = tuple(zip(keys, children))
                return result
            return tuple(child for _, child in members), build
        return None, result
    return ValueType(transform(as_type(value_type).signature, describe))


def instantiate_variant_payload(value_type, tag, payload_signature, default_point):
    """Infer the shared unbound scalar point from exact selected payload leaves."""
    declared = as_type(value_type).signature
    if declared["vtype"] != "variant":
        raise TypeError("variant requires a Variant type")
    members = dict(declared["members"])
    if not isinstance(tag, str) or tag not in members:
        raise ValueError("unknown Variant tag %r" % tag)
    pending = [(members[tag], payload_signature)]
    inferred = []
    while pending:
        wanted, actual = pending.pop()
        if wanted["vtype"] != actual["vtype"]:
            continue  # The exact signature comparison reports the mismatch.
        if wanted["vtype"] in ("scalar", "bool", "int64") and wanted.get("point") is None:
            point = actual.get("point")
            if inferred and point != inferred[0]:
                raise TypeError("variant payload has incompatible unbound scalar evaluation points")
            inferred[:] = [point]
        elif wanted["vtype"] in ("product", "variant"):
            actual_members = dict(actual["members"])
            pending.extend((child, actual_members[name]) for name, child in wanted["members"]
                           if name in actual_members)
    return instantiate(value_type, inferred[0] if inferred else default_point)


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
