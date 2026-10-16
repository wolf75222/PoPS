"""Pure structural value signatures shared by finite types and Program authoring.

This layer only describes existing SSA values. It does not construct Program nodes,
validate control flow, or import the private authoring mixins.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, TypedDict, TypeVar, cast, overload


if TYPE_CHECKING:
    _Default = TypeVar("_Default")


    class CurrentDependencyData(TypedDict):
        target: Sequence[str]
        inputs: Mapping[str, Sequence[str]]
        function: Mapping[str, object]

    class ValueSignature(Mapping[str, object]):
        """Read-only view of the existing, kind-dependent structural records.

        A signature is a partial mapping: indexing an absent field raises KeyError.
        The opcode/type validators establish which keys belong to each kind. These
        overloads describe the value of an existing key without inventing defaults
        or coercing the exact live authorities retained in the mapping.
        """

        @overload
        def __getitem__(self, key: Literal["vtype", "ownership", "identity"]) -> str: ...
        @overload
        def __getitem__(self, key: Literal["view"]) -> Literal["read", "write"]: ...
        @overload
        def __getitem__(self, key: Literal["members"]) -> Sequence[tuple[str, ValueSignature]]: ...
        @overload
        def __getitem__(self, key: Literal["element", "success", "base", "schema", "candidate", "envelope", "proposal", "checks", "read", "write", "bundle", "rows", "columns", "field_signature"]) -> ValueSignature: ...
        @overload
        def __getitem__(self, key: Literal["family", "capacity", "byte_capacity"]) -> int: ...
        @overload
        def __getitem__(self, key: Literal["time_path", "step_path"]) -> Sequence[str]: ...
        @overload
        def __getitem__(self, key: Literal["domains"]) -> Sequence[Sequence[str]]: ...
        @overload
        def __getitem__(self, key: Literal["bindings"]) -> Sequence[tuple[Sequence[str], Sequence[str]]]: ...
        @overload
        def __getitem__(self, key: Literal["fields"]) -> Sequence[tuple[object, Sequence[str]]]: ...
        @overload
        def __getitem__(self, key: Literal["scientific", "roles", "current_dependencies"]) -> Sequence[object]: ...
        @overload
        def __getitem__(self, key: str) -> object: ...
        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        @overload
        def get(self, key: Literal["current_dependencies"], default: Sequence[CurrentDependencyData] = ()) -> Sequence[CurrentDependencyData]: ...
        @overload
        def get(self, key: Literal["bundle"], default: None = None) -> ValueSignature | None: ...
        @overload
        def get(self, key: Literal["ownership"], default: None = None) -> str | None: ...
        @overload
        def get(self, key: Literal["family", "capacity"], default: None = None) -> int | None: ...
        @overload
        def get(self, key: str, default: _Default = ...) -> object | _Default: ...
        def get(self, key: str, default: object = None) -> object:
            return super().get(key, default)


def _validate_signature_shape(signature):
    """Check the structural record promised by the static view, without coercing authorities.

    Kind-specific requirements remain with the existing opcode/schema validators. Unknown
    metadata stays opaque; this boundary checks only present keys whose types the view exposes.
    Nested schema and Bundle records need not contain a value kind of their own.
    """
    active = set()
    checked = set()
    child_keys = frozenset(("element", "success", "base", "schema", "candidate", "envelope",
                            "proposal", "checks", "read", "write", "bundle", "rows", "columns", "field_signature"))

    def sequence(value, where):
        if not isinstance(value, (list, tuple)):
            raise TypeError("signature %s must be a list or tuple structural sequence" % where)
        return value

    def path(value, where):
        if any(not isinstance(key, str) for key in sequence(value, where)):
            raise TypeError("signature %s must contain string path components" % where)

    def pair(value, where):
        items = sequence(value, where)
        if len(items) != 2:
            raise TypeError("signature %s must contain pairs" % where)
        return items

    stack = [(signature, False)]
    while stack:
        record, exiting = stack.pop()
        if exiting:
            active.remove(id(record))
            checked.add(id(record))
            continue
        if not isinstance(record, Mapping) or any(not isinstance(key, str) for key in record):
            raise TypeError("signature records must be mappings with string keys")
        identity = id(record)
        if identity in active:
            raise TypeError("signature structural records cannot be cyclic")
        if identity in checked:
            continue
        active.add(identity)
        stack.append((record, True))
        for key, value in record.items():
            if key in ("vtype", "ownership", "identity"):
                if not isinstance(value, str):
                    raise TypeError("signature %s must be a string" % key)
            elif key == "view":
                if not isinstance(value, str) or value not in ("read", "write"):
                    raise TypeError("signature view must be read or write")
            elif key in ("family", "capacity", "byte_capacity"):
                if type(value) is not int:
                    raise TypeError("signature %s must be an exact integer" % key)
            elif key in child_keys:
                stack.append((value, False))
            elif key == "members":
                for row in sequence(value, key):
                    name, child = pair(row, key)
                    if not isinstance(name, str):
                        raise TypeError("signature member names must be strings")
                    stack.append((child, False))
            elif key in ("time_path", "step_path"):
                path(value, key)
            elif key == "domains":
                for item in sequence(value, key):
                    path(item, key)
            elif key in ("fields", "bindings"):
                for row in sequence(value, key):
                    first, second = pair(row, key)
                    if key == "bindings":
                        path(first, key)
                    path(second, key)
            elif key in ("scientific", "roles"):
                sequence(value, key)
            elif key == "current_dependencies":
                for item in sequence(value, key):
                    if not isinstance(item, Mapping) or not {"target", "inputs", "function"} <= set(item):
                        raise TypeError("signature current dependency requires target, inputs and function")
                    path(item["target"], key)
                    inputs = item["inputs"]
                    if not isinstance(inputs, Mapping) or any(not isinstance(name, str) for name in inputs):
                        raise TypeError("signature current dependency inputs must be named paths")
                    for item_path in inputs.values():
                        path(item_path, key)
                    function = item["function"]
                    if not isinstance(function, Mapping) or any(not isinstance(name, str) for name in function):
                        raise TypeError("signature current dependency function must be a string-keyed record")


_LEAVES = frozenset(("state", "rhs", "scalar_field", "scalar", "bool", "int64"))


def value_signature(value) -> ValueSignature:
    if value.vtype in ("product", "variant", "index_set", "index", "array", "continuation_base", "continuation_candidate", "continuation_assessment", "continuation_accepted", "field_topology"):
        return cast("ValueSignature", value.attrs["signature"])
    if value.vtype == "result":
        return cast("ValueSignature", {"vtype": "result", "success": value.attrs["success_signature"],
                "point": value.point})
    if value.vtype not in _LEAVES:
        raise TypeError("structured control cannot carry value type %r" % value.vtype)
    signature = dict(vtype=value.vtype, block=value.block, space=value.space,
                     state_ref=value.state_ref, field_context=value.field_context, point=value.point)
    if value.attrs.get("field_authority") is not None:
        signature["field_authority"] = value.attrs["field_authority"]
    return cast("ValueSignature", signature)


def same_signature(left, right):
    from pops.time.canonical_data import _json_ready
    from pops._structural_data import equal
    return equal(_json_ready(left), _json_ready(right))
