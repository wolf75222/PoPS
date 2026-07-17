"""Strict structural identities used by :class:`pops.model.Module` hashes."""
from __future__ import annotations

import inspect
import json
import math
import types
from collections.abc import Mapping
from decimal import Decimal
from fractions import Fraction
from typing import Any


def canonical_hash_data(value: Any, *, where: str = "module hash") -> Any:
    """Return deterministic JSON data, refusing every opaque/address repr fallback."""
    hook = getattr(value, "to_data", None)
    if callable(hook):
        return canonical_hash_data(hook(), where=where)

    if _is_expr(value):
        from pops._ir.visitors import _key

        return {
            "protocol": "pops.expr.key.v1",
            "value": canonical_hash_data(_key(value), where="%s expression" % where),
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("%s mapping keys must be non-empty strings" % where)
        return {
            key: canonical_hash_data(value[key], where="%s.%s" % (where, key))
            for key in sorted(value)
        }
    if isinstance(value, (tuple, list)):
        return [canonical_hash_data(item, where=where) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [canonical_hash_data(item, where=where) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )
    if isinstance(value, (Fraction, Decimal)):
        from pops._ir.literals import scalar_literal

        return {"protocol": "pops.scalar.v1", "value": scalar_literal(value).to_data()}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("%s contains a non-finite float" % where)
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(
        "%s contains opaque %s; provide strict JSON data or implement to_data()"
        % (where, type(value).__name__))


def body_identity(body: Any) -> Any:
    """Stable structural identity of data, functions, methods and callable objects."""
    return _body_identity(body, frozenset())


def _body_identity(body: Any, active: frozenset[int]) -> Any:
    """Recursive callable identity with explicit cycle references."""
    if id(body) in active:
        return _callable_reference(body)
    nested = active | {id(body)}
    if body is None:
        return {"kind": "none"}
    if inspect.isfunction(body):
        return _function_identity(body, nested)
    if inspect.ismethod(body):
        return {
            "kind": "bound_method",
            "function": _function_identity(body.__func__, nested),
            "self": _instance_identity(body.__self__, where="bound method owner"),
        }
    if inspect.isbuiltin(body):
        owner = getattr(body, "__self__", None)
        result = {
            "kind": "builtin",
            "module": getattr(body, "__module__", None),
            "qualname": getattr(body, "__qualname__", getattr(body, "__name__", None)),
        }
        if owner is not None and not inspect.ismodule(owner):
            result["self"] = _instance_identity(owner, where="builtin owner")
        return canonical_hash_data(result, where="builtin callable identity")
    if callable(body):
        if isinstance(body, type):
            raise TypeError("operator body classes are opaque; pass an instance or strict data")
        implementation = getattr(type(body), "__call__", None)
        if not inspect.isfunction(implementation):
            raise TypeError(
                "callable operator body %s has no stable Python __call__ code"
                % type(body).__name__)
        return {
            "kind": "callable_instance",
            "class": _type_identity(type(body)),
            "call": _function_identity(implementation, nested),
            "state": _instance_state(body, where="callable operator body"),
        }
    return {
        "kind": "data",
        "value": canonical_hash_data(body, where="operator body"),
    }


def _is_expr(value: Any) -> bool:
    try:
        from pops._ir.expr import Expr
    except ImportError:
        return False
    return isinstance(value, Expr)


def _type_identity(cls: type[Any]) -> Any:
    module = getattr(cls, "__module__", None)
    qualname = getattr(cls, "__qualname__", None)
    if not isinstance(module, str) or not module or not isinstance(qualname, str) or not qualname:
        raise TypeError("callable body type has no stable module/qualname identity")
    return {"module": module, "qualname": qualname}


def _function_identity(function: Any, active: frozenset[int]) -> Any:
    module = getattr(function, "__module__", None)
    qualname = getattr(function, "__qualname__", None)
    code = getattr(function, "__code__", None)
    if (not isinstance(module, str) or not module or not isinstance(qualname, str)
            or not qualname or not isinstance(code, types.CodeType)):
        raise TypeError("operator body function has no stable module/qualname/code identity")
    closure = []
    for index, cell in enumerate(function.__closure__ or ()):
        try:
            content = cell.cell_contents
        except ValueError as exc:
            raise TypeError("operator body closure cell %d is empty" % index) from exc
        closure.append(canonical_hash_data(
            content, where="operator body closure[%d]" % index))
    globals_data = {}
    for name in sorted(set(code.co_names)):
        if name not in function.__globals__:
            continue
        globals_data[name] = _global_identity(
            function.__globals__[name], active,
            where="operator body global %r" % name,
        )
    return {
        "kind": "function",
        "module": module,
        "qualname": qualname,
        "code": _code_identity(code),
        "defaults": canonical_hash_data(
            function.__defaults__, where="operator body defaults"),
        "kwdefaults": canonical_hash_data(
            function.__kwdefaults__, where="operator body keyword defaults"),
        "closure": closure,
        "globals": globals_data,
    }


def _global_identity(value: Any, active: frozenset[int], *, where: str) -> Any:
    """Identity for a name actually loaded from a function's global namespace."""
    if inspect.ismodule(value):
        name = getattr(value, "__name__", None)
        if not isinstance(name, str) or not name:
            raise TypeError("%s module has no stable name" % where)
        return {"kind": "module", "name": name}
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
        return {"kind": "callable", "value": _body_identity(value, active)}
    if callable(value):
        if isinstance(value, type):
            return {"kind": "type", "value": _type_identity(value)}
        return {"kind": "callable", "value": _body_identity(value, active)}
    return {"kind": "data", "value": canonical_hash_data(value, where=where)}


def _callable_reference(value: Any) -> Any:
    """Stable back-edge used only when callable dependency graphs are recursive."""
    if inspect.ismethod(value):
        value = value.__func__
    if inspect.isfunction(value) or inspect.isbuiltin(value):
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", getattr(value, "__name__", None))
        if not isinstance(module, str) or not module \
                or not isinstance(qualname, str) or not qualname:
            raise TypeError("recursive callable has no stable module/qualname identity")
        return {"kind": "callable_reference", "module": module, "qualname": qualname}
    if callable(value) and not isinstance(value, type):
        return {"kind": "callable_instance_reference", "class": _type_identity(type(value))}
    raise TypeError("recursive operator body dependency is not a supported callable")


def _code_identity(code: types.CodeType) -> Any:
    return {
        "bytecode": code.co_code.hex(),
        "constants": [_code_constant(item) for item in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "flags": code.co_flags,
    }


def _code_constant(value: Any) -> Any:
    if isinstance(value, types.CodeType):
        return {"code": _code_identity(value)}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if value is Ellipsis:
        return {"ellipsis": True}
    return canonical_hash_data(value, where="operator body code constant")


def _instance_state(instance: Any, *, where: str) -> Any:
    state = {}
    dictionary = getattr(instance, "__dict__", None)
    if dictionary is not None:
        if not isinstance(dictionary, Mapping):
            raise TypeError("%s exposes a non-mapping __dict__" % where)
        state.update(dictionary)
    for cls in type(instance).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            storage = _slot_storage_name(cls, slot)
            if storage in ("__dict__", "__weakref__") or storage in state:
                continue
            try:
                state[storage] = getattr(instance, storage)
            except AttributeError:
                continue
    return canonical_hash_data(state, where="%s state" % where)


def _slot_storage_name(cls: type[Any], slot: str) -> str:
    """Apply Python's private-slot name mangling exactly enough for attribute lookup."""
    if slot.startswith("__") and not slot.endswith("__"):
        class_name = cls.__name__.lstrip("_")
        if class_name:
            return "_%s%s" % (class_name, slot)
    return slot


def _instance_identity(instance: Any, *, where: str) -> Any:
    if inspect.ismodule(instance):
        name = getattr(instance, "__name__", None)
        if not isinstance(name, str) or not name:
            raise TypeError("%s module has no stable name" % where)
        return {"kind": "module", "name": name}
    return {"class": _type_identity(type(instance)), "state": _instance_state(instance, where=where)}


__all__ = ["body_identity", "canonical_hash_data"]
