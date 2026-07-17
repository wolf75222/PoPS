"""Stable, bounded projections of Python executable dependencies."""
from __future__ import annotations

import dis
import hashlib
import json
import math
import sys
import types
from typing import Any

from pops.problem._snapshot_module_dependency import (
    framework_dependency_projection as _framework_dependency_projection,
    is_cross_module_framework_dependency as _is_cross_module_framework_dependency,
    module_dependency_projection,
)
from pops.problem._snapshot_module_fingerprint import module_implementation_fingerprint


def callable_projection(
    value: Any,
    *,
    path: str,
    active: set[int],
    handle_resolver: Any,
    artifact: bool,
    canonical: Any,
    dependency_cache: dict[tuple[int, str], Any],
) -> dict[str, Any]:
    """Build an address-free digest of one callable and all dependencies it reads."""
    function = value.__func__ if isinstance(value, types.MethodType) else value
    result: dict[str, Any] = {
        "module": getattr(function, "__module__", None),
        "qualname": getattr(function, "__qualname__", getattr(function, "__name__", None)),
    }
    bound_self = getattr(value, "__self__", None)
    if bound_self is not None:
        if isinstance(bound_self, types.ModuleType):
            result["bound_self"] = {
                "module": module_implementation_fingerprint(bound_self.__name__, path=path)
            }
        elif isinstance(bound_self, type):
            result["bound_self"] = class_implementation_projection(
                bound_self,
                path=path + ".__self__",
                active=active,
                handle_resolver=handle_resolver,
                artifact=artifact,
                canonical=canonical,
                dependency_cache=dependency_cache,
            )
        else:
            result["bound_self"] = canonical(
                bound_self, path=path + ".__self__", active=active,
                handle_resolver=handle_resolver, artifact=artifact)
    code = getattr(function, "__code__", None)
    if code is None:
        result["kind"] = "builtin"
        implementation_module = getattr(function, "__module__", None)
        if not implementation_module and bound_self is not None:
            implementation_module = (
                bound_self.__module__ if isinstance(bound_self, type)
                else type(bound_self).__module__
            )
        result["implementation"] = module_implementation_fingerprint(
            implementation_module, path=path)
        return result
    result["kind"] = "python"
    result["code"] = code_projection(code)
    result["defaults"] = canonical(
        getattr(function, "__defaults__", None), path=path + ".__defaults__", active=active,
        handle_resolver=handle_resolver, artifact=artifact)
    result["kwdefaults"] = canonical(
        getattr(function, "__kwdefaults__", None), path=path + ".__kwdefaults__", active=active,
        handle_resolver=handle_resolver, artifact=artifact)
    result["annotations"] = canonical(
        getattr(function, "__annotations__", None), path=path + ".__annotations__", active=active,
        handle_resolver=handle_resolver, artifact=artifact)
    result["attributes"] = canonical(
        getattr(function, "__dict__", None), path=path + ".__dict__", active=active,
        handle_resolver=handle_resolver, artifact=artifact)
    closure_values = []
    for index, cell in enumerate(getattr(function, "__closure__", None) or ()):
        try:
            content = cell.cell_contents
        except ValueError:
            closure_values.append({"state": "empty"})
            continue
        closure_values.append({
            "state": "value",
            "value": canonical(
                content, path="%s.__closure__[%d]" % (path, index), active=active,
                handle_resolver=handle_resolver, artifact=artifact),
        })
    result["closure"] = closure_values
    globals_table = getattr(function, "__globals__", {})
    builtins_table = getattr(function, "__builtins__", {})
    if isinstance(builtins_table, types.ModuleType):
        builtins_table = vars(builtins_table)
    module_paths = _module_attribute_paths(code)
    referenced_globals = {}
    referenced_builtins = {}
    for name in _referenced_global_names(code):
        if name in referenced_globals:
            continue
        if name not in globals_table:
            if name in builtins_table:
                referenced_builtins[name] = canonical(
                    builtins_table[name], path="%s.__builtins__.%s" % (path, name),
                    active=active, handle_resolver=handle_resolver, artifact=artifact)
            continue
        item = globals_table[name]
        if isinstance(item, types.ModuleType):
            paths = module_paths.get(name, ())
            if not paths:
                raise TypeError(
                    "AuthoringSnapshot cannot structurally project module global %s at %s: "
                    "the module is used as a value rather than through explicit attributes"
                    % (name, path))
            referenced_globals[name] = module_dependency_projection(
                item,
                attribute_paths=paths,
                path="%s.__globals__.%s" % (path, name),
                active=active,
                handle_resolver=handle_resolver,
                artifact=artifact,
                canonical=canonical,
            )
            continue
        if _is_cross_module_framework_dependency(function, item):
            referenced_globals[name] = _framework_dependency_projection(
                item, path="%s.__globals__.%s" % (path, name))
            continue
        if isinstance(item, type):
            referenced_globals[name] = class_implementation_projection(
                item,
                path="%s.__globals__.%s" % (path, name),
                active=active,
                handle_resolver=handle_resolver,
                artifact=artifact,
                canonical=canonical,
                dependency_cache=dependency_cache,
            )
            continue
        if id(item) in active:
            if not isinstance(
                    item, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
                raise ValueError(
                    "AuthoringSnapshot callable global %s contains a non-callable reference cycle"
                    % name)
            referenced_globals[name] = {"recursive_callable": callable_anchor(item)}
            continue
        referenced_globals[name] = canonical(
            item, path="%s.__globals__.%s" % (path, name), active=active,
            handle_resolver=handle_resolver, artifact=artifact)
    result["globals"] = referenced_globals
    result["builtins"] = referenced_builtins
    return result


def canonical_callable_reference(
    value: Any,
    *,
    path: str,
    active: set[int],
    handle_resolver: Any,
    artifact: bool,
    canonical: Any,
    dependency_cache: dict[tuple[int, str], Any],
) -> dict[str, Any]:
    """Canonicalize one callable as a compact, memoized structural digest."""
    marker = id(value)
    if marker in active:
        return {"$recursive_callable": callable_anchor(value)}
    cache_key = (marker, "callable:%d" % artifact)
    cached = dependency_cache.get(cache_key)
    if cached is not None and cached[0] is value:
        return cached[1]
    active.add(marker)
    try:
        projection = callable_projection(
            value,
            path=path,
            active=active,
            handle_resolver=handle_resolver,
            artifact=artifact,
            canonical=canonical,
            dependency_cache=dependency_cache,
        )
        result = {"$callable": {
            "identity": {
                "module": projection.get("module"),
                "qualname": projection.get("qualname"),
                "kind": projection.get("kind"),
            },
            "sha256": _digest(projection),
        }}
        dependency_cache[cache_key] = (value, result)
        return result
    finally:
        active.remove(marker)


def code_projection(code: types.CodeType) -> dict[str, Any]:
    """Address/path-free structural projection of a Python code object."""
    result = {
        "name": code.co_name,
        "qualname": getattr(code, "co_qualname", code.co_name),
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "bytecode": code.co_code.hex(),
        "consts": [code_constant(item) for item in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }
    exception_table = getattr(code, "co_exceptiontable", None)
    if exception_table is not None:
        result["exceptiontable"] = exception_table.hex()
    return result


def code_constant(value: Any) -> Any:
    if isinstance(value, types.CodeType):
        return {"code": code_projection(value)}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("AuthoringSnapshot callable contains a non-finite float constant")
        return {"float": value.hex()}
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("AuthoringSnapshot callable contains a non-finite complex constant")
        return {"complex": [value.real.hex(), value.imag.hex()]}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, tuple):
        return {"tuple": [code_constant(item) for item in value]}
    if isinstance(value, frozenset):
        items = [code_constant(item) for item in value]
        return {"frozenset": sorted(items, key=_json_key)}
    if value is Ellipsis:
        return {"ellipsis": True}
    raise TypeError(
        "AuthoringSnapshot callable contains unsupported code constant %s.%s"
        % (type(value).__module__, type(value).__qualname__))


def class_implementation_projection(
    cls: type,
    *,
    path: str,
    active: set[int],
    handle_resolver: Any,
    artifact: bool,
    canonical: Any,
    dependency_cache: dict[tuple[int, str], Any],
) -> dict[str, Any]:
    """Digest Python and native behaviour across the complete class MRO."""
    cache_key = (id(cls), "class:%d" % artifact)
    cached = dependency_cache.get(cache_key)
    if cached is not None and cached[0] is cls:
        return cached[1]
    if _is_native_class(cls) or _is_stdlib_class(cls):
        result = {
            "type": "%s.%s" % (cls.__module__, cls.__qualname__),
            "native_implementation": module_implementation_fingerprint(cls.__module__, path=path),
        }
        dependency_cache[cache_key] = (cls, result)
        return result
    if id(cls) in active:
        return {"recursive_class": class_anchor(cls)}
    active.add(id(cls))
    try:
        bases = []
        for base in cls.__mro__:
            if base is object:
                continue
            if base is not cls and (_is_native_class(base) or _is_stdlib_class(base)):
                bases.append({
                    "type": "%s.%s" % (base.__module__, base.__qualname__),
                    "implementation": module_implementation_fingerprint(
                        base.__module__, path=path),
                })
                continue
            members = {}
            for name, member in sorted(base.__dict__.items()):
                if name.startswith("_") and name != "__call__":
                    continue
                projected = _class_member_projection(
                    member,
                    path="%s.%s.%s" % (path, base.__qualname__, name),
                    owner_module=base.__module__,
                    active=active,
                    handle_resolver=handle_resolver,
                    artifact=artifact,
                    canonical=canonical,
                    dependency_cache=dependency_cache,
                )
                if projected is not None:
                    members[name] = projected
            bases.append({
                "type": "%s.%s" % (base.__module__, base.__qualname__),
                "members": members,
            })
        metaclass = type(cls)
        metaclass_projection = {
            "type": "%s.%s" % (metaclass.__module__, metaclass.__qualname__),
            "implementation": module_implementation_fingerprint(metaclass.__module__, path=path),
            "local_code": class_anchor(metaclass),
        }
        full = {
            "type": "%s.%s" % (cls.__module__, cls.__qualname__),
            "metaclass": metaclass_projection,
            "mro": bases,
        }
        result = {"type": full["type"], "sha256": _digest(full)}
        dependency_cache[cache_key] = (cls, result)
        return result
    finally:
        active.remove(id(cls))


def _class_member_projection(member: Any, **context: Any) -> Any:
    if isinstance(member, staticmethod):
        return {"staticmethod": _canonical_member(member.__func__, **context)}
    if isinstance(member, classmethod):
        return {"classmethod": _canonical_member(member.__func__, **context)}
    if isinstance(member, property):
        return {"property": {
            "get": _canonical_member(member.fget, **context),
            "set": _canonical_member(member.fset, **context),
            "delete": _canonical_member(member.fdel, **context),
        }}
    descriptor_types = tuple(
        item for item in (
            getattr(types, "MethodDescriptorType", None),
            getattr(types, "WrapperDescriptorType", None),
            getattr(types, "GetSetDescriptorType", None),
            getattr(types, "MemberDescriptorType", None),
        ) if item is not None)
    if isinstance(member, descriptor_types):
        implementation_module = context["owner_module"]
        if isinstance(member, getattr(types, "MemberDescriptorType", ())):
            # Python ``__slots__`` use the interpreter's generic member descriptor machinery; the
            # surrounding class projection already captures the slot name and owning class.
            implementation_module = "builtins"
        return {"native_descriptor": {
            "type": "%s.%s" % (type(member).__module__, type(member).__qualname__),
            "name": getattr(member, "__name__", None),
            "implementation": module_implementation_fingerprint(
                implementation_module, path=context["path"]),
        }}
    if isinstance(member, (types.BuiltinFunctionType, types.BuiltinMethodType)):
        return {"native_callable": {
            "name": getattr(member, "__name__", None),
            "qualname": getattr(member, "__qualname__", None),
            "implementation": module_implementation_fingerprint(
                context["owner_module"], path=context["path"]),
        }}
    return _canonical_member(member, **context)


def _canonical_member(member: Any, **context: Any) -> Any:
    if member is None:
        return None
    return context["canonical"](
        member,
        path=context["path"],
        active=context["active"],
        handle_resolver=context["handle_resolver"],
        artifact=context["artifact"],
    )


def callable_anchor(value: Any) -> str:
    function = value.__func__ if isinstance(value, types.MethodType) else value
    code = getattr(function, "__code__", None)
    local = {
        "module": getattr(function, "__module__", None),
        "qualname": getattr(function, "__qualname__", getattr(function, "__name__", None)),
        "code": code_projection(code) if isinstance(code, types.CodeType) else None,
    }
    return _digest(local)


def class_anchor(cls: type) -> str:
    methods = {}
    for name, member in sorted(cls.__dict__.items()):
        if isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        elif isinstance(member, property):
            member = member.fget
        code = getattr(member, "__code__", None)
        if isinstance(code, types.CodeType):
            methods[name] = code_projection(code)
    return _digest({
        "type": "%s.%s" % (cls.__module__, cls.__qualname__),
        "mro": ["%s.%s" % (base.__module__, base.__qualname__) for base in cls.__mro__],
        "methods": methods,
    })


def _is_native_class(cls: type) -> bool:
    """Whether behaviour is authenticated by a runtime/source binary rather than Python code."""
    for base in cls.__mro__:
        for member in base.__dict__.values():
            if isinstance(member, (staticmethod, classmethod)):
                member = member.__func__
            elif isinstance(member, property):
                members = (member.fget, member.fset, member.fdel)
                if any(isinstance(getattr(item, "__code__", None), types.CodeType)
                       for item in members if item is not None):
                    return False
                continue
            if isinstance(getattr(member, "__code__", None), types.CodeType):
                return False
    return True


def _is_stdlib_class(cls: type) -> bool:
    """Whether a class is authenticated by its loaded standard-library module.

    Walking every public member of a stdlib class recursively follows implementation globals such
    as ``json.JSONEncoder``'s singleton encoder and ``Enum``'s ``str`` base.  Those are not authoring
    state: the module source (or interpreter build for builtins/frozen modules) is the complete,
    bounded implementation identity.
    """
    module_name = getattr(cls, "__module__", None)
    if not isinstance(module_name, str) or not module_name:
        return False
    return module_name.split(".", 1)[0] in sys.stdlib_module_names


def class_has_public_behavior(cls: type) -> bool:
    """Whether an otherwise stateless instance exposes authored executable behaviour."""
    for base in cls.__mro__:
        if base is object:
            continue
        for name, member in base.__dict__.items():
            if name.startswith("_") and name != "__call__":
                continue
            if isinstance(member, (staticmethod, classmethod, property)) or callable(member):
                return True
    return False


def _module_attribute_paths(code: types.CodeType) -> dict[str, tuple[tuple[str, ...], ...]]:
    found: dict[str, set[tuple[str, ...]]] = {}
    instructions = list(dis.get_instructions(code))
    for index, instruction in enumerate(instructions):
        if instruction.opname not in {"LOAD_GLOBAL", "LOAD_NAME"}:
            continue
        parts = []
        for following in instructions[index + 1:]:
            if following.opname in {"LOAD_ATTR", "LOAD_METHOD"}:
                parts.append(str(following.argval))
                continue
            break
        if parts:
            found.setdefault(str(instruction.argval), set()).add(tuple(parts))
    return {name: tuple(sorted(paths)) for name, paths in found.items()}


def _referenced_global_names(code: types.CodeType) -> tuple[str, ...]:
    names = []
    seen = set()
    for instruction in dis.get_instructions(code):
        if instruction.opname not in {"LOAD_GLOBAL", "LOAD_NAME"}:
            continue
        name = str(instruction.argval)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return tuple(names)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_key(value).encode("utf-8")).hexdigest()


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=False, separators=(",", ":"), allow_nan=False)


__all__ = [
    "canonical_callable_reference",
    "class_has_public_behavior",
    "class_implementation_projection",
]
