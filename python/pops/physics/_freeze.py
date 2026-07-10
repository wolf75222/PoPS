"""Transactional deep-freeze protocol for Python-authored physics models."""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from functools import wraps
from types import MappingProxyType
from typing import Any

from pops.problem._freeze_transaction import (
    _FREEZE_CAPABILITY, _require_freeze_capability)


def _deep_freeze(value: Any) -> Any:
    """Freeze container structure without cloning symbolic/descriptor leaves."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _walk_values(value: Any) -> Any:
    """Yield nested authoring leaves once, traversing only ordinary containers."""
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value


_FROZEN_CLASSES: dict[type, type] = {}


def _frozen_class(base: type) -> type:
    """A layout-compatible read-only subclass for a persistent Module/registry record."""
    cached = _FROZEN_CLASSES.get(base)
    if cached is not None:
        return cached

    def reject(self: Any, name: str, value: Any) -> None:
        raise RuntimeError(
            "%s is frozen with its physics model; cannot set %r after Problem.freeze()"
            % (base.__name__, name))

    def reject_delete(self: Any, name: str) -> None:
        raise RuntimeError(
            "%s is frozen with its physics model; cannot delete %r after Problem.freeze()"
            % (base.__name__, name))

    def freeze(self: Any) -> Any:
        return self

    frozen = type(
        "_PoPSFrozen%s" % base.__name__, (base,),
        {"__slots__": (), "__setattr__": reject, "__delattr__": reject_delete,
         "freeze": freeze,
         "frozen": property(lambda self: True), "_pops_physics_frozen": True,
         "_pops_unfrozen_type": base,
         "__module__": base.__module__},
    )
    _FROZEN_CLASSES[base] = frozen
    return frozen


@dataclass(frozen=True)
class _ObjectSealPlan:
    target: Any
    target_class: type | None
    replacements: tuple[tuple[str, Any], ...]
    call_freeze: bool = False

    def commit(self) -> None:
        if self.target_class is not None:
            object.__setattr__(self.target, "__class__", self.target_class)
        for name, value in self.replacements:
            object.__setattr__(self.target, name, value)
        if self.call_freeze:
            self.target.freeze()


@dataclass(frozen=True)
class _MutableNodeSnapshot:
    target: Any
    kind: str
    state: Any

    def restore(self) -> None:
        if self.kind == "object":
            original_class, attrs = self.state
            if type(self.target) is not original_class:
                object.__setattr__(self.target, "__class__", original_class)
            vars(self.target).clear()
            vars(self.target).update(attrs)
        elif self.kind == "dict":
            self.target.clear()
            self.target.update(self.state)
        elif self.kind == "list":
            self.target[:] = self.state
        elif self.kind == "set":
            self.target.clear()
            self.target.update(self.state)
        elif self.kind == "bytearray":
            self.target[:] = self.state
        elif self.kind == "deque":
            self.target.clear()
            self.target.extend(self.state)


@dataclass(frozen=True)
class _MutableGraphSnapshot:
    """Identity-preserving snapshot of one descriptor-owned mutable graph."""

    nodes: tuple[_MutableNodeSnapshot, ...]

    def restore(self) -> None:
        # Parents are recorded before children.  Restore children first so mapping keys and
        # aliased values regain their original state before their parent containers are rebuilt.
        for node in reversed(self.nodes):
            node.restore()


def _mutable_graph_snapshot(root: Any, *, stop_ids: frozenset[int]) -> _MutableGraphSnapshot:
    """Capture nested Python mutables without cloning or breaking external aliases.

    A shallow ``dict(vars(descriptor))`` cannot roll back a ``freeze()`` implementation that
    mutates ``descriptor.payload.items`` in place before raising.  This graph snapshot records
    every ordinary Python object/container by identity and restores those same objects in place.
    Tuples, frozensets and mapping proxies are traversed because they may contain mutable leaves.
    """
    nodes: list[_MutableNodeSnapshot] = []
    seen = set(stop_ids)

    def visit(value: Any) -> None:
        ident = id(value)
        if ident in seen:
            return
        seen.add(ident)

        if isinstance(value, dict):
            items = tuple(value.items())
            nodes.append(_MutableNodeSnapshot(value, "dict", items))
            for key, item in items:
                visit(key)
                visit(item)
            return
        if isinstance(value, list):
            items = tuple(value)
            nodes.append(_MutableNodeSnapshot(value, "list", items))
            for item in items:
                visit(item)
            return
        if isinstance(value, set):
            items = tuple(value)
            nodes.append(_MutableNodeSnapshot(value, "set", items))
            for item in items:
                visit(item)
            return
        if isinstance(value, bytearray):
            nodes.append(_MutableNodeSnapshot(value, "bytearray", bytes(value)))
            return
        if isinstance(value, deque):
            items = tuple(value)
            nodes.append(_MutableNodeSnapshot(value, "deque", items))
            for item in items:
                visit(item)
            return
        if isinstance(value, (MappingProxyType, tuple, frozenset)):
            items = value.values() if isinstance(value, MappingProxyType) else value
            for item in items:
                visit(item)
            return
        if isinstance(value, (str, bytes, int, float, complex, bool, type(None), type)):
            return
        if hasattr(value, "__dict__"):
            attrs = tuple(vars(value).items())
            nodes.append(_MutableNodeSnapshot(value, "object", (type(value), attrs)))
            for _name, item in attrs:
                visit(item)

    visit(root)
    return _MutableGraphSnapshot(tuple(nodes))


@dataclass(frozen=True)
class _PhysicsFreezePlan:
    target: Any
    children: tuple[Any, ...]
    external: tuple[_ObjectSealPlan, ...]
    replacements: tuple[tuple[str, Any], ...]

    def commit(self) -> None:
        for child in self.children:
            child.commit()
        for plan in self.external:
            plan.commit()
        for name, value in self.replacements:
            object.__setattr__(self.target, name, value)
        object.__setattr__(self.target, "_frozen", True)


def _container_replacements(obj: Any, *, skip: frozenset[str] = frozenset()) -> Any:
    return tuple(
        (name, _deep_freeze(value))
        for name, value in vars(obj).items()
        if name not in skip and isinstance(value, (Mapping, list, tuple, set, frozenset))
    )


def _module_plan(module: Any) -> tuple[_ObjectSealPlan, ...]:
    """Seal one board-owned persistent ``pops.model.Module`` and its registry records."""
    registry = module.operator_registry()
    operators = list(registry)
    objects = operators + [registry, module]
    return tuple(
        _ObjectSealPlan(
            obj, None if getattr(type(obj), "_pops_physics_frozen", False)
            else _frozen_class(type(obj)), _container_replacements(obj))
        for obj in objects
    )


def _module_snapshot(module: Any) -> Any:
    registry = module.operator_registry()
    objects = list(registry) + [registry, module]
    return tuple((obj, type(obj), dict(vars(obj))) for obj in objects)


def _restore_objects(snapshot: Any) -> None:
    for obj, original_class, state in reversed(snapshot):
        object.__setattr__(obj, "__class__", original_class)
        vars(obj).clear()
        vars(obj).update(state)


class PhysicsFreezable:
    """Mixin providing an idempotent, transactional deep-freeze lifecycle."""

    _frozen = False
    _physics_mutators: frozenset[str] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for method_name in cls._physics_mutators:
            method = getattr(cls, method_name, None)
            if not callable(method) or getattr(method, "_pops_freeze_guarded", False):
                continue

            @wraps(method)
            def guarded(self: Any, *args: Any, __method: Any = method,
                        __name: str = method_name, **method_kwargs: Any) -> Any:
                self._guard_mutable("call authoring method %s()" % __name)
                result = __method(self, *args, **method_kwargs)
                invalidate = getattr(self, "_invalidate_authoring_views", None)
                if callable(invalidate):
                    invalidate()
                return result

            guarded._pops_freeze_guarded = True
            setattr(cls, method_name, guarded)

    def _init_physics_freeze(self) -> None:
        object.__setattr__(self, "_frozen", False)

    @property
    def frozen(self) -> bool:
        return bool(getattr(self, "_frozen", False))

    def _guard_mutable(self, what: Any = "mutate the model") -> None:
        if self.frozen:
            raise RuntimeError(
                "%s %r is frozen by Problem.freeze(); cannot %s. Author a fresh model and "
                "recompile." % (type(self).__name__, getattr(self, "name", "?"), what))

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            self._guard_mutable("set %r" % name)
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_frozen", False):
            self._guard_mutable("delete %r" % name)
        object.__delattr__(self, name)

    def _physics_children(self) -> tuple[Any, ...]:
        seen = set()
        children = []
        for value in vars(self).values():
            for leaf in _walk_values(value):
                if isinstance(leaf, PhysicsFreezable) and leaf is not self and id(leaf) not in seen:
                    seen.add(id(leaf))
                    children.append(leaf)
        return tuple(children)

    def _physics_modules(self) -> tuple[Any, ...]:
        from pops.model import Module
        return tuple(value for value in vars(self).values() if isinstance(value, Module))

    def _descriptor_plans(self, excluded: set[int]) -> tuple[_ObjectSealPlan, ...]:
        """Deep-seal contained descriptors that already expose a freeze protocol."""
        plans = []
        seen = set(excluded)
        for value in vars(self).values():
            for leaf in _walk_values(value):
                if id(leaf) in seen or not hasattr(leaf, "__dict__"):
                    continue
                freeze = getattr(leaf, "freeze", None)
                if not callable(freeze) or isinstance(leaf, PhysicsFreezable):
                    continue
                seen.add(id(leaf))
                plans.append(_ObjectSealPlan(
                    leaf, None, _container_replacements(leaf), call_freeze=True))
        return tuple(plans)

    def _prepare_physics_freeze(self) -> _PhysicsFreezePlan:
        if self.frozen:
            return _PhysicsFreezePlan(self, (), (), ())
        children = self._physics_children()
        child_plans = tuple(child._prepare_physics_freeze() for child in children)
        modules = self._physics_modules()
        module_plans = tuple(plan for module in modules for plan in _module_plan(module))
        excluded = {id(self), *(id(child) for child in children), *(id(m) for m in modules)}
        descriptors = self._descriptor_plans(excluded)
        replacements = _container_replacements(self, skip=frozenset({"_frozen"}))
        return _PhysicsFreezePlan(self, child_plans, module_plans + descriptors, replacements)

    def freeze(self) -> Any:
        """Deep-freeze authoring containers and return ``self``; idempotent."""
        if self.frozen:
            return self
        snapshot = self._pops_freeze_snapshot(_FREEZE_CAPABILITY)
        try:
            plan = self._prepare_physics_freeze()
            plan.commit()
        except BaseException:
            self._pops_freeze_restore(_FREEZE_CAPABILITY, snapshot)
            raise
        return self

    def _pops_freeze_snapshot(self, capability: Any) -> Any:
        """Private Problem transaction hook capturing this complete physics cascade."""
        _require_freeze_capability(capability)
        children = tuple((child, child._pops_freeze_snapshot(capability))
                         for child in self._physics_children())
        modules = tuple((module, _module_snapshot(module)) for module in self._physics_modules())
        descriptors = []
        seen = {id(self), *(id(child) for child, _ in children), *(id(m) for m, _ in modules)}
        for value in vars(self).values():
            for leaf in _walk_values(value):
                if id(leaf) in seen or not hasattr(leaf, "__dict__"):
                    continue
                if callable(getattr(leaf, "freeze", None)):
                    snapshot = _mutable_graph_snapshot(leaf, stop_ids=frozenset(seen))
                    descriptors.append(snapshot)
                    seen.update(id(node.target) for node in snapshot.nodes)
        return {"self": dict(vars(self)), "children": children,
                "modules": modules, "descriptors": tuple(descriptors)}

    def _pops_freeze_restore(self, capability: Any, state: Any) -> None:
        """Private Problem rollback hook paired with :meth:`_pops_freeze_snapshot`."""
        _require_freeze_capability(capability)
        for snapshot in reversed(state["descriptors"]):
            snapshot.restore()
        for _module, snapshot in reversed(state["modules"]):
            _restore_objects(snapshot)
        for child, snapshot in reversed(state["children"]):
            child._pops_freeze_restore(capability, snapshot)
        vars(self).clear()
        vars(self).update(state["self"])


__all__ = ["PhysicsFreezable"]
