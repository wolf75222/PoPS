"""All-or-nothing freeze transactions for a :class:`pops.Problem` object graph.

Every participant is snapshotted before the first ``freeze()`` call.  PoPS Python objects are
restored by identity (attribute bindings and built-in mutable containers); native or third-party
objects must expose the explicit capability-gated private snapshot/restore hooks.  An opaque
irreversible freezer is rejected during preflight, before any member can be sealed.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable


# A process-local authority held only by the freeze coordinator.  Python cannot provide absolute
# secrecy, but identity checking makes rollback unavailable through the ordinary object API and
# prevents an accidental or guessed call from reversing an irreversible public freeze().
_FREEZE_CAPABILITY = object()


def _require_freeze_capability(capability: Any) -> None:
    if capability is not _FREEZE_CAPABILITY:
        raise RuntimeError("freeze rollback requires the private PoPS transaction capability")


class _PythonFreezeState:
    """Identity-preserving state of one trusted PoPS Python object."""

    def __init__(self, value: Any) -> None:
        self.value = value
        self.has_dict = hasattr(value, "__dict__")
        self.attributes = dict(vars(value)) if self.has_dict else {}
        self.containers: list[tuple[Any, str, Any]] = []
        self.seen: set[int] = set()
        self.slots: list[tuple[str, bool, Any]] = []
        for item in self.attributes.values():
            self._visit(item)
        for name in _slot_names(type(value)):
            present = hasattr(value, name)
            item = getattr(value, name) if present else None
            self.slots.append((name, present, item))
            if present:
                self._visit(item)
        if not self.has_dict and not self.slots:
            raise TypeError(
                "freeze transaction cannot restore opaque %s; provide private "
                "_pops_freeze_snapshot(capability) / "
                "_pops_freeze_restore(capability, state) hooks"
                % _qualified_type(value))

    def _visit(self, value: Any) -> None:
        marker = id(value)
        if marker in self.seen:
            return
        if isinstance(value, list):
            self.seen.add(marker)
            items = tuple(value)
            self.containers.append((value, "list", items))
            for item in items:
                self._visit(item)
        elif isinstance(value, dict):
            self.seen.add(marker)
            items = tuple(value.items())
            self.containers.append((value, "dict", items))
            for key, item in items:
                self._visit(key)
                self._visit(item)
        elif isinstance(value, set):
            self.seen.add(marker)
            items = frozenset(value)
            self.containers.append((value, "set", items))
            for item in items:
                self._visit(item)
        elif isinstance(value, (tuple, frozenset)):
            self.seen.add(marker)
            for item in value:
                self._visit(item)

    def restore(self) -> None:
        for container, kind, contents in reversed(self.containers):
            if kind == "list":
                container[:] = contents
            elif kind == "dict":
                container.clear()
                container.update(contents)
            else:
                container.clear()
                container.update(contents)
        if self.has_dict:
            state = object.__getattribute__(self.value, "__dict__")
            state.clear()
            state.update(self.attributes)
        for name, present, item in self.slots:
            if present:
                object.__setattr__(self.value, name, item)
            elif hasattr(self.value, name):
                object.__delattr__(self.value, name)


class _HookFreezeState:
    def __init__(self, value: Any, snapshot: Any, restore: Any) -> None:
        self.value = value
        self.restore_hook = restore
        self.state = snapshot(_FREEZE_CAPABILITY)

    def restore(self) -> None:
        self.restore_hook(_FREEZE_CAPABILITY, self.state)


def freeze_atomically(participants: Iterable[Any], commit: Callable[[], Any]) -> Any:
    """Preflight, run ``commit``, and restore every participant if any freeze fails."""
    states = [_capture(value) for value in _unique_freezable(participants)]
    try:
        return commit()
    except BaseException as failure:
        rollback_errors = []
        for state in reversed(states):
            try:
                state.restore()
            except BaseException as error:  # rollback failure is a broken internal protocol
                rollback_errors.append(error)
        if rollback_errors:
            raise RuntimeError(
                "Problem freeze failed and rollback could not restore every participant: %s"
                % "; ".join(str(error) for error in rollback_errors)) from failure
        raise


def freeze_problem_graph(problem: Any) -> None:
    """Freeze every registry/member/layout in one transaction."""
    registries = (
        problem._block_registry, problem._field_registry, problem._time_registry,
        problem._param_registry, problem._initial_registry,
    )
    participants = []
    for registry in registries:
        participants.extend(registry._freezable_members())
        participants.append(registry)
    numerical_plans = tuple(dict.fromkeys(problem._numerics_assignments.values()))
    participants.extend(numerical_plans)

    def commit() -> None:
        for registry in registries:
            registry.freeze()
        for plan in numerical_plans:
            plan.freeze()

    freeze_atomically(participants, commit)


def _capture(value: Any) -> Any:
    snapshot = getattr(value, "_pops_freeze_snapshot", None)
    restore = getattr(value, "_pops_freeze_restore", None)
    if snapshot is not None or restore is not None:
        if not callable(snapshot) or not callable(restore):
            raise TypeError(
                "%s must provide both _pops_freeze_snapshot(capability) and "
                "_pops_freeze_restore(capability, state)"
                % _qualified_type(value))
        return _HookFreezeState(value, snapshot, restore)
    if not _trusted_python_freezer(value):
        raise TypeError(
            "freeze transaction refuses opaque non-transactional %s before mutating any member; "
            "implement the private PoPS freeze snapshot/restore hooks" % _qualified_type(value))
    return _PythonFreezeState(value)


def _trusted_python_freezer(value: Any) -> bool:
    if type(value).__module__.startswith("pops."):
        return True
    try:
        from pops._descriptor_protocol import Descriptor
        from pops.descriptors import BrickDescriptor
        return isinstance(value, (Descriptor, BrickDescriptor))
    except ImportError:
        return False


def _unique_freezable(values: Iterable[Any]) -> list[Any]:
    unique = []
    seen = set()
    for value in values:
        if value is None or not callable(getattr(value, "freeze", None)):
            continue
        marker = id(value)
        if marker not in seen:
            seen.add(marker)
            unique.append(value)
    return unique


def _slot_names(cls: type) -> tuple[str, ...]:
    names = []
    for owner in cls.__mro__:
        slots = owner.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name in ("__dict__", "__weakref__"):
                continue
            if name.startswith("__") and not name.endswith("__"):
                name = "_%s%s" % (owner.__name__.lstrip("_"), name)
            if name not in names:
                names.append(name)
    return tuple(names)


def _qualified_type(value: Any) -> str:
    cls = type(value)
    return "%s.%s" % (cls.__module__, cls.__qualname__)


__all__ = ["freeze_atomically", "freeze_problem_graph"]
