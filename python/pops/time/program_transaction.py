"""Transactional guards for callback-driven Program authoring.

An authoring callback is allowed to allocate SSA ids, create region tokens and
replace immutable value records before it eventually fails validation.  Those
intermediate mutations must not leak into the Program: a retry must observe the
same ids, values and regions it would have observed had the failed call never
happened.

The snapshot deliberately preserves container identity.  Region bookkeeping
uses ``id(sub_block)`` as a key, so replacing a saved list with a copied list
would itself corrupt the restored state.  Instead, every pre-existing built-in
mutable container is restored in place and the Program's original attribute
bindings are reinstated.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator


class _AuthoringSnapshot:
    """Identity-preserving snapshot of one Program's Python authoring state."""

    def __init__(self, program: Any) -> None:
        self._attributes = dict(program.__dict__)
        self._containers: list[tuple[Any, str, Any]] = []
        self._seen: set[int] = set()
        for value in self._attributes.values():
            self._visit(value)

    def _visit(self, value: Any) -> None:
        marker = id(value)
        if marker in self._seen:
            return
        if isinstance(value, list):
            self._seen.add(marker)
            items = tuple(value)
            self._containers.append((value, "list", items))
            for item in items:
                self._visit(item)
            return
        if isinstance(value, dict):
            self._seen.add(marker)
            items = tuple(value.items())
            self._containers.append((value, "dict", items))
            for key, item in items:
                self._visit(key)
                self._visit(item)
            return
        if isinstance(value, set):
            self._seen.add(marker)
            items = frozenset(value)
            self._containers.append((value, "set", items))
            for item in items:
                self._visit(item)
            return
        if isinstance(value, (tuple, frozenset)):
            self._seen.add(marker)
            for item in value:
                self._visit(item)

    def restore(self, program: Any) -> None:
        """Restore all original containers and top-level attribute bindings."""
        for container, kind, contents in reversed(self._containers):
            if kind == "list":
                container[:] = contents
            elif kind == "dict":
                container.clear()
                container.update(contents)
            else:
                container.clear()
                container.update(contents)
        program.__dict__.clear()
        program.__dict__.update(self._attributes)


@contextmanager
def authoring_transaction(program: Any) -> Iterator[None]:
    """Roll back every Program authoring mutation if the guarded work fails."""
    snapshot = _AuthoringSnapshot(program)
    try:
        yield
    except BaseException:
        snapshot.restore(program)
        raise


def atomic_authoring(function: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate a Program method so any exception leaves its Program unchanged."""
    @wraps(function)
    def guarded(program: Any, *args: Any, **kwargs: Any) -> Any:
        with authoring_transaction(program):
            return function(program, *args, **kwargs)

    return guarded


__all__ = ["atomic_authoring", "authoring_transaction"]
