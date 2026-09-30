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

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any


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

    def require_readonly_query(self, program: Any) -> None:
        """Allow query allocation only; preserve publication and temporal metadata.

        Check saved contents rather than comparing mutable aliases or ProgramValue
        equality. Unknown/future attributes are protected by default.
        """
        append_maps = frozenset({
            "_issued_values", "_recording_regions", "_state_spaces", "_operator_registries",
            "_time_states", "_time_current_values", "_time_endpoint_handles",
        })
        counters = frozenset({"_next_id", "_next_region"})
        saved = {id(container): (kind, contents)
                 for container, kind, contents in self._containers}
        seen: set[int] = set()

        def fail(name: str) -> None:
            raise ValueError("dt_bound read-only callback changed authoring metadata %s" % name)

        def check(value: Any, name: str) -> None:
            marker = id(value)
            if marker in seen:
                return
            seen.add(marker)
            entry = saved.get(marker)
            if entry is None:
                if isinstance(value, (tuple, frozenset)):
                    for item in value:
                        check(item, name)
                return
            kind, contents = entry
            if kind == "dict":
                if len(value) != len(contents) or any(
                        key not in value or value[key] is not item for key, item in contents):
                    fail(name)
                children = (item for pair in contents for item in pair)
            elif kind == "list":
                if len(value) != len(contents) or any(
                        actual is not prior for actual, prior in zip(value, contents, strict=True)):
                    fail(name)
                children = iter(contents)
            else:
                if {id(item) for item in value} != {id(item) for item in contents}:
                    fail(name)
                children = iter(contents)
            for child in children:
                check(child, name)

        if program.__dict__.keys() != self._attributes.keys():
            fail("attribute set")
        for name, original in self._attributes.items():
            current = program.__dict__[name]
            if name in counters:
                if type(current) is not int or current < original:
                    fail(name)
                continue
            # First read of a case-owned state may establish its live Case authority.
            if name == "_case_owner_path" and original is None:
                continue
            if current is not original:
                fail(name)
            if name in append_maps:
                _, contents = saved[id(original)]
                if any(key not in current or current[key] is not item for key, item in contents):
                    fail(name)
                # New entries are query allocations; old nested regions remain protected.
                for key, item in contents:
                    check(key, name)
                    check(item, name)
            else:
                check(original, name)


@contextmanager
def authoring_transaction(program: Any) -> Iterator[None]:
    """Roll back every Program authoring mutation if the guarded work fails."""
    snapshot = _AuthoringSnapshot(program)
    try:
        yield
    except BaseException:
        snapshot.restore(program)
        raise


@contextmanager
def readonly_authoring_query(program: Any) -> Iterator[None]:
    """Authenticate a query callback's net effect using the authoring snapshot."""
    snapshot = _AuthoringSnapshot(program)
    try:
        yield
        snapshot.require_readonly_query(program)
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


__all__ = ["atomic_authoring", "authoring_transaction", "readonly_authoring_query"]
