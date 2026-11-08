"""Method memory of the temporal program (spec 0.3 T27, section 9, CN4; N8 lot 5).

``method(memory=Product(name=T, ...), role=Auxiliary())`` declares the members that a method
returns in ``Candidate(..., memory=Product(name=value, ...))``. This lot elaborates a minimal slice:

- the role ``Auxiliary()`` (CN4), one role for every member, as ``role=`` of section 2.6.8 (gap
  C51); a memory without ``role=`` is Scientific and is refused, as are the other roles and the
  argument of ``Auxiliary`` (gap C46);
- two member types: ``State``, the type placeholder of section 5.2, resolved to the type of
  ``B.state``, and a field observation, typed by the qualified field unknown it observes
  (``poisson[phi]``); no new Pi name is added;
- the default policies of CN5 and CN6: ``B.memory`` and ``policies=`` stay refused, and no
  ``schema_id`` (CN9) is computed, which is G07's.

The elaborator lowers each member to a History ring of depth one named by the member (CN11 allows
that physical encoding), stored after the assessment of the candidate, so the memory of a rejected
attempt is never published (T37). ``RunResult.continuation.memory`` (``MemoryView``) reads slot one
of each ring after a run.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from pops.time._program.traced_method import NotYetElaborated


class _StateType:
    """``State``: the type placeholder of section 5.2, resolved to the type of ``B.state``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "State"


State = _StateType()


class Auxiliary:
    """``Auxiliary()``: the CN4 role of a memory that a rejection may change and that no
    scientific claim reads (T37). Its argument (persisted) is gap C46, not elaborated here."""

    __slots__ = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs:
            raise NotYetElaborated("Auxiliary(...): the arguments of Auxiliary (gap C46) are "
                                   "elaborated with the continuation of G07")

    def __eq__(self, other: object) -> bool:
        return type(other) is Auxiliary

    def __hash__(self) -> int:
        return hash(Auxiliary)

    def __repr__(self) -> str:
        return "Auxiliary()"


class MemoryUndefined(LookupError):
    """A memory member read before the first accepted step stored it (CN6)."""


def _is_field_unknown(value: Any) -> bool:
    """A qualified field unknown, ``field[unknown]``: a field Handle that is not the field."""
    from pops.model import Handle
    from pops.problem.handles import FieldHandle

    return isinstance(value, Handle) and value.kind == "field" \
        and not isinstance(value, FieldHandle)


def declared_memory(memory: Any, role: Any) -> tuple:
    """The members ``((name, type), ...)`` of ``method(memory=..., role=...)``, checked."""
    from pops.time._program.outputs import Product

    if memory is None:
        if role is not None:
            raise TypeError("method(role=...) states the role of a memory; this method declares "
                            "none")
        return ()
    if type(memory) is not Product or not len(memory):
        raise TypeError("method(memory=...) is a Product of member types, "
                        "Product(name=State, ...), got %r" % (memory,))
    if role is None:
        raise NotYetElaborated(
            "method(memory=...) without role= is a Scientific memory (roles of section 2.6), "
            "which enters B and needs a startup (CN6); this lot elaborates "
            "role=Auxiliary() only (gap C51)")
    if type(role) is not Auxiliary:
        raise NotYetElaborated("method(role=%r): this lot elaborates the role Auxiliary() of a "
                               "method memory only (CN4, gap C51)" % (role,))
    members = []
    for name, kind in memory.items():
        if kind is not State and not _is_field_unknown(kind):
            raise TypeError(
                "method(memory=...): member %r has the type %r; this lot takes State or the "
                "qualified field unknown that the member observes, field[unknown]" % (name, kind))
        members.append((name, kind))
    return tuple(members)


class MemoryView(Mapping):
    """``RunResult.continuation.memory``: slot one of the ring of each member, read on demand.

    Reading a member gathers its ring from every rank (``history_global`` is collective), so every
    rank reads the same members in the same order. A later run supersedes this memory, and a read
    then is refused; a read before the first accepted step raises ``MemoryUndefined`` (CN6).
    """

    __slots__ = ("_runtime", "_time", "_macro_step", "_rings")

    def __init__(self, runtime: Any, time: float, macro_step: int, rings: Mapping) -> None:
        self._runtime = runtime
        self._time = time
        self._macro_step = macro_step
        self._rings = dict(rings)

    def __getitem__(self, name: str) -> Any:
        ring = self._rings[name]
        runtime = self._runtime
        if runtime.time() != self._time or runtime.macro_step() != self._macro_step:
            raise RuntimeError(
                "this memory at t=%r was superseded by a later run; read it before running again"
                % self._time)
        if runtime._executor._s.history_fill_count(ring) < 1:
            raise MemoryUndefined(
                "the memory member %r is undefined before the first accepted step stores it "
                "(CN6)" % name)
        return runtime.history_global(ring, 1)

    def __iter__(self) -> Iterator[str]:
        return iter(self._rings)

    def __len__(self) -> int:
        return len(self._rings)

    def __repr__(self) -> str:
        return "MemoryView(%s)" % ", ".join(self._rings)


__all__ = ["Auxiliary", "MemoryUndefined", "MemoryView", "State", "declared_memory"]
