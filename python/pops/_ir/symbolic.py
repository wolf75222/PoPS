"""Shared Python data-model contract for symbolic PoPS values.

Symbolic values are immutable graph nodes.  They deliberately have no Python
truth value and are never hashable: ``==`` and the ordering operators build
graph nodes, so allowing ``bool(expr)`` would make ``if``, ``and`` or a chained
comparison execute the wrong program while authoring.
"""
from __future__ import annotations

import inspect
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType
from typing import Any
from pops.provenance import ProvenanceRecord


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """A compact source location carried by a structured authoring error."""

    file: str
    line: int
    __pops_ir_immutable__ = True

    def __str__(self) -> str:
        return "%s:%d" % (self.file, self.line)


class SymbolicTruthValueError(TypeError):
    """Raised when Python attempts to truth-test a symbolic PoPS value."""

    code = "symbolic_truth_value"

    def __init__(self, value: Any, *, suggestion: str | None = None) -> None:
        self.value = value
        self.location = _truth_test_location(value)
        self.suggestions = (
            suggestion
            or "use where(...) for symbolic data selection or T.branch(...) / T.while_(...) "
            "for Program control flow; Python if/and/or cannot evaluate a symbolic predicate"
        )
        super().__init__(
            "[%s] %s has no Python truth value at %s; %s"
            % (self.code, type(value).__name__, self.location, self.suggestions)
        )


def _truth_test_location(value: Any) -> SourceLocation:
    """Return author provenance when present, otherwise the current user frame."""

    provenance = getattr(value, "provenance", None)
    if isinstance(provenance, ProvenanceRecord):
        return SourceLocation(provenance.primary.file, provenance.primary.line)

    raw = getattr(value, "source_location", None)
    if isinstance(raw, SourceLocation):
        return raw
    if isinstance(raw, str) and ":" in raw:
        file, _, line = raw.rpartition(":")
        try:
            return SourceLocation(file, int(line))
        except ValueError:
            pass

    package_root = os.path.dirname(os.path.dirname(__file__))
    for frame in inspect.stack()[2:]:
        filename = os.path.abspath(frame.filename)
        if not filename.startswith(package_root + os.sep):
            return SourceLocation(filename, frame.lineno)
    return SourceLocation("<unknown>", 0)


def freeze_symbolic_metadata(value: Any) -> Any:
    """Recursively freeze container metadata retained by a symbolic node."""
    if getattr(value, "__pops_ir_immutable__", False) is True:
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({
            freeze_symbolic_metadata(key): freeze_symbolic_metadata(item)
            for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(freeze_symbolic_metadata(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze_symbolic_metadata(item) for item in value)
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("symbolic floating metadata must be finite")
        return value
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("symbolic Decimal metadata must be finite")
        return value
    raise TypeError(
        "symbolic metadata leaf %s is mutable or opaque; use strict data or an immutable protocol"
        % type(value).__name__)


class _ImmutableSymbolicMeta(type):
    """Freeze a symbolic object after its existing constructor has completed.

    PoPS has many small node classes whose constructors predate frozen
    dataclasses.  Freezing at the metaclass boundary gives every subclass the
    same contract without a fragile, duplicated ``_freeze()`` call in each
    constructor.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        # Mirror ``type.__call__`` closely enough for extension nodes that define a
        # parameterized ``__new__``.  Calling every ``__new__`` with no arguments made
        # otherwise-valid third-party Expr subclasses impossible to construct.
        constructor: Any = cls.__new__
        instance = (constructor(cls) if constructor is object.__new__
                    else constructor(cls, *args, **kwargs))
        if not isinstance(instance, cls):
            return instance
        object.__setattr__(instance, "_pops_symbolic_initializing", True)
        try:
            cls.__init__(instance, *args, **kwargs)
        except Exception:
            object.__setattr__(instance, "_pops_symbolic_initializing", False)
            raise
        object.__setattr__(instance, "_pops_symbolic_initializing", False)
        return instance


class ImmutableSymbolic(metaclass=_ImmutableSymbolicMeta):
    """Base enforcing immutable, non-hashable, non-truthy graph semantics."""

    __slots__ = ("_pops_symbolic_initializing",)
    __pops_ir_immutable__ = True
    __hash__: Any
    __hash__ = None

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_pops_symbolic_initializing", False):
            # ``prog`` is the sole mutable owner reference of a ProgramValue; it is not graph
            # metadata and is validated by every Program boundary. All actual node fields are
            # transitively frozen or must implement the explicit immutable-value protocol.
            object.__setattr__(
                self, name, value if name == "prog" else freeze_symbolic_metadata(value))
            return
        raise AttributeError("%s is immutable" % type(self).__name__)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("%s is immutable" % type(self).__name__)

    def __bool__(self) -> bool:
        raise SymbolicTruthValueError(self)


__all__ = [
    "ImmutableSymbolic", "SourceLocation", "SymbolicTruthValueError",
    "freeze_symbolic_metadata",
]
