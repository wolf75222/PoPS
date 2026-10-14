"""Small authoring helpers shared by temporal symbolic values."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Protocol, TypeVar

    from pops.time.points import Clock

    _OwnerValue = TypeVar("_OwnerValue", bound="_ProgramValueBase", covariant=True)

    class _CoefficientPolynomial(Protocol):
        """The exact coefficient projection consumed by expression identity."""

        def to_data(self) -> Any: ...

    class _ValueProgram(Protocol[_OwnerValue]):
        """Only the owner surface read through a symbolic value, without reverse imports."""

        _values: list
        clock: Clock

        def _int64_compare(self, lhs: object, rhs: object, comparison: str) -> _OwnerValue: ...
        def _int64_binop(self, lhs: object, rhs: object, operation: str) -> _OwnerValue: ...
        def at(self, array: _ProgramValueBase, index: object) -> _OwnerValue: ...
        def _product_get(self, value: _ProgramValueBase, key: str) -> _OwnerValue: ...
        def scalar_expression(self, name: str, expression: object) -> _OwnerValue: ...
        def _replace_value(self, value: _ProgramValueBase, **changes: object) -> _OwnerValue: ...

        @property
        def owner_path(self) -> Any: ...

        def _compare(self, lhs: Any, rhs: Any, cmp: Any) -> Any: ...
        def _scalar_binop(self, a: Any, b: Any, fn: Any) -> Any: ...
        def apply(self, operator: Any = None, state: Any = None, fields: Any = None,
                  name: Any = None) -> Any: ...
        def _new(self, vtype: Any, op: Any, inputs: Any, attrs: Any, name: Any,
                 block: Any, **metadata: Any) -> Any: ...
        def _call(self, operator: Any, *args: Any, name: Any = None,
                  schedule: Any = None, **kwargs: Any) -> Any: ...
        def _canonical_value(self, value: Any) -> Any: ...
        def _subblock_value_refs(self, node: Any) -> Any: ...
        def _guard_mutable(self, operation: Any) -> None: ...
        def _require_endpoint(self, endpoint: Any, where: str) -> Any: ...


class _ProgramValueBase:
    """Private nominal marker shared by value validators without reverse imports."""

    __slots__ = ()
    vtype: str
    block: Any
    prog: _ValueProgram[_ProgramValueBase]
    id: int
    name: str

    if TYPE_CHECKING:
        def is_field(self) -> bool: ...


class _MethodCoefficientBase:
    """Nominal coefficient marker; expressions do not import the value algebra."""

    __slots__ = ()

    if TYPE_CHECKING:
        def to_polynomial(self) -> _CoefficientPolynomial: ...
        def __add__(self, other: Any) -> Any: ...
        def __neg__(self) -> Any: ...
        def __mul__(self, other: Any) -> Any: ...
        def __truediv__(self, other: Any) -> Any: ...


class _AffineExpressionBase:
    """Nominal affine marker shared without importing concrete Program values."""

    __slots__ = ()
    terms: tuple[tuple[_ProgramValueBase, _MethodCoefficientBase], ...]


def require_owned_value(program: Any, value: Any, where: str, *, vtype: Any = None) -> Any:
    """Authenticate the single issued-value table shared by expressions and Program builders."""
    if not isinstance(value, _ProgramValueBase):
        raise ValueError("%s: expected a ProgramValue, got %r" % (where, value))
    if value.prog is not program:
        raise ValueError("%s: value %r belongs to a different Program" % (where, value.name))
    if program._issued_values.get(id(value)) is not value:
        raise ValueError("%s: value %r was not authored by this Program" % (where, value.name))
    if vtype is not None and value.vtype != vtype:
        raise ValueError("%s: expected a %s value, got %s" % (where, vtype, value.vtype))
    return value


def resolve_temporal_handle(value: Any) -> Any:
    """Resolve a readable temporal handle through its Program-owned table."""
    from pops.time.handles import HistoryHandle, StageHandle

    return (
        value._as_value()
        if isinstance(value, (StageHandle, HistoryHandle))
        else value
    )


def authoring_source_location() -> Any:
    """Return the first call site outside ``pops.time`` for optional debug provenance."""
    from pathlib import Path
    import traceback

    time_root = Path(__file__).resolve().parent
    for frame in reversed(traceback.extract_stack()[:-1]):
        # Internal builders live in nested packages such as ``time._program``.  Comparing only the
        # immediate dirname leaks the first nested builder frame instead of the user authoring site.
        if not Path(frame.filename).resolve().is_relative_to(time_root):
            return "%s:%d" % (frame.filename, frame.lineno)
    return None


__all__ = ["_ProgramValueBase", "authoring_source_location", "resolve_temporal_handle"]
