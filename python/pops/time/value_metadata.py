"""Exact scalar and immutable metadata helpers for Program values."""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType
from typing import Any, cast

from pops._ir.literals import (
    ScalarLiteral,
    exact_decimal_add,
    exact_decimal_divide,
    exact_decimal_multiply,
    exact_decimal_negate,
    numeric_domains_compatible,
    scalar_data,
    scalar_literal,
)
from pops.time.value_support import _ProgramValueBase


class CoefficientLiteralError(TypeError):
    """A valid scalar literal whose metadata cannot be erased by coefficient algebra."""


def positive_scalar_literal(value: Any, *, where: str) -> Any:
    """Validate and retain a statically positive scalar without coercing its number domain."""
    try:
        literal = scalar_literal(value)
        numeric = literal.to_python()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "%s must be a statically evaluable positive scalar literal (got %r)"
            % (where, value)) from exc
    try:
        positive = numeric > 0
    except TypeError as exc:
        raise ValueError("%s must be a positive scalar literal (got %r)" % (where, value)) from exc
    if not positive:
        raise ValueError("%s must be a positive scalar literal (got %r)" % (where, value))
    return literal


def validate_program_value_identity(
    vid: Any, vtype: Any, op: Any, inputs: Any, name: Any, block: Any, region: Any,
    state_ref: Any = None,
) -> tuple[Any, ...]:
    """Validate direct SSA-node construction and return immutable inputs."""
    if isinstance(vid, bool) or not isinstance(vid, int) or vid < 0:
        raise ValueError("ProgramValue id must be a non-negative integer")
    for label, value in (("vtype", vtype), ("op", op), ("name", name)):
        if not isinstance(value, str) or not value:
            raise ValueError("ProgramValue %s must be a non-empty string" % label)
    if block is not None:
        from pops.problem.handles import BlockHandle
        if not isinstance(block, BlockHandle):
            raise TypeError("ProgramValue block must be a BlockHandle or None")
    if state_ref is not None:
        from pops.model.handles import Handle
        if not isinstance(state_ref, Handle) or state_ref.kind != "state" \
                or not state_ref.is_instance:
            raise TypeError(
                "ProgramValue state_ref must be a block-qualified state Handle or None")
        if block is None or state_ref.block_ref is not block:
            raise ValueError("ProgramValue state_ref belongs to a different block")
    if isinstance(region, bool) or not isinstance(region, int) or region < 0:
        raise ValueError("ProgramValue region must be a non-negative integer")
    frozen_inputs = tuple(inputs)
    if any(not isinstance(value, _ProgramValueBase) for value in frozen_inputs):
        raise TypeError("ProgramValue inputs must contain only ProgramValue nodes")
    return frozen_inputs


def _exact_number(value: Any) -> Any:
    """Return a coefficient without silently erasing symbolic annotations."""
    literal = scalar_literal(value)
    if literal.unit is not None or literal.target is not None:
        raise CoefficientLiteralError(
            "an affine/dt coefficient cannot silently erase a scalar unit or target annotation; "
            "use an unannotated exact coefficient or lower the annotated constant explicitly")
    if literal.kind == "algebraic":
        raise CoefficientLiteralError(
            "an algebraic scalar cannot be evaluated by the affine coefficient algebra; keep it "
            "as a symbolic Expr/Program scalar until target lowering")
    return literal.to_python()


def _exact_divide(numerator: Any, denominator: Any) -> Any:
    """Divide without turning an exact integer/rational ratio into binary64."""
    _require_compatible_domains(numerator, denominator)
    if isinstance(numerator, (int, Fraction)) and isinstance(denominator, (int, Fraction)):
        return Fraction(numerator) / Fraction(denominator)
    if isinstance(numerator, Decimal) or isinstance(denominator, Decimal):
        result = exact_decimal_divide(
            cast(Decimal | int, numerator), cast(Decimal | int, denominator)
        )
        if result is None:
            raise CoefficientLiteralError(
                "an exact Decimal quotient must terminate; use Fraction for a repeating ratio")
        return result
    return numerator / denominator


def _require_compatible_domains(left: Any, right: Any) -> None:
    """Forbid silent rational/decimal/binary64 coercion inside one polynomial coefficient."""
    if numeric_domains_compatible(left, right):
        return
    raise CoefficientLiteralError(
        "cannot mix %s and %s in one affine coefficient without an explicit target conversion"
        % (type(left).__name__, type(right).__name__))


def _exact_add(left: Any, right: Any) -> Any:
    _require_compatible_domains(left, right)
    if isinstance(left, Decimal) or isinstance(right, Decimal):
        return exact_decimal_add(left, right)
    return left + right


def _exact_multiply(left: Any, right: Any) -> Any:
    _require_compatible_domains(left, right)
    if isinstance(left, Decimal) or isinstance(right, Decimal):
        return exact_decimal_multiply(left, right)
    return left * right


def _exact_negate(value: Any) -> Any:
    value = _exact_number(value)
    return exact_decimal_negate(value) if isinstance(value, Decimal) else -value


class CoeffPolynomial(Mapping):
    """Immutable exact ``dt`` polynomial carried through Program attrs and codegen."""

    __slots__ = ("_powers",)
    __pops_ir_immutable__ = True

    def __init__(self, powers: Any) -> None:
        normalized = {int(power): _exact_number(coeff) for power, coeff in powers.items()}
        object.__setattr__(self, "_powers", MappingProxyType(dict(sorted(normalized.items()))))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("CoeffPolynomial is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("CoeffPolynomial is immutable")

    def __bool__(self) -> bool:
        return bool(self._powers)

    def __len__(self) -> int:
        return len(self._powers)

    def __iter__(self) -> Any:
        return iter(self._powers)

    def __getitem__(self, power: Any) -> Any:
        return self._powers[power]

    def items(self) -> Any:
        return self._powers.items()

    def to_data(self) -> list[Any]:
        return [[power, scalar_data(coeff)] for power, coeff in self._powers.items()]

    def __repr__(self) -> str:
        return "CoeffPolynomial(%r)" % dict(self._powers)


def _freeze_attr(value: Any) -> Any:
    """Deeply freeze Program IR metadata without changing its semantic contents."""
    if isinstance(value, CoeffPolynomial):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("ProgramValue metadata mapping keys must be non-empty strings")
        return MappingProxyType({key: _freeze_attr(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_attr(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_attr(item) for item in value)
    if value is None or isinstance(value, (bool, int, str, ScalarLiteral)):
        return value
    if isinstance(value, (float, Fraction, Decimal)):
        scalar_literal(value)  # reject non-finite leaves without changing their number domain
        return value
    if getattr(value, "__pops_ir_immutable__", False) is True:
        return value
    raise TypeError(
        "ProgramValue metadata leaf %s is not an immutable IR value; convert it to strict data"
        % type(value).__name__)


__all__ = [
    "CoeffPolynomial", "CoefficientLiteralError", "_exact_add", "_exact_divide",
    "_exact_multiply", "_exact_negate", "_exact_number", "_freeze_attr", "positive_scalar_literal",
    "validate_program_value_identity",
]
