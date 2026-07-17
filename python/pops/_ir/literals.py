"""Exact scalar literals used by symbolic expressions and time coefficients."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType
from typing import Any


_CPP_SIGNED_INTEGER_MAX = (1 << 63) - 1
CPP_INT_MAX = (1 << 31) - 1
# The native GMRES exceptional path flattens one 67-double exponent-banded payload per Arnoldi
# projection into a single MPI_Allreduce. Keep the authored restart within that signed-int count.
PREPARED_GMRES_ROBUST_DOT_PAYLOAD_WIDTH = 67
PREPARED_GMRES_MAX_RESTART = CPP_INT_MAX // PREPARED_GMRES_ROBUST_DOT_PAYLOAD_WIDTH - 1
_BINARY64_EXACT_INTEGER_MAX = 1 << 53
_CPP_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*$")


def _finite_real_token(value: Any, *, description: str) -> str:
    """Round @p value once at the explicit ``pops::Real`` target boundary."""
    try:
        lowered = float(value)
    except (OverflowError, ValueError) as exc:
        raise OverflowError(
            "%s cannot be represented by the finite pops::Real target" % description
        ) from exc
    if not math.isfinite(lowered):
        raise OverflowError(
            "%s cannot be represented by the finite pops::Real target" % description
        )
    return repr(lowered)


def _integer_chunks_cpp(value: int, cpp_type: str) -> str:
    """Spell an arbitrary integer using only signed-64-safe C++ integer tokens."""
    sign = -1 if value < 0 else 1
    digits = str(abs(value))
    first = len(digits) % 9 or 9
    chunks = [int(digits[:first])]
    chunks.extend(int(digits[index : index + 9]) for index in range(first, len(digits), 9))
    expression = "%s(%d)" % (cpp_type, chunks[0])
    for chunk in chunks[1:]:
        expression = "((%s * %s(1000000000)) + %s(%d))" % (expression, cpp_type, cpp_type, chunk)
    return "(-%s)" % expression if sign < 0 else expression


def _integer_cpp(value: int, cpp_type: str, *, bounded_real: bool) -> str:
    # A C++ decimal integer token outside the signed 64-bit range can be ill-formed before the
    # pops::Real constructor ever sees it.  Keep safe integers exact; round wider integers only here,
    # at the requested target boundary.
    if abs(value) <= _CPP_SIGNED_INTEGER_MAX:
        return "%s(%d)" % (cpp_type, value)
    if bounded_real:
        return "%s(%s)" % (cpp_type, _finite_real_token(value, description="integer literal"))
    return _integer_chunks_cpp(value, cpp_type)


def _rational_cpp(
    numerator: int,
    denominator: int,
    cpp_type: str,
    *,
    bounded_real: bool,
) -> str:
    if bounded_real:
        # Division is a single correctly-rounded binary64 operation only when both integer operands
        # reach it exactly. Wider operands would be rounded separately first (a possible 1-ulp
        # double-rounding error), so precompute the exact Fraction -> binary64 boundary once.
        if (
            abs(numerator) <= _BINARY64_EXACT_INTEGER_MAX
            and abs(denominator) <= _BINARY64_EXACT_INTEGER_MAX
        ):
            return "(%s(%d) / %s(%d))" % (cpp_type, numerator, cpp_type, denominator)
        rounded = _finite_real_token(
            Fraction(numerator, denominator), description="rational literal"
        )
        return "%s(%s)" % (cpp_type, rounded)
    if abs(numerator) <= _CPP_SIGNED_INTEGER_MAX and abs(denominator) <= _CPP_SIGNED_INTEGER_MAX:
        return "(%s(%d) / %s(%d))" % (cpp_type, numerator, cpp_type, denominator)
    return "(%s / %s)" % (
        _integer_chunks_cpp(numerator, cpp_type),
        _integer_chunks_cpp(denominator, cpp_type),
    )


def _freeze_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("custom ScalarLiteral payload mappings require string keys")
        return MappingProxyType({key: _freeze_payload(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_payload(item) for item in value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("custom ScalarLiteral floating payloads must be finite")
        return value
    raise TypeError(
        "custom ScalarLiteral payload must be strict JSON data (string-keyed mappings, "
        "lists, and scalar values)"
    )


def _data_payload(value: Any) -> Any:
    """Return a detached JSON-shaped view of an immutable custom literal payload."""
    if isinstance(value, Mapping):
        return {key: _data_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_data_payload(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ScalarLiteral:
    """A lossless authoring scalar, before target-precision lowering.

    Python ``float`` inputs remain their exact binary64 payload.  Integers,
    rationals and decimals retain their structural form; unit and target dtype
    annotations are metadata on the literal rather than reasons to coerce it.
    Algebraic values use an explicit symbolic spelling and C++ lowering.
    """

    kind: str
    payload: Any
    unit: str | None = None
    target: str | None = None
    cpp: str | None = None
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("ScalarLiteral kind must be a non-empty string")
        for name in ("unit", "target", "cpp"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError("ScalarLiteral %s must be a non-empty string or None" % name)
        if self.target is not None and _CPP_TYPE_RE.fullmatch(self.target) is None:
            raise ValueError(
                "ScalarLiteral target must be a qualified C++ scalar type name, got %r"
                % self.target
            )
        object.__setattr__(self, "payload", _freeze_payload(self.payload))
        if self.kind == "integer" and (
            isinstance(self.payload, bool) or not isinstance(self.payload, int)
        ):
            raise TypeError("integer ScalarLiteral payload must be a Python int")
        if self.kind == "rational":
            if (
                not isinstance(self.payload, tuple)
                or len(self.payload) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in self.payload)
                or self.payload[1] == 0
            ):
                raise TypeError("rational ScalarLiteral payload must be (numerator, denominator)")
            rational = Fraction(*self.payload)
            object.__setattr__(self, "payload", (rational.numerator, rational.denominator))
        if self.kind == "decimal":
            if not isinstance(self.payload, str):
                raise TypeError("decimal ScalarLiteral payload must be a decimal string")
            try:
                decimal = Decimal(self.payload)
            except Exception as exc:  # noqa: BLE001 -- normalize the public value-object boundary
                raise TypeError("decimal ScalarLiteral payload must be a decimal string") from exc
            if not decimal.is_finite():
                raise ValueError("decimal ScalarLiteral payload must be finite")
            # Decimal accepts Python conveniences such as whitespace and underscores which are not
            # valid standalone C++ numeric tokens.  Store its canonical numeric spelling so direct
            # construction cannot smuggle an ill-formed token into target lowering.
            object.__setattr__(self, "payload", str(decimal))
        if self.kind == "binary64":
            if not isinstance(self.payload, str):
                raise TypeError("binary64 ScalarLiteral payload must be a float.hex string")
            try:
                binary = float.fromhex(self.payload)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "binary64 ScalarLiteral payload must be a float.hex string"
                ) from exc
            if not math.isfinite(binary):
                raise ValueError("binary64 ScalarLiteral payload must be finite")
            object.__setattr__(self, "payload", binary.hex())
        if self.kind == "algebraic" and (
            not isinstance(self.payload, str) or not self.payload or self.cpp is None
        ):
            raise ValueError("algebraic ScalarLiteral requires string payload and C++ spelling")

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        unit: str | None = None,
        target: Any = None,
    ) -> ScalarLiteral:
        if isinstance(value, ScalarLiteral):
            if unit is None and target is None:
                return value
            return cls(
                value.kind,
                value.payload,
                value.unit if unit is None else unit,
                value.target if target is None else _target_name(target),
                value.cpp,
            )
        if isinstance(value, bool):
            raise TypeError("bool is not a real scalar literal; use a typed Boolean expression")
        if isinstance(value, int):
            return cls("integer", int(value), unit, _target_name(target))
        if isinstance(value, Fraction):
            return cls(
                "rational",
                (int(value.numerator), int(value.denominator)),
                unit,
                _target_name(target),
            )
        if isinstance(value, Decimal):
            if not value.is_finite():
                raise ValueError("Decimal scalar literal must be finite")
            return cls("decimal", str(value), unit, _target_name(target))
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("floating scalar literal must be finite")
            return cls("binary64", value.hex(), unit, _target_name(target))

        hook = getattr(value, "__pops_scalar_literal__", None)
        if callable(hook):
            data = hook()
            if not isinstance(data, dict) or "kind" not in data or "payload" not in data:
                raise TypeError("__pops_scalar_literal__() must return {kind, payload, ...}")
            if not isinstance(data["kind"], str) or not data["kind"]:
                raise TypeError("__pops_scalar_literal__() kind must be a non-empty string")
            return cls(
                data["kind"],
                _freeze_payload(data["payload"]),
                data.get("unit", unit),
                data.get("target", _target_name(target)),
                data.get("cpp"),
            )

        # Typed constants/descriptors already present in PoPS expose value, unit and dtype.
        category = getattr(value, "category", None)
        if category in ("constant", "const_param") and hasattr(value, "value"):
            return cls.from_value(
                value.value,
                unit=getattr(value, "unit", unit),
                target=getattr(value, "dtype", target),
            )
        raise TypeError("unsupported symbolic scalar literal %r" % (value,))

    @classmethod
    def algebraic(
        cls,
        expression: str,
        *,
        cpp: str,
        unit: str | None = None,
        target: Any = None,
    ) -> ScalarLiteral:
        if not isinstance(expression, str) or not expression or not isinstance(cpp, str) or not cpp:
            raise TypeError(
                "an algebraic literal requires non-empty string symbolic and C++ spellings"
            )
        return cls("algebraic", expression, unit, _target_name(target), cpp)

    def to_data(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind}
        if self.kind == "integer":
            data["value"] = str(self.payload)
        elif self.kind == "rational":
            data["numerator"] = str(self.payload[0])
            data["denominator"] = str(self.payload[1])
        else:
            data["value"] = _data_payload(self.payload)
        if self.unit is not None:
            data["unit"] = self.unit
        if self.target is not None:
            data["target"] = self.target
        if self.cpp is not None:
            data["cpp"] = self.cpp
        return data

    def to_python(self) -> Any:
        if self.kind == "integer":
            return int(self.payload)
        if self.kind == "rational":
            return Fraction(*self.payload)
        if self.kind == "decimal":
            return Decimal(self.payload)
        if self.kind == "binary64":
            return float.fromhex(self.payload)
        raise TypeError("%s literal needs target lowering before numerical evaluation" % self.kind)

    def to_cpp(self) -> str:
        if self.unit is not None:
            raise TypeError(
                "ScalarLiteral.to_cpp cannot lower unit %r without an explicit unit-system "
                "conversion; convert the quantity to an unannotated target scalar first" % self.unit
            )
        cpp_type = self.target or "pops::Real"
        bounded_real = cpp_type == "pops::Real"
        if self.kind == "integer":
            body = _integer_cpp(self.payload, cpp_type, bounded_real=bounded_real)
        elif self.kind == "rational":
            num, den = self.payload
            body = _rational_cpp(num, den, cpp_type, bounded_real=bounded_real)
        elif self.kind == "decimal":
            decimal = Decimal(self.payload)
            if bounded_real:
                _finite_real_token(decimal, description="decimal literal")
                # ``-0`` is an integer token in C++ and loses Decimal's signed-zero payload before
                # the Real cast. Force a floating token for that representational edge case.
                token = "-0.0" if decimal.is_zero() and decimal.is_signed() else self.payload
                body = "%s(%s)" % (cpp_type, token)
            elif decimal.is_zero() and decimal.is_signed():
                body = "%s(-0.0)" % cpp_type
            else:
                numerator, denominator = decimal.as_integer_ratio()
                body = _rational_cpp(numerator, denominator, cpp_type, bounded_real=False)
        elif self.kind == "binary64":
            body = repr(float.fromhex(self.payload))
            if self.target:
                body = "%s(%s)" % (cpp_type, body)
        elif self.cpp:
            body = self.cpp
        else:
            raise TypeError("literal kind %r has no C++ lowering" % self.kind)
        return body

    def __repr__(self) -> str:
        return "ScalarLiteral(%r)" % self.to_data()


def _target_name(target: Any) -> str | None:
    if target is None:
        return None
    name = getattr(target, "name", target)
    if not isinstance(name, str) or not name:
        raise TypeError("scalar target must be a non-empty string or expose a string .name")
    return name


def exact_scale_prefix(value: Any) -> str:
    """Readable, lossless coefficient prefix for symbolic reprs."""
    return "" if value == 1 else "%s*" % value


def scalar_literal(value: Any, *, unit: str | None = None, target: Any = None) -> ScalarLiteral:
    return ScalarLiteral.from_value(value, unit=unit, target=target)


def exact_numeric_scalar(value: Any, *, where: str) -> Any:
    """Return an exact Python number only when no literal annotation would be erased."""
    literal = scalar_literal(value)
    if literal.unit is not None or literal.target is not None:
        raise TypeError(
            "%s cannot erase a scalar unit or target annotation; keep the constant as an Expr "
            "until explicit target lowering" % where
        )
    try:
        return literal.to_python()
    except TypeError as exc:
        raise TypeError(
            "%s requires a statically evaluable numeric scale; keep algebraic/custom constants "
            "as Expr nodes" % where
        ) from exc


def exact_cpp_int(
    value: Any,
    *,
    where: str,
    minimum: int,
    maximum: int = CPP_INT_MAX,
) -> int:
    """Validate one exact Python integer before lowering it to a signed C++ ``int``.

    Python integers are arbitrary precision, while every native PoPS control using this helper is
    stored in a signed C++ ``int``.  Keep that target boundary explicit: booleans and coercible
    values are not integers here, and an out-of-range value must fail before source generation.
    """
    if isinstance(minimum, bool) or not isinstance(minimum, int):
        raise TypeError("exact_cpp_int minimum must be a Python int")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < minimum:
        raise TypeError("exact_cpp_int maximum must be a Python int >= minimum")
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        raise ValueError(
            "%s must be an exact Python int in [%d, %d]; got %r" % (where, minimum, maximum, value)
        )
    return value


def numeric_domains_compatible(left: Any, right: Any) -> bool:
    """Whether eager exact algebra can combine two scalar number domains losslessly."""
    if type(left) is type(right):
        return True
    if isinstance(left, int) or isinstance(right, int):
        integer = left if isinstance(left, int) else right
        other = right if isinstance(left, int) else left
        if isinstance(other, float):
            return abs(integer) <= _BINARY64_EXACT_INTEGER_MAX
        return isinstance(other, (Fraction, Decimal))
    return False


def _decimal_parts(value: Decimal | int) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return value, 0
    if not isinstance(value, Decimal) or not value.is_finite():
        raise TypeError("exact Decimal algebra accepts only finite Decimal and int operands")
    sign, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        raise TypeError("finite Decimal values must carry an integral exponent")
    coefficient = int("".join(map(str, digits))) if digits else 0
    return (-coefficient if sign else coefficient), exponent


def _decimal_from_parts(coefficient: int, exponent: int) -> Decimal:
    sign = int(coefficient < 0)
    digits = tuple(map(int, str(abs(coefficient)))) if coefficient else (0,)
    return Decimal((sign, digits, exponent))


def exact_decimal_add(left: Decimal | int, right: Decimal | int) -> Decimal:
    """Context-independent exact Decimal addition (integers are neutral)."""
    left_coeff, left_exp = _decimal_parts(left)
    right_coeff, right_exp = _decimal_parts(right)
    exponent = min(left_exp, right_exp)
    coefficient = left_coeff * 10 ** (left_exp - exponent) + right_coeff * 10 ** (
        right_exp - exponent
    )
    return _decimal_from_parts(coefficient, exponent)


def exact_decimal_negate(value: Decimal) -> Decimal:
    """Negate a Decimal payload without applying the ambient Decimal context."""
    if not isinstance(value, Decimal) or not value.is_finite():
        raise TypeError("exact Decimal negation requires a finite Decimal")
    sign, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        raise TypeError("finite Decimal values must carry an integral exponent")
    return Decimal((1 - sign, digits, exponent))


def exact_decimal_multiply(left: Decimal | int, right: Decimal | int) -> Decimal:
    """Context-independent exact Decimal multiplication (integers are neutral)."""
    left_coeff, left_exp = _decimal_parts(left)
    right_coeff, right_exp = _decimal_parts(right)
    return _decimal_from_parts(left_coeff * right_coeff, left_exp + right_exp)


def exact_decimal_divide(left: Decimal | int, right: Decimal | int) -> Decimal | None:
    """Return the exact terminating Decimal quotient, or ``None`` if it cannot terminate."""
    left_ratio = left.as_integer_ratio() if isinstance(left, Decimal) else (left, 1)
    right_ratio = right.as_integer_ratio() if isinstance(right, Decimal) else (right, 1)
    if right_ratio[0] == 0:
        raise ZeroDivisionError("division by zero")
    quotient = Fraction(left_ratio[0] * right_ratio[1], left_ratio[1] * right_ratio[0])
    denominator = quotient.denominator
    powers_two = 0
    powers_five = 0
    while denominator % 2 == 0:
        denominator //= 2
        powers_two += 1
    while denominator % 5 == 0:
        denominator //= 5
        powers_five += 1
    if denominator != 1:
        return None
    scale = max(powers_two, powers_five)
    coefficient = quotient.numerator * 5 ** (scale - powers_two) * 2 ** (scale - powers_five)
    return _decimal_from_parts(coefficient, -scale)


def multiply_exact_scalars(left: Any, right: Any, *, where: str) -> Any:
    """Multiply two scale values without silently crossing number domains."""
    a = exact_numeric_scalar(left, where=where)
    b = exact_numeric_scalar(right, where=where)
    if not numeric_domains_compatible(a, b):
        raise TypeError(
            "%s cannot mix %s and %s without an explicit target conversion"
            % (where, type(a).__name__, type(b).__name__)
        )
    if isinstance(a, Decimal) or isinstance(b, Decimal):
        return exact_decimal_multiply(a, b)
    return a * b


def scalar_data(value: Any) -> dict[str, Any]:
    return ScalarLiteral.from_value(value).to_data()


def scalar_cpp(value: Any) -> str:
    return ScalarLiteral.from_value(value).to_cpp()


def scalar_to_native(value: Any, *, where: str) -> float:
    """Round an exact authoring scalar once, at the native ``pops::Real`` ABI.

    Unit-bearing values and values pinned to another target cannot be silently
    reinterpreted as the native scalar. Algebraic/custom literals need codegen,
    not a runtime cast, and are therefore refused by this boundary as well.
    """
    literal = ScalarLiteral.from_value(value)
    if literal.unit is not None:
        raise TypeError(
            "%s cannot lower unit %r at the native pops::Real boundary; convert the "
            "quantity explicitly first" % (where, literal.unit)
        )
    if literal.target not in (None, "pops::Real"):
        raise TypeError("%s targets %r, not the native pops::Real ABI" % (where, literal.target))
    try:
        result = float(literal.to_python())
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            "%s requires a finite numeric scalar representable by native pops::Real" % where
        ) from exc
    if not math.isfinite(result):
        raise OverflowError("%s cannot be represented by finite native pops::Real" % where)
    return result


__all__ = [
    "CPP_INT_MAX",
    "PREPARED_GMRES_MAX_RESTART",
    "PREPARED_GMRES_ROBUST_DOT_PAYLOAD_WIDTH",
    "ScalarLiteral",
    "exact_decimal_add",
    "exact_decimal_divide",
    "exact_decimal_multiply",
    "exact_cpp_int",
    "exact_decimal_negate",
    "exact_numeric_scalar",
    "exact_scale_prefix",
    "multiply_exact_scalars",
    "scalar_cpp",
    "scalar_to_native",
    "numeric_domains_compatible",
    "scalar_data",
    "scalar_literal",
]
