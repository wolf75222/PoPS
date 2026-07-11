"""Registry-free parameter metadata retained by compiled artifacts.

Authoring declarations deliberately carry more than a native artifact needs: registry authority
tokens, symbolic expressions and dependency Handles.  A public :class:`CompiledModel` must retain
none of that graph.  This module projects declarations through their canonical ``bind_data()``
protocol into an immutable data value and keeps only the decoded scalar default needed by the
runtime compatibility surface.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType
from typing import Any


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({
            key: _freeze_json(item) for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        "compiled parameter metadata contains non-data value %s"
        % type(value).__name__
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _decode_scalar(data: Any) -> Any:
    """Decode one canonical parameter-default literal without retaining its source object."""
    if not isinstance(data, Mapping):
        raise TypeError("compiled parameter default literal must be a mapping")
    kind = data.get("kind")
    if kind == "boolean":
        value = data.get("value")
        if not isinstance(value, bool):
            raise TypeError("boolean compiled parameter default must contain a bool")
        return value
    if kind == "integer":
        return int(data["value"])
    if kind == "rational":
        return Fraction(int(data["numerator"]), int(data["denominator"]))
    if kind == "decimal":
        return Decimal(data["value"])
    if kind == "binary64":
        return float.fromhex(data["value"])
    raise TypeError(
        "compiled parameter default uses unsupported scalar literal kind %r" % kind
    )


def _scalar_declaration_data(name: str, value: Any) -> dict[str, Any]:
    """Normalize the narrow legacy ``{name: scalar}`` constructor input."""
    from pops.math import Bool, Integer, Real
    from pops.params import ConstParam

    if isinstance(value, bool):
        dtype = Bool
    elif isinstance(value, int):
        dtype = Integer
    elif isinstance(value, (float, Fraction, Decimal)):
        dtype = Real
    else:
        raise TypeError(
            "CompiledModel parameter %r must expose bind_data() or be a bool/int/float/"
            "Fraction/Decimal scalar, got %s" % (name, type(value).__name__)
        )
    return ConstParam(name, value, dtype=dtype).bind_data()


class CompiledParameter:
    """Immutable, registry-free projection of one parameter declaration.

    ``kind`` and ``phase`` are plain closed-string metadata.  ``domain`` and :meth:`to_data` expose
    detached data copies rather than the authoring ``Constraint``/``Expr``/``ParamHandle`` objects.
    The class intentionally has no ``expression`` attribute: the canonical serialized expression
    remains available in :meth:`to_data` for inspection and hashing, but no live graph survives.
    """

    __slots__ = ("_data", "_default")
    __pops_ir_immutable__ = True

    def __init__(self, data: Any) -> None:
        from pops.params import validate_parameter_data

        row = validate_parameter_data(data)
        default = row["default"]
        scalar = (
            _decode_scalar(default["value"])
            if default["state"] == "value"
            else None
        )
        object.__setattr__(self, "_data", _freeze_json(row))
        object.__setattr__(self, "_default", scalar)

    @property
    def name(self) -> str:
        return self._data["name"]

    @property
    def kind(self) -> str:
        return self._data["kind"]

    @property
    def phase(self) -> str:
        return self._data["phase"]

    @property
    def dtype(self) -> str:
        return self._data["dtype"]

    @property
    def unit(self) -> str | None:
        return self._data["unit"]

    @property
    def storage(self) -> str:
        return self._data["storage"]

    @property
    def invalidation(self) -> str:
        return self._data["invalidation"]

    @property
    def domain(self) -> Any:
        return self._data["domain"]

    @property
    def has_default(self) -> bool:
        return self._data["default"]["state"] == "value"

    @property
    def default(self) -> Any:
        return self._default

    @property
    def value(self) -> Any:
        """Compatibility spelling used by compiled-report inspection for const parameters."""
        return self._default

    def to_data(self) -> dict[str, Any]:
        return _thaw_json(self._data)

    def bind_data(self) -> dict[str, Any]:
        return self.to_data()

    def freeze(self) -> CompiledParameter:
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("CompiledParameter is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("CompiledParameter is immutable")

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CompiledParameter) and self.to_data() == other.to_data()

    def __hash__(self) -> int:
        return hash(json.dumps(self.to_data(), sort_keys=True, separators=(",", ":")))

    def __repr__(self) -> str:
        return "CompiledParameter(name=%r, kind=%r, phase=%r)" % (
            self.name, self.kind, self.phase)


def compiled_parameter(name: Any, value: Any) -> CompiledParameter:
    """Project one declaration/scalar without retaining ``value`` or any of its leaves."""
    if not isinstance(name, str) or not name:
        raise TypeError("CompiledModel parameter names must be non-empty strings")
    if isinstance(value, CompiledParameter):
        if value.name != name:
            raise ValueError(
                "CompiledModel parameter key %r does not match declaration name %r"
                % (name, value.name)
            )
        return value
    bind_data = getattr(value, "bind_data", None)
    data = bind_data() if callable(bind_data) else (
        dict(value) if isinstance(value, Mapping) else _scalar_declaration_data(name, value)
    )
    result = CompiledParameter(data)
    if result.name != name:
        raise ValueError(
            "CompiledModel parameter key %r does not match declaration name %r"
            % (name, result.name)
        )
    return result


def compiled_parameters(values: Any) -> Mapping[str, CompiledParameter]:
    """Return an immutable detached projection of a ``CompiledModel.params`` mapping."""
    if not isinstance(values, Mapping):
        raise TypeError("CompiledModel params must be a mapping")
    return MappingProxyType({
        name: compiled_parameter(name, value) for name, value in values.items()
    })


__all__ = ["CompiledParameter", "compiled_parameter", "compiled_parameters"]
