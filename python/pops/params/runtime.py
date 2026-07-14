"""Canonical, lossless parameter declarations.

Parameter *kind* is a closed type choice, never a stringly ``kind=`` option.  A
declaration is immutable metadata; its owner-qualified identity is issued by a
``pops.model.ParamRegistry`` as a :class:`pops.model.ParamHandle`.
"""
from __future__ import annotations

import json
from typing import Any

from pops.descriptors import Descriptor, reject_string_selector
from pops.descriptors_report import CapabilitySet
from pops.math import Real

from ._declaration_data import (
    MISSING,
    PARAM_DECLARATION_SCHEMA_VERSION,
    ParamDefaultState,
    ParamInvalidation,
    ParamKind,
    ParamPhase,
    ParamProvenance,
    ParamStorage,
    check_provenance as _check_provenance,
    checked_unit as _unit,
    dtype_name as _dtype_name,
    freeze_json as _freeze_json,
    strict_name as _strict_name,
    thaw_json as _thaw_json,
    validate_parameter_data,
    value_data as _value_data,
)


def _check_domain(name: str, domain: Any) -> Any:
    if isinstance(domain, str):
        reject_string_selector(
            domain,
            "%s domain" % name,
            "a typed pops.params constraint such as Positive(), NonNegative(), Interval() or "
            "OneOf(), not a string",
        )
    if domain is None:
        return None
    from .constraints import Constraint

    if not isinstance(domain, Constraint):
        raise TypeError("parameter %r domain must be a pops.params.Constraint or None" % name)
    return domain


def _domain_error(name: str, domain: Any, value: Any, phase: str) -> str:
    expected = domain.to_data()
    return (
        "param %r: value %r is outside the expected domain %s %s at the %s phase"
        % (name, value, domain.name, expected, phase)
    )


def _expression_data(expression: Any) -> Any:
    if isinstance(expression, str) or callable(expression):
        raise TypeError(
            "DerivedParam expression must be a PoPS Expr, not a string or Python callable"
        )
    try:
        from pops._ir.expr import Expr
    except ImportError:  # pragma: no cover - the normal package always provides it
        Expr = ()  # type: ignore[assignment]
    if isinstance(expression, Expr):
        from pops._ir.visitors import _key

        payload = _key(expression)
        return {
            "protocol": "pops.expr.key.v1",
            "value": _thaw_json(_freeze_json(payload, where="derived expression")),
        }
    if getattr(expression, "__pops_param_expression__", False):
        hook = getattr(expression, "to_data", None)
        if callable(hook):
            return _thaw_json(_freeze_json(hook(), where="derived parameter expression"))
    raise TypeError("DerivedParam expression must be a PoPS Expr")


def _dependency_rows(dependencies: Any) -> tuple[tuple[Any, ...], list[dict[str, str]]]:
    if isinstance(dependencies, (str, bytes)):
        raise TypeError("DerivedParam depends_on must contain ParamHandle values")
    try:
        values = tuple(dependencies)
    except TypeError:
        raise TypeError("DerivedParam depends_on must be an iterable of ParamHandle values") from None
    if not values:
        raise ValueError("DerivedParam depends_on must declare at least one dependency")
    from pops.model.handles import ParamHandle

    rows = []
    seen = set()
    for dependency in values:
        if not isinstance(dependency, ParamHandle):
            raise TypeError("DerivedParam depends_on entries must be ParamHandle values")
        if dependency in seen:
            raise ValueError("DerivedParam depends_on contains duplicate handle %s" % dependency)
        seen.add(dependency)
        rows.append({"name": dependency.local_id, "param_kind": dependency.param_kind})
    return values, rows


class ParameterDeclaration(Descriptor):
    """Immutable common contract of the three closed parameter kinds."""

    __slots__ = (
        "_name",
        "kind",
        "dtype",
        "unit",
        "domain",
        "default",
        "default_state",
        "storage",
        "provenance",
        "phase",
        "invalidation",
        "expression",
        "depends_on",
        "_expression_payload",
        "_resolved_value",
        "_authority_token",
        "_owner_identity",
        "_sealed",
    )

    def _initialize(
        self,
        name: Any,
        *,
        kind: ParamKind,
        dtype: Any,
        unit: Any,
        domain: Any,
        default: Any,
        default_state: ParamDefaultState,
        storage: ParamStorage,
        provenance: Any,
        phase: ParamPhase,
        invalidation: ParamInvalidation,
        expression: Any = None,
        depends_on: Any = (),
    ) -> None:
        object.__setattr__(self, "_name", _strict_name(name))
        _dtype_name(dtype)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "unit", _unit(unit))
        object.__setattr__(self, "domain", _check_domain(self._name, domain))
        object.__setattr__(self, "default", default)
        object.__setattr__(self, "default_state", default_state)
        object.__setattr__(self, "storage", storage)
        object.__setattr__(self, "provenance", _check_provenance(provenance))
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "invalidation", invalidation)
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "depends_on", tuple(depends_on))
        object.__setattr__(
            self,
            "_expression_payload",
            None if expression is None else _expression_data(expression),
        )
        object.__setattr__(self, "_resolved_value", MISSING)
        object.__setattr__(self, "_authority_token", None)
        object.__setattr__(self, "_owner_identity", None)
        object.__setattr__(self, "_sealed", True)

    @property
    def name(self) -> str:
        return self._name

    @property
    def has_default(self) -> bool:
        return self.default_state is ParamDefaultState.Value

    @property
    def owner_identity(self) -> str | None:
        """Qualified registry owner, or ``None`` until explicitly registered."""
        return self._owner_identity

    @property
    def is_owned(self) -> bool:
        return self._authority_token is not None

    @property
    def resolved_value(self) -> Any:
        if self._resolved_value is MISSING:
            raise AttributeError("parameter has no compile-resolved value")
        return self._resolved_value

    def _set_compile_resolved_value(self, value: Any) -> None:
        if self.kind is not ParamKind.Derived or self.phase is not ParamPhase.Compile:
            raise TypeError("only a compile-phase DerivedParam has a resolved value")
        if self._resolved_value is not MISSING and self._resolved_value != value:
            raise ValueError("compile-derived value cannot change after resolution")
        object.__setattr__(self, "_resolved_value", value)

    def _claim_owner(self, token: Any, owner_identity: Any) -> None:
        """Atomically bind this declaration object to one registry authority."""
        if token is None or not isinstance(owner_identity, str) or not owner_identity:
            raise TypeError("parameter owner claim requires a registry token and identity")
        if self._authority_token is not None and self._authority_token is not token:
            raise ValueError(
                "parameter %r is already owned by %s and cannot be registered by %s; "
                "sharing requires an explicit shared owner or tie"
                % (self.name, self._owner_identity, owner_identity)
            )
        object.__setattr__(self, "_authority_token", token)
        object.__setattr__(self, "_owner_identity", owner_identity)

    def _default_data(self) -> dict[str, Any]:
        result: dict[str, Any] = {"state": self.default_state.value}
        if self.has_default:
            result["value"] = _value_data(
                self.default,
                dtype=self.dtype,
                unit=self.unit,
                where="parameter %r default" % self.name,
            )
        return result

    def to_data(self) -> dict[str, Any]:
        dependencies = [
            {"name": dependency.local_id, "param_kind": dependency.param_kind}
            for dependency in self.depends_on
        ]
        return {
            "schema_version": PARAM_DECLARATION_SCHEMA_VERSION,
            "name": self.name,
            "kind": self.kind.value,
            "dtype": _dtype_name(self.dtype),
            "unit": self.unit,
            "domain": None if self.domain is None else self.domain.to_data(),
            "default": self._default_data(),
            "storage": self.storage.value,
            "provenance": None if self.provenance is None else self.provenance.to_data(),
            "expression": self._expression_payload,
            "depends_on": dependencies,
            "phase": self.phase.value,
            "invalidation": self.invalidation.value,
        }

    def bind_data(self) -> dict[str, Any]:
        """Lossless declaration metadata consumed by bind-schema construction."""
        return self.to_data()

    def artifact_data(self) -> dict[str, Any]:
        """Semantic compile identity, excluding runtime values and report-only provenance."""
        data = self.to_data()
        data.pop("provenance")
        if self.kind is ParamKind.Runtime:
            data.pop("default")
        return data

    def options(self) -> dict[str, Any]:
        return self.to_data()

    def validate(self, context: Any = None) -> bool:
        super().validate(context)
        validate_parameter_data(self.to_data())
        if self.has_default:
            _value_data(
                self.default,
                dtype=self.dtype,
                unit=self.unit,
                where="parameter %r default" % self.name,
            )
            if self.domain is not None:
                try:
                    self.domain.check(self.default, who="%s default" % self.name)
                except (TypeError, ValueError):
                    raise ValueError(
                        _domain_error(self.name, self.domain, self.default, "compile")
                    ) from None
        return True

    def freeze(self) -> ParameterDeclaration:
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("%s is immutable" % type(self).__name__)
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("%s is immutable" % type(self).__name__)

    def __eq__(self, other: Any) -> bool:
        return type(self) is type(other) and self.to_data() == other.to_data()

    def __hash__(self) -> int:
        return hash(json.dumps(self.to_data(), sort_keys=True, separators=(",", ":")))


class RuntimeParam(ParameterDeclaration):
    category = "runtime_param"

    def __init__(
        self,
        name: Any,
        dtype: Any = Real,
        default: Any = MISSING,
        domain: Any = None,
        *,
        unit: Any = None,
        provenance: Any = None,
    ) -> None:
        self._initialize(
            name,
            kind=ParamKind.Runtime,
            dtype=dtype,
            unit=unit,
            domain=domain,
            default=default,
            default_state=(
                ParamDefaultState.Missing if default is MISSING else ParamDefaultState.Value
            ),
            storage=ParamStorage.RuntimeSlot,
            provenance=provenance,
            phase=ParamPhase.Bind,
            invalidation=ParamInvalidation.PerBind,
        )
        self.validate()

    def capabilities(self) -> Any:
        return CapabilitySet({"runtime": True, "compile_time": False})

    def check_bind(self, value: Any = MISSING) -> bool:
        if value is MISSING:
            if not self.has_default:
                raise ValueError(
                    "param %r: a value is required at the bind phase (no default declared)"
                    % self.name
                )
            value = self.default
        _value_data(value, dtype=self.dtype, unit=self.unit, where="param %r bind value" % self.name)
        if self.domain is not None:
            try:
                self.domain.check(value, who=self.name)
            except (TypeError, ValueError):
                raise ValueError(_domain_error(self.name, self.domain, value, "bind")) from None
        return True


class ConstParam(ParameterDeclaration):
    category = "const_param"

    def __init__(
        self,
        name: Any,
        value: Any,
        dtype: Any = Real,
        domain: Any = None,
        *,
        unit: Any = None,
        provenance: Any = None,
    ) -> None:
        self._initialize(
            name,
            kind=ParamKind.Const,
            dtype=dtype,
            unit=unit,
            domain=domain,
            default=value,
            default_state=ParamDefaultState.Value,
            storage=ParamStorage.Inline,
            provenance=provenance,
            phase=ParamPhase.Compile,
            invalidation=ParamInvalidation.Never,
        )
        self.validate()

    @property
    def value(self) -> Any:
        return self.default

    def capabilities(self) -> Any:
        return CapabilitySet({"runtime": False, "compile_time": True, "in_cache_key": True})


class DerivedParam(ParameterDeclaration):
    category = "derived_param"

    def __init__(
        self,
        name: Any,
        expression: Any,
        *,
        depends_on: Any,
        phase: Any,
        storage: Any,
        invalidation: Any,
        dtype: Any = Real,
        unit: Any = None,
        domain: Any = None,
        provenance: Any = None,
    ) -> None:
        if not isinstance(phase, ParamPhase):
            raise TypeError("DerivedParam phase must be a ParamPhase")
        if not isinstance(storage, ParamStorage):
            raise TypeError("DerivedParam storage must be a ParamStorage")
        if storage not in (ParamStorage.Inline, ParamStorage.DerivedCache):
            raise ValueError("DerivedParam storage must be Inline or DerivedCache")
        if not isinstance(invalidation, ParamInvalidation):
            raise TypeError("DerivedParam invalidation must be a ParamInvalidation")
        dependencies, _ = _dependency_rows(depends_on)
        self._initialize(
            name,
            kind=ParamKind.Derived,
            dtype=dtype,
            unit=unit,
            domain=domain,
            default=MISSING,
            default_state=ParamDefaultState.Derived,
            storage=storage,
            provenance=provenance,
            phase=phase,
            invalidation=invalidation,
            expression=expression,
            depends_on=dependencies,
        )
        self.validate()


__all__ = [
    "MISSING",
    "PARAM_DECLARATION_SCHEMA_VERSION",
    "ParamDefaultState",
    "ParamInvalidation",
    "ParamKind",
    "ParamPhase",
    "ParamProvenance",
    "ParamStorage",
    "ParameterDeclaration",
    "RuntimeParam",
    "ConstParam",
    "DerivedParam",
    "validate_parameter_data",
]
