"""Prepared Krylov method providers used by Program lowering.

The compiler core knows only universal stopping controls and mathematical problem facts.  Every
method-specific option is an opaque canonical mapping owned, validated and emitted by its provider.
Builtin presets and external methods use the same append-only registry and exact authority record;
numerical execution remains entirely native C++.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
from threading import RLock
from typing import Any

from pops._frozen_data import freeze_data
from pops.identity import (
    ScalarLiteral,
    canonical_bytes,
    exact_cpp_int,
    exact_numeric_scalar,
    scalar_cpp,
    scalar_data,
)
from pops.native_components import PreparedNativeComponent

from ._native_contract import PREPARED_GMRES_MAX_RESTART


_PROVIDER_SCHEMA_VERSION = 2


def _exact_nonempty(value: Any, *, where: str) -> str:
    if type(value) is not str or not value:
        raise TypeError("%s must be a non-empty exact string" % where)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class PreparedKrylovMethodUse:
    """Universal solve facts plus provider-owned, opaque method options."""

    rel_tol: Any
    abs_tol: Any
    max_iterations: int
    components: int
    input_ghosts: int
    preconditioned: bool
    operator_properties: Mapping[str, bool]
    declared_nullspace: bool
    method_options: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("prepared Krylov max_iterations must be positive")
        if type(self.components) is not int or self.components < 1:
            raise ValueError("prepared Krylov components must be positive")
        if type(self.input_ghosts) is not int or self.input_ghosts < 0:
            raise ValueError("prepared Krylov input_ghosts must be non-negative")
        if type(self.preconditioned) is not bool or type(self.declared_nullspace) is not bool:
            raise TypeError("prepared Krylov problem flags must be exact booleans")
        properties = freeze_data(
            dict(self.operator_properties), "prepared Krylov operator properties"
        )
        options = freeze_data(dict(self.method_options), "prepared Krylov method options")
        canonical_bytes(_plain(properties))
        canonical_bytes(_plain(options))
        object.__setattr__(self, "operator_properties", properties)
        object.__setattr__(self, "method_options", options)


PreparedKrylovOptionPreparer = Callable[[Mapping[str, Any]], Mapping[str, Any]]
PreparedKrylovValidator = Callable[[PreparedKrylovMethodUse, str], None]
PreparedKrylovEmitter = Callable[[Any, Mapping[str, Any]], str]


@dataclass(frozen=True, slots=True)
class PreparedKrylovMethodProvider:
    provider_id: str
    interface_version: int
    options_schema: str
    emitter_id: str
    capabilities: Any
    native_component: PreparedNativeComponent
    option_preparer: PreparedKrylovOptionPreparer = field(repr=False, compare=False)
    validator: PreparedKrylovValidator = field(repr=False, compare=False)
    emitter: PreparedKrylovEmitter = field(repr=False, compare=False)

    def authority(self) -> dict[str, Any]:
        capabilities = freeze_data(self.capabilities, "prepared Krylov capabilities")
        canonical_bytes(_plain(capabilities))
        return {
            "schema_version": _PROVIDER_SCHEMA_VERSION,
            "provider_id": self.provider_id,
            "interface_version": self.interface_version,
            "options_schema": self.options_schema,
            "emitter_id": self.emitter_id,
            "capabilities": _plain(capabilities),
            "native_component": self.native_component.authority(),
        }

    def prepare_options(self, options: Any) -> dict[str, Any]:
        if not isinstance(options, Mapping):
            raise TypeError("prepared Krylov method options must be a mapping")
        prepared = self.option_preparer(dict(options))
        if not isinstance(prepared, Mapping):
            raise TypeError("prepared Krylov option preparer must return a mapping")
        frozen = freeze_data(dict(prepared), "prepared Krylov method options")
        canonical_bytes(_plain(frozen))
        return _plain(frozen)

    def authenticate_use(self, use: PreparedKrylovMethodUse) -> None:
        if self.prepare_options(use.method_options) != _plain(use.method_options):
            raise ValueError("prepared Krylov method options are not canonical")

    def validate_use(self, use: PreparedKrylovMethodUse, *, where: str) -> None:
        self.authenticate_use(use)
        result = self.validator(use, where)
        if result is not None:
            raise TypeError("prepared Krylov provider validator must return None")

    def emit_cpp(self, node: Any) -> str:
        options = self.prepare_options(node.attrs.get("method_options"))
        expression = self.emitter(node, options)
        if type(expression) is not str or not expression:
            raise TypeError("prepared Krylov provider emitter must return a C++ expression")
        return expression


_registry_lock = RLock()
_providers_by_id: dict[str, PreparedKrylovMethodProvider] = {}


def register_prepared_krylov_method_provider(
    provider: PreparedKrylovMethodProvider,
) -> PreparedKrylovMethodProvider:
    if type(provider) is not PreparedKrylovMethodProvider:
        raise TypeError("prepared Krylov plugins must register an exact provider record")
    for name in ("provider_id", "options_schema", "emitter_id"):
        _exact_nonempty(getattr(provider, name), where="prepared Krylov %s" % name)
    if type(provider.interface_version) is not int or provider.interface_version < 1:
        raise ValueError("prepared Krylov interface_version must be positive")
    if type(provider.native_component) is not PreparedNativeComponent:
        raise TypeError("prepared Krylov provider requires a PreparedNativeComponent")
    for name in ("option_preparer", "validator", "emitter"):
        if not callable(getattr(provider, name)):
            raise TypeError("prepared Krylov provider is missing callable %s" % name)
    provider.authority()
    with _registry_lock:
        if provider.provider_id in _providers_by_id:
            raise ValueError("prepared Krylov provider %r is already registered" % provider.provider_id)
        _providers_by_id[provider.provider_id] = provider
    return provider


def prepared_krylov_method_provider_by_id(provider_id: Any) -> PreparedKrylovMethodProvider:
    provider_id = _exact_nonempty(provider_id, where="prepared Krylov provider_id")
    with _registry_lock:
        provider = _providers_by_id.get(provider_id)
    if provider is None:
        raise NotImplementedError("prepared Krylov provider %r is not registered" % provider_id)
    return provider


def prepared_krylov_method_provider_from_identity(
    identity: Any,
) -> PreparedKrylovMethodProvider:
    if not isinstance(identity, Mapping):
        raise TypeError("prepared Krylov provider identity must be a mapping")
    provider_id = identity.get("provider_id")
    if type(provider_id) is not str:
        raise TypeError("prepared Krylov provider identity requires an exact provider_id")
    provider = prepared_krylov_method_provider_by_id(provider_id)
    if identity != provider.authority():
        raise ValueError("prepared Krylov provider identity is inconsistent")
    return provider


def prepared_krylov_method_provider_from_attrs(
    attrs: Mapping[str, Any],
) -> PreparedKrylovMethodProvider:
    if not isinstance(attrs, Mapping):
        raise TypeError("solve_linear attributes must be a mapping")
    provider = prepared_krylov_method_provider_from_identity(attrs.get("method_provider"))
    prepared = provider.prepare_options(attrs.get("method_options"))
    if prepared != attrs.get("method_options"):
        raise ValueError("solve_linear method options are not canonical")
    return provider


def _common(use: PreparedKrylovMethodUse, where: str) -> None:
    try:
        rel_tol = float(use.rel_tol)
        abs_tol = float(use.abs_tol)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("%s controls must be finite scalar values" % where) from exc
    if not math.isfinite(rel_tol) or not 0 <= rel_tol < 1:
        raise ValueError("%s rel_tol must be finite and in [0, 1)" % where)
    if not math.isfinite(abs_tol) or abs_tol < 0:
        raise ValueError("%s abs_tol must be finite and non-negative" % where)
    if rel_tol == 0 and abs_tol == 0:
        raise ValueError("%s requires at least one non-zero stopping tolerance" % where)


def _empty_options(options: Mapping[str, Any]) -> Mapping[str, Any]:
    if options:
        raise ValueError("this prepared Krylov provider accepts no method options")
    return {}


def _gmres_options(options: Mapping[str, Any]) -> Mapping[str, Any]:
    if set(options) != {"restart"}:
        raise ValueError("prepared GMRES options require exactly restart")
    return {
        "restart": exact_cpp_int(
            options["restart"],
            where="GMRES restart (MPI Arnoldi reduction count requires restart + 1)",
            minimum=1,
            maximum=PREPARED_GMRES_MAX_RESTART,
        )
    }


def _richardson_options(options: Mapping[str, Any]) -> Mapping[str, Any]:
    if set(options) != {"relaxation"}:
        raise ValueError("prepared Richardson options require exactly relaxation")
    raw = options["relaxation"]
    if isinstance(raw, Mapping):
        kind = raw.get("kind")
        if kind == "integer" and set(raw) == {"kind", "value"}:
            relaxation_literal = ScalarLiteral("integer", int(raw["value"]))
        elif kind == "rational" and set(raw) == {"kind", "numerator", "denominator"}:
            relaxation_literal = ScalarLiteral(
                "rational", (int(raw["numerator"]), int(raw["denominator"]))
            )
        elif kind in ("decimal", "binary64") and set(raw) == {"kind", "value"}:
            relaxation_literal = ScalarLiteral(kind, raw["value"])
        else:
            raise ValueError("prepared Richardson relaxation literal is not canonical")
        if scalar_data(relaxation_literal) != dict(raw):
            raise ValueError("prepared Richardson relaxation literal is not canonical")
    else:
        relaxation_literal = ScalarLiteral.from_value(
            exact_numeric_scalar(raw, where="Richardson relaxation")
        )
    relaxation = relaxation_literal.to_python()
    if not math.isfinite(float(relaxation)) or relaxation <= 0:
        raise ValueError("prepared Richardson relaxation must be finite and positive")
    return {"relaxation": scalar_data(relaxation_literal)}


def _validate_cg(use: PreparedKrylovMethodUse, where: str) -> None:
    _common(use, where)
    if use.preconditioned:
        raise ValueError("%s CG has no prepared preconditioner slot" % where)
    props = use.operator_properties
    positive = (
        props.get("symmetric") is True
        and (
            props.get("positive_definite_on_nullspace_complement") is True
            if use.declared_nullspace
            else props.get("positive_definite") is True
        )
    )
    if not positive:
        required = (
            "positive_definite_on_nullspace_complement"
            if use.declared_nullspace
            else "positive_definite"
        )
        raise ValueError("%s CG requires an authenticated %s certificate" % (where, required))


def _validate_generic(use: PreparedKrylovMethodUse, where: str) -> None:
    _common(use, where)


def _validate_richardson(use: PreparedKrylovMethodUse, where: str) -> None:
    _common(use, where)
    if use.preconditioned:
        raise ValueError("%s Richardson has no prepared preconditioner slot" % where)


def _builtin(expression: str) -> PreparedKrylovEmitter:
    return lambda _node, _options: expression


def _emit_gmres(_node: Any, options: Mapping[str, Any]) -> str:
    return "pops::gmres_krylov_method(%d)" % options["restart"]


def _emit_richardson(_node: Any, options: Mapping[str, Any]) -> str:
    prepared = _richardson_options(options)["relaxation"]
    kind = prepared["kind"]
    if kind == "integer":
        literal = ScalarLiteral("integer", int(prepared["value"]))
    elif kind == "rational":
        literal = ScalarLiteral(
            "rational", (int(prepared["numerator"]), int(prepared["denominator"]))
        )
    else:
        literal = ScalarLiteral(kind, prepared["value"])
    return "pops::richardson_krylov_method(static_cast<pops::Real>(%s))" % scalar_cpp(
        literal
    )


_BUILTIN_COMPONENT = PreparedNativeComponent.pops_builtin(
    "pops.prepared-krylov-methods",
    entry_headers=("pops/numerics/elliptic/linear/generic_krylov.hpp",),
)


def _register_builtins() -> None:
    records = (
        (
            "pops.krylov.cg",
            "pops.krylov.cg.options@1",
            _empty_options,
            _validate_cg,
            _builtin("pops::cg_krylov_method()"),
            False,
        ),
        (
            "pops.krylov.bicgstab",
            "pops.krylov.bicgstab.options@1",
            _empty_options,
            _validate_generic,
            _builtin("pops::bicgstab_krylov_method()"),
            True,
        ),
        (
            "pops.krylov.gmres",
            "pops.krylov.gmres.options@1",
            _gmres_options,
            _validate_generic,
            _emit_gmres,
            True,
        ),
        (
            "pops.krylov.richardson",
            "pops.krylov.richardson.options@1",
            _richardson_options,
            _validate_richardson,
            _emit_richardson,
            False,
        ),
    )
    for (
        provider_id,
        options_schema,
        preparer,
        validator,
        emitter,
        left_preconditioning,
    ) in records:
        register_prepared_krylov_method_provider(
            PreparedKrylovMethodProvider(
                provider_id=provider_id,
                interface_version=1,
                options_schema=options_schema,
                emitter_id=provider_id + "@1",
                capabilities={
                    "contract_version": 1,
                    "left_preconditioning": left_preconditioning,
                },
                native_component=_BUILTIN_COMPONENT,
                option_preparer=preparer,
                validator=validator,
                emitter=emitter,
            )
        )


_register_builtins()


__all__ = [
    "PreparedKrylovMethodProvider",
    "PreparedKrylovMethodUse",
    "prepared_krylov_method_provider_by_id",
    "prepared_krylov_method_provider_from_attrs",
    "prepared_krylov_method_provider_from_identity",
    "register_prepared_krylov_method_provider",
]
