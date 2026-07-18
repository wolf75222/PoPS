"""Builtin field-nullspace providers registered through the public generic protocol."""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from ._prepared_field_nullspace_registry import (
    PreparedFieldNullspaceBinding,
    PreparedFieldNullspaceDefaultPolicy,
    PreparedFieldNullspaceFacts,
    PreparedFieldNullspaceProvider,
    PreparedFieldNullspaceResolution,
    register_prepared_field_nullspace_provider,
    register_prepared_field_nullspace_default_policy,
)


_NATIVE_PROVIDER_ROUTE = "pops.field-nullspace.operator-topology-derived"
_NATIVE_OPTION_SCHEMA = "pops.field-nullspace.operator-topology-derived.options@1"


def _native_real(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s must be a finite real" % where)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s cannot be represented by the native real type" % where) from exc
    if not math.isfinite(result):
        raise ValueError("%s must lower to a finite native real" % where)
    return result


def _native_contract(gauge_value: float) -> dict[str, Any]:
    return {
        "provider_route": _NATIVE_PROVIDER_ROUTE,
        "schema_identity": _NATIVE_OPTION_SCHEMA,
        "options": {"gauge.value": gauge_value},
    }


def _nonsingular_author(
    options: Mapping[str, Any],
    gauge: Any,
    facts: PreparedFieldNullspaceFacts,
    where: str,
) -> PreparedFieldNullspaceResolution:
    if options or gauge is not None:
        raise ValueError("%s nonsingular field requires no nullspace options or gauge" % where)
    if facts.kernel_components != 0:
        raise ValueError("%s nonsingular assertion contradicts the derived kernel" % where)
    return PreparedFieldNullspaceResolution(_native_contract(0.0), False)


def _constant_author(
    options: Mapping[str, Any],
    gauge: Any,
    facts: PreparedFieldNullspaceFacts,
    where: str,
) -> PreparedFieldNullspaceResolution:
    if options:
        raise ValueError("%s constant field nullspace takes no options" % where)
    from pops.fields.gauges import MeanValueGauge

    if type(gauge) is not MeanValueGauge:
        raise TypeError("%s constant field nullspace requires exactly MeanValueGauge(value)" % where)
    value = _native_real(gauge.value, where="%s MeanValueGauge value" % where)
    if facts.kernel_components != 1:
        raise ValueError("%s constant nullspace requires exactly one derived kernel component" % where)
    return PreparedFieldNullspaceResolution(_native_contract(value), True)


def _validate_native_contract(
    binding: PreparedFieldNullspaceBinding,
    *,
    where: str,
) -> float:
    contract = binding.resolution.native_contract
    expected = {"provider_route", "schema_identity", "options"}
    if not isinstance(contract, Mapping) or set(contract) != expected:
        raise ValueError("%s field nullspace native contract has an invalid shape" % where)
    if contract["provider_route"] != _NATIVE_PROVIDER_ROUTE:
        raise ValueError("%s field nullspace native provider route changed" % where)
    if contract["schema_identity"] != _NATIVE_OPTION_SCHEMA:
        raise ValueError("%s field nullspace native option schema changed" % where)
    options = contract["options"]
    if not isinstance(options, Mapping) or set(options) != {"gauge.value"}:
        raise ValueError("%s field nullspace native options have an invalid shape" % where)
    return _native_real(options["gauge.value"], where="%s gauge.value" % where)


def _validate_nonsingular_resolution(
    binding: PreparedFieldNullspaceBinding,
    where: str,
) -> None:
    value = _validate_native_contract(binding, where=where)
    if binding.facts.kernel_components != 0 or binding.resolution.singular:
        raise ValueError("%s nonsingular provider contradicts its topology facts" % where)
    if value != 0.0:
        raise ValueError("%s nonsingular provider requires its inert gauge value to be zero" % where)


def _validate_constant_resolution(
    binding: PreparedFieldNullspaceBinding,
    where: str,
) -> None:
    _validate_native_contract(binding, where=where)
    if binding.facts.kernel_components != 1 or not binding.resolution.singular:
        raise ValueError("%s constant provider contradicts its topology facts" % where)


def _install_registered(context: Any, binding: PreparedFieldNullspaceBinding) -> None:
    context.install_registered_nullspace(binding)


_NONSINGULAR = register_prepared_field_nullspace_provider(
    PreparedFieldNullspaceProvider(
        provider_id="pops.field-nullspace.nonsingular",
        version=1,
        resolver_id="pops.field-nullspace.nonsingular.resolve@1",
        resolution_validator_id="pops.field-nullspace.nonsingular.validate-resolution@1",
        installer_id="pops.field-nullspace.nonsingular.install@1",
        capabilities={"kernel_components": 0, "gauge": "none"},
        author=_nonsingular_author,
        resolution_validator=_validate_nonsingular_resolution,
        native_installer=_install_registered,
    )
)


_CONSTANT = register_prepared_field_nullspace_provider(
    PreparedFieldNullspaceProvider(
        provider_id="pops.field-nullspace.constant",
        version=1,
        resolver_id="pops.field-nullspace.constant.resolve@1",
        resolution_validator_id="pops.field-nullspace.constant.validate-resolution@1",
        installer_id="pops.field-nullspace.constant.install@1",
        capabilities={"kernel_components": 1, "gauge": "mean-value"},
        author=_constant_author,
        resolution_validator=_validate_constant_resolution,
        native_installer=_install_registered,
    )
)


def _resolve_builtin_default(
    facts: PreparedFieldNullspaceFacts,
) -> tuple[PreparedFieldNullspaceProvider, Mapping[str, Any]]:
    if facts.kernel_components == 0:
        return _NONSINGULAR, {}
    if facts.kernel_components == 1:
        return _CONSTANT, {}
    raise NotImplementedError(
        "builtin field nullspace policy has no provider for %d kernel components"
        % facts.kernel_components
    )


_DEFAULT_POLICY = register_prepared_field_nullspace_default_policy(
    PreparedFieldNullspaceDefaultPolicy(
        policy_id="pops.field-nullspace.operator-topology-default",
        version=1,
        resolver=_resolve_builtin_default,
    )
)


def constant_field_nullspace_provider() -> PreparedFieldNullspaceProvider:
    return _CONSTANT


__all__ = ["constant_field_nullspace_provider"]
