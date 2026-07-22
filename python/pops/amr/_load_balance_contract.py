"""Canonical data boundary for open AMR load-balance providers."""
from __future__ import annotations

import json
import math
from typing import Any

from pops.identity import make_identity


_LOAD_BALANCE_KEYS = {
    "schema_version", "provider_type", "provider_id", "provider_identity",
    "native_route", "option_schema_identity", "options", "weight_capability",
}


def _validate_option(value: Any, *, key: str) -> None:
    if type(value) is bool or isinstance(value, str):
        return
    if type(value) is int:
        if value < -(1 << 63) or value > (1 << 64) - 1:
            raise OverflowError("AMR load-balance option %r exceeds the native integer ABI" % key)
        return
    if type(value) is dict and set(value) == {"binary64"} \
            and isinstance(value["binary64"], str):
        try:
            decoded = float.fromhex(value["binary64"])
        except ValueError as exc:
            raise ValueError(
                "AMR load-balance option %r has invalid canonical binary64 data" % key
            ) from exc
        if not math.isfinite(decoded) or decoded.hex() != value["binary64"]:
            raise ValueError(
                "AMR load-balance option %r has non-canonical binary64 data" % key)
        return
    raise TypeError(
        "AMR load-balance option %r must be bool, int64/uint64, str or canonical binary64"
        % key)


def validate_load_balance_provider_data(value: Any) -> dict[str, Any]:
    """Validate detached provider data without dispatching on route or provider name."""
    if type(value) is not dict or set(value) != _LOAD_BALANCE_KEYS \
            or type(value.get("schema_version")) is not int \
            or value["schema_version"] != 1 \
            or value.get("provider_type") != "amr_load_balance_provider":
        raise TypeError(
            "AMR load balance requires the exact amr_load_balance_provider schema-v1")
    for key in (
            "provider_id", "provider_identity", "native_route",
            "option_schema_identity"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise TypeError("AMR load-balance %s must be non-empty text" % key)
    if type(value.get("options")) is not dict:
        raise TypeError("AMR load-balance options must be one exact mapping")
    for key, option in value["options"].items():
        if not isinstance(key, str) or not key:
            raise TypeError("AMR load-balance option keys must be non-empty text")
        _validate_option(option, key=key)
    capability = value.get("weight_capability")
    if type(capability) is not dict \
            or set(capability) != {"authenticated", "consumed"} \
            or type(capability.get("authenticated")) is not bool \
            or type(capability.get("consumed")) is not bool \
            or not capability["authenticated"]:
        raise TypeError("AMR load balance must authenticate its exact weight capability")
    try:
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("AMR load balance must expose strict JSON data") from exc
    identity_payload = {
        key: item for key, item in value.items() if key != "provider_identity"
    }
    expected_identity = make_identity(
        "amr-load-balance-provider", identity_payload).token
    if value["provider_identity"] != expected_identity:
        raise ValueError(
            "AMR load-balance provider_identity does not authenticate its options and "
            "capabilities")
    return value


def load_balance_provider_data(authority: Any) -> dict[str, Any]:
    """Invoke and authenticate the single-method public extension protocol."""
    protocol = getattr(authority, "load_balance_provider_data", None)
    if not callable(protocol):
        raise TypeError(
            "AMR.load_balance must implement load_balance_provider_data()")
    first, second = protocol(), protocol()
    if first != second:
        raise TypeError("AMR.load_balance must expose deterministic mapping data")
    return validate_load_balance_provider_data(first)


__all__ = ["load_balance_provider_data", "validate_load_balance_provider_data"]
