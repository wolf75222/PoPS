"""Structural scientific-output provider contract.

Providers are extension objects, not members of a central format registry.  The runtime retains the
immutable provider for behavior and authenticates only its canonical ``consumer_data`` projection.
"""
from __future__ import annotations

from typing import Any

from pops.identity import canonical_bytes


_REQUIRED = frozenset({"schema_version", "provider_id", "extension", "parallel_mode"})
_MODES = frozenset({"serial", "collective", "per_rank"})


def consumer_format_data(provider: Any, *, where: str = "output format") -> dict[str, Any]:
    """Validate a provider structurally and return its deterministic canonical evidence."""
    if getattr(provider, "__pops_ir_immutable__", False) is not True:
        raise TypeError("%s provider must declare immutable semantic state" % where)
    consumer_data = getattr(provider, "consumer_data", None)
    writer_factory = getattr(provider, "writer", None)
    if not callable(consumer_data) or not callable(writer_factory):
        raise TypeError("%s provider must implement consumer_data() and writer()" % where)
    first, second = consumer_data(), consumer_data()
    if type(first) is not dict or type(second) is not dict or first != second:
        raise TypeError("%s consumer_data() must return one deterministic dict" % where)
    if set(first) < _REQUIRED:
        raise ValueError(
            "%s consumer_data() lacks required keys %s"
            % (where, sorted(_REQUIRED - set(first)))
        )
    if first["schema_version"] != 1:
        raise ValueError("%s consumer_data schema_version must be 1" % where)
    provider_id, extension, mode = (
        first["provider_id"], first["extension"], first["parallel_mode"])
    if not isinstance(provider_id, str) or not provider_id or provider_id.strip() != provider_id:
        raise TypeError("%s provider_id must be canonical text" % where)
    if not isinstance(extension, str) or not extension.startswith(".") \
            or extension.strip() != extension or "/" in extension:
        raise TypeError("%s extension must be a canonical file suffix" % where)
    if mode not in _MODES:
        raise ValueError("%s parallel_mode must be serial, collective, or per_rank" % where)
    # The canonical encoder refuses opaque provider state and proves deterministic serialization.
    canonical_bytes(first)
    writer = writer_factory()
    if not callable(getattr(writer, "prepare", None)):
        raise TypeError("%s writer() must return an object implementing prepare()" % where)
    return first


__all__ = ["consumer_format_data"]
