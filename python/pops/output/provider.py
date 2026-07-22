"""Structural scientific-output provider contract.

Providers are extension objects, not members of a central format registry.  The runtime retains the
immutable provider for behavior, authenticates its canonical ``consumer_data`` projection, and
requires the returned writer to preflight its own execution dependencies before native install.
"""
from __future__ import annotations

from typing import Any

from pops.identity import canonical_bytes


_REQUIRED = frozenset({
    "schema_version", "provider_id", "format_name", "extension", "parallel_mode",
})
_MODES = frozenset({"serial", "root", "collective", "per_rank"})
_LAYOUT_CARDINALITIES = frozenset({"single", "multiple"})
_TARGET_POLICIES = frozenset({"literal", "immutable_sample"})


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
    missing = _REQUIRED - set(first)
    if missing:
        raise ValueError(
            "%s consumer_data() lacks required keys %s"
            % (where, sorted(missing))
        )
    if first["schema_version"] != 1:
        raise ValueError("%s consumer_data schema_version must be 1" % where)
    provider_id, format_name, extension, mode = (
        first["provider_id"], first["format_name"], first["extension"],
        first["parallel_mode"],
    )
    if not isinstance(provider_id, str) or not provider_id or provider_id.strip() != provider_id:
        raise TypeError("%s provider_id must be canonical text" % where)
    if not isinstance(format_name, str) or not format_name \
            or format_name.strip() != format_name:
        raise TypeError("%s format_name must be canonical text" % where)
    if not isinstance(extension, str) or not extension.startswith(".") \
            or extension.strip() != extension or "/" in extension or "\\" in extension \
            or extension.endswith(".series"):
        raise TypeError("%s extension must be a canonical file suffix" % where)
    if mode not in _MODES:
        raise ValueError(
            "%s parallel_mode must be serial, root, collective, or per_rank" % where)
    target_policy = first.get("target_policy", "literal")
    if target_policy not in _TARGET_POLICIES:
        raise ValueError(
            "%s target_policy must be literal or immutable_sample" % where)
    selection_contract = first.get("selection_contract")
    if selection_contract is not None:
        if type(selection_contract) is not dict or set(selection_contract) != {
                "schema_version", "layout_cardinality"}:
            raise TypeError("%s selection_contract has an unknown schema" % where)
        if selection_contract["schema_version"] != 1:
            raise ValueError("%s selection_contract schema_version must be 1" % where)
        if selection_contract["layout_cardinality"] not in _LAYOUT_CARDINALITIES:
            raise ValueError(
                "%s selection_contract layout_cardinality must be single or multiple"
                % where)
    # The canonical encoder refuses opaque provider state and proves deterministic serialization.
    canonical_bytes(first)
    # Do not instantiate the writer while projecting semantic data.  ``consumer_format_data`` is
    # used in local constructors as well as distributed runtime paths; a rank-local factory failure
    # here could let peers enter a collective writer.  Runtime preflight and publication construct
    # the writer inside explicit all-rank error consensus before any backend collective begins.
    return first


def immutable_sample_targets(format_data: Any) -> bool:
    """Return whether an authenticated format versions even explicitly suffixed targets."""
    if type(format_data) is not dict:
        raise TypeError("format_data must be an authenticated exact dict")
    policy = format_data.get("target_policy", "literal")
    if policy not in _TARGET_POLICIES:
        raise ValueError("format_data has an unsupported target_policy")
    return policy == "immutable_sample"


__all__ = ["consumer_format_data", "immutable_sample_targets"]
