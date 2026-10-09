"""Exact checkpoint origin schema shared by consumers and runtime envelopes."""
from collections.abc import Mapping
from typing import Any

BOUND_INITIAL_CHECKPOINT_SCHEMA_VERSION = 2
RUN_ORIGIN_CHECKPOINT_SCHEMA_VERSION = 1
_BOUND_INITIAL_ORIGIN = {"schema_version": 1, "kind": "bound_initial"}


def _is_bound_initial_origin(value: Any) -> bool:
    return isinstance(value, Mapping) and set(value) == {"schema_version", "kind"} \
        and type(value["schema_version"]) is int and value == _BOUND_INITIAL_ORIGIN
