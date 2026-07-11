"""Alias validation kept separate from the BindSchema orchestration value."""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pops._manifest_protocol import strict_string
from pops.model.handles import ParamHandle


def validate_authoring_aliases(aliases: Any, by_handle: Mapping) -> Mapping[str, ParamHandle]:
    if not isinstance(aliases, Mapping):
        raise TypeError("BindSchema aliases must be a ParamHandle mapping")
    checked = {}
    for alias, canonical in aliases.items():
        if not isinstance(alias, ParamHandle) or not isinstance(canonical, ParamHandle):
            raise TypeError("BindSchema aliases must map ParamHandle to ParamHandle")
        if canonical not in by_handle:
            raise ValueError("BindSchema alias targets an unknown canonical ParamHandle")
        alias_key = alias.qualified_id
        previous = checked.get(alias_key)
        if previous is not None and previous != canonical:
            raise ValueError("BindSchema alias resolves to multiple parameter slots")
        checked[alias_key] = canonical
    return MappingProxyType(checked)


def validate_serialized_aliases(aliases: Any, slots: Any) -> Mapping[str, ParamHandle]:
    if not isinstance(aliases, Mapping):
        raise TypeError("BindSchema aliases payload must be a mapping")
    by_qid = {slot.qid: slot.handle for slot in slots}
    checked = {}
    for alias_qid, target_qid in aliases.items():
        strict_string(alias_qid, where="BindSchema alias qid")
        strict_string(target_qid, where="BindSchema alias target qid")
        target = by_qid.get(target_qid)
        if target is None:
            raise ValueError(
                "BindSchema alias %r targets unknown parameter slot %r" % (alias_qid, target_qid))
        checked[alias_qid] = target
    return MappingProxyType(checked)


__all__ = ["validate_authoring_aliases", "validate_serialized_aliases"]
