"""Explicit cold-path allocation authority for a differently partitioned archive.

The installed budget still describes the destination.  The restart workspace is
additional temporary source-storage authority; archive headers never enlarge it.
Scientific schemas, ranked image tables and native store capacities are validated
unchanged after this inert decode and before publication.
"""
from __future__ import annotations

from typing import Any
from ._checkpoint_contract import CheckpointResourceBudget, _capacity, require_checkpoint_resource_budget

_DEFAULT_WORKSPACE = 64 * 1024 * 1024


def restart_workspace_capacity(owner: Any, requested: int | None, *, bit_identical: bool) -> int | None:
    if type(bit_identical) is not bool:
        raise TypeError("restart bit_identical must be an exact bool")
    if requested is not None:
        if type(requested) is not int:
            raise TypeError("restart workspace_byte_capacity must be an exact integer")
        _capacity(requested, where="restart workspace_byte_capacity", positive=True)
    budget = require_checkpoint_resource_budget(owner)
    if bit_identical or budget.runtime_kind not in {"amr", "multi_layout_amr"}:
        return None
    return _DEFAULT_WORKSPACE if requested is None else requested


def restart_checkpoint_resource_budget(owner: Any, requested: int | None = None, *,
                                       bit_identical: bool = False) -> CheckpointResourceBudget:
    base = require_checkpoint_resource_budget(owner)
    workspace = restart_workspace_capacity(owner, requested, bit_identical=bit_identical)
    if workspace is None:
        return base

    def extra(value: int, addition: int) -> int:
        return _capacity(value + addition, where="restart source archive capacity", positive=True)

    # A central directory entry has a 46-byte fixed prefix. Unicode manifests
    # consume four bytes per code point. These framing bounds precede ZipInfo,
    # NPY and JSON allocation, and do not depend on claimed source rank counts.
    return CheckpointResourceBudget(
        base.runtime_kind, extra(base.max_members, workspace // 46),
        extra(base.max_manifest_characters, workspace // 4),
        extra(base.max_array_bytes, workspace), extra(base.max_uncompressed_bytes, workspace),
        extra(base.max_archive_bytes, workspace),
        base.authority + ":restart-source-workspace:" + str(workspace),
    )
