"""Shared exact block ownership for consumer resources and their runtime collectives."""
from __future__ import annotations

from typing import Any

from pops.model.ownership import OwnerKind, OwnerPath


def consumer_block_owner(
    reference: Any, block_ids: tuple[str, ...], *, case_owner: OwnerPath,
) -> str | None:
    """Resolve an explicit block or a same-case field in a singleton block assembly.

    Case-owned fields intentionally have no block_ref. The existing singleton
    adapter can carry them on its sole block, but neither a foreign-case field nor
    an ambiguous multi-block field acquires an execution owner by a local name.
    """
    candidates = frozenset(block_ids)
    block_id = getattr(getattr(reference, "block_ref", None), "qualified_id", None)
    if block_id is not None:
        return block_id if block_id in candidates else None
    if getattr(reference, "kind", None) == "block":
        block_id = reference.qualified_id
        return block_id if block_id in candidates else None
    if getattr(reference, "kind", None) != "field" or len(candidates) != 1:
        return None
    root = case_owner.nodes[0]
    if root.kind is not OwnerKind.CASE or reference.owner_path.nodes[0] != root:
        return None
    return next(iter(candidates))


__all__ = ["consumer_block_owner"]
