"""Checkpoint roles emitted with the same resolved owner authority as the Program."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pops.identity.presentation import PresentationMap


@dataclass(frozen=True, slots=True)
class _CheckpointProvenance:
    presentation: PresentationMap
    cache_identity: str


_provenance: ContextVar[_CheckpointProvenance | None] = ContextVar(
    "checkpoint_provenance", default=None)


@contextmanager
def checkpoint_provenance(presentation: PresentationMap | None) -> Iterator[None]:
    from pops.identity import make_identity

    provenance = None
    if presentation is not None:
        # The emitted roles depend on the projection recipes and their declared order.
        # The calculated lookup index is reconstructed from these immutable inputs.
        declaration = {item.name: getattr(presentation, item.name)
                       for item in fields(presentation) if item.init}
        identity = make_identity("checkpoint-provenance", declaration)
        provenance = _CheckpointProvenance(presentation, identity.token)
    token = _provenance.set(provenance)
    try:
        yield
    finally:
        _provenance.reset(token)


def model_cache_components(target: str) -> dict[str, str]:
    provenance = _provenance.get()
    if provenance is None or target != "system":
        return {}
    return {"checkpoint_projection": provenance.cache_identity}


def owner_role(owner: str) -> str | None:
    provenance = _provenance.get()
    if provenance is None:
        return None
    return provenance.presentation.project_owner_string(owner)


def provider_role(producer: str, owner: str, space: str, component: str) -> str:
    provenance = _provenance.get()
    if provenance is None:
        return ""
    if producer.startswith("pops.handle.v1::"):
        producer = provenance.presentation.project_qualified_id(producer)
    from pops.identity import make_identity
    return make_identity("checkpoint-auxiliary-provider", {
        "producer": producer, "owner": owner, "space": space, "component": component,
    }).token


def consumer_role(owner: str, suffix: str) -> str:
    projected = owner_role(owner)
    return "" if projected is None else projected + suffix
