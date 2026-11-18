"""Checkpoint roles emitted with the same resolved owner authority as the Program."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pops.identity.presentation import PresentationMap


_presentation: ContextVar[PresentationMap | None] = ContextVar(
    "checkpoint_presentation", default=None)


@contextmanager
def checkpoint_provenance(presentation: PresentationMap | None) -> Iterator[None]:
    token = _presentation.set(presentation)
    try:
        yield
    finally:
        _presentation.reset(token)


def owner_role(owner: str) -> str | None:
    presentation = _presentation.get()
    if presentation is None:
        return None
    return presentation.project_owner_string(owner)


def qualified_role(qid: str) -> str:
    presentation = _presentation.get()
    if presentation is None:
        return qid
    return presentation.project_qualified_id(qid)


def provider_role(producer: str, owner: str, space: str, component: str) -> str:
    presentation = _presentation.get()
    if presentation is None:
        return ""
    if producer.startswith("pops.handle.v1::"):
        producer = presentation.project_qualified_id(producer)
    from pops.identity import make_identity
    return make_identity("pops.checkpoint-auxiliary-provider", {
        "producer": producer, "owner": owner, "space": space, "component": component,
    }).token


def consumer_role(owner: str, suffix: str) -> str:
    projected = owner_role(owner)
    return "" if projected is None else projected + suffix
