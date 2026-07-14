"""Private adapter from accepted consumer effects to exact format writers."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pops.identity import make_identity
from pops.output._consumer_contracts import ParallelMode
from pops.output.data import OutputRequest, OutputSnapshot
from pops.output.provider import consumer_format_data
from pops.output.writers import PreparedOutputFile

from ._consumer import (
    AcceptedSideEffect,
    ConsumerPublisher,
    PreparedPublication,
    PublicationReceipt,
)

@dataclass(frozen=True, slots=True)
class OutputPreparation:
    """Exact writer input resolved from one already accepted side effect."""

    format: Any
    snapshot: OutputSnapshot
    request: OutputRequest
    target: Any
    communicator: Any = None

    def __post_init__(self) -> None:
        format_data = consumer_format_data(
            self.format, where="OutputPreparation.format")
        if type(self.snapshot) is not OutputSnapshot or type(self.request) is not OutputRequest:
            raise TypeError("output preparation requires exact snapshot/request values")
        if self.request.parallel != (format_data["parallel_mode"] == "collective"):
            raise ValueError("resolved output request parallel mode differs from its format")


class PreparedConsumerOutput(PreparedPublication):
    """Bind a verified temporary file to the identities of its accepted effect."""

    __slots__ = ("_effect", "_prepared", "_publisher_id")

    def __init__(self, effect: AcceptedSideEffect, prepared: PreparedOutputFile,
                 publisher_id: str) -> None:
        self._effect = effect
        self._prepared = prepared
        self._publisher_id = publisher_id

    @property
    def effect_identity(self):
        return self._effect.identity

    @property
    def payload_identity(self):
        return self._effect.payload.identity

    @property
    def temporary(self):
        return self._prepared.temporary

    @property
    def target(self):
        return self._prepared.target

    def publish(self) -> PublicationReceipt:
        local = self._prepared.publish()
        artifact = make_identity("scientific-output-artifact", {
            "output_identity": local.output_identity.to_data(),
            "target": local.path.as_posix(),
            "format": local.format,
        })
        return PublicationReceipt(
            self.effect_identity,
            self.payload_identity,
            self._publisher_id,
            artifact.token,
        )

    def discard(self) -> None:
        self._prepared.discard()

    def rollback(self) -> None:
        self._prepared.rollback()


class ConsumerOutputPublisher(ConsumerPublisher):
    """Dispatch only accepted scientific-output effects to their exact writer."""

    __slots__ = ("_resolve", "publisher_id")

    def __init__(self, resolve: Callable[[AcceptedSideEffect], OutputPreparation], *,
                 publisher_id: str = "pops.exact-output.v1") -> None:
        if not callable(resolve):
            raise TypeError("ConsumerOutputPublisher resolver must be callable")
        if not isinstance(publisher_id, str) or not publisher_id or publisher_id.strip() != publisher_id:
            raise TypeError("ConsumerOutputPublisher publisher_id must be canonical text")
        self._resolve = resolve
        self.publisher_id = publisher_id

    def prepare(self, effect: AcceptedSideEffect) -> PreparedConsumerOutput:
        if type(effect) is not AcceptedSideEffect:
            raise TypeError("ConsumerOutputPublisher requires an exact AcceptedSideEffect")
        preparation = self._resolve(effect)
        if type(preparation) is not OutputPreparation:
            raise TypeError("output effect resolver must return an exact OutputPreparation")
        if preparation.request.consumer_id != effect.consumer_id:
            raise ValueError("output request consumer identity differs from its accepted effect")
        if consumer_format_data(
                preparation.format, where="resolved output format") != \
                dict(effect.target.output_format):
            raise ValueError("resolved output format differs from its accepted target")
        collective = effect.target.parallel_mode is ParallelMode.COLLECTIVE
        if preparation.request.parallel != collective:
            raise ValueError("resolved output parallel mode differs from its accepted target")
        writer = preparation.format.writer()
        prepared = writer.prepare(
            preparation.snapshot,
            preparation.request,
            preparation.target,
            communicator=preparation.communicator,
        )
        if not isinstance(prepared, PreparedOutputFile):
            raise TypeError("exact output writer must return PreparedOutputFile")
        return PreparedConsumerOutput(effect, prepared, self.publisher_id)


__all__ = ["ConsumerOutputPublisher", "OutputPreparation", "PreparedConsumerOutput"]
