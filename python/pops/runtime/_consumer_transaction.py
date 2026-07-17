"""Prepare/accept/reject transaction for ConsumerGraph side effects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pops.identity import Identity, make_identity

from pops.output._consumer_contracts import (
    ConsumerCursorSet,
    ParallelMode,
    Retry,
    SkipSampleReported,
)
from ._consumer_effects import AcceptedSideEffect, EffectPlan


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TypeError("%s must be non-empty canonical text" % where)
    return value


def _identity(value: Any, domain: str, where: str) -> Identity:
    if type(value) is not Identity or value.domain != domain:
        raise TypeError("%s must be an exact %s Identity" % (where, domain))
    return value


@dataclass(frozen=True, slots=True)
class PublicationReceipt:
    """Completion certificate returned only after one atomic artifact publication."""

    effect_identity: Identity
    payload_identity: Identity
    publisher_id: str
    artifact_id: str
    parallel_mode: ParallelMode = ParallelMode.SERIAL
    rank_artifacts: tuple[tuple[int, str], ...] = ()
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        _identity(self.effect_identity, "accepted-side-effect", "PublicationReceipt.effect_identity")
        _identity(self.payload_identity, "consumer-payload", "PublicationReceipt.payload_identity")
        _text(self.publisher_id, "PublicationReceipt.publisher_id")
        _text(self.artifact_id, "PublicationReceipt.artifact_id")
        if type(self.parallel_mode) is not ParallelMode:
            raise TypeError("PublicationReceipt.parallel_mode must be an exact ParallelMode")
        rows = self.rank_artifacts or ((0, self.artifact_id),)
        if not isinstance(rows, tuple):
            raise TypeError("PublicationReceipt.rank_artifacts must be a tuple")
        normalized = []
        for row in rows:
            if not isinstance(row, tuple) or len(row) != 2:
                raise TypeError("PublicationReceipt rank artifact must be a (rank, id) tuple")
            rank, artifact = row
            if isinstance(rank, bool) or type(rank) is not int or rank < 0:
                raise TypeError("PublicationReceipt artifact rank must be an integer >= 0")
            _text(artifact, "PublicationReceipt.rank_artifacts[].artifact_id")
            normalized.append((rank, artifact))
        normalized = sorted(normalized)
        if len({rank for rank, _ in normalized}) != len(normalized):
            raise ValueError("PublicationReceipt contains duplicate rank artifacts")
        ranks = tuple(rank for rank, _ in normalized)
        if self.parallel_mode is ParallelMode.PER_RANK:
            if len(ranks) < 2 or ranks != tuple(range(len(ranks))):
                raise ValueError(
                    "PER_RANK receipt must aggregate one artifact for every contiguous rank")
        elif normalized != [(0, self.artifact_id)]:
            raise ValueError(
                "%s receipt must authenticate the sole shared rank-0 artifact"
                % self.parallel_mode.name)
        object.__setattr__(self, "rank_artifacts", tuple(normalized))
        object.__setattr__(self, "identity", make_identity("consumer-publication-receipt", self._payload()))

    def _payload(self) -> dict[str, Any]:
        return {
            "effect_identity": self.effect_identity.to_data(),
            "payload_identity": self.payload_identity.to_data(),
            "publisher_id": self.publisher_id,
            "artifact_id": self.artifact_id,
            "parallel_mode": self.parallel_mode.value,
            "rank_artifacts": [
                {"rank": rank, "artifact_id": artifact}
                for rank, artifact in self.rank_artifacts
            ],
        }

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity.to_data()}


class PreparedPublication(ABC):
    """ADC-686 seam: an opaque temporary that is not yet a complete artifact."""

    @property
    @abstractmethod
    def effect_identity(self) -> Identity:
        raise NotImplementedError

    @property
    @abstractmethod
    def payload_identity(self) -> Identity:
        raise NotImplementedError

    @abstractmethod
    def publish(self) -> PublicationReceipt:
        """Atomically make the artifact visible, then return its completion receipt."""
        raise NotImplementedError

    @abstractmethod
    def discard(self) -> None:
        """Idempotently remove every temporary artifact owned by this preparation."""
        raise NotImplementedError

    @abstractmethod
    def rollback(self) -> None:
        """Idempotently remove this preparation, including an artifact it published."""
        raise NotImplementedError

    def finalize(self) -> None:
        """Release rollback-only resources after the enclosing transaction commits."""
        return None

    @property
    def recoveries(self) -> tuple[Any, ...]:
        """Typed quarantine authorities retained by a failed cleanup operation."""
        return ()


class ConsumerPublisher(ABC):
    """ADC-686 writer dispatch: format-specific work begins only behind this seam."""

    @abstractmethod
    def prepare(self, effect: AcceptedSideEffect) -> PreparedPublication:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class SkippedSampleReport:
    effect_identity: Identity
    consumer_id: str
    phase: str
    attempts: int
    reason: str

    def __post_init__(self) -> None:
        _identity(self.effect_identity, "accepted-side-effect", "SkippedSampleReport.effect_identity")
        _text(self.consumer_id, "SkippedSampleReport.consumer_id")
        if self.phase not in ("prepare", "publish"):
            raise ValueError("SkippedSampleReport.phase must be prepare or publish")
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int) \
                or self.attempts < 1:
            raise ValueError("SkippedSampleReport.attempts must be positive")
        _text(self.reason, "SkippedSampleReport.reason")

    def to_data(self) -> dict[str, Any]:
        return {
            "effect_identity": self.effect_identity.to_data(),
            "consumer_id": self.consumer_id,
            "phase": self.phase,
            "attempts": self.attempts,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ConsumerTransactionReport:
    status: str
    cursors: ConsumerCursorSet
    staged_effects: tuple[str, ...]
    published: tuple[PublicationReceipt, ...] = ()
    skipped: tuple[SkippedSampleReport, ...] = ()
    rolled_back_effects: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("accepted", "rejected", "failed"):
            raise ValueError("ConsumerTransactionReport.status is unsupported")
        if type(self.cursors) is not ConsumerCursorSet:
            raise TypeError("ConsumerTransactionReport.cursors must be an exact ConsumerCursorSet")
        if any(type(value) is not PublicationReceipt for value in self.published):
            raise TypeError("ConsumerTransactionReport.published contains an invalid receipt")
        if any(type(value) is not SkippedSampleReport for value in self.skipped):
            raise TypeError("ConsumerTransactionReport.skipped contains an invalid report")
        if self.status == "rejected" and self.published:
            raise ValueError("a rejected attempt cannot contain published artifacts")

    def to_data(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "cursors": self.cursors.to_data(),
            "staged_effects": list(self.staged_effects),
            "published": [value.to_data() for value in self.published],
            "skipped": [value.to_data() for value in self.skipped],
            "rolled_back_effects": list(self.rolled_back_effects),
            "diagnostics": list(self.diagnostics),
        }


class ConsumerPublicationError(RuntimeError):
    def __init__(self, message: str, *, report: ConsumerTransactionReport) -> None:
        super().__init__(message)
        self.report = report


class ConsumerTransaction:
    """Own temporaries until a step controller explicitly accepts or rejects the attempt."""

    __slots__ = (
        "_plan", "_publisher", "_initial_cursors", "_prepared", "_accepted",
        "_cursor_updates", "_skipped", "_state", "_finalize_pending", "_recoveries",
    )

    def __init__(
        self,
        plan: EffectPlan,
        cursors: ConsumerCursorSet,
        publisher: ConsumerPublisher,
    ) -> None:
        if type(plan) is not EffectPlan:
            raise TypeError("ConsumerTransaction requires an exact EffectPlan")
        if type(cursors) is not ConsumerCursorSet:
            raise TypeError("ConsumerTransaction requires an exact ConsumerCursorSet")
        if not isinstance(publisher, ConsumerPublisher):
            raise TypeError("ConsumerTransaction publisher must implement ConsumerPublisher")
        for effect in plan.effects:
            if cursors.for_consumer(effect.consumer_id) != effect.cursor_before:
                raise ValueError("EffectPlan cursor snapshot is stale for %s" % effect.consumer_id)
        self._plan = plan
        self._publisher = publisher
        self._initial_cursors = cursors
        self._prepared: list[tuple[AcceptedSideEffect, PreparedPublication, int]] = []
        self._accepted: list[
            tuple[AcceptedSideEffect, PreparedPublication, PublicationReceipt]
        ] = []
        self._cursor_updates = ()
        self._skipped: list[SkippedSampleReport] = []
        self._finalize_pending: list[
            tuple[AcceptedSideEffect, PreparedPublication, PublicationReceipt]
        ] = []
        self._recoveries: list[Any] = []
        self._state = "preparing"
        self._prepare_all()
        self._state = "staged"

    def _attempt_limit(self, effect: AcceptedSideEffect) -> int:
        action = effect.failure_action
        return action.max_attempts if type(action) is Retry else 1

    def _validate_prepared(
        self, effect: AcceptedSideEffect, prepared: Any,
    ) -> PreparedPublication:
        if not isinstance(prepared, PreparedPublication):
            raise TypeError("ConsumerPublisher.prepare must return PreparedPublication")
        if prepared.effect_identity != effect.identity \
                or prepared.payload_identity != effect.payload.identity:
            raise ValueError("prepared publication does not authenticate its exact effect payload")
        return prepared

    def _prepare_one(
        self, effect: AcceptedSideEffect, start_attempt: int = 0,
    ) -> tuple[PreparedPublication | None, int, Exception | None]:
        attempts, last_error = start_attempt, None
        while attempts < self._attempt_limit(effect):
            attempts += 1
            try:
                return self._validate_prepared(effect, self._publisher.prepare(effect)), attempts, None
            except Exception as exc:  # writer failures are classified by the typed action
                last_error = exc
        return None, attempts, last_error

    @staticmethod
    def _reason(error: Exception | None) -> str:
        if error is None:
            return "consumer publication failed without diagnostic"
        return "%s: %s" % (type(error).__name__, error)

    def _retain_recoveries(self, prepared: PreparedPublication) -> str | None:
        try:
            recoveries = prepared.recoveries
            if type(recoveries) is not tuple:
                raise TypeError("PreparedPublication.recoveries must return a tuple")
            for recovery in recoveries:
                if not any(value is recovery for value in self._recoveries):
                    self._recoveries.append(recovery)
        except Exception as exc:
            return "recovery ownership transfer failed: %s" % self._reason(exc)
        return None

    def _discard(self, prepared: PreparedPublication) -> str | None:
        failure = None
        try:
            prepared.discard()
        except Exception as exc:
            failure = "discard failed: %s" % self._reason(exc)
        recovery_failure = self._retain_recoveries(prepared)
        if recovery_failure is not None:
            failure = recovery_failure if failure is None else failure + "; " + recovery_failure
        return failure

    def _rollback(self, prepared: PreparedPublication) -> str | None:
        failure = None
        try:
            prepared.rollback()
        except Exception as exc:
            failure = "publication rollback failed: %s" % self._reason(exc)
        recovery_failure = self._retain_recoveries(prepared)
        if recovery_failure is not None:
            failure = recovery_failure if failure is None else failure + "; " + recovery_failure
        return failure

    def _discard_staged(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        rolled_back, diagnostics = [], []
        for effect, prepared, _ in reversed(self._prepared):
            failure = self._discard(prepared)
            if failure is None:
                rolled_back.append(effect.identity.token)
            else:
                diagnostics.append("%s: %s" % (effect.consumer_id, failure))
        self._prepared.clear()
        return tuple(rolled_back), tuple(diagnostics)

    def _rollback_accepted(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        rolled_back, diagnostics = [], []
        for effect, prepared, _ in reversed(self._accepted):
            failure = self._rollback(prepared)
            if failure is None:
                rolled_back.append(effect.identity.token)
            else:
                diagnostics.append("%s: %s" % (effect.consumer_id, failure))
        self._accepted.clear()
        self._cursor_updates = ()
        return tuple(rolled_back), tuple(diagnostics)

    def _failed(
        self,
        effect: AcceptedSideEffect,
        error: Exception | None,
        *,
        cursors: ConsumerCursorSet,
        diagnostics: tuple[str, ...] = (),
        rolled_back: tuple[str, ...] = (),
    ) -> ConsumerPublicationError:
        staged_rollback, cleanup = self._discard_staged()
        report = ConsumerTransactionReport(
            "failed",
            cursors,
            tuple(value.identity.token for value in self._plan.effects),
            (),
            tuple(self._skipped),
            rolled_back + staged_rollback,
            diagnostics + cleanup + (self._reason(error),),
        )
        self._state = "failed"
        return ConsumerPublicationError(
            "consumer %s failed under %s" % (
                effect.consumer_id, type(effect.failure_action).__name__),
            report=report,
        )

    def _prepare_all(self) -> None:
        for effect in self._plan.effects:
            prepared, attempts, error = self._prepare_one(effect)
            if prepared is not None:
                self._prepared.append((effect, prepared, attempts))
                continue
            if type(effect.failure_action) is SkipSampleReported:
                self._skipped.append(SkippedSampleReport(
                    effect.identity, effect.consumer_id, "prepare", attempts,
                    self._reason(error),
                ))
                continue
            raise self._failed(effect, error, cursors=self._initial_cursors) from error

    def reject(self) -> ConsumerTransactionReport:
        if self._state != "staged":
            raise RuntimeError("ConsumerTransaction is already resolved")
        rolled_back, diagnostics = self._discard_staged()
        self._state = "rejected" if not diagnostics else "failed"
        report = ConsumerTransactionReport(
            self._state,
            self._initial_cursors,
            tuple(value.identity.token for value in self._plan.effects),
            skipped=tuple(self._skipped),
            rolled_back_effects=rolled_back,
            diagnostics=diagnostics,
        )
        if diagnostics:
            raise ConsumerPublicationError("consumer rollback left unremoved temporaries", report=report)
        return report

    def accept(self) -> ConsumerTransactionReport:
        if self._state != "staged":
            raise RuntimeError("ConsumerTransaction is already resolved")
        cursors, published = self._initial_cursors, []
        pending = list(self._prepared)
        self._prepared.clear()
        while pending:
            effect, prepared, attempts = pending.pop(0)
            try:
                receipt = prepared.publish()
                if type(receipt) is not PublicationReceipt:
                    raise TypeError("PreparedPublication.publish must return PublicationReceipt")
                if receipt.effect_identity != effect.identity \
                        or receipt.payload_identity != effect.payload.identity:
                    raise ValueError("PublicationReceipt does not authenticate its exact effect payload")
                if receipt.parallel_mode is not effect.target.parallel_mode:
                    raise ValueError(
                        "PublicationReceipt parallel mode differs from its accepted target")
            except Exception as exc:
                error = exc
                cleanup = self._rollback(prepared)
                rolled_back = (effect.identity.token,) if cleanup is None else ()
                if cleanup is None and type(effect.failure_action) is Retry \
                        and attempts < self._attempt_limit(effect):
                    replacement, attempts, prepare_error = self._prepare_one(effect, attempts)
                    if replacement is not None:
                        pending.insert(0, (effect, replacement, attempts))
                        continue
                    error = prepare_error
                if type(effect.failure_action) is SkipSampleReported:
                    self._skipped.append(SkippedSampleReport(
                        effect.identity, effect.consumer_id, "publish", attempts,
                        self._reason(error),
                    ))
                    if cleanup is not None:
                        self._prepared.extend(pending)
                        accepted_rollback, accepted_cleanup = self._rollback_accepted()
                        raise self._failed(
                            effect, error, cursors=self._initial_cursors,
                            diagnostics=(cleanup,) + accepted_cleanup,
                            rolled_back=accepted_rollback,
                        ) from error
                    continue
                self._prepared.extend(pending)
                accepted_rollback, accepted_cleanup = self._rollback_accepted()
                diagnostics = ((cleanup,) if cleanup is not None else ()) + accepted_cleanup
                raise self._failed(
                    effect, error, cursors=self._initial_cursors,
                    diagnostics=diagnostics,
                    rolled_back=rolled_back + accepted_rollback,
                ) from error
            published.append(receipt)
            self._accepted.append((effect, prepared, receipt))
            cursors = cursors.replace(effect.cursor_after)
        self._state = "accepted"
        self._cursor_updates = tuple(effect.cursor_after for effect, _, _ in self._accepted)
        return ConsumerTransactionReport(
            "accepted",
            cursors,
            tuple(value.identity.token for value in self._plan.effects),
            tuple(published),
            tuple(self._skipped),
        )

    @property
    def cursor_updates(self) -> tuple[Any, ...]:
        if self._state not in ("accepted", "sealed"):
            raise RuntimeError("consumer cursor updates exist only after acceptance")
        return self._cursor_updates

    @property
    def finalize_pending(self) -> bool:
        """Whether this sealed transaction still owns release-only resources."""
        return bool(self._finalize_pending)

    @property
    def recoveries(self) -> tuple[Any, ...]:
        """Typed recovery authorities retained independently of report diagnostics."""
        return tuple(self._recoveries)

    def rollback_accepted(self) -> ConsumerTransactionReport:
        if self._state != "accepted":
            raise RuntimeError("ConsumerTransaction has no accepted publication to roll back")
        rolled_back, diagnostics = self._rollback_accepted()
        self._state = "rejected" if not diagnostics else "failed"
        report = ConsumerTransactionReport(
            self._state,
            self._initial_cursors,
            tuple(value.identity.token for value in self._plan.effects),
            skipped=tuple(self._skipped),
            rolled_back_effects=rolled_back,
            diagnostics=diagnostics,
        )
        if diagnostics:
            raise ConsumerPublicationError(
                "accepted consumer publication could not be compensated", report=report)
        return report

    def seal(self) -> tuple[str, ...]:
        """Drop rollback ownership post-commit; retain failed releases for an idempotent retry."""
        if self._state == "accepted":
            self._finalize_pending = list(self._accepted)
            self._accepted.clear()
            # This transition precedes every release attempt: a finalizer is never allowed to
            # reopen compensation after the enclosing native transaction has committed.
            self._state = "sealed"
        elif self._state != "sealed":
            raise RuntimeError("only an accepted ConsumerTransaction can be sealed")
        failures = []
        pending = []
        for effect, prepared, receipt in self._finalize_pending:
            try:
                if prepared.finalize() is not None:
                    raise TypeError("PreparedPublication.finalize() must return None")
            except BaseException as error:
                pending.append((effect, prepared, receipt))
                failures.append(
                    "%s: %s: %s" % (effect.consumer_id, type(error).__name__, error))
            recovery_failure = self._retain_recoveries(prepared)
            if recovery_failure is not None:
                if not pending or pending[-1][1] is not prepared:
                    pending.append((effect, prepared, receipt))
                failures.append("%s: %s" % (effect.consumer_id, recovery_failure))
        self._finalize_pending = pending
        return tuple(failures)

    def abort(self) -> ConsumerTransactionReport | None:
        """Reject staged work or compensate accepted work; resolved failures are already clean."""
        if self._state == "staged":
            return self.reject()
        if self._state == "accepted":
            return self.rollback_accepted()
        return None


__all__ = [
    "ConsumerPublicationError", "ConsumerPublisher", "ConsumerTransaction",
    "ConsumerTransactionReport", "PreparedPublication", "PublicationReceipt",
    "SkippedSampleReport",
]
