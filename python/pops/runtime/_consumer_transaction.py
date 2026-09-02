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
            if not ranks or ranks != tuple(range(len(ranks))):
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


class ConsumerTransactionWorkspace:
    """Bounded bind-time reservation for compensable consumer transactions.

    The runtime owns one of these for the lifetime of its frozen ConsumerGraph.  It is a
    deliberately small ownership registry, not a planner: a transaction must fit the exact
    graph shape and retry budget that was authenticated at bind.  A retained finalizer keeps its
    reservation, so repeated release failures cannot silently grow a per-step transaction
    registry.
    """

    __slots__ = ("max_effects", "max_prepared", "_owners")

    def __init__(self, *, max_effects: int, max_prepared: int, max_transactions: int) -> None:
        for value, name in (
            (max_effects, "max_effects"),
            (max_prepared, "max_prepared"),
            (max_transactions, "max_transactions"),
        ):
            if isinstance(value, bool) or type(value) is not int or value < 0:
                raise TypeError("ConsumerTransactionWorkspace.%s must be an integer >= 0" % name)
        if max_prepared < max_effects:
            raise ValueError("ConsumerTransactionWorkspace.max_prepared is below max_effects")
        self.max_effects = max_effects
        self.max_prepared = max_prepared
        # Slots are allocated once at bind and reused only after their transaction releases them.
        self._owners: list[ConsumerTransaction | None] = [None] * max_transactions

    def reserve(self, transaction: ConsumerTransaction, *, effects: int, prepared: int) -> None:
        if effects > self.max_effects or prepared > self.max_prepared:
            raise ValueError(
                "consumer transaction exceeds its frozen bind-time effect/retry budget"
            )
        for index, owner in enumerate(self._owners):
            if owner is None:
                self._owners[index] = transaction
                return
        raise RuntimeError("consumer transaction workspace is retained by an unfinished finalizer")

    def release(self, transaction: ConsumerTransaction) -> None:
        for index, owner in enumerate(self._owners):
            if owner is transaction:
                self._owners[index] = None
                return


class ConsumerTransaction:
    """Own temporaries until a step controller explicitly accepts or rejects the attempt."""

    __slots__ = (
        "_plan", "_publisher", "_initial_cursors", "_prepared", "_accepted",
        "_cursor_updates", "_skipped", "_state", "_finalize_pending", "_recoveries",
        "_workspace",
    )

    def __init__(
        self,
        plan: EffectPlan,
        cursors: ConsumerCursorSet,
        publisher: ConsumerPublisher,
        *,
        workspace: ConsumerTransactionWorkspace | None = None,
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
        if workspace is not None and type(workspace) is not ConsumerTransactionWorkspace:
            raise TypeError("ConsumerTransaction workspace must be an exact ConsumerTransactionWorkspace")
        self._workspace = workspace
        if workspace is not None:
            workspace.reserve(
                self,
                effects=len(plan.effects),
                prepared=sum(self._attempt_limit(effect) for effect in plan.effects),
            )
        self._state = "preparing"
        try:
            self._prepare_all()
        except BaseException:
            self._release_workspace()
            raise
        self._state = "staged"

    def _release_workspace(self) -> None:
        if self._workspace is not None:
            self._workspace.release(self)

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

    def _prepare_alternatives(
        self, effect: AcceptedSideEffect,
    ) -> tuple[tuple[tuple[AcceptedSideEffect, PreparedPublication, int], ...], Exception | None]:
        """Prepare every bounded publication alternative before hidden publication.

        A writer preparation may allocate a temporary artifact and therefore belongs to the
        candidate phase.  In particular, publication retries must consume this frozen sequence;
        calling ``prepare`` from ``accept`` would run scientific capture after native hidden
        publication, when the native writer lock intentionally prevents ordinary reads.
        """
        alternatives = []
        last_error = None
        for attempt in range(1, self._attempt_limit(effect) + 1):
            try:
                prepared = self._validate_prepared(effect, self._publisher.prepare(effect))
            except Exception as exc:  # writer failures are classified by the typed action
                last_error = exc
                continue
            alternatives.append((effect, prepared, attempt))
            # A successful final preparation means there is no preparation failure to report if
            # publication later exhausts the frozen alternatives.
            last_error = None
        return tuple(alternatives), last_error

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

    def _discard_rows(
        self,
        rows: tuple[tuple[AcceptedSideEffect, PreparedPublication, int], ...],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        rolled_back, diagnostics = [], []
        for effect, prepared, _ in reversed(rows):
            failure = self._discard(prepared)
            if failure is None:
                if effect.identity.token not in rolled_back:
                    rolled_back.append(effect.identity.token)
            else:
                diagnostics.append("%s: %s" % (effect.consumer_id, failure))
        return tuple(rolled_back), tuple(diagnostics)

    def _discard_staged(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        rows = tuple(self._prepared)
        self._prepared.clear()
        return self._discard_rows(rows)

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
        # Dispose every still-staged alternative before compensating accepted publications.  The
        # preparation order is the ownership order, so this keeps the complete cleanup walk in
        # exact reverse order even when a later effect fails after an earlier one was published.
        staged_rollback, cleanup = self._discard_staged()
        accepted_rollback, accepted_cleanup = self._rollback_accepted()
        rolled_back_effects = []
        for token in rolled_back + staged_rollback + accepted_rollback:
            if token not in rolled_back_effects:
                rolled_back_effects.append(token)
        report = ConsumerTransactionReport(
            "failed",
            cursors,
            tuple(value.identity.token for value in self._plan.effects),
            (),
            tuple(self._skipped),
            tuple(rolled_back_effects),
            diagnostics + cleanup + accepted_cleanup + (self._reason(error),),
        )
        self._state = "failed"
        self._release_workspace()
        reason = self._reason(error)
        return ConsumerPublicationError(
            "consumer %s failed under %s: %s" % (
                effect.consumer_id, type(effect.failure_action).__name__, reason),
            report=report,
        )

    def _prepare_all(self) -> None:
        for effect in self._plan.effects:
            alternatives, error = self._prepare_alternatives(effect)
            if alternatives:
                self._prepared.extend(alternatives)
                continue
            if type(effect.failure_action) is SkipSampleReported:
                self._skipped.append(SkippedSampleReport(
                    effect.identity, effect.consumer_id, "prepare", self._attempt_limit(effect),
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
            self._release_workspace()
            raise ConsumerPublicationError("consumer rollback left unremoved temporaries", report=report)
        self._release_workspace()
        return report

    def accept(self) -> ConsumerTransactionReport:
        if self._state != "staged":
            raise RuntimeError("ConsumerTransaction is already resolved")
        cursors, published = self._initial_cursors, []
        pending = tuple(self._prepared)
        self._prepared.clear()
        # ``_prepare_all`` appends alternatives contiguously for each effect.  Grouping here
        # lets a publication retry consume the next frozen preparation for the same effect before
        # moving to the next effect, while retaining one global creation order for cleanup.
        row_index = 0
        while row_index < len(pending):
            effect = pending[row_index][0]
            group_end = row_index + 1
            while group_end < len(pending) and pending[group_end][0].identity == effect.identity:
                group_end += 1
            alternative_index = row_index
            while alternative_index < group_end:
                _, prepared, attempts = pending[alternative_index]
                try:
                    receipt = prepared.publish()
                    if type(receipt) is not PublicationReceipt:
                        raise TypeError("PreparedPublication.publish must return PublicationReceipt")
                    if receipt.effect_identity != effect.identity \
                            or receipt.payload_identity != effect.payload.identity:
                        raise ValueError(
                            "PublicationReceipt does not authenticate its exact effect payload")
                    if receipt.parallel_mode is not effect.target.parallel_mode:
                        raise ValueError(
                            "PublicationReceipt parallel mode differs from its accepted target")
                except Exception as exc:
                    error = exc
                    cleanup = self._rollback(prepared)
                    rolled_back = (effect.identity.token,) if cleanup is None else ()
                    # Every retry alternative was prepared before this method entered.  A failed
                    # rollback is unsafe to retry, so leave the remaining frozen alternatives for
                    # the exact reverse-order discard path below.
                    if cleanup is None and alternative_index + 1 < group_end:
                        alternative_index += 1
                        continue
                    if type(effect.failure_action) is SkipSampleReported:
                        self._skipped.append(SkippedSampleReport(
                            effect.identity, effect.consumer_id, "publish", attempts,
                            self._reason(error),
                        ))
                        if cleanup is not None:
                            self._prepared.extend(pending[alternative_index + 1:])
                            raise self._failed(
                                effect, error, cursors=self._initial_cursors,
                                diagnostics=(cleanup,),
                            ) from error
                        break
                    self._prepared.extend(pending[alternative_index + 1:])
                    diagnostics = ((cleanup,) if cleanup is not None else ())
                    raise self._failed(
                        effect, error, cursors=self._initial_cursors,
                        diagnostics=diagnostics,
                        rolled_back=rolled_back,
                    ) from error

                published.append(receipt)
                self._accepted.append((effect, prepared, receipt))
                cursors = cursors.replace(effect.cursor_after)
                # The successful alternative wins.  Discard all later alternatives for this
                # effect immediately; they can never be published after its cursor advances.
                _, unused_diagnostics = self._discard_rows(
                    pending[alternative_index + 1:group_end]
                )
                if unused_diagnostics:
                    self._prepared.extend(pending[group_end:])
                    cleanup_error = RuntimeError(
                        "unused consumer publication alternative cleanup failed"
                    )
                    raise self._failed(
                        effect, cleanup_error, cursors=self._initial_cursors,
                        diagnostics=unused_diagnostics,
                    ) from cleanup_error
                break
            row_index = group_end
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
            self._release_workspace()
            raise ConsumerPublicationError(
                "accepted consumer publication could not be compensated", report=report)
        self._release_workspace()
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
        if not self._finalize_pending:
            self._release_workspace()
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
    "ConsumerTransactionWorkspace",
    "ConsumerTransactionReport", "PreparedPublication", "PublicationReceipt",
    "SkippedSampleReport",
]
