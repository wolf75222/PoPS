"""Bounded post-commit worker for irreversible live-observer delivery.

This module intentionally has no dependency on ``RuntimeInstance``.  Integration must create an
``ObserverFrame`` only after native step finalization and submit it here.  Keeping that splice
explicit prevents a live packet from masquerading as a compensatable ConsumerTransaction artifact.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any

from pops.identity import Identity, make_identity
from pops.output.observers import (
    ObserverFrame,
    ObserverReceipt,
    ObserverRun,
    authenticate_observer_session,
    detach_observer_frame,
)


def _reason(error: BaseException) -> str:
    return "%s: %s" % (type(error).__name__, error)


@dataclass(frozen=True, slots=True)
class ObserverDeliveryReport:
    """Terminal, non-compensating result of one submitted accepted frame."""

    consumer_id: str
    run_identity: Identity
    sequence: int
    frame_identity: Identity
    status: str
    attempts: int
    receipt: ObserverReceipt | None = None
    reason: str | None = None
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.consumer_id, str) or not self.consumer_id \
                or self.consumer_id.strip() != self.consumer_id:
            raise TypeError("observer report consumer_id must be non-empty canonical text")
        if type(self.run_identity) is not Identity or self.run_identity.domain != "run":
            raise TypeError("observer report run_identity must be an exact run Identity")
        if self.status not in {"delivered", "skipped"}:
            raise ValueError("observer delivery status must be delivered or skipped")
        if isinstance(self.sequence, bool) or type(self.sequence) is not int \
                or self.sequence < 0:
            raise TypeError("observer delivery sequence must be an integer >= 0")
        if isinstance(self.attempts, bool) or type(self.attempts) is not int \
                or self.attempts < 1:
            raise TypeError("observer delivery attempts must be a positive integer")
        if self.status == "delivered":
            if type(self.receipt) is not ObserverReceipt or self.reason is not None:
                raise ValueError("delivered observer report requires only its receipt")
            if self.receipt.frame_identity != self.frame_identity:
                raise ValueError("observer receipt authenticates a different frame")
        elif self.receipt is not None or not isinstance(self.reason, str) or not self.reason:
            raise ValueError("skipped observer report requires only a non-empty reason")
        object.__setattr__(self, "identity", make_identity(
            "observer-delivery-report", self._payload()))

    def _payload(self) -> dict[str, Any]:
        return {
            "consumer_id": self.consumer_id,
            "run_identity": self.run_identity.to_data(),
            "sequence": self.sequence,
            "frame_identity": self.frame_identity.to_data(),
            "status": self.status,
            "attempts": self.attempts,
            "receipt": None if self.receipt is None else self.receipt.to_data(),
            "reason": self.reason,
        }

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity.to_data()}

    def to_collective_data(self) -> dict[str, Any]:
        """Return the authenticated report in the byte-free MPI control language."""

        return {
            "consumer_id": self.consumer_id,
            "run_identity": self.run_identity.token,
            "sequence": self.sequence,
            "frame_identity": self.frame_identity.token,
            "status": self.status,
            "attempts": self.attempts,
            "receipt": None if self.receipt is None else self.receipt.to_collective_data(),
            "reason": self.reason,
            "identity": self.identity.token,
        }

    @classmethod
    def from_data(cls, data: Any) -> ObserverDeliveryReport:
        if not isinstance(data, dict) or set(data) != {
                "consumer_id", "run_identity", "sequence", "frame_identity", "status",
                "attempts", "receipt", "reason", "identity"}:
            raise TypeError("observer delivery report data has an unsupported schema")
        receipt = None if data["receipt"] is None else ObserverReceipt.from_data(data["receipt"])
        result = cls(
            data["consumer_id"],
            Identity.from_data(data["run_identity"]),
            data["sequence"],
            Identity.from_data(data["frame_identity"]),
            data["status"],
            data["attempts"],
            receipt=receipt,
            reason=data["reason"],
        )
        if result.identity != Identity.from_data(data["identity"]) \
                or result.to_data() != data:
            raise ValueError("observer delivery report data is not canonical")
        return result

    @classmethod
    def from_collective_data(cls, data: Any) -> ObserverDeliveryReport:
        if not isinstance(data, dict) or set(data) != {
                "consumer_id", "run_identity", "sequence", "frame_identity", "status",
                "attempts", "receipt", "reason", "identity"}:
            raise TypeError("observer delivery report collective data has an unsupported schema")
        receipt = None if data["receipt"] is None \
            else ObserverReceipt.from_collective_data(data["receipt"])
        result = cls(
            data["consumer_id"],
            Identity.from_token(data["run_identity"]),
            data["sequence"],
            Identity.from_token(data["frame_identity"]),
            data["status"],
            data["attempts"],
            receipt=receipt,
            reason=data["reason"],
        )
        if result.identity != Identity.from_token(data["identity"]) \
                or result.to_collective_data() != data:
            raise ValueError("observer delivery report collective data is not canonical")
        return result


class _SubmissionGate:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._resolved = False
        self._error: BaseException | None = None

    def arm(self) -> None:
        with self._lock:
            if self._resolved:
                raise RuntimeError("observer submission gate is already resolved")
            self._resolved = True
            self._event.set()

    def cancel(self, error: BaseException) -> None:
        if not isinstance(error, BaseException):
            raise TypeError("observer submission cancellation requires an exception")
        with self._lock:
            if self._resolved:
                raise RuntimeError("observer submission gate is already resolved")
            self._error = error
            self._resolved = True
            self._event.set()

    def wait(self) -> BaseException | None:
        self._event.wait()
        return self._error


@dataclass(frozen=True, slots=True)
class _PreparedObserverSubmission:
    sequence: int
    _gate: _SubmissionGate = field(repr=False, compare=False)

    def arm(self) -> None:
        self._gate.arm()

    def cancel(self, error: BaseException) -> None:
        self._gate.cancel(error)


@dataclass(frozen=True, slots=True)
class _Job:
    sequence: int
    frame: ObserverFrame
    journal: Any = None
    journal_record: Any = None
    gate: _SubmissionGate = field(default_factory=_SubmissionGate, repr=False, compare=False)


_DETACHED_FRAME_PROOF = object()


@dataclass(frozen=True, slots=True)
class _DetachedObserverFrame:
    """Runtime-private proof that one frame crossed the native ownership boundary.

    The wrapper is deliberately not part of the public observer API.  Public callers submit an
    :class:`ObserverFrame` and the queue detaches it itself.  The accepted-step publisher, which
    must detach before native finalization, carries this exact proof into the private submission
    seam so the queue cannot accidentally copy the same (potentially large) frame twice.
    """

    frame: ObserverFrame
    _proof: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.frame) is not ObserverFrame or self._proof is not _DETACHED_FRAME_PROOF:
            raise TypeError("detached observer-frame ownership proof is invalid")


def _detach_owned_observer_frame(frame: ObserverFrame) -> _DetachedObserverFrame:
    """Detach one frame exactly once and seal the internal ownership evidence."""

    return _DetachedObserverFrame(detach_observer_frame(frame), _DETACHED_FRAME_PROOF)


def _authenticated_detached_frame(value: Any) -> ObserverFrame:
    if type(value) is not _DetachedObserverFrame or value._proof is not _DETACHED_FRAME_PROOF:
        raise TypeError("post-commit submission requires authenticated detached-frame ownership")
    if type(value.frame) is not ObserverFrame:
        raise TypeError("detached observer-frame proof contains an invalid frame")
    return value.frame


_STOP = object()


@dataclass(frozen=True, slots=True)
class _SharedWorkerTask:
    operation: Any
    on_failure: Any


class PostCommitObserverWorker:
    """One process-local FIFO for every post-commit session in a runtime run.

    MPI-capable writers such as parallel HDF5 may carry process-global state even when each
    observer owns a different duplicated communicator; serial Catalyst also owns process-global
    lifecycle state.  Letting one thread per observer enter those libraries permits inconsistent
    local ordering.  This worker makes initialization, execution and finalization follow the exact
    main-thread submission order on each process while keeping them off the simulation thread.
    """

    def __init__(self, *, thread_name: str = "pops-post-commit-worker") -> None:
        if not isinstance(thread_name, str) or not thread_name:
            raise TypeError("post-commit worker thread_name must be non-empty text")
        self._jobs: queue.Queue[Any] = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name=thread_name, daemon=False)
        self._thread.start()

    def submit(self, operation: Any, on_failure: Any) -> None:
        if not callable(operation) or not callable(on_failure):
            raise TypeError("post-commit worker tasks require callable operation/failure routes")
        with self._lock:
            if self._closed:
                raise RuntimeError("post-commit worker is closed")
            self._jobs.put(_SharedWorkerTask(operation, on_failure))

    def call(self, operation: Any) -> Any:
        """Run one lifecycle operation in FIFO order and return or re-raise on the caller."""

        if not callable(operation):
            raise TypeError("post-commit worker call requires a callable operation")
        done = threading.Event()
        result: list[Any] = []
        failure: list[BaseException] = []

        def invoke() -> None:
            try:
                result.append(operation())
            except BaseException as error:
                failure.append(error)
            finally:
                done.set()

        def failed(error: BaseException) -> None:
            failure.append(error)
            done.set()

        self.submit(invoke, failed)
        done.wait()
        if failure:
            raise failure[0]
        if len(result) != 1:
            raise RuntimeError("post-commit worker call lost its result")
        return result[0]

    def close(self) -> None:
        with self._lock:
            was_closed = self._closed
            if not was_closed:
                self._closed = True
                self._jobs.put(_STOP)
        if not was_closed:
            self._thread.join()

    def _run(self) -> None:
        while True:
            item = self._jobs.get()
            try:
                if item is _STOP:
                    return
                if type(item) is not _SharedWorkerTask:
                    raise TypeError("post-commit worker received an invalid internal task")
                try:
                    item.operation()
                except BaseException as error:
                    item.on_failure(error)
            finally:
                self._jobs.task_done()


class PostCommitObserverQueue:
    """One bounded, dedicated-worker, report-only observer queue.

    Delivery failures are recorded as ``skipped``; they never roll back numerical state.  The
    queue applies blocking backpressure when full, so it cannot silently lose frames or grow
    without bound.  ``max_attempts`` provides bounded retry inside the worker.
    """

    def __init__(
        self,
        session: Any,
        run: ObserverRun,
        *,
        consumer_id: str,
        capacity: int = 1,
        max_attempts: int = 1,
        thread_name: str = "pops-post-commit-observer",
        worker_communicator: Any = None,
        shared_worker: PostCommitObserverWorker | None = None,
    ) -> None:
        if type(run) is not ObserverRun:
            raise TypeError("PostCommitObserverQueue requires an exact ObserverRun")
        if not isinstance(consumer_id, str) or not consumer_id \
                or consumer_id.strip() != consumer_id:
            raise TypeError("observer queue consumer_id must be non-empty canonical text")
        if isinstance(capacity, bool) or type(capacity) is not int or capacity < 1:
            raise ValueError("observer queue capacity must be an integer >= 1")
        if isinstance(max_attempts, bool) or type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("observer max_attempts must be an integer >= 1")
        if not isinstance(thread_name, str) or not thread_name:
            raise TypeError("observer thread_name must be non-empty text")
        authority = authenticate_observer_session(session)
        if worker_communicator is not None:
            from pops._native_collectives import require_communicator

            require_communicator(worker_communicator, allow_world=False)
            if max_attempts != 1:
                raise ValueError(
                    "MPI observer queues require max_attempts=1 after a collective call")
        elif authority["worker_mpi"]:
            raise ValueError(
                "an MPI observer session requires an explicit duplicated worker lane")
        if shared_worker is not None and type(shared_worker) is not PostCommitObserverWorker:
            raise TypeError(
                "observer queue shared_worker must be an exact PostCommitObserverWorker")
        if worker_communicator is not None and shared_worker is None:
            raise ValueError(
                "an MPI observer queue requires the runtime's shared post-commit worker")
        self._session = session
        self._worker_communicator = worker_communicator
        self._shared_worker = shared_worker
        self._provider_id = authority["provider_id"]
        self._run = run
        self._consumer_id = consumer_id
        self._max_attempts = max_attempts
        self._capacity = capacity
        self._jobs: queue.Queue[Any] | None = (
            None if shared_worker is not None else queue.Queue(maxsize=capacity))
        self._condition = threading.Condition()
        self._reports: list[ObserverDeliveryReport] = []
        self._next_sequence = 0
        self._pending = 0
        self._closed = False
        self._lifecycle_error: BaseException | None = None
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        if shared_worker is None:
            self._thread = threading.Thread(
                target=self._worker, name=thread_name, daemon=False)
            self._thread.start()
            self._ready.wait()
        else:
            try:
                shared_worker.call(self._initialize_session)
            except BaseException as error:
                self._set_lifecycle_error(error)
        if self._lifecycle_error is not None:
            if self._thread is not None:
                self._thread.join()
            raise RuntimeError(
                "observer session initialization failed: "
                + _reason(self._lifecycle_error)) from self._lifecycle_error

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def pending(self) -> int:
        with self._condition:
            return self._pending

    @property
    def reports(self) -> tuple[ObserverDeliveryReport, ...]:
        with self._condition:
            return tuple(self._reports)

    def submit(
        self,
        frame: ObserverFrame,
        *,
        journal: Any = None,
        journal_record: Any = None,
    ) -> int:
        """Submit and detach one already committed frame with bounded backpressure."""

        if type(frame) is not ObserverFrame:
            raise TypeError("observer queue accepts only exact ObserverFrame values")
        return self._submit_detached(
            _detach_owned_observer_frame(frame),
            journal=journal,
            journal_record=journal_record,
        )

    def _submit_detached(
        self,
        owned: _DetachedObserverFrame,
        *,
        journal: Any = None,
        journal_record: Any = None,
    ) -> int:
        """Submit runtime-authenticated owned storage without a second deep copy."""

        submission = self._enqueue_detached(
            owned, journal=journal, journal_record=journal_record, deferred=False)
        return submission.sequence

    def _prepare_detached(
        self,
        owned: _DetachedObserverFrame,
        *,
        journal: Any = None,
        journal_record: Any = None,
    ) -> _PreparedObserverSubmission:
        """Enqueue one frame behind a gate resolved by main-thread MPI consensus."""

        if self._shared_worker is None:
            raise RuntimeError("deferred observer submission requires the shared worker")
        return self._enqueue_detached(
            owned, journal=journal, journal_record=journal_record, deferred=True)

    def _enqueue_detached(
        self,
        owned: _DetachedObserverFrame,
        *,
        journal: Any,
        journal_record: Any,
        deferred: bool,
    ) -> _PreparedObserverSubmission:
        frame = _authenticated_detached_frame(owned)
        if (journal is None) != (journal_record is None):
            raise TypeError("durable observer submission requires both journal and record")
        if journal is not None:
            from pops.output._durable_journal import DurableJournal

            if type(journal) is not DurableJournal:
                raise TypeError("durable observer submission requires an exact DurableJournal")
            record_frame = getattr(journal_record, "frame", None)
            record_state = getattr(journal_record, "state", None)
            if type(record_frame) is not ObserverFrame or record_frame.identity != frame.identity \
                    or record_state not in {"pending", "delivered"}:
                raise ValueError(
                    "durable observer record does not authenticate the submitted frame")
        with self._condition:
            while self._shared_worker is not None and self._pending >= self._capacity \
                    and not self._closed and self._lifecycle_error is None:
                self._condition.wait()
            self._require_available_locked()
            sequence = self._next_sequence
            self._next_sequence += 1
            self._pending += 1
        gate = _SubmissionGate()
        if not deferred:
            gate.arm()
        job = _Job(sequence, frame, journal, journal_record, gate)
        try:
            if self._shared_worker is None:
                if self._jobs is None:  # pragma: no cover - constructor establishes this invariant
                    raise RuntimeError("observer queue lost its private worker queue")
                self._jobs.put(job, block=True)
            else:
                self._shared_worker.submit(
                    lambda: self._process_job(job),
                    lambda error: self._fail_job(job, error),
                )
        except BaseException:
            with self._condition:
                self._pending -= 1
                self._condition.notify_all()
            raise
        return _PreparedObserverSubmission(sequence, gate)

    def flush(self) -> tuple[ObserverDeliveryReport, ...]:
        """Wait until every accepted frame submitted so far has a terminal report."""

        with self._condition:
            while self._pending:
                self._condition.wait()
            if self._lifecycle_error is not None:
                raise RuntimeError(
                    "observer worker is unavailable: "
                    + _reason(self._lifecycle_error)) from self._lifecycle_error
            return tuple(self._reports)

    def close(self) -> tuple[ObserverDeliveryReport, ...]:
        """Drain frames, finalize the optional backend, and join its non-daemon worker."""

        with self._condition:
            was_closed = self._closed
            if not was_closed:
                self._closed = True
        if was_closed:
            return self.reports
        self.flush()
        if self._shared_worker is None:
            if self._jobs is None or self._thread is None:  # pragma: no cover - invariant
                raise RuntimeError("observer queue lost its private worker")
            self._jobs.put(_STOP)
            self._thread.join()
        else:
            try:
                self._shared_worker.call(self._finalize_session)
            except BaseException as error:
                self._set_lifecycle_error(error)
        if self._lifecycle_error is not None:
            raise RuntimeError(
                "observer session finalization failed: "
                + _reason(self._lifecycle_error)) from self._lifecycle_error
        return self.reports

    def _record(self, report: ObserverDeliveryReport) -> None:
        with self._condition:
            if self._pending < 1:
                raise RuntimeError("observer queue completed a job it did not own")
            self._reports.append(report)
            self._pending -= 1
            self._condition.notify_all()

    def _require_available_locked(self) -> None:
        if self._closed:
            raise RuntimeError("observer queue is closed")
        if self._lifecycle_error is not None:
            raise RuntimeError(
                "observer worker is unavailable: " + _reason(self._lifecycle_error))

    def _set_lifecycle_error(self, error: BaseException) -> None:
        with self._condition:
            if self._lifecycle_error is None:
                self._lifecycle_error = error
            self._condition.notify_all()

    def _skipped_job(self, job: _Job, error: BaseException) -> ObserverDeliveryReport:
        return ObserverDeliveryReport(
            self._consumer_id,
            job.frame.snapshot.provenance.run_identity,
            job.sequence,
            job.frame.identity,
            "skipped",
            self._max_attempts,
            reason=_reason(error),
        )

    def _fail_job(self, job: _Job, error: BaseException) -> None:
        self._set_lifecycle_error(error)
        self._record(self._skipped_job(job, error))

    def _process_job(self, job: _Job) -> None:
        gate_error = job.gate.wait()
        if gate_error is not None:
            self._record(self._skipped_job(job, gate_error))
            return
        with self._condition:
            unavailable = self._lifecycle_error
        if unavailable is not None:
            self._record(self._skipped_job(job, unavailable))
            return
        try:
            report = self._deliver(job)
        except BaseException as error:
            self._set_lifecycle_error(error)
            try:
                self._session.abort()
            except BaseException as abort_error:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    add_note("observer abort also failed: %s" % _reason(abort_error))
            report = self._skipped_job(job, error)
        self._record(report)

    def _deliver(self, job: _Job) -> ObserverDeliveryReport:
        gate_error: BaseException | None = None
        if self._worker_communicator is not None:
            try:
                from pops._native_collectives import allgather_value, rank, size

                request = job.frame.request.to_data()
                request.pop("rank")
                gate = {
                    "rank": rank(self._worker_communicator),
                    "consumer_id": self._consumer_id,
                    "run_identity": job.frame.snapshot.provenance.run_identity.token,
                    "sequence": job.sequence,
                    "clock": job.frame.snapshot.clock.to_data(),
                    "request": request,
                }
                rows = allgather_value(self._worker_communicator, gate)
                if len(rows) != size(self._worker_communicator) or any(
                        not isinstance(row, dict) or row.get("rank") != owner
                        for owner, row in enumerate(rows)):
                    raise RuntimeError(
                        "MPI observer frame gate returned malformed rank evidence")
                canonical = dict(rows[0])
                canonical.pop("rank")
                if any(
                        {key: value for key, value in row.items() if key != "rank"} != canonical
                        for row in rows[1:]):
                    raise RuntimeError(
                        "MPI observer ranks submitted different accepted frame authorities")
            except BaseException as caught:
                gate_error = caught
        for attempt in range(1, self._max_attempts + 1):
            error = gate_error
            receipt = None
            if error is None:
                try:
                    receipt = self._session.execute(job.frame)
                    if type(receipt) is not ObserverReceipt:
                        raise TypeError("observer execute() must return an exact ObserverReceipt")
                    if receipt.frame_identity != job.frame.identity:
                        raise ValueError("observer receipt authenticates a different frame")
                    if receipt.provider_id != self._provider_id:
                        raise ValueError(
                            "observer receipt provider_id differs from authenticated session "
                            "authority")
                except BaseException as caught:
                    error = caught
            if self._worker_communicator is not None:
                from pops._native_collectives import allgather_value, rank

                rows = allgather_value(self._worker_communicator, {
                    "rank": rank(self._worker_communicator),
                    "error": None if error is None else _reason(error),
                })
                failures = []
                for owner, row in enumerate(rows):
                    if not isinstance(row, dict):
                        failures.append("rank %d: malformed execution evidence" % owner)
                    elif row.get("rank") != owner:
                        failures.append("rank %d: mismatched rank evidence" % owner)
                    elif row.get("error") is not None:
                        failures.append("rank %d: %s" % (owner, row["error"]))
                if failures:
                    error = RuntimeError(
                        "MPI observer execution failed collectively: " + "; ".join(failures))
                else:
                    error = None
            if error is None and job.journal is not None:
                try:
                    job.journal.delivered(job.journal_record)
                except BaseException as caught:
                    error = caught
                if self._worker_communicator is not None:
                    from pops._native_collectives import allgather_value, rank

                    rows = allgather_value(self._worker_communicator, {
                        "rank": rank(self._worker_communicator),
                        "error": None if error is None else _reason(error),
                    })
                    failures = []
                    for owner, row in enumerate(rows):
                        if not isinstance(row, dict):
                            failures.append(
                                "rank %d: malformed journal evidence" % owner)
                        elif row.get("rank") != owner:
                            failures.append("rank %d: mismatched rank evidence" % owner)
                        elif row.get("error") is not None:
                            failures.append("rank %d: %s" % (owner, row["error"]))
                    if failures:
                        error = RuntimeError(
                            "MPI observer journal acknowledgement failed collectively: "
                            + "; ".join(failures))
                    else:
                        error = None
            if error is None:
                return ObserverDeliveryReport(
                    self._consumer_id,
                    job.frame.snapshot.provenance.run_identity,
                    job.sequence,
                    job.frame.identity,
                    "delivered",
                    attempt,
                    receipt=receipt,
                )
        if error is None:  # max_attempts validation makes this unreachable
            error = RuntimeError("observer delivery failed without diagnostic")
        return ObserverDeliveryReport(
            self._consumer_id,
            job.frame.snapshot.provenance.run_identity,
            job.sequence,
            job.frame.identity,
            "skipped",
            self._max_attempts,
            reason=_reason(error),
        )

    def _worker_agreement(
        self, phase: str, error: BaseException | None,
    ) -> BaseException | None:
        """Make one worker lifecycle result uniform before any rank leaves the lane."""

        if self._worker_communicator is None:
            return error
        try:
            from pops._native_collectives import allgather_value, rank, size

            rows = allgather_value(self._worker_communicator, {
                "rank": rank(self._worker_communicator),
                "error": None if error is None else _reason(error),
            })
            if len(rows) != size(self._worker_communicator) or any(
                    not isinstance(row, dict)
                    or set(row) != {"rank", "error"}
                    or row["rank"] != owner
                    or (row["error"] is not None and not isinstance(row["error"], str))
                    for owner, row in enumerate(rows)):
                return RuntimeError(
                    "MPI observer %s returned malformed lifecycle evidence" % phase)
            failures = [
                "rank %d: %s" % (owner, row["error"])
                for owner, row in enumerate(rows) if row["error"] is not None
            ]
            if failures:
                return RuntimeError(
                    "MPI observer %s failed collectively: %s"
                    % (phase, "; ".join(failures)))
            return None
        except BaseException as agreement_error:
            return agreement_error

    def _initialize_session(self) -> None:
        initialized = False
        initialization_error: BaseException | None = None
        try:
            if self._worker_communicator is not None:
                from pops._native_collectives import barrier

                barrier(self._worker_communicator)
            result = self._session.initialize(self._run)
            if result is not None:
                raise TypeError("observer initialize() must return None")
            initialized = True
        except BaseException as error:
            initialization_error = error
        initialization_error = self._worker_agreement(
            "initialization", initialization_error)
        if initialization_error is not None:
            try:
                self._session.abort()
            except BaseException as abort_error:
                add_note = getattr(initialization_error, "add_note", None)
                if callable(add_note):
                    add_note(
                        "observer abort after initialization failure also failed: %s"
                        % _reason(abort_error))
            raise initialization_error
        if not initialized:  # defensive: collective agreement cannot clear a local failure
            raise RuntimeError("observer initialization lost its local failure evidence")

    def _finalize_session(self) -> None:
        finalization_error: BaseException | None = None
        try:
            result = self._session.finalize()
            if result is not None:
                raise TypeError("observer finalize() must return None")
        except BaseException as error:
            finalization_error = error
        finalization_error = self._worker_agreement(
            "finalization", finalization_error)
        if finalization_error is not None:
            try:
                self._session.abort()
            except BaseException as abort_error:
                add_note = getattr(finalization_error, "add_note", None)
                if callable(add_note):
                    add_note(
                        "observer abort after finalization failure also failed: %s"
                        % _reason(abort_error))
            raise finalization_error

    def _worker(self) -> None:
        if self._jobs is None:  # pragma: no cover - constructor establishes this invariant
            self._set_lifecycle_error(RuntimeError("observer queue lost its private jobs"))
            self._ready.set()
            return
        try:
            self._initialize_session()
        except BaseException as error:
            self._set_lifecycle_error(error)
            self._ready.set()
            return
        self._ready.set()
        try:
            while True:
                item = self._jobs.get()
                try:
                    if item is _STOP:
                        break
                    if type(item) is not _Job:
                        raise TypeError("observer queue received an invalid internal job")
                    self._process_job(item)
                finally:
                    self._jobs.task_done()
            self._finalize_session()
        except BaseException as error:
            self._set_lifecycle_error(error)

    def __enter__(self) -> PostCommitObserverQueue:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()


__all__ = [
    "ObserverDeliveryReport", "PostCommitObserverQueue", "PostCommitObserverWorker",
]
