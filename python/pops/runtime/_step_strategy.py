"""Execution protocol for the four typed :mod:`pops.time` step strategies."""
from __future__ import annotations

import bisect
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeVar

from pops._bootstrap import StepAttemptRejected
from pops.time._step.strategy import (
    AdaptiveCFL,
    ErrorControlledDt,
    ExternalTimeGrid,
    FixedDt,
    StepStrategy,
)
from pops.time._step.transaction import StepTransactionReport


_StepStrategyT = TypeVar("_StepStrategyT", bound=StepStrategy)
ControllerFactory = Callable[
    [StepStrategy, Mapping[str, Any] | None], "StepController[Any]"
]
_CONTROLLER_FACTORIES: dict[type[StepStrategy], ControllerFactory] = {}
_MISSING = object()


def register_step_controller_factory(
    strategy_type: type[StepStrategy],
) -> Callable[[ControllerFactory], ControllerFactory]:
    """Register the sole runtime materializer for one exact authoring descriptor type."""
    if not isinstance(strategy_type, type) or not issubclass(strategy_type, StepStrategy) \
            or strategy_type is StepStrategy:
        raise TypeError("step controller adapters require a concrete StepStrategy type")

    def register(factory: ControllerFactory) -> ControllerFactory:
        if not callable(factory):
            raise TypeError("step controller factory must be callable")
        existing = _CONTROLLER_FACTORIES.get(strategy_type)
        if existing is not None and existing is not factory:
            raise ValueError(
                "runtime controller factory already registered for %s" % strategy_type.__name__)
        _CONTROLLER_FACTORIES[strategy_type] = factory
        return factory

    return register


def _stores(engine: Any) -> tuple[str, ...]:
    plan = getattr(engine, "_step_transaction_plan", None)
    return tuple(store.value for store in plan.stores) if plan is not None else ()


def _phase(error: BaseException) -> str:
    value = getattr(error, "phase", _MISSING)
    if value is not _MISSING:
        if callable(value):
            value = value()
        if isinstance(value, str) and value in {
            "prepare", "stage", "solve", "synchronize", "guard", "effect", "commit",
        }:
            return value
        # Native providers may expose a phase more precise than the report vocabulary.  Keep that
        # exact value on the exception, but never infer a different phase from its display string.
        return "solve"
    message = str(error)
    for phase in ("prepare", "stage", "solve", "synchronize", "guard", "effect", "commit"):
        if " during %s:" % phase in message:
            return phase
    return "solve"


def _control_identity(controls: Mapping[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    values = {} if controls is None else dict(controls)
    return tuple(sorted(
        (name, tuple(value) if isinstance(value, (tuple, list)) else value)
        for name, value in values.items()
    ))


def _attempt_world(engine: Any) -> Any:
    """Return the authenticated MPI world for one installed engine, if distributed."""
    if getattr(engine, "_collective_step_envelope_active", False) is not True:
        return None
    context = getattr(engine, "_execution_context", None)
    resource = getattr(context, "communicator", None)
    identity = getattr(resource, "identity", None)
    handle = getattr(resource, "handle", None)
    if identity is None:
        # Small direct-engine and unit-test seams have no installed execution authority.
        return None
    if identity == "serial":
        if handle is not None:
            raise ValueError("serial step execution hides a communicator handle")
        return None
    if identity != "MPI_COMM_WORLD":
        raise ValueError(
            "distributed step execution requires the authenticated MPI_COMM_WORLD resource")
    from pops._native_collectives import require_world

    return require_world(handle)


def _attempt_error_record(error: BaseException) -> dict[str, Any]:
    rejected = isinstance(error, StepAttemptRejected)

    def attribute(name: str, default: Any) -> Any:
        try:
            value = getattr(error, name, default)
            return value() if callable(value) else value
        except BaseException:
            return default

    def text(value: Any, default: str) -> str:
        try:
            return str(value)
        except BaseException:
            return default

    message = text(error, "<unprintable rank-local exception>")
    try:
        phase = _phase(error)
    except BaseException:
        phase = "solve"
    try:
        reason_code = int(attribute("reason_code", 0)) if rejected else None
    except (TypeError, ValueError, OverflowError):
        reason_code = 0

    return {
        "kind": "reject" if rejected else "fail",
        "type": type(error).__name__,
        "message": message,
        "status": text(
            attribute("status", "invalid_evaluation"), "invalid_evaluation",
        ) if rejected else None,
        "phase": phase,
        "detail": text(attribute("detail", message), message) if rejected else None,
        "disposition": text(attribute("disposition", "reject"), "reject")
        if rejected else None,
        "reason_code": reason_code,
    }


def _collective_attempt_error(
    engine: Any, local_error: BaseException | None,
) -> BaseException | None:
    """Turn rank-local attempt results into one deterministic pre-publication decision."""
    world = _attempt_world(engine)
    if world is None or int(world.size) == 1:
        return local_error

    from pops._native_collectives import allgather_value

    rows = allgather_value(world, {
        "rank": int(world.rank),
        "error": None if local_error is None else _attempt_error_record(local_error),
    })
    failures: list[tuple[int, Mapping[str, Any]]] = []
    expected_error_keys = {
        "kind", "type", "message", "status", "phase", "detail", "disposition", "reason_code",
    }
    for rank, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"rank", "error"} \
                or row["rank"] != rank:
            return RuntimeError("collective step attempt returned an invalid rank envelope")
        error = row["error"]
        if error is None:
            continue
        if not isinstance(error, Mapping) or set(error) != expected_error_keys:
            return RuntimeError(
                "collective step attempt returned an invalid error envelope on rank %d" % rank)
        failures.append((rank, error))
    if not failures:
        return None

    fatal = tuple(row for row in failures if row[1]["kind"] == "fail")
    rank_divergent = len(failures) != len(rows)
    rejection_controls = {
        (
            error["status"],
            error["phase"],
            error["disposition"],
            error["reason_code"],
        )
        for _, error in failures
        if error["kind"] == "reject"
    }
    rejection_divergent = len(rejection_controls) > 1
    selected_rank, selected = (fatal or tuple(failures))[0]
    diagnostics = "; ".join(
        "rank %d %s: %s" % (rank, error["type"], error["message"])
        for rank, error in failures
    )
    phase = str(selected["phase"])
    # A controller may retry only when every rank rejected and therefore every native backend
    # already restored its attempt-local state.  A mixed success/rejection is escalated to FailRun;
    # the enclosing RuntimeInstance transaction then rolls every rank back before publication.
    if fatal or rank_divergent or rejection_divergent:
        return RuntimeError(
            "collective step attempt failed during %s: %s" % (phase, diagnostics))

    rejection = StepAttemptRejected(
        "collective step attempt rejected during %s: %s" % (phase, diagnostics))
    rejection.status = str(selected["status"])
    rejection.phase = phase
    rejection.detail = diagnostics
    rejection.disposition = str(selected["disposition"])
    rejection.reason_code = int(selected["reason_code"])
    rejection.failed_rank = selected_rank
    return rejection


def _record_failure(engine: Any, error: BaseException, attempts: int) -> None:
    rejected = isinstance(error, StepAttemptRejected)
    stores = _stores(engine)
    engine._last_step_transaction_report = StepTransactionReport(
        status="rejected" if rejected else "failed",
        phase=_phase(error),
        action="reject_attempt" if rejected else "fail_run",
        attempts=attempts,
        staged_effects=stores,
        rolled_back_effects=stores,
        diagnostics=(str(error),),
    )


def _native_attempt(engine: Any, native: Any, advance: Any) -> Any:
    temporal = getattr(engine, "_temporal_restart_state", None)
    before_time, before_step = native.time(), native.macro_step()
    if temporal is not None:
        temporal.before_attempt(time=before_time, macro_step=before_step)
    result = None
    local_error = None
    try:
        result = advance()
    except BaseException as error:
        local_error = error
    error = _collective_attempt_error(engine, local_error)
    if error is not None:
        if temporal is not None:
            from pops.runtime._temporal_restart import is_rejected_attempt
            recorder = temporal.reject if is_rejected_attempt(error) else temporal.fail
            # Record the unsuccessful attempt at the last accepted boundary. A composite target may
            # have advanced one child before another failed; querying its live clock here can then
            # raise a divergence error and mask the initiating numerical exception. The enclosing
            # transaction owns native rollback, while the temporal envelope remains at this captured
            # pre-attempt clock by definition.
            recorder(time=before_time, macro_step=before_step)
        raise error
    if temporal is not None:
        temporal.accept(
            before_time=before_time, before_step=before_step,
            time=native.time(), macro_step=native.macro_step())
    return result


class StepController(ABC, Generic[_StepStrategyT]):
    """Small runtime protocol; implementations choose dt, native executors advance fields."""

    def __init__(
        self, strategy: _StepStrategyT, controls: Mapping[str, Any] | None = None,
    ) -> None:
        self.strategy = strategy
        self.controls = _control_identity(controls)
        self.attempts = 0

    def matches(self, strategy: StepStrategy, controls: Mapping[str, Any] | None) -> bool:
        return self.strategy == strategy and self.controls == _control_identity(controls)

    def restore_temporal_state(self, temporal: Any) -> None:
        """Restore provider-owned proposal state; stateless controllers need no action."""
        return None

    def prepare_attempts(
        self, engine: Any, native: Any, *, t_end: float,
    ) -> _PreparedStepAttempts:
        """Prepare one opaque extension attempt.

        Registered third-party controllers that do not retry remain valid through this default
        adapter.  A controller that retries must override this method so the RuntimeInstance can
        put every retry behind its own native transaction boundary.
        """
        return _PreparedStepAttempts(
            engine=engine,
            controller=self,
            attempt=lambda: self.execute(engine, native, t_end=t_end),
        )

    @abstractmethod
    def execute(self, engine: Any, native: Any, *, t_end: float) -> int:
        """Execute until one macro-step is accepted and return the number of native attempts."""


class _PreparedStepAttempts:
    """Detached retry cursor whose individual executions are transaction-sized.

    The cursor is deliberately not stored on the runtime engine: native rollback restores the
    accepted controller snapshot after every rejected attempt, while this local cursor retains only
    the next provisional proposal and the attempt count.  ``accept`` is the sole operation allowed
    to publish controller state.
    """

    def __init__(
        self,
        *,
        engine: Any,
        controller: StepController[Any],
        attempt: Callable[[], int | None],
        retry: Callable[[BaseException, int], bool] | None = None,
        accept: Callable[[int], None] | None = None,
    ) -> None:
        self.engine = engine
        self.controller = controller
        self._attempt = attempt
        self._retry = retry
        self._accept = accept
        self.attempts = 0
        self._accepted = False

    def execute(self) -> None:
        if self._accepted:
            raise RuntimeError("prepared step-attempt sequence is already accepted")
        self.attempts += 1
        reported = self._attempt()
        if reported is None:
            return
        if isinstance(reported, bool) or not isinstance(reported, int) or reported <= 0:
            raise RuntimeError("step controller returned an invalid native-attempt count")
        if reported != 1:
            raise RuntimeError(
                "a retrying step-controller extension must override prepare_attempts() "
                "so every native attempt receives its own transaction")

    def retry(self, error: BaseException) -> bool:
        if self._accepted or self._retry is None:
            return False
        return bool(self._retry(error, self.attempts))

    def accept(self) -> None:
        if self._accepted:
            raise RuntimeError("prepared step-attempt sequence was accepted more than once")
        if self._accept is not None:
            self._accept(self.attempts)
        self._accepted = True


def _execute_prepared_attempts(sequence: _PreparedStepAttempts) -> int:
    """Compatibility executor for direct controller calls outside RuntimeInstance."""
    while True:
        try:
            sequence.execute()
        except StepAttemptRejected as error:
            if sequence.retry(error):
                continue
            sequence.controller.attempts = sequence.attempts
            raise
        sequence.accept()
        sequence.controller.attempts = sequence.attempts
        return sequence.attempts


class FixedDtController(StepController[FixedDt]):
    def prepare_attempts(
        self, engine: Any, native: Any, *, t_end: float,
    ) -> _PreparedStepAttempts:
        dt = min(self.strategy.dt, t_end - float(native.time()))
        if not dt > 0.0:
            raise RuntimeError("FixedDt has no positive interval left before the final time")

        def attempt() -> None:
            _native_attempt(engine, native, lambda: native.step(dt))

        return _PreparedStepAttempts(
            engine=engine,
            controller=self,
            attempt=attempt,
        )

    def execute(self, engine: Any, native: Any, *, t_end: float) -> int:
        return _execute_prepared_attempts(
            self.prepare_attempts(engine, native, t_end=t_end))


class AdaptiveCFLController(StepController[AdaptiveCFL]):
    def prepare_attempts(
        self, engine: Any, native: Any, *, t_end: float,
    ) -> _PreparedStepAttempts:
        remaining = t_end - float(native.time())
        caps = [remaining]
        if self.strategy.max_dt is not None:
            caps.append(self.strategy.max_dt)
        controls = dict(self.controls)
        if "dt_max" in controls:
            caps.append(float(controls["dt_max"]))
        max_dt = min(caps)
        min_dt = float(controls.get("dt_min", 0.0))

        def attempt() -> None:
            _native_attempt(
                engine,
                native,
                lambda: native.step_cfl(
                    self.strategy.cfl, max_dt=max_dt, min_dt=min_dt),
            )

        return _PreparedStepAttempts(
            engine=engine,
            controller=self,
            attempt=attempt,
        )

    def execute(self, engine: Any, native: Any, *, t_end: float) -> int:
        return _execute_prepared_attempts(
            self.prepare_attempts(engine, native, t_end=t_end))


class ErrorControlledDtController(StepController[ErrorControlledDt]):
    def __init__(self, strategy: ErrorControlledDt) -> None:
        super().__init__(strategy)
        self.next_dt = strategy.dt_init

    def restore_temporal_state(self, temporal: Any) -> None:
        if temporal is None or not getattr(temporal, "_restored_pending", False):
            return
        last_hex = getattr(temporal, "controller_state", {}).get("last_accepted_dt")
        if last_hex is None:
            raise RuntimeError(
                "ErrorControlledDt restart lacks the accepted dt needed for the next proposal")
        self.next_dt = min(
            self.strategy.dt_max,
            float.fromhex(last_hex) * self.strategy.growth,
        )

    def prepare_attempts(
        self, engine: Any, native: Any, *, t_end: float,
    ) -> _PreparedStepAttempts:
        proposal = [
            min(self.next_dt, self.strategy.dt_max, t_end - float(native.time()))
        ]

        def attempt() -> None:
            _native_attempt(
                engine,
                native,
                lambda: native.step(proposal[0]),
            )

        def retry(_error: BaseException, attempts: int) -> bool:
            if attempts > self.strategy.max_rejections:
                return False
            reduced = proposal[0] * self.strategy.shrink
            if reduced < self.strategy.dt_min:
                return False
            proposal[0] = reduced
            return True

        def accept(attempts: int) -> None:
            current = getattr(engine, "_step_controller", None)
            if current is None:
                current = self
            if type(current) is not ErrorControlledDtController \
                    or current.strategy != self.strategy:
                raise RuntimeError(
                    "ErrorControlledDt accepted through a different controller authority")
            current.next_dt = min(
                current.strategy.dt_max, proposal[0] * current.strategy.growth)
            current.attempts = attempts

        return _PreparedStepAttempts(
            engine=engine,
            controller=self,
            attempt=attempt,
            retry=retry,
            accept=accept,
        )

    def execute(self, engine: Any, native: Any, *, t_end: float) -> int:
        return _execute_prepared_attempts(
            self.prepare_attempts(engine, native, t_end=t_end))


class ExternalTimeGridController(StepController[ExternalTimeGrid]):
    def __init__(self, strategy: ExternalTimeGrid, grid: tuple[float, ...]) -> None:
        super().__init__(strategy, {strategy.grid_id: grid})
        self.grid = grid

    @staticmethod
    def _same_time(left: float, right: float) -> bool:
        scale = max(1.0, abs(left), abs(right))
        return abs(left - right) <= 4.0 * math.ulp(scale)

    def prepare_attempts(
        self, engine: Any, native: Any, *, t_end: float,
    ) -> _PreparedStepAttempts:
        now = float(native.time())
        index = bisect.bisect_left(self.grid, now)
        if index == len(self.grid) or not self._same_time(self.grid[index], now):
            if index and self._same_time(self.grid[index - 1], now):
                index -= 1
            else:
                raise RuntimeError("ExternalTimeGrid current time is not a declared grid point")
        if index + 1 >= len(self.grid):
            raise RuntimeError("ExternalTimeGrid is exhausted")
        next_time = self.grid[index + 1]
        if next_time > t_end and not self._same_time(next_time, t_end):
            raise RuntimeError("ExternalTimeGrid final time is not a declared grid point")

        def attempt() -> None:
            _native_attempt(engine, native, lambda: native.step(next_time - now))

        return _PreparedStepAttempts(
            engine=engine,
            controller=self,
            attempt=attempt,
        )

    def execute(self, engine: Any, native: Any, *, t_end: float) -> int:
        return _execute_prepared_attempts(
            self.prepare_attempts(engine, native, t_end=t_end))


@register_step_controller_factory(FixedDt)
def _fixed_dt_controller(
    strategy: StepStrategy, controls: Mapping[str, Any] | None,
) -> StepController[Any]:
    del controls
    if type(strategy) is not FixedDt:
        raise TypeError("FixedDt controller factory received another strategy type")
    return FixedDtController(strategy)


@register_step_controller_factory(AdaptiveCFL)
def _adaptive_cfl_controller(
    strategy: StepStrategy, controls: Mapping[str, Any] | None,
) -> StepController[Any]:
    if type(strategy) is not AdaptiveCFL:
        raise TypeError("AdaptiveCFL controller factory received another strategy type")
    return AdaptiveCFLController(strategy, controls)


@register_step_controller_factory(ErrorControlledDt)
def _error_controlled_dt_controller(
    strategy: StepStrategy, controls: Mapping[str, Any] | None,
) -> StepController[Any]:
    del controls
    if type(strategy) is not ErrorControlledDt:
        raise TypeError("ErrorControlledDt controller factory received another strategy type")
    return ErrorControlledDtController(strategy)


@register_step_controller_factory(ExternalTimeGrid)
def _external_time_grid_controller(
    strategy: StepStrategy, controls: Mapping[str, Any] | None,
) -> StepController[Any]:
    values = {} if controls is None else dict(controls)
    if type(strategy) is not ExternalTimeGrid:
        raise TypeError("ExternalTimeGrid controller factory received another strategy type")
    return ExternalTimeGridController(
        strategy, tuple(float(value) for value in values[strategy.grid_id]))


def materialize_step_controller(
    strategy: StepStrategy, controls: Mapping[str, Any] | None = None,
) -> StepController[Any]:
    """Materialize a runtime controller through the exact registered adapter only."""
    strategy.validate_runtime_controls(controls)
    factory = _CONTROLLER_FACTORIES.get(type(strategy))
    if factory is None:
        raise TypeError(
            "no runtime controller adapter is registered for StepStrategy type %s"
            % type(strategy).__name__)
    controller = factory(strategy, controls)
    if not isinstance(controller, StepController):
        raise TypeError("step controller factory must return a StepController")
    if controller.strategy is not strategy:
        raise TypeError("step controller factory must preserve the exact strategy authority")
    return controller


def resolve_run_strategy(engine: Any) -> StepStrategy:
    """Resolve the sole strategy authenticated by the installed Program."""
    from pops.time._step.transaction import ensure_step_strategy

    selected = getattr(engine, "_step_strategy", None)
    try:
        selected = ensure_step_strategy(selected)
    except TypeError:
        raise TypeError(
            "run requires an exact registered StepStrategy from a Program.step_strategy(...) "
            "contract authenticated at installation"
        ) from None
    return selected


def _controller(
    engine: Any, strategy: StepStrategy, controls: Mapping[str, Any] | None,
) -> StepController[Any]:
    strategy.validate_runtime_controls(controls)
    current = getattr(engine, "_step_controller", None)
    if current is None or not current.matches(strategy, controls):
        current = materialize_step_controller(strategy, controls)
        current.restore_temporal_state(getattr(engine, "_temporal_restart_state", None))
        engine._step_controller = current
    return current


def prepare_step_controller(
    engine: Any,
    strategy: StepStrategy,
    controls: Mapping[str, Any] | None = None,
) -> StepController[Any]:
    """Validate the complete execution contract before any attempt or side effect."""
    return _controller(engine, strategy, controls)


def prepare_step_attempts(
    engine: Any,
    native: Any,
    strategy: StepStrategy,
    *,
    t_end: float,
    controls: Mapping[str, Any] | None = None,
) -> _PreparedStepAttempts:
    """Prepare a detached retry cursor before opening the first native transaction."""
    controller = _controller(engine, strategy, controls)
    sequence = controller.prepare_attempts(engine, native, t_end=float(t_end))
    if type(sequence) is not _PreparedStepAttempts:
        raise TypeError("step controller prepare_attempts() must return _PreparedStepAttempts")
    if sequence.engine is not engine or sequence.controller is not controller:
        raise TypeError("prepared step attempts must retain their exact runtime authorities")
    return sequence


def run_prepared_step_attempt(
    sequence: _PreparedStepAttempts,
) -> StepTransactionReport:
    """Execute exactly one native attempt from a prepared retry cursor."""
    engine = sequence.engine
    try:
        sequence.execute()
        sequence.accept()
    except BaseException as error:
        _record_failure(engine, error, sequence.attempts)
        raise
    current = getattr(engine, "_step_controller", None)
    if current is not None:
        current.attempts = sequence.attempts
    stores = _stores(engine)
    report = StepTransactionReport(
        status="accepted", phase="commit", action="commit", attempts=sequence.attempts,
        staged_effects=stores, committed_effects=stores,
    )
    engine._last_step_transaction_report = report
    return report


def run_step_attempt(
    engine: Any,
    native: Any,
    strategy: StepStrategy,
    *,
    t_end: float,
    controls: Mapping[str, Any] | None = None,
) -> StepTransactionReport:
    """Execute one accepted step without an enclosing RuntimeInstance transaction.

    RuntimeInstance uses :func:`prepare_step_attempts` and
    :func:`run_prepared_step_attempt` directly so every retry receives a distinct native
    transaction.  This compatibility helper retains the small standalone controller protocol used
    by low-level tests and extension adapters.
    """
    sequence = prepare_step_attempts(
        engine, native, strategy, t_end=float(t_end), controls=controls)
    while True:
        try:
            return run_prepared_step_attempt(sequence)
        except StepAttemptRejected as error:
            if sequence.retry(error):
                continue
            raise


def run_control_payload(
    strategy: StepStrategy, controls: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stable execution manifest preserving the exact strategy and runtime controls."""
    strategy.validate_runtime_controls(controls)
    values = {} if controls is None else dict(controls)
    return {
        "strategy": strategy.to_data(),
        "controls": strategy.runtime_controls_data(values),
    }


__all__ = [
    "AdaptiveCFLController", "ErrorControlledDtController", "ExternalTimeGridController",
    "FixedDtController", "StepAttemptRejected", "StepController", "materialize_step_controller",
    "prepare_step_attempts", "prepare_step_controller", "register_step_controller_factory",
    "resolve_run_strategy", "run_control_payload", "run_prepared_step_attempt",
    "run_step_attempt",
]
