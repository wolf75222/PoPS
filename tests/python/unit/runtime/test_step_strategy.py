"""ADC-666: all final StepStrategy descriptors execute through one controller protocol."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import pytest

from pops.runtime._step_strategy import (
    StepController,
    StepAttemptRejected,
    prepare_step_controller,
    resolve_run_strategy,
    run_control_payload,
    run_step_attempt,
)
from pops.runtime._temporal_restart import TemporalRestartState
from pops.time import AdaptiveCFL, ErrorControlledDt, ExternalTimeGrid, FixedDt, StepStrategy
from pops.time.step_strategy import register_step_strategy_type
from pops.time.step_strategy import validate_step_strategy_manifest


class _Engine:
    def __init__(self, strategy=None):
        self._step_strategy = strategy
        self._step_transaction_plan = None
        self._step_controller = None
        self._last_step_transaction_report = None


class _Native:
    def __init__(self, *, reject=0):
        self.t = 0.0
        self.cursor = 0
        self.reject = reject
        self.calls = []

    def time(self):
        return self.t

    def macro_step(self):
        return self.cursor

    def step(self, dt):
        self.calls.append(("step", float(dt)))
        if self.reject:
            self.reject -= 1
            raise StepAttemptRejected("step attempt rejected during guard: test error estimate")
        self.t += float(dt)
        self.cursor += 1
        return float(dt)

    def step_cfl(self, cfl, *, max_dt, min_dt):
        self.calls.append(("step_cfl", float(cfl), float(max_dt), float(min_dt)))
        dt = min(0.25, float(max_dt))
        if dt < float(min_dt):
            raise RuntimeError("stability bound is below declared min_dt")
        self.t += dt
        self.cursor += 1
        return dt


def _error_strategy():
    return ErrorControlledDt(
        dt_init=0.2,
        rtol=1.0e-4,
        atol=1.0e-8,
        dt_min=0.01,
        dt_max=0.5,
        max_rejections=3,
        shrink=0.5,
        growth=1.5,
    )


def test_step_strategy_is_closed_exact_and_validated():
    with pytest.raises(TypeError, match="StepStrategy is closed"):
        StepStrategy()

    class Forged(FixedDt):
        pass

    with pytest.raises(TypeError, match="StepStrategy"):
        resolve_run_strategy(_Engine(Forged(0.1)))
    for bad in (0.0, -1.0, float("nan"), True):
        with pytest.raises((TypeError, ValueError)):
            FixedDt(bad)
    with pytest.raises(ValueError, match="strictly increasing"):
        ExternalTimeGrid("grid").validate_runtime_controls({"grid": [0.0, 0.0]})


def test_resolve_requires_the_installed_authored_strategy():
    with pytest.raises(TypeError, match=r"Program\.step_strategy"):
        resolve_run_strategy(_Engine())
    authored = FixedDt(0.1)
    assert resolve_run_strategy(_Engine(authored)) is authored


def test_all_four_controllers_execute_real_native_attempts():
    fixed_native = _Native()
    fixed_report = run_step_attempt(
        _Engine(), fixed_native, FixedDt(0.1), t_end=1.0)
    assert fixed_report.attempts == 1
    assert fixed_native.calls == [("step", 0.1)]

    cfl_native = _Native()
    cfl_report = run_step_attempt(
        _Engine(), cfl_native, AdaptiveCFL(0.4, max_dt=0.2), t_end=1.0,
        controls={"dt_min": 0.01, "dt_max": 0.15})
    assert cfl_report.attempts == 1
    assert cfl_native.calls == [("step_cfl", 0.4, 0.15, 0.01)]
    assert cfl_native.time() == 0.15

    error_native = _Native(reject=1)
    error_engine = _Engine()
    error_report = run_step_attempt(
        error_engine, error_native, _error_strategy(), t_end=1.0)
    assert error_report.attempts == 2
    assert error_native.calls == [("step", 0.2), ("step", 0.1)]
    assert error_native.time() == 0.1

    grid_native = _Native()
    grid = ExternalTimeGrid("forcing_times")
    grid_report = run_step_attempt(
        _Engine(), grid_native, grid, t_end=0.5,
        controls={"forcing_times": [0.0, 0.125, 0.5]})
    assert grid_report.attempts == 1
    assert grid_native.calls == [("step", 0.125)]


def test_error_controlled_exhaustion_preserves_rejection_and_exact_attempt_count():
    native = _Native(reject=4)
    engine = _Engine()
    with pytest.raises(StepAttemptRejected):
        run_step_attempt(engine, native, _error_strategy(), t_end=1.0)
    assert engine._last_step_transaction_report.status == "rejected"
    assert engine._last_step_transaction_report.phase == "guard"
    assert engine._last_step_transaction_report.attempts == 4
    assert native.time() == 0.0


def test_controls_are_validated_before_controller_or_manifest_publication():
    strategy = ExternalTimeGrid("forcing_times")
    engine = _Engine()
    with pytest.raises(ValueError, match="forcing_times"):
        prepare_step_controller(engine, strategy, {"other": [0.0, 1.0]})
    assert engine._step_controller is None

    payload = run_control_payload(strategy, {"forcing_times": (0.0, 0.5, 1.0)})
    assert payload["strategy"] == {"kind": "external_time_grid", "grid_id": "forcing_times"}
    assert [row["value"] for row in payload["controls"]["forcing_times"]] == [
        (0.0).hex(), (0.5).hex(), (1.0).hex(),
    ]


def test_controller_identity_normalizes_external_grid_list_and_tuple():
    strategy = ExternalTimeGrid("grid")
    engine = SimpleNamespace(_step_controller=None)
    first = prepare_step_controller(engine, strategy, {"grid": [0.0, 1.0]})
    second = prepare_step_controller(engine, strategy, {"grid": (0.0, 1.0)})
    assert second is first


def test_error_controller_restores_the_exact_next_proposal_after_restart():
    strategy = _error_strategy()
    temporal = TemporalRestartState(
        strategy=run_control_payload(strategy),
        controller_state={"last_accepted_dt": (0.1).hex()},
        _restored_pending=True,
    )
    engine = SimpleNamespace(
        _step_controller=None,
        _temporal_restart_state=temporal,
        _step_transaction_plan=None,
        _last_step_transaction_report=None,
    )
    controller = prepare_step_controller(engine, strategy)

    assert controller.next_dt == pytest.approx(0.15)
    native = _Native()
    run_step_attempt(engine, native, strategy, t_end=1.0)
    assert native.calls == [("step", pytest.approx(0.15))]


def test_registered_strategy_and_controller_own_extension_and_restart_protocols():
    class Controller(StepController):
        def __init__(self, strategy):
            super().__init__(strategy)
            self.restored = False

        def restore_temporal_state(self, temporal):
            self.restored = temporal.marker

        def execute(self, engine, native, *, t_end):
            native.step(min(self.strategy.dt, t_end - native.time()))
            return 1

    @register_step_strategy_type
    @dataclass(frozen=True, slots=True)
    class Registered(StepStrategy):
        dt: float
        kind: ClassVar[str] = "test_registered_strategy"

        def to_data(self):
            return {"kind": self.kind, "dt": self.dt}

        def runtime_controller(self, controls=None):
            self.validate_runtime_controls(controls)
            return Controller(self)

    strategy = Registered(0.25)
    engine = _Engine(strategy)
    engine._temporal_restart_state = SimpleNamespace(marker=True)

    assert resolve_run_strategy(engine) is strategy
    controller = prepare_step_controller(engine, strategy)
    assert controller.restored is True


def test_registered_strategy_provider_owns_strict_restart_reconstruction():
    from pops.ir.literals import scalar_data

    @register_step_strategy_type
    @dataclass(frozen=True, slots=True)
    class Restartable(StepStrategy):
        dt: float
        kind: ClassVar[str] = "test_restartable_strategy"

        def to_data(self):
            return {"kind": self.kind, "dt": scalar_data(self.dt)}

        @classmethod
        def from_data(cls, payload):
            if set(payload) != {"kind", "dt"} or payload["kind"] != cls.kind:
                raise ValueError("invalid Restartable manifest")
            value = payload["dt"]
            if set(value) != {"kind", "value"} or value["kind"] != "binary64":
                raise ValueError("invalid Restartable dt")
            return cls(float.fromhex(value["value"]))

        def runtime_controller(self, controls=None):
            self.validate_runtime_controls(controls)
            raise NotImplementedError

    payload = run_control_payload(Restartable(0.25))
    assert validate_step_strategy_manifest(payload) == payload
    forged = {"strategy": {**payload["strategy"], "extra": True}, "controls": {}}
    with pytest.raises(ValueError, match="invalid Restartable manifest"):
        validate_step_strategy_manifest(forged)
