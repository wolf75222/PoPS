"""ADC-666: run controllers are explicit typed StepStrategy objects."""
from __future__ import annotations

import inspect

import pytest

from pops.runtime._step_strategy import resolve_run_strategy, run_control_payload, run_step_attempt
from pops.runtime.system import System
from pops.time import AdaptiveCFL, ErrorControlledDt, ExternalTimeGrid, FixedDt, StepStrategy


class _Engine:
    def __init__(self):
        self._program_cadence_cfl = None


class _Native:
    def __init__(self):
        self.t = 0.0
        self.calls = []

    def time(self):
        return self.t

    def step(self, dt):
        self.calls.append(("step", dt))
        self.t += dt

    def step_cfl(self, cfl):
        self.calls.append(("step_cfl", cfl))
        self.t += 0.25


def test_step_strategy_is_closed_and_validated():
    with pytest.raises(TypeError, match="StepStrategy is closed"):
        StepStrategy()
    assert FixedDt(0.125).to_data()["dt"]["value"] == "0x1.0000000000000p-3"
    assert AdaptiveCFL(0.4).to_data()["cfl"]["value"] == "0x1.999999999999ap-2"
    assert ErrorControlledDt(1e-3, 1e-7).to_data()["rtol"]["kind"] == "binary64"
    assert ExternalTimeGrid("grid").grid_id == "grid"
    for bad in (0.0, -1.0, float("nan"), True):
        with pytest.raises((TypeError, ValueError)):
            FixedDt(bad)
    with pytest.raises(ValueError, match="strictly increasing"):
        ExternalTimeGrid("grid").validate_runtime_controls({"grid": [0.1, 0.1]})


def test_resolve_run_strategy_rejects_mixed_controls_and_preserves_legacy_default():
    engine = _Engine()
    assert resolve_run_strategy(engine, None, None) == AdaptiveCFL(0.4)
    engine._program_cadence_cfl = 0.2
    assert resolve_run_strategy(engine, None, None) == AdaptiveCFL(0.2)
    assert resolve_run_strategy(engine, None, 0.7) == AdaptiveCFL(0.7)
    with pytest.raises(ValueError, match="cfl=.*strategy"):
        resolve_run_strategy(engine, FixedDt(0.1), 0.4)
    with pytest.raises(TypeError, match="StepStrategy"):
        resolve_run_strategy(engine, object(), None)


def test_run_step_attempt_dispatches_only_the_declared_controller():
    native = _Native()
    run_step_attempt(_Engine(), native, AdaptiveCFL(0.3), t_end=1.0)
    run_step_attempt(_Engine(), native, FixedDt(0.1), t_end=1.0)
    assert native.calls == [("step_cfl", 0.3), ("step", 0.1)]
    with pytest.raises(NotImplementedError, match="ErrorControlledDt"):
        run_step_attempt(_Engine(), native, ErrorControlledDt(0.1, 1e-3), t_end=1.0)
    with pytest.raises(NotImplementedError, match="ExternalTimeGrid"):
        run_step_attempt(_Engine(), native, ExternalTimeGrid("grid"), t_end=1.0)


def test_strategy_payload_is_reportable_and_run_signature_keeps_cfl_default():
    assert run_control_payload(FixedDt(0.1)) == {"strategy": "fixed_dt", "dt": 0.1}
    assert run_control_payload(AdaptiveCFL(0.4)) == {"strategy": "adaptive_cfl", "cfl": 0.4}
    assert inspect.signature(System.run).parameters["cfl"].default is None
    assert "strategy" in inspect.signature(System.run).parameters
