"""Strict checkpoint codec for the shared Uniform/AMR Program cadence window."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from typing import Any


_SUBSTEPS = "program_cadence_substeps"
_STRIDE = "program_cadence_stride"
_WINDOW_STEPS = "program_cadence_window_steps"
_WINDOW_DT = "program_cadence_window_dt"
_WINDOW_START_TIME = "program_cadence_window_start_time"
_LAST_DT = "program_last_dt"
PROGRAM_CADENCE_CHECKPOINT_KEYS = frozenset(
    {_SUBSTEPS, _STRIDE, _WINDOW_STEPS, _WINDOW_DT, _WINDOW_START_TIME, _LAST_DT}
)


@dataclass(frozen=True, slots=True)
class ProgramCadenceCheckpointState:
    """Authenticated configuration and partially accumulated stride window."""

    substeps: int
    stride: int
    held_steps: int
    accumulated_dt: float
    window_start_time: float
    last_dt: float

    def to_payload(self) -> dict[str, Any]:
        return {
            _SUBSTEPS: self.substeps,
            _STRIDE: self.stride,
            _WINDOW_STEPS: self.held_steps,
            _WINDOW_DT: self.accumulated_dt,
            _WINDOW_START_TIME: self.window_start_time,
            _LAST_DT: self.last_dt,
        }

    def to_data(self) -> dict[str, Any]:
        return {
            "substeps": self.substeps,
            "stride": self.stride,
            "held_steps": self.held_steps,
            "accumulated_dt": self.accumulated_dt.hex(),
            "window_start_time": self.window_start_time.hex(),
            "last_dt": self.last_dt.hex(),
        }


def _payload_files(payload: Any) -> set[str]:
    files = getattr(payload, "files", None)
    if files is not None:
        file_values = files() if callable(files) else files
        if not isinstance(file_values, Iterable):
            raise TypeError("checkpoint payload files must be iterable")
        return {str(key) for key in file_values}
    keys = getattr(payload, "keys", None)
    if not callable(keys):
        raise TypeError("checkpoint payload exposes neither files nor keys()")
    key_values = keys()
    if not isinstance(key_values, Iterable):
        raise TypeError("checkpoint payload keys() must return an iterable")
    return {str(key) for key in key_values}


def _exact_integer_scalar(payload: Any, key: str) -> int:
    import numpy as np

    value = np.asarray(payload[key])
    if value.ndim != 0 or value.dtype.kind not in "iu":
        raise TypeError("restart : %s must be an exact integer scalar" % key)
    return int(value.item())


def _exact_float_scalar(payload: Any, key: str) -> float:
    import numpy as np

    value = np.asarray(payload[key])
    if value.ndim != 0 or value.dtype.kind != "f" or value.dtype.itemsize != 8:
        raise TypeError("restart : %s must be an exact binary64 scalar" % key)
    return float(value.item())


def _validate_state(
    state: ProgramCadenceCheckpointState,
    *,
    macro_step: int,
    accepted_time: float,
    phase: str,
) -> None:
    if state.substeps < 1 or state.stride < 1:
        raise ValueError("%s : Program cadence substeps and stride must be positive" % phase)
    if macro_step < 0:
        raise ValueError("%s : Program cadence macro_step must be non-negative" % phase)
    if not isfinite(accepted_time):
        raise ValueError("%s : Program cadence accepted time must be finite" % phase)
    if state.held_steps < 0 or state.held_steps >= state.stride:
        raise ValueError(
            "%s : Program cadence held-step count %d lies outside [0, %d)"
            % (phase, state.held_steps, state.stride)
        )
    expected_phase = macro_step % state.stride
    if state.held_steps != expected_phase:
        raise ValueError(
            "%s : Program cadence held-step count %d differs from macro-step phase %d"
            % (phase, state.held_steps, expected_phase)
        )
    if not isfinite(state.accumulated_dt) or state.accumulated_dt < 0.0:
        raise ValueError(
            "%s : Program cadence accumulated duration must be finite and non-negative" % phase
        )
    if (state.held_steps == 0) != (state.accumulated_dt == 0.0):
        raise ValueError(
            "%s : Program cadence accumulated duration must be zero exactly when no step is held"
            % phase
        )
    if not isfinite(state.window_start_time):
        raise ValueError("%s : Program cadence window start time must be finite" % phase)
    if state.held_steps == 0 and state.window_start_time != 0.0:
        raise ValueError(
            "%s : inactive Program cadence window must use the canonical zero start time" % phase
        )
    if state.held_steps != 0 and not state.window_start_time < accepted_time:
        raise ValueError(
            "%s : active Program cadence window start must precede accepted time" % phase
        )
    if not isfinite(state.last_dt) or state.last_dt < 0.0:
        raise ValueError("%s : Program accepted last dt must be finite and non-negative" % phase)


def capture_program_cadence(sim: Any, *, macro_step: int) -> ProgramCadenceCheckpointState:
    """Capture the exact native cadence image; never infer held duration from the clock."""

    required = (
        "time",
        "program_substeps",
        "program_stride",
        "program_cadence_window_steps",
        "program_cadence_window_dt",
        "program_cadence_window_start_time",
        "program_last_dt",
    )
    missing = [name for name in required if not callable(getattr(sim, name, None))]
    if missing:
        raise TypeError("checkpoint engine lacks strict Program cadence accessors %r" % missing)
    state = ProgramCadenceCheckpointState(
        substeps=int(sim.program_substeps()),
        stride=int(sim.program_stride()),
        held_steps=int(sim.program_cadence_window_steps()),
        accumulated_dt=float(sim.program_cadence_window_dt()),
        window_start_time=float(sim.program_cadence_window_start_time()),
        last_dt=float(sim.program_last_dt()),
    )
    _validate_state(
        state,
        macro_step=macro_step,
        accepted_time=float(sim.time()),
        phase="checkpoint",
    )
    return state


def prepare_program_cadence(
    sim: Any,
    payload: Any,
    *,
    macro_step: int,
    accepted_time: float,
) -> ProgramCadenceCheckpointState:
    """Validate a complete payload against the already installed native cadence."""

    missing = sorted(PROGRAM_CADENCE_CHECKPOINT_KEYS - _payload_files(payload))
    if missing:
        raise ValueError(
            "restart : strict checkpoint is missing Program cadence key(s) %s" % ", ".join(missing)
        )
    state = ProgramCadenceCheckpointState(
        substeps=_exact_integer_scalar(payload, _SUBSTEPS),
        stride=_exact_integer_scalar(payload, _STRIDE),
        held_steps=_exact_integer_scalar(payload, _WINDOW_STEPS),
        accumulated_dt=_exact_float_scalar(payload, _WINDOW_DT),
        window_start_time=_exact_float_scalar(payload, _WINDOW_START_TIME),
        last_dt=_exact_float_scalar(payload, _LAST_DT),
    )
    _validate_state(
        state,
        macro_step=macro_step,
        accepted_time=accepted_time,
        phase="restart",
    )
    current_substeps = int(sim.program_substeps())
    current_stride = int(sim.program_stride())
    if (state.substeps, state.stride) != (current_substeps, current_stride):
        raise ValueError(
            "restart : checkpoint Program cadence (%d substeps, stride %d) differs from "
            "the installed cadence (%d substeps, stride %d)"
            % (
                state.substeps,
                state.stride,
                current_substeps,
                current_stride,
            )
        )
    if not callable(getattr(sim, "restore_program_cadence_window", None)):
        raise TypeError("restart engine lacks the strict Program cadence restore seam")
    return state


def restore_program_cadence(
    sim: Any,
    state: ProgramCadenceCheckpointState,
    *,
    macro_step: int,
    accepted_time: float,
) -> None:
    """Stage the authenticated window immediately before the matching clock restore."""

    if type(state) is not ProgramCadenceCheckpointState:
        raise TypeError("Program cadence restore requires its exact prepared state")
    _validate_state(
        state,
        macro_step=macro_step,
        accepted_time=accepted_time,
        phase="restart",
    )
    sim.restore_program_cadence_window(
        state.accumulated_dt,
        state.held_steps,
        state.window_start_time,
        state.last_dt,
        accepted_time,
        macro_step,
    )


__all__ = [
    "PROGRAM_CADENCE_CHECKPOINT_KEYS",
    "ProgramCadenceCheckpointState",
    "capture_program_cadence",
    "prepare_program_cadence",
    "restore_program_cadence",
]
