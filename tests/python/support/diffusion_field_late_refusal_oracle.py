"""Independent two accepted Euler intervals following one refused predictor."""

import json
from pathlib import Path
import numpy as np


def check_adaptive_saved(directory):
    folder = Path(directory)
    inputs = json.loads((folder / "inputs.json").read_text())
    nx, ny = inputs["cells"]
    dt = inputs["dt"]
    kappa = inputs["kappa"]
    x = (np.arange(nx, dtype=np.float64) + 0.5) / nx
    mode = np.broadcast_to(np.sinc(1 / nx) * np.sin(2 * np.pi * x), (1, ny, nx))
    lam = 4 * nx**2 * np.sin(np.pi / nx) ** 2
    z = dt * kappa * lam
    assert 2 * kappa * (2 * dt) * (nx * nx + ny * ny) < 1
    mean, amplitude = inputs["mean"], inputs["amplitude"]
    accepted_gain = (1 - z) ** 2
    reference = {
        "initial": mean + amplitude * mode,
        "predictor": mean + amplitude * accepted_gain * mode,
        "accepted": mean + amplitude * accepted_gain * mode,
        "phi_stage": (inputs["rhs_offset"] + 2 * dt)
        * (mean + amplitude * accepted_gain * mode / (1 + lam)),
    }
    state_bound = 4096 * np.finfo(np.float64).eps * (abs(mean) + abs(amplitude))
    field_bound = (
        nx
        * ny
        * (inputs["rhs_offset"] + 2 * dt)
        * (abs(mean) + abs(amplitude))
        * inputs["solver_rtol"]
        + inputs["solver_atol"]
        + state_bound
    )
    comparisons = {}
    for name, expected in reference.items():
        actual = np.load(folder / (name + ".npy"), allow_pickle=False)
        assert (
            actual.shape == expected.shape
            and actual.dtype == np.dtype("float64")
            and np.isfinite(actual).all()
        )
        bound = field_bound if name == "phi_stage" else state_bound
        error = float(np.max(np.abs(actual - expected)))
        assert error <= bound, (name, error, bound)
        comparisons[name] = {"max_abs_error": error, "bound": float(bound)}
    # These bounds are declared without Native results and use the original mode.
    norm_initial = 2 * dt * kappa * lam * abs(amplitude) * float(np.linalg.norm(mode))
    norm_retry = dt * kappa * lam * abs(amplitude) * float(np.linalg.norm(mode))
    assert (
        0.14 < norm_initial < 0.16
        and 0.07 < norm_retry < 0.08
        and norm_retry < 0.1 < norm_initial
    )
    return dict(
        status="PASS",
        accepted_intervals=2,
        effective_dt=dt,
        rejected_initial_dt=2 * dt,
        accepted_gain=float(accepted_gain),
        comparisons=comparisons,
    )


def check_two_fixed_intervals_saved(directory):
    """Two declared public FixedDt Euler invocations; no adaptive-retry inference."""
    folder = Path(directory)
    inputs = json.loads((folder / "inputs.json").read_text())
    assert inputs["method"] == "euler"
    assert inputs["accepted_intervals"] == 2 and inputs["history_sample_slot"] == 1
    assert inputs["final_time"] == 2 * inputs["dt"]
    nx, ny = inputs["cells"]
    dt, kappa = inputs["dt"], inputs["kappa"]
    faces = 2 * np.pi * np.arange(nx + 1, dtype=np.float64) / nx
    mode = np.broadcast_to(
        (np.cos(faces[:-1]) - np.cos(faces[1:])) / (2 * np.pi / nx), (1, ny, nx)
    )
    lam = 4 * nx**2 * np.sin(np.pi / nx) ** 2
    z = dt * kappa * lam
    gain = (1 - z) ** 2
    mean, amplitude = inputs["mean"], inputs["amplitude"]
    stage = inputs["rhs_offset"] + 2 * dt
    reference = {
        "initial": mean + amplitude * mode,
        "predictor": mean + amplitude * gain * mode,
        "accepted": mean + amplitude * gain * mode,
        "phi_stage": stage * (mean + amplitude * gain * mode / (1 + lam)),
    }
    state_bound = 4096 * np.finfo(np.float64).eps * (abs(mean) + abs(amplitude))
    field_bound = (
        nx * ny * stage * (abs(mean) + abs(amplitude)) * inputs["solver_rtol"]
        + inputs["solver_atol"]
        + state_bound
    )
    comparisons = {}
    for name, expected in reference.items():
        actual = np.load(folder / (name + ".npy"), allow_pickle=False)
        assert actual.shape == expected.shape and actual.dtype == np.dtype("float64")
        assert np.isfinite(actual).all()
        bound = field_bound if name == "phi_stage" else state_bound
        error = float(np.max(np.abs(actual - expected)))
        assert error <= bound, (name, error, bound)
        comparisons[name] = {"max_abs_error": error, "bound": float(bound)}
    first_norm = float(np.linalg.norm(mean + amplitude * (1 - z) * mode))
    second_norm = float(np.linalg.norm(mean + amplitude * gain * mode))
    assert 20 < second_norm < 30 and 20 < first_norm < 30
    return dict(
        status="PASS",
        accepted_intervals=2,
        dt=dt,
        final_time=2 * dt,
        accepted_gain=float(gain),
        first_predictor_norm=first_norm,
        second_predictor_norm=second_norm,
        comparisons=comparisons,
    )
