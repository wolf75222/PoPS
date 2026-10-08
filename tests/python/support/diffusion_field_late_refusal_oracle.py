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
