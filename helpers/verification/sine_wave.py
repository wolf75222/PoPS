"""Exact finite-volume references and small diagnostics for periodic sine waves.

The helpers deliberately operate on NumPy arrays only: they are usable by both
the PoPS data producer and plot-only/report-only consumers.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ErrorNorms:
    """Volume-normalised errors against an independent finite-volume oracle."""

    l1: float
    l2: float
    linf: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _positive_ints(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    items = tuple(values)
    if not items or any(isinstance(value, (bool, np.bool_)) for value in items):
        raise ValueError("%s must contain positive extents" % name)
    if any(not isinstance(value, (int, np.integer)) or value < 1 for value in items):
        raise ValueError("%s must contain positive extents" % name)
    return tuple(int(value) for value in items)


def direction_velocity(mode: str, dimension: int) -> tuple[float, ...]:
    """Return an integer propagation direction, rejecting transverse modes by rank."""
    if (
        isinstance(dimension, (bool, np.bool_))
        or not isinstance(dimension, (int, np.integer))
        or dimension not in (1, 2, 3)
    ):
        raise ValueError("dimension must be 1, 2, or 3")
    directions = {
        "x": (1, 0, 0),
        "y": (0, 1, 0),
        "z": (0, 0, 1),
        "diagonal": (1, 1, 1),
    }
    if mode not in directions:
        raise ValueError("mode must be x, y, z, or diagonal")
    values = directions[mode]
    if mode != "diagonal" and any(values[index] for index in range(dimension, 3)):
        raise ValueError("mode %r is unavailable in dimension %d" % (mode, dimension))
    return tuple(float(value) for value in values[:dimension])


def sine_wave_cell_averages(
    resolution: Sequence[int],
    wave_numbers: Sequence[int],
    *,
    epsilon: float,
    displacement: Sequence[float] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return exact cell averages and x-first coordinate vectors.

    A tensor-product cell average multiplies the sine amplitude by
    ``prod(sinc(k_i / n_i))``.  Arrays follow PoPS' public extraction order:
    the x physical coordinate is the last NumPy axis.
    """
    cells = _positive_ints(resolution, name="resolution")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in wave_numbers
    ):
        raise ValueError("wave_numbers must contain exact integers")
    waves = tuple(int(value) for value in wave_numbers)
    if len(waves) != len(cells):
        raise ValueError("wave_numbers and resolution must have the same rank")
    if not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite")
    shift = (0.0,) * len(cells) if displacement is None else tuple(displacement)
    if len(shift) != len(cells) or not np.isfinite(shift).all():
        raise ValueError("displacement must be finite and match the rank")

    coordinates = tuple((np.arange(count, dtype=np.float64) + 0.5) / count for count in cells)
    numpy_axes = np.meshgrid(*coordinates[::-1], indexing="ij")
    phase = np.zeros(tuple(reversed(cells)), dtype=np.float64)
    for numpy_axis, coefficient, offset in zip(numpy_axes, waves[::-1], shift[::-1], strict=True):
        phase += coefficient * (numpy_axis - offset)
    average_factor = float(np.prod(np.sinc(np.asarray(waves, dtype=np.float64) / cells)))
    return 1.0 + epsilon * average_factor * np.sin(2.0 * np.pi * phase), coordinates


def weighted_error_norms(
    numerical: np.ndarray,
    exact: np.ndarray,
    volumes: np.ndarray | float,
) -> ErrorNorms:
    """Compute L1/L2/Linf with positive finite weights and overflow checks."""
    numerical64 = np.asarray(numerical, dtype=np.float64)
    exact64 = np.asarray(exact, dtype=np.float64)
    weights = np.broadcast_to(np.asarray(volumes, dtype=np.float64), numerical64.shape)
    if numerical64.shape != exact64.shape or numerical64.size == 0:
        raise ValueError("numerical and exact fields must be non-empty and equally shaped")
    if not (np.isfinite(numerical64).all() and np.isfinite(exact64).all()):
        raise ValueError("fields must be finite")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("volumes must be finite and strictly positive")
    error = np.abs(numerical64 - exact64)
    total = float(np.sum(weights, dtype=np.float64))
    l1_sum = float(np.sum(weights * error, dtype=np.float64))
    l2_sum = float(np.sum(weights * error * error, dtype=np.float64))
    if not np.isfinite((total, l1_sum, l2_sum)).all() or total <= 0.0:
        raise ValueError("weighted reduction overflowed or has zero volume")
    return ErrorNorms(
        l1=l1_sum / total,
        l2=float(np.sqrt(l2_sum / total)),
        linf=float(np.max(error)),
    )


def sine_diagnostics(
    field: np.ndarray,
    exact: np.ndarray,
    volumes: np.ndarray | float,
) -> dict[str, float | bool | None | dict[str, float]]:
    """Return conservation, amplitude and phase diagnostics from finite-volume data."""
    field64 = np.asarray(field, dtype=np.float64)
    exact64 = np.asarray(exact, dtype=np.float64)
    weights = np.broadcast_to(np.asarray(volumes, dtype=np.float64), field64.shape)
    norms = weighted_error_norms(field64, exact64, weights)
    total = float(np.sum(weights, dtype=np.float64))
    mean = float(np.sum(weights * field64, dtype=np.float64) / total)
    exact_mean = float(np.sum(weights * exact64, dtype=np.float64) / total)
    centred = field64 - mean
    reference = exact64 - exact_mean
    amplitude = float(np.sqrt(np.sum(weights * centred * centred) / total))
    reference_amplitude = float(np.sqrt(np.sum(weights * reference * reference) / total))
    correlation = float(np.sum(weights * centred * reference) / total)
    phase_defined = bool(
        amplitude > np.finfo(float).eps and reference_amplitude > np.finfo(float).eps
    )
    phase_cosine = (
        float(np.clip(correlation / (amplitude * reference_amplitude), -1.0, 1.0))
        if phase_defined
        else None
    )
    return {
        "errors": norms.to_dict(),
        "mass": float(np.sum(weights * field64, dtype=np.float64)),
        "exact_mass": float(np.sum(weights * exact64, dtype=np.float64)),
        "amplitude_rms": amplitude,
        "exact_amplitude_rms": reference_amplitude,
        "phase_cosine": phase_cosine,
        "phase_defined": phase_defined,
    }


def convergence_orders(resolutions: Iterable[int], errors: Iterable[float]) -> list[float | None]:
    """Observed adjacent-grid orders; incompatible/non-positive entries remain null."""
    counts = list(_positive_ints(tuple(resolutions), name="resolutions"))
    values = [float(value) for value in errors]
    if len(counts) != len(values):
        raise ValueError("resolutions and errors must have the same length")
    result: list[float | None] = [None]
    for old_count, new_count, old_error, new_error in zip(
        counts[:-1], counts[1:], values[:-1], values[1:], strict=True
    ):
        if (
            old_count < 1
            or new_count <= old_count
            or not np.isfinite((old_error, new_error)).all()
            or old_error <= 0.0
            or new_error <= 0.0
        ):
            result.append(None)
        else:
            result.append(float(np.log(old_error / new_error) / np.log(new_count / old_count)))
    return result
