"""Private diagnostics, provenance, and publication support for the sine-wave case.

This module deliberately contains the non-pedagogical mechanics.  The readable
case in :mod:`generate_data` passes every scientific constant explicitly and is
the sole authority that constructs or runs a PoPS simulation.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from helpers.verification import (  # noqa: E402
    sine_diagnostics,
    sine_wave_cell_averages,
    weighted_error_norms,
)

BUILD_SOURCE_ROOTS = (
    "CMakeLists.txt",
    "cmake",
    "include",
    "src",
    "python",
    "pyproject.toml",
    "schemas",
    "scripts",
)
BUILD_SOURCE_EXCLUDED_NAMES = frozenset(
    {
        ".git",
        ".pytest_cache",
        "__pycache__",
        "build",
        "cache",
        "figures",
        "report",
        "results",
    }
)


def _git_value(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments), cwd=repository_root, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _git_content_sha256(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments), cwd=repository_root, capture_output=True, check=False
    )
    return (
        hashlib.sha256(completed.stdout).hexdigest() if completed.returncode == 0 else "unavailable"
    )


def _active_mask(
    patch_rows: tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...],
    *,
    cells: tuple[int, ...],
    level: int,
    refinement_ratio: int,
) -> np.ndarray:
    """Composite leaf mask; patch upper bounds are inclusive and NumPy axes are reversed."""
    level_cells = tuple(value * refinement_ratio**level for value in cells)
    mask = (
        np.ones(tuple(reversed(level_cells)), dtype=bool)
        if level == 0
        else np.zeros(tuple(reversed(level_cells)), dtype=bool)
    )

    def slices(lower: tuple[int, ...], upper: tuple[int, ...], scale: int = 1) -> tuple[slice, ...]:
        return tuple(
            slice(lo // scale, hi // scale + 1)
            for lo, hi in zip(reversed(lower), reversed(upper), strict=True)
        )

    if level:
        for patch_level, lower, upper in patch_rows:
            if patch_level == level:
                mask[slices(lower, upper)] = True
    for patch_level, lower, upper in patch_rows:
        if patch_level == level + 1:
            mask[slices(lower, upper, refinement_ratio)] = False
    return mask


def _interface_mask(active: np.ndarray) -> np.ndarray:
    """Mark active leaves sharing a face with an inactive same-level cell, periodically."""
    mask = np.asarray(active)
    if mask.dtype != np.bool_ or mask.ndim not in (1, 2, 3):
        raise ValueError("active leaf mask must be a rank-1/2/3 boolean array")
    interface = np.zeros_like(mask)
    for axis in range(mask.ndim):
        interface |= mask & ~np.roll(mask, 1, axis=axis)
        interface |= mask & ~np.roll(mask, -1, axis=axis)
    return interface


def collect_snapshot(
    simulation,
    *,
    cells: tuple[int, ...],
    velocity: tuple[float, ...],
    waves: tuple[int, ...],
    layout: str,
    time: float,
    epsilon: float,
    refinement_ratio: int,
):
    """Collect all global fields collectively; the caller decides which rank writes them."""
    if layout == "uniform":
        numerical = np.asarray(simulation.state_global("tracer"), dtype=np.float64).reshape(
            tuple(reversed(cells))
        )
        initial, _ = sine_wave_cell_averages(cells, waves, epsilon=epsilon)
        exact, coordinates = sine_wave_cell_averages(
            cells,
            waves,
            epsilon=epsilon,
            displacement=tuple(speed * time for speed in velocity),
        )
        quarter_displacement = [speed * time for speed in velocity]
        quarter_displacement[0] += 0.25 / waves[0]
        shifted_quarter, _ = sine_wave_cell_averages(
            cells,
            waves,
            epsilon=epsilon,
            displacement=tuple(quarter_displacement),
        )
        return {
            "initial": initial,
            "numeric": numerical,
            "exact": exact,
            "quadrature": 2.0 - shifted_quarter,
            "mask": np.ones_like(numerical, dtype=bool),
            "interface_mask": np.zeros_like(numerical, dtype=bool),
            "coordinates": coordinates,
            "patch_rows": (),
            "local_boxes": tuple(simulation.local_boxes("tracer")),
            "regrid_count": 0,
            "topology_epoch": 0,
        }

    patch_rows = tuple(
        (int(level), tuple(lower), tuple(upper)) for level, lower, upper in simulation.patch_boxes()
    )
    regrid_report = simulation.amr.explain_regrid()
    result: dict[str, object] = {
        "patch_rows": patch_rows,
        "regrid_count": int(regrid_report.regrid_count),
        "topology_epoch": int(regrid_report.topology_epoch),
    }
    for level in range(int(simulation.n_levels())):
        level_cells = tuple(value * refinement_ratio**level for value in cells)
        numerical = np.asarray(
            simulation.block_level_state_global("tracer", level), dtype=np.float64
        ).reshape(tuple(reversed(level_cells)))
        initial, _ = sine_wave_cell_averages(level_cells, waves, epsilon=epsilon)
        exact, coordinates = sine_wave_cell_averages(
            level_cells,
            waves,
            epsilon=epsilon,
            displacement=tuple(speed * time for speed in velocity),
        )
        quarter_displacement = [speed * time for speed in velocity]
        quarter_displacement[0] += 0.25 / waves[0]
        shifted_quarter, _ = sine_wave_cell_averages(
            level_cells,
            waves,
            epsilon=epsilon,
            displacement=tuple(quarter_displacement),
        )
        result["initial_level_%d" % level] = initial
        result["numeric_level_%d" % level] = numerical
        result["exact_level_%d" % level] = exact
        result["quadrature_level_%d" % level] = 2.0 - shifted_quarter
        active_mask = _active_mask(
            patch_rows,
            cells=cells,
            level=level,
            refinement_ratio=refinement_ratio,
        )
        result["mask_level_%d" % level] = active_mask
        result["interface_mask_level_%d" % level] = _interface_mask(active_mask)
        result["coordinates_level_%d" % level] = coordinates
    return result


def coverage_witnesses(
    *,
    dimension: int,
    velocity: tuple[float, ...],
    layout: str,
    resolution: tuple[int, ...],
    mode: str,
    cycles: int,
    obligations: tuple[str, ...],
    timeline_times: tuple[float, ...],
    timeline_snapshots: tuple[dict[str, object], ...],
    metrics: dict[str, object],
    witness_reference_point: tuple[float, ...],
    patch_velocity: tuple[float, ...],
    patch_center: tuple[float, ...] | None = None,
    patch_half_width: tuple[float, ...] | None = None,
    wave_numbers: tuple[int, ...],
    final_time: float,
    refinement_ratio: int,
) -> dict[str, dict[str, object]]:
    """Attach named, falsifiable coverage witnesses to one completed case.

    This is deliberately diagnostic-only: the readable case passes its
    scientific constants explicitly, and this function neither constructs nor
    advances a PoPS simulation.  A crossing is accepted only if actual native
    ``local_boxes`` expose every internal plane, the selected material
    reference intersects those planes at one exact time, and two saved
    timeline times enclose that time.
    """
    if (
        dimension not in (1, 2, 3)
        or len(velocity) != dimension
        or len(resolution) != dimension
        or len(witness_reference_point) < dimension
        or len(patch_velocity) < dimension
        or len(wave_numbers) != dimension
        or not np.isfinite(final_time)
        or final_time <= 0.0
        or type(refinement_ratio) is not int
        or refinement_ratio < 2
    ):
        raise ValueError("coverage witness inputs do not describe one finite dimensional case")
    speed_axes = tuple(index for index, speed in enumerate(velocity) if speed != 0.0)
    native_boxes = tuple(timeline_snapshots[0].get("local_boxes", ()))

    if layout == "amr-mobile":
        if patch_center is None or patch_half_width is None:
            raise ValueError(
                "amr-mobile coverage requires the prescribed patch_center and patch_half_width"
            )
        if (
            len(patch_center) < dimension
            or len(patch_half_width) < dimension
            or any(
                not np.isfinite(value)
                for values in (patch_center[:dimension], patch_half_width[:dimension])
                for value in values
            )
            or any(value <= 0.0 or value >= 0.5 for value in patch_half_width[:dimension])
        ):
            raise ValueError("amr-mobile prescribed patch geometry is invalid")

    def internal_planes(axis: int) -> tuple[float, ...]:
        bounds = {
            endpoint[axis]
            for lower, upper in native_boxes
            for endpoint in (lower, upper)
            if 0 < endpoint[axis] < resolution[axis]
        }
        return tuple(sorted(value / resolution[axis] for value in bounds))

    def enclosed(time: float) -> bool:
        return any(
            left < time < right
            for left, right in zip(timeline_times[:-1], timeline_times[1:], strict=True)
        )

    def crossing(active_axes: tuple[int, ...]) -> dict[str, object]:
        if layout != "uniform" or not active_axes or len(native_boxes) == 0:
            return {"observed": False, "time": None, "planes": []}
        candidates: list[tuple[float, tuple[float, ...]]] = []
        first_axis = active_axes[0]
        for plane in internal_planes(first_axis):
            time = (plane - witness_reference_point[first_axis]) / velocity[first_axis]
            if not 0.0 < time < final_time:
                continue
            planes = [plane]
            for axis in active_axes[1:]:
                matching = next(
                    (
                        candidate
                        for candidate in internal_planes(axis)
                        if abs((candidate - witness_reference_point[axis]) / velocity[axis] - time)
                        < 1.0e-12
                    ),
                    None,
                )
                if matching is None:
                    break
                planes.append(matching)
            else:
                candidates.append((time, tuple(planes)))
        for time, planes in candidates:
            incident = any(
                all(
                    lower[axis] == round(plane * resolution[axis])
                    or upper[axis] == round(plane * resolution[axis])
                    for axis, plane in zip(active_axes, planes, strict=True)
                )
                for lower, upper in native_boxes
            )
            if incident and enclosed(time):
                return {"observed": True, "time": time, "planes": list(planes)}
        return {"observed": False, "time": None, "planes": []}

    face_crossing = crossing((speed_axes[0],)) if speed_axes else {"observed": False}
    edge_crossing = crossing((0, 1)) if mode == "xy" else {"observed": False}
    corner_crossing = (
        crossing((0, 1, 2)) if mode == "diagonal" and dimension == 3 else {"observed": False}
    )

    def periodic_crossing() -> dict[str, object]:
        if layout != "uniform" or not speed_axes or len(native_boxes) == 0:
            return {"observed": False, "time": None, "planes": []}
        times = tuple((1.0 - witness_reference_point[axis]) / velocity[axis] for axis in speed_axes)
        if any(not 0.0 < time < final_time or not enclosed(time) for time in times):
            return {"observed": False, "time": None, "planes": []}
        touches_boundary = all(
            any(lower[axis] == 0 for lower, _ in native_boxes)
            and any(upper[axis] == resolution[axis] for _, upper in native_boxes)
            for axis in speed_axes
        )
        return {
            "observed": touches_boundary,
            "time": list(times),
            "planes": [1.0 for _ in speed_axes],
        }

    boundary_crossing = periodic_crossing()
    regrid_counts = tuple(int(snapshot["regrid_count"]) for snapshot in timeline_snapshots)
    topology_epochs = tuple(int(snapshot["topology_epoch"]) for snapshot in timeline_snapshots)

    def frozen_patch_crossing() -> dict[str, object]:
        """Witness repeated characteristic entry/exit through native static fine boxes.

        The point follows the material characteristic from the fixed reference
        point, wrapped on the periodic domain.  Every transition is bracketed
        by two saved native topology snapshots.  We accept neither a merely
        requested number of cycles nor a predicted tag window: the hierarchy
        must publish unchanged level>0 boxes and the sampled material path
        must alternately enter and leave their actual union at least once per
        period.
        """
        if layout != "amr-frozen" or cycles < 3 or len(timeline_snapshots) != len(timeline_times):
            return {"observed": False, "transition_brackets": [], "native_fine_boxes": []}
        patch_rows = tuple(timeline_snapshots[0].get("patch_rows", ()))
        if not patch_rows or any(
            tuple(snapshot.get("patch_rows", ())) != patch_rows
            for snapshot in timeline_snapshots[1:]
        ):
            return {"observed": False, "transition_brackets": [], "native_fine_boxes": []}
        fine_boxes = tuple(row for row in patch_rows if int(row[0]) > 0)
        if not fine_boxes or not all(
            left < right
            for left, right in zip(timeline_times[:-1], timeline_times[1:], strict=True)
        ):
            return {"observed": False, "transition_brackets": [], "native_fine_boxes": []}

        def in_native_fine_box(time: float) -> bool:
            point = tuple(
                (witness_reference_point[axis] + velocity[axis] * time) % 1.0
                for axis in range(dimension)
            )
            for level, lower, upper in fine_boxes:
                cells = tuple(
                    resolution[axis] * refinement_ratio ** int(level) for axis in range(dimension)
                )
                if all(
                    lower[axis] / cells[axis] <= point[axis] < (upper[axis] + 1) / cells[axis]
                    for axis in range(dimension)
                ):
                    return True
            return False

        inside = tuple(in_native_fine_box(time) for time in timeline_times)
        transitions = [
            {
                "left_time": left_time,
                "right_time": right_time,
                "from_inside": left_inside,
                "to_inside": right_inside,
            }
            for left_time, right_time, left_inside, right_inside in zip(
                timeline_times[:-1],
                timeline_times[1:],
                inside[:-1],
                inside[1:],
                strict=True,
            )
            if left_inside != right_inside
        ]
        entries = sum(not row["from_inside"] and row["to_inside"] for row in transitions)
        exits = sum(row["from_inside"] and not row["to_inside"] for row in transitions)
        observed = entries >= cycles and exits >= cycles and len(transitions) >= 2 * cycles
        return {
            "observed": observed,
            "entries": entries,
            "exits": exits,
            "transition_brackets": transitions,
            "native_fine_boxes": [
                {"level": int(level), "lower": list(lower), "upper": list(upper)}
                for level, lower, upper in fine_boxes
            ],
            "diagnostic": "static_native_fine_boxes_plus_periodic_material_characteristic",
        }

    frozen_crossing = frozen_patch_crossing()

    def prescribed_mobile_regrid() -> dict[str, object]:
        """Authenticate native fine boxes against the prescribed periodic trajectory.

        A regrid counter alone only proves that *some* topology changed.  At
        the initial snapshot and at each snapshot publishing a new regrid or
        topology epoch, the actual level>0 boxes must contain the window centre
        at ``(center + velocity * time) mod 1`` on every active periodic axis.
        This rejects a fixed window or motion along a wrong axis even when the
        regrid counter continues to increase.
        """
        if layout != "amr-mobile":
            return {"observed": False, "snapshots": []}
        if len(timeline_snapshots) != len(timeline_times) or not timeline_snapshots:
            return {"observed": False, "snapshots": []}
        if not all(
            left < right
            for left, right in zip(timeline_times[:-1], timeline_times[1:], strict=True)
        ):
            return {"observed": False, "snapshots": []}
        if any(
            later < earlier
            for earlier, later in zip(regrid_counts[:-1], regrid_counts[1:], strict=True)
        ) or any(
            later < earlier
            for earlier, later in zip(topology_epochs[:-1], topology_epochs[1:], strict=True)
        ):
            return {"observed": False, "snapshots": []}

        def expected_center(time: float) -> tuple[float, ...]:
            return tuple(
                (patch_center[axis] + patch_velocity[axis] * time) % 1.0
                for axis in range(dimension)
            )

        def contains(
            row: tuple[int, tuple[int, ...], tuple[int, ...]], center: tuple[float, ...]
        ) -> bool:
            level, lower, upper = row
            if int(level) <= 0 or len(lower) != dimension or len(upper) != dimension:
                return False
            level_cells = tuple(
                resolution[axis] * refinement_ratio ** int(level) for axis in range(dimension)
            )
            return all(
                0 <= lower[axis] <= upper[axis] < level_cells[axis]
                and lower[axis] / level_cells[axis]
                <= center[axis]
                <= (upper[axis] + 1) / level_cells[axis]
                for axis in range(dimension)
            )

        def volume(row: tuple[int, tuple[int, ...], tuple[int, ...]]) -> int:
            _, lower, upper = row
            return int(
                np.prod(tuple(high - low + 1 for low, high in zip(lower, upper, strict=True)))
            )

        def window_probes(center: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
            """Sample the centre and both directions of every periodic window axis.

            The half-width probes remain inside the prescribed window rather
            than exactly on its cell-discretized edge.  They therefore prove
            both position and nonzero extent without claiming an impossible
            exact match to the clustering boxes.
            """
            probes = [center]
            for axis in range(dimension):
                for sign in (-1.0, 1.0):
                    point = list(center)
                    point[axis] = (point[axis] + sign * 0.5 * patch_half_width[axis]) % 1.0
                    probes.append(tuple(point))
            return tuple(probes)

        pertinent = [0]
        pertinent.extend(
            index
            for index in range(1, len(timeline_snapshots))
            if regrid_counts[index] > regrid_counts[index - 1]
            or topology_epochs[index] > topology_epochs[index - 1]
        )
        observed_snapshots: list[dict[str, object]] = []
        for index in pertinent:
            center = expected_center(timeline_times[index])
            fine_boxes = tuple(
                row
                for row in timeline_snapshots[index].get("patch_rows", ())
                if int(row[0]) > 0
            )
            candidates = tuple(row for row in fine_boxes if contains(row, center))
            uncovered = [
                point for point in window_probes(center) if not any(contains(row, point) for row in fine_boxes)
            ]
            if not candidates or uncovered:
                return {
                    "observed": False,
                    "snapshots": observed_snapshots,
                    "missing_snapshot": index,
                    "expected_center": list(center),
                    "uncovered_window_probes": [list(point) for point in uncovered],
                    "diagnostic": "native_fine_boxes_follow_prescribed_periodic_center",
                }
            level, lower, upper = min(candidates, key=volume)
            level_cells = tuple(
                resolution[axis] * refinement_ratio ** int(level) for axis in range(dimension)
            )
            observed_center = tuple(
                (lower[axis] + upper[axis] + 1) / (2.0 * level_cells[axis])
                for axis in range(dimension)
            )
            observed_snapshots.append(
                {
                    "index": index,
                    "time": timeline_times[index],
                    "expected_center": list(center),
                    "window_probe_count": len(window_probes(center)),
                    "box": {"level": int(level), "lower": list(lower), "upper": list(upper)},
                    "box_center": list(observed_center),
                }
            )
        regridded = any(count > regrid_counts[0] for count in regrid_counts[1:]) and any(
            epoch > topology_epochs[0] for epoch in topology_epochs[1:]
        )
        return {
            "observed": bool(regridded and len(observed_snapshots) == len(pertinent)),
            "snapshots": observed_snapshots,
            "expected_trajectory": {
                "center": list(patch_center[:dimension]),
                "velocity": list(patch_velocity[:dimension]),
                "periodic_axes": list(range(dimension)),
                "formula": "(center + velocity * time) mod 1",
            },
            "diagnostic": "native_fine_boxes_follow_prescribed_periodic_center",
        }

    mobile_regrid = prescribed_mobile_regrid()
    qualification = metrics["qualification"]
    witnesses: dict[str, dict[str, object]] = {
        "block_face": {
            "applicable": layout == "uniform" and len(speed_axes) >= 1,
            **face_crossing,
            "diagnostic": "native_local_boxes_plus_timeline_enclosed_face_crossing",
        },
        "block_edge_3d": {
            "applicable": dimension == 3 and mode == "xy" and layout == "uniform",
            **edge_crossing,
            "diagnostic": "native_local_boxes_plus_timeline_enclosed_edge_intersection",
        },
        "block_corner_3d": {
            "applicable": dimension == 3 and mode == "diagonal" and layout == "uniform",
            **corner_crossing,
            "diagnostic": "native_local_boxes_plus_timeline_enclosed_corner_intersection",
        },
        "coarse_fine_interface": {
            "applicable": layout != "uniform",
            "observed": bool(qualification["coarse_fine_interface_seen"]),
            "diagnostic": "composite_leaf_interface_mask",
        },
        "periodic_boundary": {
            "applicable": bool(speed_axes),
            **boundary_crossing,
            "diagnostic": "native_local_boxes_plus_timeline_enclosed_periodic_boundary_crossing",
        },
        "prescribed_mobile_regrid": {
            "applicable": layout == "amr-mobile",
            **mobile_regrid,
        },
        "repeated_patch_crossing": {
            "applicable": layout == "amr-frozen" and cycles >= 3,
            **frozen_crossing,
        },
        "second_order_convergence": {
            "applicable": layout == "uniform" and mode == "x",
            "observed": False,
            "deferred": {
                "scope": "matrix_complete",
                "qualifier": "_convergence_receipt",
                "obligation": "second_order_convergence",
            },
            "diagnostic": "matrix_complete_postprocess_l1_order",
        },
    }
    for obligation in obligations:
        witness = witnesses[obligation]
        if not witness["applicable"]:
            raise RuntimeError("coverage obligation %s is inapplicable to this case" % obligation)
        if obligation == "second_order_convergence":
            # A single resolution has an error, not an observed order.  Its
            # deliberately negative receipt is consumed only once all three
            # compatible resolutions reach the matrix-level completion gate.
            continue
        if not witness["observed"]:
            raise RuntimeError("coverage obligation %s has no observed witness" % obligation)
    return witnesses


def _composite_vectors(
    snapshot: dict[str, object],
    *,
    cells: tuple[int, ...],
    layout: str,
    refinement_ratio: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten composite leaf values and their physical cell volumes."""
    numerical_parts: list[np.ndarray] = []
    exact_parts: list[np.ndarray] = []
    initial_parts: list[np.ndarray] = []
    volume_parts: list[np.ndarray] = []
    if layout == "uniform":
        numerical_parts.append(np.asarray(snapshot["numeric"])[np.asarray(snapshot["mask"])])
        exact_parts.append(np.asarray(snapshot["exact"])[np.asarray(snapshot["mask"])])
        initial_parts.append(np.asarray(snapshot["initial"])[np.asarray(snapshot["mask"])])
        volume_parts.append(np.full(numerical_parts[-1].shape, 1.0 / np.prod(cells)))
    else:
        for level in range(sum(key.startswith("numeric_level_") for key in snapshot)):
            mask = np.asarray(snapshot["mask_level_%d" % level])
            numerical_parts.append(np.asarray(snapshot["numeric_level_%d" % level])[mask])
            exact_parts.append(np.asarray(snapshot["exact_level_%d" % level])[mask])
            initial_parts.append(np.asarray(snapshot["initial_level_%d" % level])[mask])
            volume_parts.append(
                np.full(
                    numerical_parts[-1].shape,
                    1.0 / (np.prod(cells) * refinement_ratio ** (len(cells) * level)),
                )
            )
    return (
        np.concatenate(numerical_parts),
        np.concatenate(exact_parts),
        np.concatenate(initial_parts),
        np.concatenate(volume_parts),
    )


def _composite_reference(
    snapshot: dict[str, object], *, layout: str, name: str
) -> np.ndarray | None:
    """Flatten one optional analytic reference on the same composite leaf ordering."""
    parts: list[np.ndarray] = []
    if layout == "uniform":
        if name not in snapshot:
            return None
        parts.append(np.asarray(snapshot[name])[np.asarray(snapshot["mask"])])
    else:
        level_count = sum(key.startswith("numeric_level_") for key in snapshot)
        for level in range(level_count):
            key = "%s_level_%d" % (name, level)
            if key not in snapshot:
                return None
            parts.append(np.asarray(snapshot[key])[np.asarray(snapshot["mask_level_%d" % level])])
    return np.concatenate(parts)


def _signed_phase_error_cycles(
    snapshot: dict[str, object],
    numerical: np.ndarray,
    exact: np.ndarray,
    volumes: np.ndarray,
    *,
    layout: str,
) -> float | None:
    """Return signed phase lag from weighted sine/cosine projections, if identifiable."""
    quadrature = _composite_reference(snapshot, layout=layout, name="quadrature")
    if quadrature is None:
        return None
    weight_sum = float(np.sum(volumes))

    def centered(values: np.ndarray) -> np.ndarray:
        return values - float(np.sum(volumes * values) / weight_sum)

    numerical_mode = centered(numerical)
    exact_mode = centered(exact)
    quadrature_mode = centered(quadrature)
    if float(np.sum(volumes * numerical_mode * numerical_mode)) == 0.0:
        return None
    basis = (exact_mode, quadrature_mode)
    gram = np.asarray(
        [[float(np.sum(volumes * left * right)) for right in basis] for left in basis]
    )
    projection = np.asarray(
        [float(np.sum(volumes * numerical_mode * reference)) for reference in basis]
    )
    try:
        sine_coefficient, cosine_coefficient = np.linalg.solve(gram, projection)
    except np.linalg.LinAlgError:
        return None
    phase = np.arctan2(-cosine_coefficient, sine_coefficient) / (2.0 * np.pi)
    return float(phase) if np.isfinite(phase) else None


def _topology_summary(snapshot: dict[str, object], *, layout: str) -> dict[str, object]:
    if layout == "uniform":
        return {
            "levels": 1,
            "patches": 0,
            "active_cells_by_level": [int(np.asarray(snapshot["mask"]).sum())],
            "has_coarse_fine_interface": False,
        }
    level_count = sum(key.startswith("mask_level_") for key in snapshot)
    active = [
        int(np.asarray(snapshot["mask_level_%d" % level]).sum()) for level in range(level_count)
    ]
    return {
        "levels": level_count,
        "patches": len(snapshot["patch_rows"]),
        "active_cells_by_level": active,
        "has_coarse_fine_interface": bool(
            len(active) > 1 and active[0] > 0 and sum(active[1:]) > 0
        ),
    }


def compute_metrics(
    initial_snapshot: dict[str, object],
    snapshot: dict[str, object],
    probe_snapshot: dict[str, object],
    *,
    cells: tuple[int, ...],
    layout: str,
    initial_mass: float,
    probe_mass: float,
    final_mass: float,
    final_time: float,
    probe_phase_cycles: float,
    probe_time: float,
    base_cfl: float,
    refinement_ratio: int,
    history_times: tuple[float, ...] | None = None,
    history_snapshots: tuple[dict[str, object], ...] | None = None,
    history_masses: tuple[float, ...] | None = None,
) -> dict[str, object]:
    initial_numerical, initial_exact, _, initial_volumes = _composite_vectors(
        initial_snapshot,
        cells=cells,
        layout=layout,
        refinement_ratio=refinement_ratio,
    )
    numerical, exact, _, volumes = _composite_vectors(
        snapshot,
        cells=cells,
        layout=layout,
        refinement_ratio=refinement_ratio,
    )
    probe_numerical, probe_exact, probe_initial, probe_volumes = _composite_vectors(
        probe_snapshot,
        cells=cells,
        layout=layout,
        refinement_ratio=refinement_ratio,
    )

    def validate_composite(
        numerical_values: np.ndarray,
        volumes: np.ndarray,
        runtime_mass: float,
        *,
        where: str,
    ) -> None:
        represented_volume = float(np.sum(volumes, dtype=np.float64))
        if not np.isclose(represented_volume, 1.0, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(
                "%s composite leaf mask represents volume %.17g instead of 1"
                % (
                    where,
                    represented_volume,
                )
            )
        reconstructed_mass = float(np.sum(volumes * numerical_values, dtype=np.float64))
        tolerance = 5.0e-11 * max(1.0, abs(runtime_mass))
        if abs(reconstructed_mass - runtime_mass) > tolerance:
            raise RuntimeError(
                "%s composite mass %.17g disagrees with native integral %.17g"
                % (
                    where,
                    reconstructed_mass,
                    runtime_mass,
                )
            )

    validate_composite(initial_numerical, initial_volumes, initial_mass, where="initial")
    validate_composite(numerical, volumes, final_mass, where="final")
    validate_composite(probe_numerical, probe_volumes, probe_mass, where="probe")
    initial_diagnostics = sine_diagnostics(initial_numerical, initial_exact, initial_volumes)
    diagnostics = sine_diagnostics(numerical, exact, volumes)
    probe_diagnostics = sine_diagnostics(probe_numerical, probe_exact, probe_volumes)
    probe_against_initial = sine_diagnostics(probe_numerical, probe_initial, probe_volumes)
    if not np.isfinite(probe_phase_cycles):
        raise ValueError("probe_phase_cycles must be finite")
    phase_cosine = float(np.cos(2.0 * np.pi * probe_phase_cycles))
    # sin(theta + phi) = 2 cos(phi) sin(theta) - sin(theta - phi).
    # The analytic phase keeps this exact on asymmetric composite AMR masks too.
    reverse_reference = 1.0 + 2.0 * phase_cosine * (probe_initial - 1.0) - (probe_exact - 1.0)
    probe_against_reverse = sine_diagnostics(probe_numerical, reverse_reference, probe_volumes)
    mass_scale = max(abs(initial_mass), np.finfo(float).tiny)
    probe_drift = abs(probe_mass - initial_mass) / mass_scale
    final_drift = abs(final_mass - initial_mass) / mass_scale
    initial_topology = _topology_summary(initial_snapshot, layout=layout)
    topology = _topology_summary(snapshot, layout=layout)
    probe_topology = _topology_summary(probe_snapshot, layout=layout)
    interface_seen = bool(
        initial_topology["has_coarse_fine_interface"]
        or topology["has_coarse_fine_interface"]
        or probe_topology["has_coarse_fine_interface"]
    )
    topology_passed = layout == "uniform" or interface_seen
    probe_l2 = probe_diagnostics["errors"]["l2"]
    initial_l2 = probe_against_initial["errors"]["l2"]
    reverse_l2 = probe_against_reverse["errors"]["l2"]
    comparison_tolerance = 64.0 * np.finfo(float).eps
    transport_passed = bool(
        probe_l2 + comparison_tolerance < initial_l2
        and probe_l2 + comparison_tolerance < reverse_l2
    )
    if history_times is None or history_snapshots is None or history_masses is None:
        history_times = (0.0, probe_time, final_time)
        history_snapshots = (initial_snapshot, probe_snapshot, snapshot)
        history_masses = (initial_mass, probe_mass, final_mass)
    if not (
        len(history_times) == len(history_snapshots) == len(history_masses)
        and len(history_times) >= 3
    ):
        raise ValueError("time-history times, snapshots, and masses must have one common length")
    history_time_array = np.asarray(history_times, dtype=np.float64)
    if (
        not np.isfinite(history_time_array).all()
        or history_time_array[0] != 0.0
        or not np.isclose(history_time_array[-1], final_time, rtol=0.0, atol=1.0e-14)
        or np.any(np.diff(history_time_array) <= 0.0)
    ):
        raise ValueError("time-history coordinates must increase strictly from zero to final_time")

    time_history = {
        "time": [],
        "mass": [],
        "mass_relative_drift": [],
        "amplitude_rms": [],
        "exact_amplitude_rms": [],
        "amplitude_retention": [],
        "phase_cosine": [],
        "phase_error_cycles": [],
        "l1": [],
        "l2": [],
        "linf": [],
    }
    for history_time, history_snapshot, history_mass in zip(
        history_times, history_snapshots, history_masses, strict=True
    ):
        history_numerical, history_exact, _, history_volumes = _composite_vectors(
            history_snapshot,
            cells=cells,
            layout=layout,
            refinement_ratio=refinement_ratio,
        )
        validate_composite(
            history_numerical,
            history_volumes,
            history_mass,
            where="history t=%.17g" % history_time,
        )
        history_diagnostics = sine_diagnostics(history_numerical, history_exact, history_volumes)
        history_errors = weighted_error_norms(
            history_numerical, history_exact, history_volumes
        ).to_dict()
        time_history["time"].append(float(history_time))
        time_history["mass"].append(float(history_mass))
        time_history["mass_relative_drift"].append(
            float((history_mass - initial_mass) / mass_scale)
        )
        time_history["amplitude_rms"].append(float(history_diagnostics["amplitude_rms"]))
        time_history["exact_amplitude_rms"].append(
            float(history_diagnostics["exact_amplitude_rms"])
        )
        history_exact_amplitude = float(history_diagnostics["exact_amplitude_rms"])
        time_history["amplitude_retention"].append(
            float(history_diagnostics["amplitude_rms"] / history_exact_amplitude)
            if history_exact_amplitude > 0.0
            else None
        )
        phase_cosine_value = history_diagnostics["phase_cosine"]
        time_history["phase_cosine"].append(
            None if phase_cosine_value is None else float(phase_cosine_value)
        )
        time_history["phase_error_cycles"].append(
            _signed_phase_error_cycles(
                history_snapshot,
                history_numerical,
                history_exact,
                history_volumes,
                layout=layout,
            )
        )
        for name in ("l1", "l2", "linf"):
            time_history[name].append(float(history_errors[name]))

    amplitude_retentions = [
        value for value in time_history["amplitude_retention"] if value is not None
    ]
    final_exact_amplitude = float(diagnostics["exact_amplitude_rms"])
    final_amplitude_retention = (
        float(diagnostics["amplitude_rms"] / final_exact_amplitude)
        if final_exact_amplitude > 0.0
        else None
    )
    history_max_drift = max(abs(float(value)) for value in time_history["mass_relative_drift"])
    return {
        "case": "periodic_sine_wave_advection",
        "method": {
            "time": "SSPRK2",
            "reconstruction": "MUSCL(VanLeer)",
            "riemann": "ScalarUpwind",
            "cfl": base_cfl / len(cells),
            "cfl_base": base_cfl,
            "cfl_effective": base_cfl / len(cells),
            "cfl_formula": "base_cfl / dimension",
        },
        "errors": weighted_error_norms(numerical, exact, volumes).to_dict(),
        "probe_errors": weighted_error_norms(probe_numerical, probe_exact, probe_volumes).to_dict(),
        # Accuracy is deliberately metric-only: no arbitrary pass threshold turns a very
        # diffusive but correctly routed run into a scientifically accurate result.
        "accuracy": {
            "final_errors": weighted_error_norms(numerical, exact, volumes).to_dict(),
            "probe_errors": weighted_error_norms(
                probe_numerical, probe_exact, probe_volumes
            ).to_dict(),
            "final_amplitude_retention": final_amplitude_retention,
            "minimum_amplitude_retention": (
                float(min(amplitude_retentions)) if amplitude_retentions else None
            ),
        },
        "diagnostics": diagnostics,
        "initial_diagnostics": initial_diagnostics,
        "probe_diagnostics": probe_diagnostics,
        "probe_against_initial": probe_against_initial,
        "probe_against_reverse": probe_against_reverse,
        "initial_topology": initial_topology,
        "topology": topology,
        "probe_topology": probe_topology,
        "qualification": {
            "kind": "run_integrity",
            "passed": topology_passed and transport_passed,
            "run_integrity_passed": topology_passed and transport_passed,
            "accuracy_assessed": False,
            "topology_passed": topology_passed,
            "transport_probe_passed": transport_passed,
            "coarse_fine_interface_seen": interface_seen,
            "reason": (
                "AMR materialized as all-coarse or all-fine; no interface was exercised"
                if not topology_passed
                else (
                    "the non-periodic probe is not more phase-aligned with the transported "
                    "oracle than with the initial wave"
                    if not transport_passed
                    else ("run-integrity checks passed; numerical accuracy is reported separately")
                )
            ),
        },
        "conservation": {
            "analytic_mass": 1.0,
            "initial_mass": initial_mass,
            "probe_mass": probe_mass,
            "final_mass": final_mass,
            "initial_analytic_error": initial_mass - 1.0,
            "probe_relative_drift": probe_drift,
            "final_relative_drift": final_drift,
            "max_relative_drift": history_max_drift,
            "timeline_samples": len(time_history["time"]),
        },
        "time_history": time_history,
        "note": (
            "La qualification vérifie uniquement l'intégrité du routage/topologie, jamais "
            "la précision. Les métriques d'accuracy et l'historique temporel quantifient "
            "séparément diffusion et erreur; aucun pas interne n'est piloté en Python."
        ),
    }


def _storage_payload(snapshot: dict[str, object], *, prefix: str = "") -> dict[str, np.ndarray]:
    """Convert one public runtime snapshot to flat, pickle-free NPZ arrays."""
    payload = {
        prefix + key: np.asarray(value)
        for key, value in snapshot.items()
        if key not in {"patch_rows", "coordinates", "local_boxes"}
        and not key.startswith("coordinates_level_")
    }
    for coordinate_name, coordinate in zip(
        ("x", "y", "z"), snapshot.get("coordinates", ()), strict=False
    ):
        payload[prefix + coordinate_name] = np.asarray(coordinate)
    level_count = sum(key.startswith("coordinates_level_") for key in snapshot)
    for level in range(level_count):
        for coordinate_name, coordinate in zip(
            ("x", "y", "z"),
            snapshot["coordinates_level_%d" % level],
            strict=False,
        ):
            payload["%s%s_level_%d" % (prefix, coordinate_name, level)] = np.asarray(coordinate)
    payload[prefix + "patch_boxes"] = np.asarray(
        [(*lower, *upper, level) for level, lower, upper in snapshot["patch_rows"]],
        dtype=np.int64,
    )
    payload[prefix + "local_boxes"] = np.asarray(
        [(*lower, *upper) for lower, upper in snapshot.get("local_boxes", ())], dtype=np.int64
    )
    return payload


def _sha256(path: Path) -> str:
    """Return the content digest recorded by the mandatory JSON sidecar."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    """Hash one JSON-compatible value with deterministic, fail-closed encoding."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_regular_file(path: Path, *, where: str) -> Path:
    """Resolve one provenance authority without following a leaf symlink."""
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("%s must be a regular non-symlink file: %s" % (where, path))
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError("cannot resolve %s: %s" % (where, path)) from error


def _build_source_tree(repository_root: Path) -> dict[str, object]:
    """Digest every build-relevant leaf, including relevant untracked files.

    Git's tracked diff cannot authenticate an untracked CMake or generated source
    leaf which affects a later native build. This deterministic inventory covers
    the build roots, skips output/cache trees and every symlink, and hashes regular
    leaves directly instead of trusting the index.
    """
    root = repository_root.resolve()
    files: dict[str, str] = {}
    for relative_root in BUILD_SOURCE_ROOTS:
        candidate = root / relative_root
        if candidate.is_symlink() or not candidate.exists():
            raise RuntimeError("build source root is missing or symlinked: %s" % candidate)
        if candidate.is_file():
            files[relative_root] = _sha256(candidate)
            continue
        if not candidate.is_dir():
            raise RuntimeError("build source root is not a regular file or directory: %s" % candidate)
        for directory, directory_names, filenames in os.walk(candidate, followlinks=False):
            directory_path = Path(directory)
            directory_names[:] = sorted(
                name
                for name in directory_names
                if name not in BUILD_SOURCE_EXCLUDED_NAMES
                and not (directory_path / name).is_symlink()
            )
            for filename in sorted(filenames):
                path = directory_path / filename
                if (
                    filename in BUILD_SOURCE_EXCLUDED_NAMES
                    or path.is_symlink()
                    or not path.is_file()
                ):
                    continue
                relative = path.relative_to(root).as_posix()
                if relative in files:
                    raise RuntimeError("duplicate build source authority: %s" % relative)
                files[relative] = _sha256(path)
    if not files:
        raise RuntimeError("build source authority cannot be empty")
    receipt: dict[str, object] = {
        "schema_version": "pops.sine-wave.build-source-tree.v1",
        "roots": list(BUILD_SOURCE_ROOTS),
        "files": files,
    }
    receipt["fingerprint"] = _canonical_sha256(receipt)
    return receipt


def native_receipt(module: object) -> dict[str, object]:
    """Authenticate the exact selected extension against its local manifest.

    The receipt deliberately keeps only the canonical path within ``pops/_native``.
    Absolute installation paths are validation-only and never become part of a result
    identity, so the same wheel can be compared across hosts and staging locations.
    """
    origin = getattr(module, "__file__", None)
    if type(origin) is not str or not origin:
        raise RuntimeError("selected PoPS native module has no file origin")
    module_path = _require_regular_file(Path(origin), where="native extension")
    if module_path.parent.name not in {"dim1", "dim2", "dim3"}:
        raise RuntimeError("native extension is not stored in a canonical dimN directory")
    native_root = module_path.parent.parent
    if native_root.name != "_native":
        raise RuntimeError("native extension is not stored below pops/_native")
    manifest_path = _require_regular_file(
        native_root / "variants.json", where="native variants manifest"
    )
    try:
        relative_path = module_path.relative_to(native_root).as_posix()
    except ValueError as error:
        raise RuntimeError("native extension escapes its package native root") from error

    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot read native variants manifest") from error
    if (
        type(document) is not dict
        or set(document) != {"schema_version", "variants"}
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 2
        or type(document["variants"]) is not list
    ):
        raise RuntimeError("unsupported native variants manifest schema")

    expected_keys = {
        "dimension",
        "path",
        "sha256",
        "version",
        "abi_key",
        "build_fingerprint",
        "has_mpi",
        "has_kokkos",
    }
    matching_rows: list[dict[str, object]] = []
    for raw_row in document["variants"]:
        if type(raw_row) is not dict or set(raw_row) != expected_keys:
            raise RuntimeError("native variant row has an invalid schema")
        if type(raw_row["dimension"]) is not int or raw_row["dimension"] not in (1, 2, 3):
            raise RuntimeError("native variant row has an invalid dimension")
        path_text = raw_row["path"]
        if type(path_text) is not str or not path_text:
            raise RuntimeError("native variant path must be canonical relative text")
        relative = PurePosixPath(path_text)
        expected_leafs = {"_pops" + suffix for suffix in importlib.machinery.EXTENSION_SUFFIXES}
        if (
            relative.is_absolute()
            or str(relative) != path_text
            or any(part in ("", ".", "..") for part in relative.parts)
            or len(relative.parts) != 2
            or relative.parts[0] != "dim%d" % raw_row["dimension"]
            or relative.parts[1] not in expected_leafs
        ):
            raise RuntimeError("native variant path is not canonical dimN/_pops<suffix>")
        sha256 = raw_row["sha256"]
        if (
            type(sha256) is not str
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise RuntimeError("native variant sha256 must be lowercase hexadecimal")
        build_fingerprint = raw_row["build_fingerprint"]
        if (
            type(build_fingerprint) is not str
            or len(build_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in build_fingerprint)
        ):
            raise RuntimeError("native variant build_fingerprint must be lowercase hexadecimal")
        if (
            type(raw_row["version"]) is not str
            or not raw_row["version"]
            or type(raw_row["abi_key"]) is not str
            or not raw_row["abi_key"]
            or type(raw_row["has_mpi"]) is not bool
            or type(raw_row["has_kokkos"]) is not bool
        ):
            raise RuntimeError("native variant row has invalid runtime facts")
        if path_text == relative_path:
            matching_rows.append(raw_row)
    if len({row["build_fingerprint"] for row in document["variants"]}) != 1:
        raise RuntimeError("native variant rows disagree on their common build fingerprint")
    if len(matching_rows) != 1:
        raise RuntimeError("native variants manifest does not identify the selected extension")
    row = matching_rows[0]
    if _sha256(module_path) != row["sha256"]:
        raise RuntimeError("native extension bytes differ from variants.json")
    if getattr(module, "__native_dimension__", None) != row["dimension"]:
        raise RuntimeError("selected native module dimension differs from variants.json")
    if getattr(module, "__version__", None) != row["version"]:
        raise RuntimeError("selected native module version differs from variants.json")
    abi_key = getattr(module, "abi_key", None)
    if not callable(abi_key) or abi_key() != row["abi_key"]:
        raise RuntimeError("selected native module ABI differs from variants.json")
    if getattr(module, "__build_fingerprint__", None) != row["build_fingerprint"]:
        raise RuntimeError("selected native module build fingerprint differs from variants.json")
    if (
        getattr(module, "__has_mpi__", None) is not row["has_mpi"]
        or getattr(module, "__has_kokkos__", None) is not row["has_kokkos"]
    ):
        raise RuntimeError("selected native module backend facts differ from variants.json")
    return {
        "manifest_sha256": _sha256(manifest_path),
        "path": relative_path,
        "sha256": row["sha256"],
        "dimension": row["dimension"],
        "version": row["version"],
        "abi_key": row["abi_key"],
        "build_fingerprint": row["build_fingerprint"],
        "has_mpi": row["has_mpi"],
        "has_kokkos": row["has_kokkos"],
    }


def source_provenance(
    *,
    repository_root: Path,
    generator_path: Path,
    support_path: Path,
    source_schema_version: str,
    native_module: object | None = None,
) -> dict[str, object]:
    """Fingerprint source, build tree, imported benchmark authority, and native leaf."""
    root = repository_root.resolve()
    generator = _require_regular_file(generator_path, where="benchmark generator")
    support = _require_regular_file(support_path, where="benchmark support")
    source_paths = (
        generator,
        support,
        root / "helpers" / "__init__.py",
        root / "helpers" / "verification" / "__init__.py",
        root / "helpers" / "verification" / "sine_wave.py",
    )
    files: dict[str, str] = {}
    for source_path in source_paths:
        authority = _require_regular_file(source_path, where="imported benchmark authority")
        try:
            relative = authority.relative_to(root).as_posix()
        except ValueError as error:
            raise RuntimeError("imported benchmark authority escapes repository root") from error
        if relative in files:
            raise RuntimeError("duplicate imported benchmark provenance authority: %s" % relative)
        files[relative] = _sha256(authority)
    if native_module is None:
        from pops._native_selector import selected_native_module

        native_module = selected_native_module(required=True)
    repository_status = _git_value(root, "status", "--porcelain=v1", "--untracked-files=all")
    source: dict[str, object] = {
        "schema_version": source_schema_version,
        "repository_sha": _git_value(root, "rev-parse", "HEAD"),
        "repository_dirty": bool(repository_status),
        "tracked_diff_sha256": _git_content_sha256(root, "diff", "--binary", "HEAD", "--"),
        "files": files,
        "build_tree": _build_source_tree(root),
        "native": native_receipt(native_module),
    }
    source["fingerprint"] = _canonical_sha256(source)
    return source


def execution_provenance(runtime_report: dict[str, object]) -> dict[str, object]:
    """Keep backend/scheduler facts needed to interpret CPU, GPU, and MPI campaigns."""
    runtime_keys = (
        "has_kokkos",
        "kokkos_backend",
        "kokkos_device",
        "kokkos_shared_space",
        "field_memory_space",
        "kokkos_concurrency",
        "mpi_compiled",
        "mpi_active",
        "mpi_ranks",
        "communicator",
    )
    environment_keys = (
        "OMP_NUM_THREADS",
        "OMP_PROC_BIND",
        "OMP_PLACES",
        "KOKKOS_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NTASKS",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS",
        "SLURM_JOB_GPUS",
    )
    return {
        "runtime": {key: runtime_report.get(key) for key in runtime_keys},
        "environment": {key: os.environ.get(key) for key in environment_keys},
        "host": {
            "node": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }


def publish_result(
    *,
    output: Path,
    schema_version: str,
    pops_version: str,
    configuration: dict[str, object],
    metrics: dict[str, object],
    execution: dict[str, object],
    source: dict[str, object],
    artifact: dict[str, str],
    initial_snapshot: dict[str, object],
    probe_snapshot: dict[str, object],
    final_snapshot: dict[str, object],
    timeline_times: tuple[float, ...],
    timeline_snapshots: tuple[dict[str, object], ...],
    segmented_native_runs: int,
) -> tuple[Path, Path]:
    """Authenticate and atomically publish one complete rank-zero result pair."""
    timeline = {
        "frames": len(timeline_times),
        "times": timeline_times,
        "storage_prefix": "timeline_{index:04d}_",
        "segmented_native_runs": segmented_native_runs,
    }
    amr_diagnostics = {
        "interface_mask": {
            "definition": ("active composite leaf sharing a face with an inactive same-level cell"),
            "connectivity": "face",
            "periodic": True,
            "storage": "{snapshot_prefix}interface_mask[_level_L]",
        },
        "regrid_events": {
            "source": "simulation.amr.explain_regrid()",
            "storage": "{snapshot_prefix}regrid_count and topology_epoch",
            "time_semantics": (
                "a counter increase occurred after the previous snapshot and no later "
                "than the labelled snapshot; the internal event time is not claimed"
            ),
        },
    }
    complete_configuration = {
        **configuration,
        "timeline": timeline,
        "amr_diagnostics": amr_diagnostics,
    }
    result_identity_inputs = {
        "schema_version": schema_version,
        "configuration": complete_configuration,
        "method": metrics["method"],
        "execution": execution,
        "source_fingerprint": source["fingerprint"],
        "artifact": artifact,
    }
    result_identity = _canonical_sha256(result_identity_inputs)

    cycles = int(configuration["cycles"])
    cycle_tag = "" if cycles == 1 else "_cycles%d" % cycles
    resolution = tuple(int(value) for value in configuration["resolution"])
    stem = "sine_dim%d_n%s%s_%s_%s_%s_block%d_mpi%d_np%d_ts%d_rid%s" % (
        int(configuration["dimension"]),
        "x".join(map(str, resolution)),
        cycle_tag,
        configuration["mode"],
        configuration["layout"],
        configuration["subcycling"],
        int(configuration["block_size"]),
        int(bool(configuration["mpi"])),
        int(configuration["mpi_ranks"]),
        len(timeline_times),
        result_identity[:16],
    )
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / (stem + ".npz")
    metadata_path = output / (stem + ".json")
    if data_path.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite an existing authenticated result: %s" % stem)

    payload = _storage_payload(final_snapshot)
    payload["schema_version"] = np.asarray(schema_version)
    payload["result_identity"] = np.asarray(result_identity)
    payload.update(_storage_payload(probe_snapshot, prefix="probe_"))
    payload.update(_storage_payload(initial_snapshot, prefix="runtime_initial_"))
    payload["timeline_time"] = np.asarray(timeline_times, dtype=np.float64)
    for index, timeline_snapshot in enumerate(timeline_snapshots):
        payload.update(_storage_payload(timeline_snapshot, prefix="timeline_%04d_" % index))

    temporary_suffix = ".tmp.%d" % os.getpid()
    temporary_data_path = data_path.with_name(".%s%s" % (data_path.name, temporary_suffix))
    temporary_metadata_path = metadata_path.with_name(
        ".%s%s" % (metadata_path.name, temporary_suffix)
    )
    with temporary_data_path.open("xb") as stream:
        np.savez_compressed(stream, **payload)
    data_sha256 = _sha256(temporary_data_path)

    metadata = {
        "schema_version": schema_version,
        "result_identity": result_identity,
        "result_identity_inputs": result_identity_inputs,
        "source_fingerprint": source["fingerprint"],
        "case": configuration["case"],
        "dimension": configuration["dimension"],
        "resolution": configuration["resolution"],
        "mode": configuration["mode"],
        "wave_numbers": configuration["wave_numbers"],
        "velocity": configuration["velocity"],
        "epsilon": configuration["epsilon"],
        "probe_time": configuration["probe_time"],
        "period": configuration["period"],
        "cycles": configuration["cycles"],
        "final_time": configuration["final_time"],
        "layout": configuration["layout"],
        "subcycling": configuration["subcycling"],
        "block_size": configuration["block_size"],
        "amr_block_size": (
            None if configuration["layout"] == "uniform" else configuration["block_size"]
        ),
        "patch_marker": configuration["patch_marker"],
        "coverage": configuration.get("coverage"),
        "mpi": configuration["mpi"],
        "mpi_ranks": configuration["mpi_ranks"],
        "mpi_topology": configuration.get("mpi_topology"),
        "time_snapshots": len(timeline_times),
        "data": data_path.name,
        "data_sha256": data_sha256,
        "timeline": timeline,
        "amr_diagnostics": amr_diagnostics,
        "metrics": metrics,
        "provenance": {
            "repository_sha": source["repository_sha"],
            "repository_dirty": source["repository_dirty"],
            "pops_version": pops_version,
            "date_utc": datetime.now(UTC).isoformat(),
            "execution": execution,
            "source": source,
            "artifact": artifact,
            "campaign": {
                "mpi_ranks": configuration["mpi_ranks"],
                "time_snapshots": len(timeline_times),
                "timeline_times": timeline_times,
            },
        },
    }

    published_data = False
    try:
        with temporary_metadata_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        # Hard-link publication is atomic and refuses an existing target, unlike replace().
        os.link(temporary_data_path, data_path)
        published_data = True
        os.link(temporary_metadata_path, metadata_path)
    except Exception:
        if published_data:
            data_path.unlink(missing_ok=True)
        raise
    finally:
        temporary_data_path.unlink(missing_ok=True)
        temporary_metadata_path.unlink(missing_ok=True)
    return data_path, metadata_path
