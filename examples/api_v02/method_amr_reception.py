"""Read-only AMR reception oracles; never used by generated numerical kernels.

The supported geometry is a periodic Cartesian unit square with refinement 2.
Dense level arrays use (component, y, x), and patch upper bounds are inclusive.
Malformed, incomplete or nonconservative observations fail closed.
"""
from __future__ import annotations

import math
import numpy as np


def composite_base_projection(level_values, patch_boxes, *, cells):
    """Project valid, uncovered scalar cells conservatively onto base cells.

    All levels must be gathered collectively before this pure host operation.
    A non-owner returns None only if it owns none of the requested level arrays.
    The base grid is fully present, as required by the public Cartesian AMR
    runtime contract; finer grids are present only inside the reported patches.
    """
    if len(cells) != 2 or any(type(n) is not int or n < 1 for n in cells):
        raise ValueError("T08 projection requires two positive base-grid extents")
    levels = tuple(level_values)
    if not levels:
        raise ValueError("T08 projection requires at least one native level")
    absent = tuple(value is None for value in levels)
    if all(absent):
        return None
    if any(absent):
        raise ValueError("T08 projection requires one owner of every gathered level")
    nx, ny = cells
    valid = [np.ones((ny, nx), dtype=bool)] + [
        np.zeros((ny*2**level, nx*2**level), dtype=bool)
        for level in range(1, len(levels))]
    boxes = []
    listed_base = np.zeros((ny, nx), dtype=bool)
    for row in patch_boxes:
        if len(row) != 3:
            raise ValueError("T08 patch requires level/lower/upper")
        level, lower, upper = row
        if type(level) is not int or not 0 <= level < len(levels):
            raise ValueError("T08 patch level is outside gathered levels")
        if len(lower) != 2 or len(upper) != 2:
            raise ValueError("T08 patch bounds require exactly two axes")
        if any(type(index) is not int for index in (*lower, *upper)):
            raise ValueError("T08 patch indices must be exact integers")
        scale = 2**level
        if any(not 0 <= lo <= hi < count*scale
               for lo, hi, count in zip(lower, upper, cells, strict=True)):
            raise ValueError("T08 patch lies outside its level domain")
        if level and any(lo % 2 or (hi+1) % 2
                         for lo, hi in zip(lower, upper, strict=True)):
            raise ValueError("T08 child patch must align with whole parent cells")
        x0, y0 = lower
        x1, y1 = upper
        listed = valid[level] if level else listed_base
        window = (slice(y0, y1+1), slice(x0, x1+1))
        if listed[window].any():
            raise ValueError("T08 patch boxes of the same level overlap or repeat")
        listed[window] = True
        boxes.append((level, tuple(lower), tuple(upper)))
    if any(not mask.any() for mask in valid):
        raise ValueError("T08 gathered level has no valid patch")
    active = [mask.copy() for mask in valid]
    for level, (x0, y0), (x1, y1) in boxes:
        if not level:
            continue
        parent_slice = (slice(y0//2, y1//2+1), slice(x0//2, x1//2+1))
        if not valid[level-1][parent_slice].all():
            raise ValueError("T08 child patch refines cells without a parent patch")
        active[level-1][parent_slice] = False
    means = np.zeros((ny, nx), dtype=float)
    weights = np.zeros_like(means)
    active_counts = []
    for level, (raw, mask) in enumerate(zip(levels, active, strict=True)):
        scale = 2**level
        values = np.asarray(raw, dtype=float)
        if values.size != nx*ny*scale*scale:
            raise ValueError("T08 reception requires exactly one scalar component")
        values = values.reshape(ny*scale, nx*scale)
        if not np.isfinite(values[mask]).all():
            raise ValueError("T08 composite includes a nonfinite active state")
        iy, ix = np.nonzero(mask)
        fraction = 1.0/(scale*scale)
        np.add.at(means, (iy//scale, ix//scale), values[iy, ix]*fraction)
        np.add.at(weights, (iy//scale, ix//scale), fraction)
        active_counts.append(int(mask.sum()))
    # Dyadic fractions and these finite grid sizes make this sum exact. A hole
    # or duplicate volume must never be hidden by renormalizing the projection.
    if not np.array_equal(weights, np.ones_like(weights)):
        raise ValueError("T08 composite does not cover every base cell exactly once")
    return {"means": means, "weights": weights, "active_counts": tuple(active_counts)}


def initial_base_means(cells):
    nx, ny = cells
    x, y = np.meshgrid((np.arange(nx)+.5)/nx, (np.arange(ny)+.5)/ny)
    return 1-.1*np.cos(2*np.pi*x)-.1*np.cos(2*np.pi*y)


def translation_metrics(base_means, *, cells, time):
    """Compare native composite means to an independent exact translation.

    The initial coarse samples are exact cell averages of the sum of two
    cosines with continuous amplitudes -.1/sinc(1/nx) and -.1/sinc(1/ny).
    After unit x-advection, averaging on the same base cells therefore gives
    the expression below; the y cosine is stationary.
    Acceptance requires an improvement over doing nothing, exceeding a fixed
    floating-point roundoff allowance. No empirically fitted error threshold
    or post-execution calibration is used.
    """
    if not math.isfinite(time) or time <= 0:
        raise ValueError("T08 translation assessment requires positive finite time")
    nx, ny = cells
    actual = np.asarray(base_means, dtype=float)
    if actual.shape != (ny, nx) or not np.isfinite(actual).all():
        raise ValueError("T08 projected state has the wrong shape or nonfinite values")
    x, y = np.meshgrid((np.arange(nx)+.5)/nx, (np.arange(ny)+.5)/ny)
    initial = initial_base_means(cells)
    exact = 1-.1*np.cos(2*np.pi*(x-time))-.1*np.cos(2*np.pi*y)
    error = float(np.sqrt(np.mean((actual-exact)**2)))
    stationary_error = float(np.sqrt(np.mean((initial-exact)**2)))
    change = float(np.sqrt(np.mean((actual-initial)**2)))
    roundoff = 64*np.finfo(float).eps*max(1., float(np.max(abs(initial))))
    # If the chosen time makes the exact translation numerically stationary,
    # the witness has no resolving power and must fail closed.
    passed = bool(stationary_error > 4*roundoff and change > roundoff
                  and error+roundoff < stationary_error)
    return dict(passed=passed, composite_translation_error=error,
        stationary_translation_error=stationary_error,
        composite_state_change=change, comparison_roundoff=roundoff,
        translation_error_ratio=error/stationary_error if stationary_error else None)
