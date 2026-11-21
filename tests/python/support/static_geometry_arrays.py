"""The static geometry arrays of a layout, generated natively without compiling.

N17 L1b: the elaborated layout of a case (``pops.Graded`` gives the static case of the
geometry-evolution contract) is normalized, its native spatial layout is resolved, and the arrays
are generated and certified in a bare native System of the requested rank, as ``pops.bind`` does
before the first block. Nothing is compiled, and the case itself need not resolve: probe 3 does
not yet, its value relations under a diffusion waiting on G04.

N17 L2b-2: a layout written below the surface takes the same route with the owner of its case;
the bare System also evaluates the centred slope of the arrays (``centroid_slope``) from given
differences, through its runtime-private seam. The NumPy side of the slope is here too: the
neighbour centroids of every cell, read from the snapshot alone, and the rounding floor of a
solve of rank one to three.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def layout_static_system(case: Any = None, *, dimension: int, layout: Any = None,
                         owner: Any = None) -> Any:
    """The bare native System that holds the static geometry arrays of a layout: the elaborated
    layout of the surface case ``case``, or ``layout`` written below the surface for the case
    ``owner``."""
    from pops._native_selector import select_native_dimension
    from pops.codegen._layout_resolution import resolve_native_spatial_layouts
    from pops.mesh import normalize_layout_plan
    from pops.runtime._runtime_mesh_lowering import (
        install_static_geometry, system_config_from_layout)

    select_native_dimension(dimension)
    from pops.runtime._system import System

    if case is not None:
        elaborated = case.elaborated
        layout, owner = elaborated.layout, elaborated.case
    plan = normalize_layout_plan(layout, owner=owner.owner_path.canonical())
    resolve_native_spatial_layouts(plan)
    (resolved,) = plan.layouts
    engine = System(system_config_from_layout(resolved.native_spatial_layout))
    install_static_geometry(engine, resolved)
    return engine._s


def layout_static_arrays(case: Any = None, *, dimension: int, layout: Any = None,
                         owner: Any = None) -> dict:
    """The snapshot of the static geometry arrays of a layout (``layout_static_system``)."""
    return layout_static_system(case, dimension=dimension, layout=layout,
                                owner=owner)._static_geometry_snapshot()


def native(values: Any) -> np.ndarray:
    """An array of the snapshot (NumPy order, the last axis the first native axis) indexed by the
    native axes, [i0, i1, ...]."""
    return np.asarray(values).transpose()


def centroids(arrays: dict) -> np.ndarray:
    """The cell centroids, (Dim, n0, n1, ...)."""
    return np.stack([native(component) for component in arrays["centroid"]])


def neighbour_centroids(arrays: dict, axis: int, side: int) -> np.ndarray:
    """The centroid of the neighbour of every cell across its face of ``axis`` on ``side`` (+1 or
    -1), read from the snapshot alone: the stored centroid inside the domain; across the seam of a
    periodic axis the centroid of the cell at the other end, translated by the seam (+T across the
    upper seam, -T across the lower one); across a physical end the centroid of the cell reflected
    through the centroid of the end face, 2 m_f - c_K."""
    centre = centroids(arrays)
    rank = centre.shape[0]
    cells = centre.shape[1 + axis]
    result = np.roll(centre, -side, axis=1 + axis)
    edge = [slice(None)] * (rank + 1)
    edge[1 + axis] = cells - 1 if side > 0 else 0
    edge = tuple(edge)
    if arrays["periodic"][axis]:
        translation = np.asarray(arrays["seam_translation"][axis], dtype=float)
        result[edge] += (side * translation).reshape((rank,) + (1,) * (rank - 1))
        return result
    faces = np.stack([native(component) for component in arrays["face_centroid"][axis]])
    face = [slice(None)] * (rank + 1)
    face[1 + axis] = cells if side > 0 else 0
    result[edge] = 2 * faces[tuple(face)] - centre[edge]
    return result


def centroid_rows(arrays: dict) -> np.ndarray:
    """The rows of delta X of every cell, (n0, ..., Dim, Dim): row a is c(K + 1_a) - c(K - 1_a)."""
    rank = len(arrays["centroid"])
    rows = np.stack([neighbour_centroids(arrays, axis, +1) - neighbour_centroids(arrays, axis, -1)
                     for axis in range(rank)])
    return np.moveaxis(rows, (0, 1), (-2, -1))


def affine_differences(arrays: dict, offset: float, slope: Any) -> np.ndarray:
    """The differences e(K + 1_a) - e(K - 1_a) of the affine field e = offset + slope . x read at
    the neighbour centroids, (Dim, n0, n1, ...)."""
    rank = len(arrays["centroid"])
    slope = np.asarray(slope, dtype=float).reshape((rank,) + (1,) * rank)

    def value(points):
        return offset + np.sum(slope * points, axis=0)

    return np.stack([value(neighbour_centroids(arrays, axis, +1))
                     - value(neighbour_centroids(arrays, axis, -1)) for axis in range(rank)])


def largest_value(arrays: dict, offset: float, slope: Any) -> float:
    """The largest |e| of the affine field e = offset + slope . x at a neighbour centroid of any
    cell."""
    rank = len(arrays["centroid"])
    points = np.stack([neighbour_centroids(arrays, axis, side)
                       for axis in range(rank) for side in (-1, 1)])
    values = offset + np.tensordot(np.asarray(slope, dtype=float), points, axes=([0], [1]))
    return float(np.max(np.abs(values)))


def ranked(differences: np.ndarray) -> np.ndarray:
    """Differences indexed [a, i0, i1, ...] in the NumPy order of the System, (Dim, ..., n1, n0)."""
    return np.ascontiguousarray(np.stack([native(component) for component in differences]))


def rounding_floor(rows: np.ndarray, magnitude: Any, slope: Any) -> np.ndarray:
    """The rounding floor of the slope of every cell, (n0, ...): the solve of delta X g = delta e
    by elimination is backward stable, delta e is the difference of two values of magnitude at
    most ``magnitude``, so |g - b| <= ||delta X^-1|| (|delta(delta e)| + ||delta(delta X)|| |b|),
    bounded here by 32 eps ||delta X^-1|| (magnitude + ||delta X|| |b|) in the infinity norm."""
    inverse = np.linalg.inv(rows)
    size = np.max(np.sum(np.abs(rows), axis=-1), axis=-1)
    inverse_size = np.max(np.sum(np.abs(inverse), axis=-1), axis=-1)
    largest = np.max(np.abs(np.asarray(slope, dtype=float)))
    return 32 * np.finfo(float).eps * inverse_size * (np.asarray(magnitude) + size * largest)


__all__ = ["affine_differences", "centroid_rows", "centroids", "largest_value",
           "layout_static_arrays", "layout_static_system", "native", "neighbour_centroids",
           "ranked", "rounding_floor"]
