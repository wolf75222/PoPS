"""Autonomous finite-support product/reduction/lift reference, no PoPS import.

Coordinates, vector components, source owners and destination owners are
separate. No output constitutes a native state or an external owner receipt.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import math

import numpy as np


def need(condition, message):
    if not condition:
        raise ValueError("product contract: " + message)


def coordinates(shape):
    need(all(type(n) is int and n > 0 for n in shape), "positive exact extents required")
    return tuple(product(*(range(n) for n in shape)))


def weights_for(shape, axes, retained, weights):
    need(len(axes) == len(shape) and len(set(axes)) == len(axes), "source axes are not distinct")
    need(
        len(set(retained)) == len(retained) and set(retained) < set(axes),
        "retained support is not a strict subset",
    )
    removed = set(axes) - set(retained)
    need(set(weights) == removed, "each eliminated axis needs exactly one quadrature")
    result = {}
    for axis in removed:
        row = tuple(Fraction(value) for value in weights[axis])
        need(len(row) == shape[axes.index(axis)], "one weight per eliminated cell required")
        result[axis] = row
    return result


def native_weight_admission(weights):
    """Binary64 realization has a separate admission from rational mathematics."""
    for row in weights.values():
        for authored in row:
            value = Fraction(authored)
            try:
                binary = float(value)
            except OverflowError as error:
                raise ValueError("product contract: weight overflow") from error
            need(
                math.isfinite(binary) and (value == 0 or binary != 0),
                "nonfinite/underflowed weight",
            )


def reduce_exact(values, axes, retained, weights):
    values = np.asarray(values)
    need(
        values.dtype == np.dtype("float64")
        and values.ndim == len(axes) + 1
        and values.shape[0] > 0
        and np.isfinite(values).all(),
        "finite vector product field required",
    )
    shape = tuple(values.shape[1:])
    spatial = coordinates(shape)
    factors = weights_for(shape, axes, retained, weights)
    output_shape = tuple(shape[axes.index(axis)] for axis in retained)
    sums = {
        (c, *point): Fraction(0)
        for c in range(values.shape[0])
        for point in coordinates(output_shape)
    }
    for point in spatial:
        target = tuple(point[axes.index(axis)] for axis in retained)
        factor = Fraction(1)
        for axis, row in factors.items():
            factor *= row[point[axes.index(axis)]]
        for c in range(values.shape[0]):
            sums[(c, *target)] += factor * Fraction(float(values[(c, *point)]))
    result = np.empty((values.shape[0], *output_shape), dtype=np.float64)
    for point, value in sums.items():
        try:
            result[point] = float(value)
        except OverflowError as error:
            raise ValueError("product contract: reduction overflow") from error
    need(np.isfinite(result).all(), "reduction overflow")
    return result


def lift(field, axes, target_axes, target_shape):
    field = np.asarray(field)
    need(
        field.dtype == np.dtype("float64")
        and field.ndim == len(axes) + 1
        and field.shape[0] > 0
        and np.isfinite(field).all(),
        "finite vector retained field required",
    )
    need(
        len(set(target_axes)) == len(target_axes)
        and len(set(axes)) == len(axes)
        and set(axes) < set(target_axes)
        and len(target_axes) == len(target_shape),
        "independent extension support required",
    )
    need(
        all(
            field.shape[1 + i] == target_shape[target_axes.index(axis)]
            for i, axis in enumerate(axes)
        ),
        "retained extent changed during lift",
    )
    result = np.empty((field.shape[0], *target_shape), dtype=np.float64)
    for point in coordinates(target_shape):
        source = tuple(point[target_axes.index(axis)] for axis in axes)
        for c in range(field.shape[0]):
            result[(c, *point)] = field[(c, *source)]
    return result


@dataclass(frozen=True)
class Stamp:
    subject: str
    support: tuple  # exact (coordinate, domain identity), independent of native embedding
    point: str
    generation: int
    components: tuple
    units: tuple


@dataclass(frozen=True)
class Cell:
    rank: int
    index: tuple
    values: tuple
    stamp: Stamp


def owner_table(shape, ranks, owner):
    need(type(ranks) is int and ranks > 0, "positive rank space required")
    result = {point: owner(point) for point in coordinates(shape)}
    need(
        all(type(rank) is int and 0 <= rank < ranks for rank in result.values()),
        "foreign owner rank",
    )
    return result


def owned_dense(cells, shape, owners, ranks, expected_stamp, *, replicated=False):
    support = set(coordinates(shape))
    need(
        set(owners) == support
        and all(type(rank) is int and 0 <= rank < ranks for rank in owners.values()),
        "owner table must represent every physical cell",
    )
    need(
        len(expected_stamp.support) == len(shape)
        and len(set(expected_stamp.support)) == len(shape),
        "support authority",
    )
    need(
        type(expected_stamp.generation) is int
        and expected_stamp.generation >= 0
        and len(expected_stamp.components) > 0
        and len(set(expected_stamp.components)) == len(expected_stamp.components)
        and len(expected_stamp.units) == len(expected_stamp.components),
        "component/unit/generation authority",
    )
    seen, by_cell = set(), {}
    for row in cells:
        need(
            type(row.rank) is int
            and 0 <= row.rank < ranks
            and type(row.index) is tuple
            and all(type(i) is int for i in row.index)
            and row.index in support,
            "foreign rank/cell",
        )
        need(
            row.stamp == expected_stamp
            and len(row.values) == len(expected_stamp.components)
            and all(math.isfinite(value) for value in row.values),
            "stale subject/point/support/components or nonfinite cell",
        )
        key = (row.rank, row.index)
        need(key not in seen, "duplicate rank-local physical cell")
        seen.add(key)
        if not replicated:
            need(
                row.rank == owners[row.index] and row.index not in by_cell,
                "wrong or duplicate physical owner",
            )
            by_cell[row.index] = row.values
        else:
            if row.index in by_cell:
                need(row.values == by_cell[row.index], "replicas disagree")
            else:
                by_cell[row.index] = row.values
    expected = (
        {(rank, point) for rank in range(ranks) for point in support}
        if replicated
        else {(owners[point], point) for point in support}
    )
    need(seen == expected, "missing owner or partial replication")
    result = np.empty((len(expected_stamp.components), *shape), dtype=np.float64)
    for point, values in by_cell.items():
        for c, value in enumerate(values):
            result[(c, *point)] = value
    return result


def publish_owned(values, owners, ranks):
    shape = tuple(values.shape[1:])
    support = set(coordinates(shape))
    need(
        set(owners) == support
        and all(type(rank) is int and 0 <= rank < ranks for rank in owners.values()),
        "destination ownership incomplete or foreign",
    )
    # Validate everything before constructing the publication value.
    return tuple(
        (
            rank,
            tuple(
                (point, tuple(float(values[(c, *point)]) for c in range(values.shape[0])))
                for point in coordinates(shape)
                if owners[point] == rank
            ),
        )
        for rank in range(ranks)
    )


def reduce_owned(
    cells,
    shape,
    axes,
    retained,
    weights,
    source_owners,
    destination_owners,
    ranks,
    stamp,
    *,
    replicated=False,
):
    values = owned_dense(cells, shape, source_owners, ranks, stamp, replicated=replicated)
    need(
        {axis for axis, _ in stamp.support} == set(axes),
        "native embedding disagrees with source support",
    )
    native_weight_admission(weights)
    reduced = reduce_exact(values, axes, retained, weights)
    return publish_owned(reduced, destination_owners, ranks)


def lift_owned(
    cells,
    shape,
    axes,
    target_axes,
    target_shape,
    source_owners,
    destination_owners,
    ranks,
    source_stamp,
    target_stamp,
    *,
    replicated=False,
):
    """Independent record-consistency oracle, never a native authority issuer."""
    values = owned_dense(cells, shape, source_owners, ranks, source_stamp, replicated=replicated)
    source_domains = dict(source_stamp.support)
    target_domains = dict(target_stamp.support)
    need(
        len(source_domains) == len(source_stamp.support)
        and len(target_domains) == len(target_stamp.support)
        and set(source_domains) == set(axes)
        and set(target_domains) == set(target_axes),
        "extension embedding/support mismatch",
    )
    need(
        all(target_domains.get(axis) == domain for axis, domain in source_domains.items()),
        "extension changed retained physical domain",
    )
    need(
        source_stamp.point == target_stamp.point
        and source_stamp.generation == target_stamp.generation
        and source_stamp.components == target_stamp.components
        and source_stamp.units == target_stamp.units,
        "extension point/components/units mismatch",
    )
    extended = lift(values, axes, target_axes, target_shape)
    return publish_owned(extended, destination_owners, ranks)
