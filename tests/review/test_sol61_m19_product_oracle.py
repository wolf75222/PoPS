"""Finite mathematical arrays and ownership model, never native saved states."""

from dataclasses import replace
from fractions import Fraction
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "m19_independent_reference", Path(__file__).with_name("sol61_m19_product_oracle.py")
)
oracle = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oracle
spec.loader.exec_module(oracle)


def values(shape, width):
    result = np.empty((width, *shape))
    for point in oracle.coordinates(shape):
        # Distinct signed values in every component and physical coordinate.
        cell = sum((i + 1) * (k + 1) ** 2 for i, k in enumerate(point))
        for c in range(width):
            result[(c, *point)] = (-1.0) ** (c + sum(point)) * (11 * (c + 1) + cell) / 8
    return result


def weights(length, *, signed=True):
    return tuple(Fraction((i + 1) * (-1 if signed and i % 2 else 1), 8) for i in range(length))


@pytest.mark.parametrize(
    "shape,axes,retained",
    [
        ((2, 5), ("x", "v"), ("x",)),
        ((7, 3), ("x", "v"), ("x",)),
        ((3, 2, 4), ("y", "v", "x"), ("x", "y")),
        ((5, 3, 2), ("v", "x", "w"), ("x",)),
    ],
)
@pytest.mark.parametrize("width", [1, 3, 5])
def test_tensor_reduction_axis_permutation_and_full_vector_lift(shape, axes, retained, width):
    f = values(shape, width)
    factors = {axis: weights(shape[i]) for i, axis in enumerate(axes) if axis not in retained}
    reduced = oracle.reduce_exact(f, axes, retained, factors)
    permutation = tuple(reversed(range(len(shape))))
    permuted_axes = tuple(axes[i] for i in permutation)
    permuted = f.transpose((0, *(i + 1 for i in permutation)))
    assert np.array_equal(oracle.reduce_exact(permuted, permuted_axes, retained, factors), reduced)
    lifted = oracle.lift(reduced, retained, axes, shape)
    for point in oracle.coordinates(shape):
        x = tuple(point[axes.index(axis)] for axis in retained)
        for c in range(width):
            assert lifted[(c, *point)] == reduced[(c, *x)]
    factor = math.prod(sum(row) for row in factors.values())
    assert np.array_equal(
        oracle.reduce_exact(lifted, axes, retained, factors), reduced * float(factor)
    )
    # A lift is a constant extension, not an inverse or a component-zero copy.
    assert not np.array_equal(lifted, f)
    if width > 1:
        assert not np.array_equal(lifted[0], lifted[-1])


def stamp(width=3):
    return oracle.Stamp(
        "accepted population@case-A",
        (("x", "space-A"), ("v", "velocity-A")),
        "macro:7+1/3",
        4,
        tuple("q%d" % c for c in range(width)),
        ("density",) * width,
    )


def cells(f, owners, authority, *, replicas=0):
    result = []
    for point in oracle.coordinates(tuple(f.shape[1:])):
        for rank in range(replicas) if replicas else (owners[point],):
            result.append(
                oracle.Cell(
                    rank, point, tuple(float(f[(c, *point)]) for c in range(f.shape[0])), authority
                )
            )
    return result


def flattened_publication(result):
    return {point: values for _, rows in result for point, values in rows}


@pytest.mark.parametrize("ranks", [1, 2, 3])
@pytest.mark.parametrize("split", ["x", "v", "checkerboard"])
def test_cross_rank_fibres_reduce_into_independently_owned_x_field(ranks, split):
    shape, width = (7, 5), 3
    f = values(shape, width)
    coordinate = {"x": lambda p: p[0], "v": lambda p: p[1], "checkerboard": lambda p: sum(p)}[split]
    source_owners = oracle.owner_table(shape, ranks, lambda p: coordinate(p) % ranks)
    destination_owners = oracle.owner_table((shape[0],), ranks, lambda p: (p[0] + 1) % ranks)
    result = oracle.reduce_owned(
        cells(f, source_owners, stamp(width)),
        shape,
        ("x", "v"),
        ("x",),
        {"v": weights(shape[1])},
        source_owners,
        destination_owners,
        ranks,
        stamp(width),
    )
    dense = oracle.reduce_exact(f, ("x", "v"), ("x",), {"v": weights(shape[1])})
    actual = flattened_publication(result)
    assert len(actual) == shape[0]
    assert all(actual[p] == tuple(dense[:, p[0]]) for p in actual)
    assert all(destination_owners[p] == rank for rank, rows in result for p, _ in rows)
    if ranks > 1 and split == "v":
        # Dropping a peer's part of each velocity fibre changes the original sum.
        wrong = oracle.reduce_exact(
            f * np.array([j % ranks == 0 for j in range(shape[1])]),
            ("x", "v"),
            ("x",),
            {"v": weights(shape[1])},
        )
        assert not np.array_equal(dense, wrong)


def test_replicas_validate_all_ranks_but_count_one_physical_owner():
    f, ranks = values((3, 5), 3), 2
    owners = oracle.owner_table((3, 5), ranks, lambda _: 0)
    rows = cells(f, owners, stamp(), replicas=ranks)
    expected = oracle.owned_dense(rows, (3, 5), owners, ranks, stamp(), replicated=True)
    assert np.array_equal(expected, f)
    reversed_rows = oracle.owned_dense(
        tuple(reversed(rows)), (3, 5), owners, ranks, stamp(), replicated=True
    )
    assert np.array_equal(reversed_rows, expected)
    bad = list(rows)
    bad[1] = replace(bad[1], values=(99.0, *bad[1].values[1:]))
    with pytest.raises(ValueError, match="replicas disagree"):
        oracle.owned_dense(bad, (3, 5), owners, ranks, stamp(), replicated=True)
    with pytest.raises(ValueError, match="partial replication"):
        oracle.owned_dense(rows[:-1], (3, 5), owners, ranks, stamp(), replicated=True)


@pytest.mark.parametrize(
    "attack",
    [
        "duplicate",
        "missing",
        "wrong_rank",
        "outside",
        "bool_index",
        "point",
        "generation",
        "subject",
        "domain",
        "components",
        "units",
        "nan",
    ],
)
def test_ownership_and_authority_injections_refused_before_publication(attack):
    f, shape, ranks = values((3, 5), 3), (3, 5), 2
    owners = oracle.owner_table(shape, ranks, lambda p: p[1] % ranks)
    rows = cells(f, owners, stamp())
    if attack == "duplicate":
        rows.append(rows[0])
    elif attack == "missing":
        rows.pop()
    elif attack == "wrong_rank":
        rows[0] = replace(rows[0], rank=1 - rows[0].rank)
    elif attack == "outside":
        rows[0] = replace(rows[0], index=(3, 0))
    elif attack == "bool_index":
        rows[0] = replace(rows[0], index=(False, 0))
    elif attack == "nan":
        rows[0] = replace(rows[0], values=(np.nan, 1.0, 2.0))
    else:
        mutations = dict(
            point="macro:7+2/3",
            generation=5,
            subject="foreign case population",
            domain=(("x", "foreign-space"), ("v", "velocity-A")),
            components=("q2", "q1", "q0"),
            units=("time",) * 3,
        )
        key = "support" if attack == "domain" else attack
        rows[0] = replace(rows[0], stamp=replace(rows[0].stamp, **{key: mutations[attack]}))
    target_owners = oracle.owner_table((3,), ranks, lambda p: p[0] % ranks)
    with pytest.raises(ValueError):
        oracle.reduce_owned(
            rows,
            shape,
            ("x", "v"),
            ("x",),
            {"v": weights(5)},
            owners,
            target_owners,
            ranks,
            stamp(),
        )
    assert np.array_equal(f, values(shape, 3))


def test_weighted_adjoint_is_declared_measure_dependent_not_an_inverse():
    f, g = values((3, 5), 3), values((3,), 3)
    dv = tuple(Fraction(i + 1, 8) for i in range(5))
    dx = tuple(Fraction(2 * i + 1, 16) for i in range(3))
    rf = oracle.reduce_exact(f, ("x", "v"), ("x",), {"v": dv})
    lg = oracle.lift(g, ("x",), ("x", "v"), (3, 5))
    left = sum(
        dx[x] * dv[v] * Fraction(float(f[c, x, v])) * Fraction(float(lg[c, x, v]))
        for c in range(3)
        for x in range(3)
        for v in range(5)
    )
    right = sum(
        dx[x] * Fraction(float(rf[c, x])) * Fraction(float(g[c, x]))
        for c in range(3)
        for x in range(3)
    )
    assert left == right
    # Changing to a signed moment changes the adjoint equation, not the physical measure.
    signed = oracle.reduce_exact(f, ("x", "v"), ("x",), {"v": weights(5)})
    wrong_adjoint = sum(
        dx[x] * Fraction(float(signed[c, x])) * Fraction(float(g[c, x]))
        for c in range(3)
        for x in range(3)
    )
    assert left != wrong_adjoint


@pytest.mark.parametrize(
    "attack",
    [
        "missing_weight",
        "wrong_length",
        "lost_component",
        "wrong_extent",
        "weight_overflow",
        "weight_underflow",
    ],
)
def test_map_shape_and_binary64_weight_admission_are_separate(attack):
    f = values((3, 5), 3)
    with pytest.raises(ValueError):
        if attack == "missing_weight":
            oracle.reduce_exact(f, ("x", "v"), ("x",), {})
        elif attack == "wrong_length":
            oracle.reduce_exact(f, ("x", "v"), ("x",), {"v": weights(4)})
        elif attack == "lost_component":
            oracle.owned_dense(
                cells(f, {(x, v): 0 for x in range(3) for v in range(5)}, stamp()),
                (3, 5),
                {(x, v): 0 for x in range(3) for v in range(5)},
                1,
                stamp(1),
            )
        elif attack == "wrong_extent":
            oracle.lift(values((3,), 3), ("x",), ("x", "v"), (4, 5))
        else:
            oracle.native_weight_admission(
                {"v": (Fraction(10**500) if attack == "weight_overflow" else Fraction(1, 10**500),)}
            )


@pytest.mark.parametrize("ranks", [1, 2, 4])
@pytest.mark.parametrize("width", [1, 3, 5])
def test_lift_routes_full_vectors_to_product_owners_including_empty_rank(ranks, width):
    field = values((3,), width)
    source_stamp = oracle.Stamp(
        "field@case-A",
        (("x", "space-A"),),
        "accepted:9",
        3,
        tuple(f"e{c}" for c in range(width)),
        ("acceleration",) * width,
    )
    target_stamp = replace(
        source_stamp, subject="lift@case-A", support=(("v", "velocity-A"), ("x", "space-A"))
    )
    # Rank3 is empty at the source for ranks4 but owns destination cells.
    source_owners = oracle.owner_table((3,), ranks, lambda p: p[0] % ranks)
    target_owners = oracle.owner_table((5, 3), ranks, lambda p: (p[0] + 2 * p[1]) % ranks)
    rows = cells(field, source_owners, source_stamp)
    result = oracle.lift_owned(
        rows,
        (3,),
        ("x",),
        ("v", "x"),
        (5, 3),
        source_owners,
        target_owners,
        ranks,
        source_stamp,
        target_stamp,
    )
    actual = flattened_publication(result)
    assert len(actual) == 15
    assert all(actual[v, x] == tuple(field[:, x]) for v, x in actual)
    assert all(target_owners[p] == rank for rank, rows in result for p, _ in rows)
    if ranks == 4:
        assert not any(row.rank == 3 for row in rows)
        assert result[3][1]


@pytest.mark.parametrize(
    "attack",
    [
        "domain",
        "point",
        "generation",
        "components",
        "units",
        "destination_missing",
        "destination_foreign",
    ],
)
def test_lift_destination_attacks_refuse_without_changing_source(attack):
    field = values((3,), 3)
    original = field.copy()
    source_stamp = replace(stamp(), support=(("x", "space-A"),))
    target_stamp = replace(stamp(), subject="lift@case-A")
    source_owners = oracle.owner_table((3,), 2, lambda p: p[0] % 2)
    target_owners = oracle.owner_table((3, 5), 2, lambda p: p[1] % 2)
    changes = dict(
        domain={"support": (("x", "other-space"), ("v", "velocity-A"))},
        point={"point": "macro:8"},
        generation={"generation": 5},
        components={"components": ("q2", "q1", "q0")},
        units={"units": ("time",) * 3},
    )
    if attack in changes:
        target_stamp = replace(target_stamp, **changes[attack])
    elif attack == "destination_missing":
        target_owners.pop((2, 4))
    else:
        target_owners[2, 4] = 2
    with pytest.raises(ValueError):
        oracle.lift_owned(
            cells(field, source_owners, source_stamp),
            (3,),
            ("x",),
            ("x", "v"),
            (3, 5),
            source_owners,
            target_owners,
            2,
            source_stamp,
            target_stamp,
        )
    assert np.array_equal(field, original)


def test_zero_sum_signed_moment_and_tensor_mass_balance_countermodels():
    f = values((3, 5), 3)
    dx = (Fraction(1, 4), Fraction(3, 8), Fraction(7, 16))
    dv = (Fraction(1, 8), Fraction(3, 16), Fraction(1, 4), Fraction(3, 8), Fraction(1, 2))
    reduced = oracle.reduce_exact(f, ("x", "v"), ("x",), {"v": dv})
    full_mass = tuple(
        sum(dx[x] * dv[v] * Fraction(float(f[c, x, v])) for x in range(3) for v in range(5))
        for c in range(3)
    )
    retained_mass = tuple(
        sum(dx[x] * Fraction(float(reduced[c, x])) for x in range(3)) for c in range(3)
    )
    assert full_mass == retained_mass
    unweighted = oracle.reduce_exact(f, ("x", "v"), ("x",), {"v": (1,) * 5})
    assert not np.array_equal(unweighted, reduced)
    assert not np.array_equal(reduced[::-1], reduced)
    g = values((3,), 3)
    extended = oracle.lift(g, ("x",), ("x", "v"), (3, 5))
    zero_sum = (Fraction(1), Fraction(-2), Fraction(0), Fraction(3), Fraction(-2))
    assert sum(zero_sum) == 0
    assert not np.array_equal(extended, np.zeros_like(extended))
    assert np.array_equal(
        oracle.reduce_exact(extended, ("x", "v"), ("x",), {"v": zero_sum}), np.zeros_like(g)
    )
