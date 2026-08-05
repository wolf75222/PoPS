"""The Python auxiliary channel mirrors the exact native 1D/2D/3D rank."""
from __future__ import annotations

import pytest

from pops.physics.aux import (
    AUX_CANONICAL_NAMES,
    AUX_NAMED_MAX,
    AuxLayout,
    aux_component_index,
    aux_layout,
    aux_total_n_aux,
)


@pytest.mark.parametrize(
    ("dimension", "axes", "canonical", "base", "named_base"),
    (
        (1, ("x",), {"phi": 0, "grad_x": 1, "B_z": 2, "T_e": 3}, 2, 4),
        (
            2,
            ("x", "y"),
            {"phi": 0, "grad_x": 1, "grad_y": 2, "B_z": 3, "T_e": 4},
            3,
            5,
        ),
        (
            3,
            ("x", "y", "z"),
            {
                "phi": 0,
                "grad_x": 1,
                "grad_y": 2,
                "grad_z": 3,
                "B_z": 4,
                "T_e": 5,
            },
            4,
            6,
        ),
    ),
)
def test_aux_layout_is_exactly_ranked(
    dimension, axes, canonical, base, named_base
):
    layout = aux_layout(dimension)

    assert isinstance(layout, AuxLayout)
    assert layout.dimension == dimension
    assert layout.axes == axes
    assert dict(layout.canonical) == canonical
    assert layout.base_components == base
    assert layout.named_base == named_base
    assert layout.max_components == named_base + AUX_NAMED_MAX
    assert set(layout.canonical).issubset(AUX_CANONICAL_NAMES)


@pytest.mark.parametrize(
    ("dimension", "canonical_names", "named", "expected"),
    (
        (1, (), (), 2),
        (1, ("B_z",), (), 3),
        (1, (), ("kappa",), 5),
        (2, (), (), 3),
        (2, ("B_z",), ("kappa",), 6),
        (3, (), (), 4),
        (3, ("grad_z",), ("kappa",), 7),
    ),
)
def test_aux_width_uses_the_ranked_named_base(
    dimension, canonical_names, named, expected
):
    assert aux_total_n_aux(
        canonical_names, named, dimension=dimension
    ) == expected


@pytest.mark.parametrize(
    ("dimension", "name"),
    ((1, "grad_y"), (1, "grad_z"), (2, "grad_z")),
)
def test_gradient_components_outside_the_rank_are_rejected(dimension, name):
    with pytest.raises(ValueError, match="outside the .*D canonical layout"):
        aux_component_index(name, dimension=dimension)

    with pytest.raises(ValueError, match="outside the .*D canonical layout"):
        aux_total_n_aux((name,), (), dimension=dimension)


@pytest.mark.parametrize("dimension", (True, 0, 4, 2.0, "2"))
def test_aux_layout_refuses_an_ambiguous_or_unsupported_rank(dimension):
    error = TypeError if dimension in (True, 2.0, "2") else ValueError
    with pytest.raises(error):
        aux_layout(dimension)

