"""Exact-rank isotropic AMR ratios follow the compiled native dimension."""

from __future__ import annotations

import pytest

from pops.amr.authoring import AMRHierarchy, resolve_transition_ratios
from pops.runtime_environment import RuntimeCapabilityError, validate_amr_refinement_ratio


@pytest.mark.parametrize(
    ("dimension", "expected"),
    (
        (1, (2,)),
        (2, (2, 2)),
        (3, (2, 2, 2)),
    ),
)
def test_isotropic_scalar_expands_to_the_native_rank(monkeypatch, dimension, expected):
    monkeypatch.setattr("pops.runtime_environment.native_dimension", lambda: dimension)

    assert validate_amr_refinement_ratio(2) == expected
    assert validate_amr_refinement_ratio(expected) == expected


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_anisotropic_sequence_must_match_the_native_rank(monkeypatch, dimension):
    monkeypatch.setattr("pops.runtime_environment.native_dimension", lambda: dimension)
    ranked = tuple(1 if axis else 2 for axis in range(dimension))
    wrong = ranked + (2,)

    assert validate_amr_refinement_ratio(ranked) == ranked
    with pytest.raises(RuntimeCapabilityError, match="exactly %d axes" % dimension):
        validate_amr_refinement_ratio(wrong)


@pytest.mark.parametrize(
    ("dimension", "wrong"),
    (
        (1, (2, 2)),
        (2, (2,)),
        (2, (2, 2, 2)),
        (3, (2, 2)),
    ),
)
def test_cross_rank_isotropic_defaults_are_refused(monkeypatch, dimension, wrong):
    monkeypatch.setattr("pops.runtime_environment.native_dimension", lambda: dimension)

    with pytest.raises(RuntimeCapabilityError, match="exactly %d axes" % dimension):
        validate_amr_refinement_ratio(wrong)


@pytest.mark.parametrize(
    ("dimension", "expected"),
    (
        (1, ((2,),)),
        (2, ((2, 2),)),
        (3, ((2, 2, 2),)),
    ),
)
def test_authoring_scalar_ratios_resolve_to_exact_rank(dimension, expected):
    assert resolve_transition_ratios((2,), dimension=dimension) == expected
    hierarchy = AMRHierarchy(max_levels=2, ratios=(2,))
    assert hierarchy.resolved_ratios(dimension) == expected


def test_authoring_preserves_anisotropic_dim_length_ratios():
    assert resolve_transition_ratios(((2, 1, 2),), dimension=3) == ((2, 1, 2),)
    with pytest.raises(ValueError, match="rank 2 but the owning mesh has rank 3"):
        resolve_transition_ratios(((2, 2),), dimension=3)
