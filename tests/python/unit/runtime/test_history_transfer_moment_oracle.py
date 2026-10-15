"""Counterexamples for the exact native history receipt, without running a surrogate solver."""
from copy import deepcopy
from fractions import Fraction

import pytest

from tests.python.support.history_transfer_oracles import (
    assert_projected_quadratic_flux, quadratic_moment_weights,
)


def _fixture():
    times = (.04, .12, .08)
    targets = tuple(Fraction(.16)-lag*Fraction(.04)/2 for lag in (3, 1, 2))
    parent = [[dict(identity=1, level=0, runtime_block=0, weights=[(0, Fraction(1))],
                    snapshot=dict(kind=1, level=0, identity=f"source-{i}", source_identity=f"native-{i}"))]
              for i in range(3)]
    # Build by the moment system, then separately perturb coefficients and provenance below.
    child = [[dict(identity=i, level=1, runtime_block=0, weights=[(0, weight)],
                   snapshot=dict(kind=2, level=1, identity=f"projection-{i}", parent=deepcopy(parent[i][0]["snapshot"])))
              for i, weight in enumerate(quadratic_moment_weights(times, target)) if weight]
             for target in targets]
    return parent, child, times, targets


def test_equal_rounded_target_is_not_an_equal_exact_source_time():
    parent, child, times, targets = _fixture()
    assert float(targets[2]) == times[1]
    assert targets[2] != Fraction(times[1])
    assert [len(slot) for slot in child] == [3, 3, 3]
    assert_projected_quadratic_flux(parent, child, times, targets)
    for target in targets:
        coefficients = quadratic_moment_weights(times, target)
        for power in range(3):
            assert sum(w*Fraction(t)**power for w, t in zip(coefficients, times, strict=True)) == target**power


def test_exact_source_target_has_one_basis_without_a_cardinality_exception():
    parent, child, times, targets = _fixture()
    targets = (*targets[:2], Fraction(times[1]))
    child[2] = [deepcopy(child[2][1])]
    child[2][0]["weights"] = [(0, Fraction(1))]
    assert_projected_quadratic_flux(parent, child, times, targets)


@pytest.mark.parametrize("mutation", ("drop_tiny", "round_coefficients", "swap", "change_source"))
def test_exact_receipt_rejects_lost_coefficients_and_false_source_authority(mutation):
    parent, child, times, targets = _fixture()
    if mutation == "drop_tiny":
        assert abs(child[2][0]["weights"][0][1]) < Fraction(1, 10**15)
        del child[2][0]
    elif mutation == "round_coefficients":
        for slot in child:
            for basis in slot:
                basis["weights"] = [(power, Fraction(float(weight))) for power, weight in basis["weights"]]
    elif mutation == "swap":
        child[0][0]["weights"], child[0][1]["weights"] = child[0][1]["weights"], child[0][0]["weights"]
    else:
        child[0][0]["snapshot"]["parent"]["source_identity"] = "other-source"
    with pytest.raises(AssertionError):
        assert_projected_quadratic_flux(parent, child, times, targets)
