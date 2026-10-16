"""Adversarial checks of reception logic, without native builds or executions."""
import numpy as np
import pytest

from examples.api_v02.method_amr_reception import (
    composite_base_projection, initial_base_means, translation_metrics)


def _partial_two_level_observation(time=0.):
    cells = (4, 4)
    levels = []
    amplitude = -.1/np.sinc(1/4)
    for level in range(2):
        count = 4*2**level
        x, y = np.meshgrid((np.arange(count)+.5)/count,
                           (np.arange(count)+.5)/count)
        values = (1+amplitude*np.sinc(1/count)
                  *(np.cos(2*np.pi*(x-time))+np.cos(2*np.pi*y)))
        levels.append(values)
    # The fine patch covers the left half. Deliberately poison covered coarse
    # cells and absent fine cells: neither belongs to the physical composite.
    levels[0][:, :2] = 1000.
    levels[1][:, 4:] = np.nan
    return cells, levels, ((0, (0, 0), (3, 3)), (1, (0, 0), (3, 7)))


def test_t08_projection_uses_only_valid_uncovered_volume():
    cells, levels, boxes = _partial_two_level_observation()
    projected = composite_base_projection(levels, boxes, cells=cells)
    np.testing.assert_allclose(projected["means"], initial_base_means(cells),
                               rtol=0, atol=2.e-15)
    np.testing.assert_array_equal(projected["weights"], np.ones((4, 4)))
    assert projected["active_counts"] == (8, 32)
    assert abs(projected["means"].mean()-1) < 2.e-15


def test_t08_reception_rejects_immobile_mass_conserving_output():
    cells, levels, boxes = _partial_two_level_observation()
    projected = composite_base_projection(levels, boxes, cells=cells)
    # This observation has two nonempty active levels and exactly the correct
    # composite mass. Advancing only the runtime clock cannot make it pass.
    assert abs(projected["means"].mean()-1) < 2.e-15
    metrics = translation_metrics(projected["means"], cells=cells, time=.01)
    assert not metrics["passed"]
    assert metrics["composite_state_change"] <= metrics["comparison_roundoff"]


def test_t08_reception_accepts_exact_translated_composite():
    cells, levels, boxes = _partial_two_level_observation(time=.01)
    projected = composite_base_projection(levels, boxes, cells=cells)
    metrics = translation_metrics(projected["means"], cells=cells, time=.01)
    assert metrics["passed"]
    assert metrics["composite_translation_error"] < 2.e-15
    assert metrics["composite_state_change"] > 1.e-3


def test_t08_projection_refuses_incomplete_or_unaligned_observation():
    cells, levels, boxes = _partial_two_level_observation()
    with pytest.raises(ValueError, match="one owner"):
        composite_base_projection((levels[0], None), boxes, cells=cells)
    with pytest.raises(ValueError, match="align"):
        composite_base_projection(levels, ((1, (0, 0), (2, 7)),), cells=cells)


def test_t08_reference_with_no_resolvable_motion_fails_closed():
    cells = (4, 4)
    metrics = translation_metrics(initial_base_means(cells), cells=cells, time=1.)
    assert not metrics["passed"]


def test_t08_projection_refuses_duplicated_patch_volume():
    cells, levels, boxes = _partial_two_level_observation()
    with pytest.raises(ValueError, match="overlap or repeat"):
        composite_base_projection(levels, (*boxes, boxes[-1]), cells=cells)
