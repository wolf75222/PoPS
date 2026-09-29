"""Mathematical checks independent of the installed PoPS native solver."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCIENTIFIC = Path(__file__).resolve().parents[4] / "examples/migration/scientific"
sys.path.insert(0, str(SCIENTIFIC))
from api040_m08_oracle import (  # noqa: E402
    cell_means, centered_divergence, field_metrics, guiding_velocity,
    negative_laplacian, poisson_discrete, ssprk2_reference,
)


@pytest.mark.parametrize("n", (32, 64))
def test_exact_cell_means_and_independent_periodic_poisson(n):
    q, c = cell_means(n)
    assert q.shape == c.shape == (n, n)
    assert abs(float(q.mean())) < 1.e-16
    assert abs(float(c.mean()) - 1.) < 1.e-15
    phi = poisson_discrete(q)
    np.testing.assert_allclose(negative_laplacian(phi), q, atol=2.e-14, rtol=0)
    ux, uy = guiding_velocity(phi)
    np.testing.assert_allclose(centered_divergence(ux, uy), 0., atol=2.e-15, rtol=0)
    metric = field_metrics(q, phi, np.stack((-uy, ux)))
    assert max(metric.values()) < 2.e-14


def test_same_time_modified_charge_cannot_reuse_old_field():
    q, _ = cell_means(32)
    phi = poisson_discrete(q)
    changed = 1.1 * q
    fresh = poisson_discrete(changed)
    np.testing.assert_allclose(fresh, 1.1 * phi, atol=2.e-16, rtol=0)
    assert np.max(np.abs(fresh - phi)) > 1.e-3
    assert field_metrics(changed, phi, np.stack((-guiding_velocity(phi)[1],
                                                guiding_velocity(phi)[0])))[
                                                    "poisson_residual_max"] > 1.e-3


def test_periodic_poisson_rejects_incompatible_mean():
    with pytest.raises(ValueError, match="zero mean"):
        poisson_discrete(np.ones((32, 32)))


def test_fv_ssprk2_conservation_and_stale_stage_discrimination():
    q, c = cell_means(32)
    fresh_q, fresh_c = ssprk2_reference(q, c, dt=.02, steps=2)
    stale_q, stale_c = ssprk2_reference(q, c, dt=.02, steps=2,
                                        stale_second_stage=True)
    np.testing.assert_allclose([fresh_q.mean(), fresh_c.mean()],
                               [q.mean(), c.mean()], atol=2.e-15, rtol=0)
    assert np.max(np.abs(fresh_q - q)) > 1.e-7
    assert np.max(np.abs(fresh_c - c)) > 1.e-7
    assert max(np.max(np.abs(fresh_q - stale_q)),
               np.max(np.abs(fresh_c - stale_c))) > 1.e-10
