"""Independent full phase-space and original field-equation checks."""

import numpy as np

from tests.python.support.gyrokinetic_nd_oracle import gyrokinetic_reference


def test_full_f_five_dimensional_oracle_activates_geometry_and_induction():
    initial, result, records = gyrokinetic_reference(collision_frequency=0.02)
    assert initial.shape == (2, 3, 3, 3, 4, 3)
    assert result.min() > 0
    assert abs(result - initial).max() > 1e-8
    np.testing.assert_allclose(
        np.sum(result - initial, axis=(1, 2, 3, 4, 5)), 0, atol=1e-13, rtol=0
    )
    for row in records:
        assert row["phi_residual"] < 1e-20
        assert row["apar_residual"] < 1e-20
        assert row["derivative_residual"] < 1e-20
        assert abs(row["neutrality"]) < 1e-13
        assert abs(row["mean_current"]) < 1e-13
        assert row["min_induction_mass"] > 0
        assert min(min(a) for a in row["characteristic_amplitudes"]) > 1e-5
        assert row["max_curvature"] > 0.01
        assert row["max_nonlinear_hamiltonian"] > 1e-10
        assert abs(row["adot"]).max() > 1e-5
        assert max(row["collision_defects_before"]) > 1e-5
        assert max(row["collision_defects_after"]) < 1e-15
    assert records[-1]["max_flutter"] > 1e-9


def test_full_f_coefficient_and_hamiltonian_change_together():
    _, nonlinear, _ = gyrokinetic_reference(nonlinear_polarization=True)
    _, linear, _ = gyrokinetic_reference(nonlinear_polarization=False)
    assert abs(nonlinear - linear).max() > 1e-12


def test_finite_box_collision_flux_needs_an_explicit_conservation_discretization():
    _, projected, records = gyrokinetic_reference(collision_frequency=0.02)
    _, raw, raw_records = gyrokinetic_reference(
        collision_frequency=0.02, collision_projection=False
    )
    assert abs(raw - projected).max() > 1e-9
    assert max(max(row["collision_defects_after"]) for row in raw_records) > 1e-5
    assert max(max(row["collision_defects_after"]) for row in records) < 1e-15


def test_projected_collision_does_not_claim_unconditional_positivity():
    # The compiled Assess also checks positivity before publishing the candidate.
    # Its native rejection is a separate reception from this independent witness.
    _, result, _ = gyrokinetic_reference(collision_frequency=0.02, dt=1.0, steps=1)
    assert result.min() < -1e-4
