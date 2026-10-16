"""Divergence identity, Hall skewness and distinct Ohmic heating in genuine 2D."""

import numpy as np

from tests.python.support.hall_mhd_ct_oracle import hall_mhd_reference


def test_ct_preserves_divergence_with_nonlinear_hall_and_ohm_in_two_axes():
    initial, final, records = hall_mhd_reference()
    for axis in (1, 2):
        assert np.max(abs(np.roll(initial, -1, axis) - initial)) > 0.01
    assert np.max(abs(final - initial)) > 1e-5
    for record in records:
        assert record["divergence_after"] < 1e-13
        assert record["inventory_defect"] < 1e-14
        assert abs(record["magnetic_energy_change"] + record["integrated_work"]) < 1e-13
        for stage in record["stages"]:
            assert stage["pressure_minimum"] > 0
            assert stage["divergence_rate"] < 1e-13
            assert abs(stage["hall_work"]) < 1e-14
            assert stage["ohmic_heat"] > 1e-3
            assert abs(stage["magnetic_rate"] + stage["field_work"]) < 1e-14


def test_hall_changes_topology_without_dissipation_and_ohm_has_a_distinct_budget():
    _, hall, records = hall_mhd_reference(ohmic=0.0)
    _, ideal, _ = hall_mhd_reference(hall=0.0, ohmic=0.0)
    _, resistive, _ = hall_mhd_reference(hall=0.0, ohmic=0.01)
    assert abs(hall - ideal).max() > 1e-6
    assert abs(resistive - ideal).max() > 1e-6
    assert all(stage["ohmic_heat"] == 0 for row in records for stage in row["stages"])


def test_a_non_ct_induction_perturbation_breaks_the_divergence_identity():
    _, _, records = hall_mhd_reference(steps=1)
    derivative = records[0]["stages"][0]["rhs"].reshape(8, 6, 8)
    broken = derivative.copy()
    broken[5, 1, 2] += 0.01

    def divergence(q):
        return 6 * (np.roll(q[5], -1, 0) - q[5]) + 8 * (np.roll(q[6], -1, 1) - q[6])

    assert abs(divergence(derivative)).max() < 1e-13
    assert abs(divergence(broken)).max() > 0.05
