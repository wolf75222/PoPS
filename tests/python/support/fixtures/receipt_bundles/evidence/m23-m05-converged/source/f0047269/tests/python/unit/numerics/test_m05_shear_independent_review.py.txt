"""Independent dense stencil and fail-closed receipt checks; no native run."""
from pathlib import Path
import sys

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[4] / "examples/migration/scientific"
sys.path.insert(0, str(EXAMPLES))
from api040_m05_periodic_shear import _ledger_metrics  # noqa: E402
from api040_m05_shear_oracle import (  # noqa: E402
    energy, forward_euler_energy_defect, periodic_faces, periodic_rate, semidiscrete_work,
)


def test_arbitrary_periodic_state_matches_dense_operator_and_invariances():
    size, viscosity = 17, .03
    values = np.random.default_rng(912).normal(size=size)
    operator = np.zeros((size, size))
    for i in range(size):
        operator[i, i] = -2
        operator[i, (i-1) % size] = 1
        operator[i, (i+1) % size] = 1
    operator *= viscosity * size**2
    rate = periodic_rate(values, viscosity)
    np.testing.assert_allclose(rate, operator @ values, rtol=0, atol=2e-14)
    for shifted in (values + 2, np.roll(values, 4), values[::-1]):
        np.testing.assert_allclose(periodic_rate(shifted, viscosity), operator @ shifted, rtol=0, atol=2e-14)
    work, dissipation = semidiscrete_work(values, viscosity)
    assert abs(work - values @ operator @ values / size) < 2e-14
    assert abs(work - dissipation) < 2e-14
    dt = .001
    increment, split = forward_euler_energy_defect(values, viscosity, dt)
    assert abs(increment - (energy(values + dt * (operator @ values)) - energy(values))) < 2e-15
    assert abs(increment - split) < 2e-15
    assert increment > dt * work


def rows_and_states():
    values = np.random.default_rng(716).normal(size=11)
    dt = .001
    fluxes = periodic_faces(values, .03)
    rows = []
    for i in range(len(values)):
        for side in (0, 1):
            sign = -1 if side == 0 else 1
            flux = fluxes[(i-1) % len(values) if side == 0 else i]
            rows.append(dict(
                quadrature_identity=f"cell:{i}/axis:0/side:{side}",
                occurrence_identity="one_occurrence", evaluation_context="last_step",
                orientation=sign, face_measure=1., temporal_weight=dt,
                multiplicity=1, numerical_flux=flux, integrated_amount=sign*flux*dt,
            ))
    return rows, values, values + dt * periodic_rate(values, .03), dt


@pytest.mark.parametrize("field", ("numerical_flux", "face_measure", "temporal_weight", "integrated_amount"))
def test_nonfinite_ledger_fields_are_refused_before_metrics(field):
    rows, before, after, dt = rows_and_states()
    rows[0][field] = float("nan")
    with pytest.raises(AssertionError, match="nonfinite"):
        _ledger_metrics(rows, before, after, dt)
