"""Multidimensional geometric-conservation witnesses independent of generated code."""
import numpy as np
import pytest

from tests.python.support.ale_nd_oracle import ale_reference


@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
def test_moving_tensor_geometric_conservation_and_shared_faces(dimension):
    options = dict(shape=(5,)*dimension, amplitudes=(.02,)*dimension,
                   velocities=(.3,)*dimension)
    result, records = ale_reference(**options)
    assert all(row["gcl"] < 2e-16 and row["mass"] < 1e-14 and row["uniform"] < 2e-15
               and row["cfl_margin"] > 0 for row in records)
    np.testing.assert_allclose(result["uniform"], .8, atol=2e-15, rtol=0)
    mutant, bad = ale_reference(**options, old_volume_mutant=True)
    assert max(row["uniform"] for row in bad) > 1e-5
    assert np.max(abs(mutant["density"]-result["density"])) > 1e-5
