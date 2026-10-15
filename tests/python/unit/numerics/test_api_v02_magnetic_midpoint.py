"""Equation/oracle and source admission tests; no substituted native execution."""
import math
import numpy as np
import pops
import pytest

from examples.api_v02.models_magnetic_midpoint import build_m09_midpoint
from tests.python.support import magnetic_midpoint_oracle as reference


@pytest.mark.parametrize("cells", ((3, 3), (4, 3), (5, 4)))
@pytest.mark.parametrize("omega", (0., 4., 1.e5))
def test_original_midpoint_energy_internal_energy_and_gauge(cells, omega):
    lengths = (2*math.pi, 3*math.pi)
    old = reference.initial(cells, lengths=lengths, omega=omega)
    new, residual = reference.source(old, .03, lengths=lengths, alpha=1.)
    assert residual < 2.e-11
    assert abs(reference.energy(new, lengths)-reference.energy(old, lengths)) < 1.e-10
    internal = lambda q: q[3]-.5*(q[1]**2+q[2]**2)/q[0]
    np.testing.assert_allclose(internal(new), internal(old), rtol=0, atol=3.e-15)
    assert np.max(np.abs(reference.gauss(old, lengths))) < 1.e-15
    assert abs(new[4].mean()) < 1.e-14
    assert np.max(abs(new[1:3]-old[1:3])) > 1.e-6


def test_full_strang_evolves_density_and_preserves_energy_with_visible_gauss_defect():
    old = reference.initial((4, 3))
    new, residuals = reference.evolve(old, .003, 3)
    assert max(residuals) < 1.e-12
    assert np.max(abs(new[0]-old[0])) > 1.e-5
    assert abs(reference.energy(new)-reference.energy(old)) < 1.e-11
    # The source paper discusses this splitting defect. Receiving it separately
    # prevents a replacement Poisson solve from passing as the midpoint method.
    assert np.max(abs(reference.gauss(new))) > 1.e-6


@pytest.mark.parametrize("solver", ("schur", "monolithic"))
def test_whole_program_admission_uses_only_native_algebra(solver):
    spec = build_m09_midpoint(cells=(3, 3), solver=solver)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    assert not plan.blocks  # the complete state lives in the accepted Continuation
    encoded = spec.case._time._serialize(include_provenance=False)
    assert "continuation_publish" in str(encoded)
    assert "finite_map" in str(encoded)
    assert "python_callback" not in str(encoded)


@pytest.mark.parametrize("kwargs", ({"cells": (2, 3)}, {"dt": 0.}, {"alpha": -1.},
                                    {"solver": "CG"}, {"omega": float("nan")}))
def test_invalid_equation_data_rejected(kwargs):
    with pytest.raises(ValueError):
        build_m09_midpoint(**kwargs)
