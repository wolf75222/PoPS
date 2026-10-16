"""An independent Gram oracle checks extension beyond the original order catalogue."""
import numpy as np
import pytest

from examples.api_v02.models_M15_orders import (
    hankel_margins, hyqmom_flux, initial_moments, reference_flux, reference_spectrum,
)


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_parametric_closure_agrees_with_independent_gram_system(order):
    raw = initial_moments(order, 16, "smooth")
    np.testing.assert_allclose(np.asarray(hyqmom_flux(raw)), reference_flux(raw),
                               atol=2e-12, rtol=2e-12)
    assert min(float(np.min(v)) for v in hankel_margins(raw)) > 0
    for column in raw.T:
        spectrum = reference_spectrum(column)
        assert len(spectrum) == order+1
        assert np.min(np.diff(spectrum)) > 1e-5
