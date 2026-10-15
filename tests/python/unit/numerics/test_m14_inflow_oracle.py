"""Independent M14 face oracle distinguishes prescribed trace from mirrored ghosts."""
import numpy as np

from examples.api_v02.models_M14 import CAPACITANCE, RESERVOIR, _reference


def test_reservoir_trace_uses_riemann_state_not_centered_ghost():
    initial = np.broadcast_to((.5*RESERVOIR)[:, None, None], (4, 1, 4)).copy()
    dt = .001
    actual = _reference(initial, dt=dt, steps=1, capacitance=CAPACITANCE)[0]
    wrong = _reference(initial, dt=dt, steps=1, capacitance=CAPACITANCE,
                       mirrored_reservoir=True)[0]
    # Independent hand calculation: alpha=3 and left=RESERVOIR, right=.5*RESERVOIR.
    # Rusanov ion/electron number fluxes are 2.25 and 3 at the inlet, 1 and 1.5 outside.
    expected_ion = initial[0, 0].copy()
    expected_electron = initial[2, 0].copy()
    expected_ion[0] += dt*4*(2.25-1)
    expected_electron[0] += dt*4*(3-1.5)
    np.testing.assert_allclose(actual[0, 0], expected_ion, atol=1.e-15, rtol=0)
    np.testing.assert_allclose(actual[2, 0], expected_electron, atol=1.e-15, rtol=0)
    assert np.max(abs(actual-wrong)) > .001
    assert abs(np.mean(actual[0, 0]-initial[0, 0])-dt*(2.25-1)) < 1.e-16
    assert abs(np.mean(actual[2, 0]-initial[2, 0])-dt*(3-1.5)) < 1.e-16
