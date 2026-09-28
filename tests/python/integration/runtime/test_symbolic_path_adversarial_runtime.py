"""Native three-state path transport against a separate analytic FV oracle."""

import numpy as np
import pytest

import pops
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.unit.numerics.test_symbolic_path_adversarial import _three_component_case


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


@pytest.mark.parametrize("bent", [False, True])
def test_three_state_two_axis_path_matches_independent_one_step_oracle(
        isolated_native_cache, native_cxx, kokkos_root, bent):
    del isolated_native_cache, native_cxx, kokkos_root
    cells, dt = 8, 1.e-4
    case, layout, _ = _three_component_case(bent=bent)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    x = (np.arange(cells) + .5) / cells
    alpha = np.broadcast_to(2. + .1*np.sin(2*np.pi*x), (cells, cells)).copy()
    beta = np.broadcast_to(3. + .1*np.cos(2*np.pi*x), (cells, cells)).copy()
    gamma = np.broadcast_to(4. + .05*np.sin(2*np.pi*x + .4), (cells, cells)).copy()
    initial = np.stack((alpha, beta, gamma))

    right = np.roll(initial, -1, axis=2)
    jump = right - initial
    maximum_state = np.maximum(np.max(np.abs(initial), axis=0),
                               np.max(np.abs(right), axis=0))
    speed = 10. + 3.*maximum_state
    flux = .1*(initial + right) - .5*speed*jump
    integral = np.zeros_like(initial)
    integral[0] = 1.7*jump[0]*(.5*(initial[1] + right[1])
                                 + (jump[0]/3. if bent else 0.))
    integral[1] = -.4*.5*(initial[2] + right[2])*jump[2]
    integral[2] = .3*(.5*(initial[0] + right[0])*jump[1]
                        - (jump[0]**2/3. if bent else 0.))
    expected = initial + dt*cells*(np.roll(flux, 1, axis=2) - flux
                                   - .5*(integral + np.roll(integral, 1, axis=2)))

    runtime = pops.bind(artifact, initial_state={"triangle": initial},
                        resources={"execution_context": artifact_execution_context(artifact)})
    report = pops.run(runtime, t_end=dt, max_steps=1)
    assert report.accepted_steps == 1
    actual = np.asarray(runtime.state_global("triangle")).reshape(initial.shape)
    np.testing.assert_allclose(actual, expected, rtol=0., atol=8.e-14)
    assert not np.allclose(actual, initial, rtol=0., atol=1.e-10)
