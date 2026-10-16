import numpy as np
import pytest

from examples.api_v02.relaxed_composite_reference import rotation_reference, trajectory_metrics

CELLS = (4, 4)
BOXES = ((0, (0, 0), (3, 3)), (1, (0, 0), (3, 7)))


def fields(states, masks):
    result = []
    for level in (0, 1):
        value = np.zeros((3, 4*2**level, 4*2**level))
        value[:2] = states[level, :, None, None]
        value[2] = .5
        result.append(value)
    result[0][:2, ~masks[0]] = states[1, :, None]
    return result


@pytest.mark.parametrize("ratio,nominal,end", [(1, 1., 1.), (2, 1., 1.), (2, .4, .7), (2, .75, 1.3)])
def test_reference_reaches_endpoint_and_detects_noop_or_component_permutation(ratio, nominal, end):
    state, masks, steps, time = rotation_reference(BOXES, cells=CELLS, ratio=ratio,
                                                 requested=nominal, end=end)
    assert time == end and len(steps) > 1
    assert steps[0] == min(nominal, end)
    assert steps[-1] < steps[0]
    actual = fields(state, masks)
    arguments = dict(cells=CELLS, ratio=ratio, requested=nominal, end=end, initial_passive=.5)
    assert trajectory_metrics(actual, BOXES, **arguments)["passed"]
    unchanged = fields(np.array([[1., 0.], [1., 0.]]), masks)
    assert not trajectory_metrics(unchanged, BOXES, **arguments)["passed"]
    permuted = [value[[1, 0, 2]] for value in actual]
    assert not trajectory_metrics(permuted, BOXES, **arguments)["passed"]
    perturbed_mass = [value.copy() for value in actual]
    for value in perturbed_mass:
        value[2] += 1.e-5
    assert not trajectory_metrics(perturbed_mass, BOXES, **arguments)["passed"]


def test_first_nominal_step_matches_independent_heun_polynomial():
    # With synchronous levels the relaxation map is the Cayley rotation.
    h = .5
    gamma = 1/(1+h*h/4)
    state, _, steps, time = rotation_reference(BOXES, cells=CELLS, ratio=1,
                                              requested=h, end=gamma*h)
    # The clipped request is smaller than h, so it must NOT return the legacy
    # nominal one-step endpoint (.88235294, .47058824).
    assert len(steps) > 1 and steps[0] == gamma*h and time == gamma*h
    legacy = np.array([1-gamma*h*h/2, gamma*h])
    assert np.max(abs(state[0]-legacy)) > 1.e-3
