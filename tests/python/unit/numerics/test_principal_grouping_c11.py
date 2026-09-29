"""C11: the same symmetric principal law must not lose cross-state inputs."""

import numpy as np
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D


def _independent_rusanov_rhs(matrix, values, *, speed=None):
    """A separate periodic, first-order method-of-lines oracle in x."""
    speed = float(max(abs(np.linalg.eigvalsh(matrix)))) if speed is None else speed
    right = np.roll(values, -1, axis=1)
    face = .5 * (matrix @ values + matrix @ right) - .5 * speed * (right - values)
    return values.shape[1] * (np.roll(face, 1, axis=1) - face)


def test_offdiagonal_symmetric_law_has_observable_cross_derivative():
    matrix = np.array(((1., .5), (.5, 1.)))
    values = np.array(((1., 0., 0., 0.), (0., 0., 1., 0.)))
    speed = 1.5  # spectral bound of the full symmetric matrix in both probes
    coupled = _independent_rusanov_rhs(matrix, values, speed=speed)
    uncoupled = _independent_rusanov_rhs(np.diag(np.diag(matrix)), values, speed=speed)
    np.testing.assert_allclose(coupled, ((-6., 4., 0., 2.), (0., 2., -6., 4.)),
                               rtol=0., atol=1.e-15)
    assert np.max(np.abs(coupled - uncoupled)) >= 1.
    np.testing.assert_allclose(coupled.sum(axis=1), 0., rtol=0., atol=1.e-15)


def test_two_scalar_states_capture_both_inputs_of_one_principal_system():
    """Expected C11 public contract; currently red on per-state flux signatures."""
    frame = Rectangle("c11-principal-box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    x, y = frame.axes

    vector_model = pops.Model("c11-vector", frame=frame)
    vector_state = vector_model.state("U", components=("u", "v"))
    vu, vv = vector_state
    vector_flux = vector_model.flux("F", frame=frame, state=vector_state,
        components={x: (vu + .5 * vv, .5 * vu + vv), y: (0., 0.)})
    assert vector_model.module.operator_handle(vector_flux.reg_name).signature.inputs == (
        vector_state.space,)

    scalar_model = pops.Model("c11-scalars", frame=frame)
    u_state = scalar_model.species("first", state=("u",))
    v_state = scalar_model.species("second", state=("v",))
    u, v = u_state["u"], v_state["v"]
    first_flux = scalar_model.flux("F_first", frame=frame, state=u_state,
        components={x: (u + .5 * v,), y: (0.,)})
    second_flux = scalar_model.flux("F_second", frame=frame, state=v_state,
        components={x: (.5 * u + v,), y: (0.,)})
    expected_inputs = (u_state.space, v_state.space)
    first = scalar_model.module.operator_handle(first_flux.reg_name)
    second = scalar_model.module.operator_handle(second_flux.reg_name)
    assert first.signature.inputs == expected_inputs, (
        "C11: F_first reads v but its public operator signature omits the v state")
    assert second.signature.inputs == expected_inputs, (
        "C11: F_second reads u but its public operator signature omits the u state")
