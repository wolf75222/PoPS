"""N3a: the Rusanov, upwind and HLL faces of pops.lib.numerics are Python bodies, not native routes.

Source-level checks only; the native parity runs in
tests/python/integration/runtime/test_api_v02_python_faces_native.py.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import pops
from pops.codegen.user_riemann_lowering import emit_user_face_policy
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.lib.numerics import hll, rusanov, upwind
from pops.numerics import reconstruction, riemann
from pops.numerics.riemann.user import authenticated_user_face


def _transport_model(width=1, *, signed_wave_speeds=False):
    frame = Rectangle("n3a_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("n3a_transport", frame=frame)
    U = model.state("U", components=tuple("q%d" % k for k in range(width)))
    speeds = (.7, .3)
    flux = model.flux("transport", state=U, frame=frame,
        components={axis: tuple(a*q for q in U)
                    for axis, a in zip(frame.axes, speeds, strict=True)},
        waves={axis: (a,)*width for axis, a in zip(frame.axes, speeds, strict=True)})
    if signed_wave_speeds:
        model.wave_speeds(flux, frame=frame,
                          values={axis: (a, a) for axis, a in zip(frame.axes, speeds, strict=True)})
    return model, U, flux


def _transport(width=1):
    return _transport_model(width)[1:]


def _policy(face):
    return emit_user_face_policy(SimpleNamespace(_user_face=face))


def _same_face(face, by_hand):
    """CF4: same body identity, hence the same generated C++ face policy."""
    assert face.options["source_identity"] == by_hand.options["source_identity"]
    assert _policy(face) == _policy(by_hand)


def test_rusanov_is_the_face_a_case_would_write_by_hand():
    U, flux = _transport(width=3)
    method = rusanov(flux, U)
    face = authenticated_user_face(method.riemann)
    assert face.scheme == "source_face" and face.brick_type == "generated"
    assert face.native_id != riemann.Rusanov().native_id
    _same_face(face, riemann.User(
        body=lambda left, right, fl, fr, s: tuple(
            .5*(a + b) - .5*s*(r - l) for l, r, a, b in zip(left, right, fl, fr, strict=True)),
        state=U, stability=lambda left, right, fl, fr, s: s))
    assert method.flux is flux and method.variables.options["state"] is U
    assert method.reconstruction == reconstruction.FirstOrder()


def test_rusanov_speed_option_changes_the_body_and_the_recorded_speed():
    U, flux = _transport()
    default = rusanov(flux, U).riemann
    _same_face(default, rusanov(flux, U, smax=lambda uL, uR, fL, fR, s: s).riemann)
    doubled = authenticated_user_face(rusanov(flux, U, smax=lambda uL, uR, fL, fR, s: 2*s).riemann)
    _same_face(doubled, riemann.User(
        body=lambda left, right, fl, fr, s: tuple(
            .5*(a + b) - .5*(2*s)*(r - l) for l, r, a, b in zip(left, right, fl, fr, strict=True)),
        state=U, stability=lambda left, right, fl, fr, s: 2*s))
    assert doubled.options["stability"] != default.options["stability"]
    with pytest.raises(TypeError, match="speed function"):
        rusanov(flux, U, smax=1.)


def test_rusanov_keeps_a_given_reconstruction():
    U, flux = _transport()
    method = rusanov(flux, U, reconstruction=reconstruction.MUSCL(reconstruction.Minmod()))
    assert method.reconstruction == reconstruction.MUSCL(reconstruction.Minmod())


def test_upwind_without_speed_is_rusanov_with_the_spectral_radius():
    """For a scalar flux linear in u, Rusanov with the R9 speed is the upwind flux (section 4.4)."""
    U, flux = _transport(width=2)
    _same_face(upwind(flux, U).riemann, rusanov(flux, U).riemann)


def test_upwind_with_a_stated_speed_is_the_face_a_case_would_write_by_hand():
    U, flux = _transport(width=2)
    face = authenticated_user_face(upwind(flux, U, speed=lambda uL, uR, fL, fR, s: s).riemann)
    _same_face(face, riemann.User(
        body=lambda left, right, fl, fr, s: tuple(
            pops.math.where(s > 0, lambda a=a: a, lambda b=b: b)
            for a, b in zip(fl, fr, strict=True)),
        state=U, stability=lambda left, right, fl, fr, s: s))
    with pytest.raises(TypeError, match="speed function"):
        upwind(flux, U, speed=.7)


def test_hll_with_stated_speeds_is_the_face_a_case_would_write_by_hand():
    U, flux = _transport(width=2)
    def speeds(uL, uR, fL, fR, s):
        return -s, 2*s

    face = authenticated_user_face(hll(flux, U, speeds=speeds).riemann)
    assert face.native_id != riemann.HLL().native_id

    def body(left, right, fl, fr, s):
        lower, upper = speeds(left, right, fl, fr, s)
        return tuple(pops.math.where(lower >= 0, lambda a=a: a, lambda a=a, b=b, l=l, r=r:
                     pops.math.where(upper <= 0, lambda: b, lambda: (
                         upper*a - lower*b + lower*upper*(r - l))*(1./(upper - lower))))
                     for l, r, a, b in zip(left, right, fl, fr, strict=True))

    _same_face(face, riemann.User(body=body, state=U,
                                  stability=lambda left, right, fl, fr, s: s))


def test_hll_without_stated_speeds_is_refused_by_the_library():
    """The library's own refusal (fluxes.py): ``hll(F, Uh)`` has no derived default yet.

    It fires before any model is read, for every flux. The CF15 evidence is in the next two tests.
    """
    U, flux = _transport()
    with pytest.raises(NotImplementedError, match="no derived default"):
        hll(flux, U)
    with pytest.raises(TypeError, match="speeds"):
        hll(flux, U, speeds=(0., 1.))
    with pytest.raises(TypeError, match="pair"):
        hll(flux, U, speeds=lambda uL, uR, fL, fR, s: s)


def test_a_user_face_body_cannot_read_the_face_normal():
    """CF15 failing graph (ROM-99): ``riemann.User`` gives its body no face normal.

    Section 4.4 writes the face speed ``smax(uL, uR, n)``, a stated upwind speed as the velocity
    dotted with the face normal, and the derived HLL from the declared speeds along n. The body
    and the stability bound of ``riemann.User`` receive the five face inputs only, so a face that
    reads n is refused, here on the scalar transport. When the face input lands, this graph
    builds, and the test becomes the parity test of the derived ``hll(F, Uh)`` against
    ``riemann.HLL()`` on M03 and M16.
    """
    U, flux = _transport()

    def upwind_along_the_normal(uL, uR, fL, fR, s, n):
        speed = .7*n[0] + .3*n[1]
        return tuple(pops.math.where(speed > 0, lambda a=a: a, lambda b=b: b)
                     for a, b in zip(fL, fR, strict=True))

    with pytest.raises(TypeError, match="missing 1 required positional argument: 'n'"):
        riemann.User(body=upwind_along_the_normal, state=U,
                     stability=lambda uL, uR, fL, fR, s: s)
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'n'"):
        riemann.User(body=lambda uL, uR, fL, fR, s: tuple(fL), state=U,
                     stability=lambda uL, uR, fL, fR, s, n: s)


def _declared_pairs(model, state):
    """The declared signed pair (sL, sR) along each canonical normal at every cell (R9)."""
    emitted = model.__pops_compiler_lowering__().emit_model
    samples = np.asarray(state, dtype=float).reshape(len(state), -1)
    return {axis: tuple(np.asarray(value, dtype=float)
                        for value in emitted.eval_wave_speeds(samples, {}, axis))
            for axis in ("x", "y")}


def test_m03_and_m16_declare_signed_speeds_that_the_face_inputs_do_not_give():
    """The two structures of CF15 (ROM-99), against the scalar transport of M01.

    The face of ``riemann.User`` receives one speed, the spectral radius s = max(|sL|, |sR|) of
    R9, and evaluates one body on both axes. On the transport, the declared pair along each normal
    is (s, s), so a body states it from s (``m01_hll``). On M03 (v.n - c, v.n + c) and on M16 (the
    eigenvalues of the flux Jacobian along n, which have no closed form), at each case's own
    initial state, sL < 0 < sR on every cell, sL + sR is not zero, and the pair differs between
    the two normals: neither (s, s) nor (-s, s) gives it, and the derived ``hll(F, Uh)`` needs the
    face input. M03 still has a Python HLL whose interval the case states from its own physics
    (``m03_hll``, bitwise equal to ``riemann.HLL()`` in test_api_v02_python_faces_native.py).
    """
    from examples.api_v02.models_M01_M14 import _domain, _euler, build_m03
    from examples.api_v02.models_M15_M18 import _m16_model, build_m16

    transport = _transport_model(signed_wave_speeds=True)[0]
    for lower, upper in _declared_pairs(transport, np.ones((1, 4))).values():
        speed = np.maximum(np.abs(lower), np.abs(upper))
        assert np.array_equal(lower, speed) and np.array_equal(upper, speed)

    frame, _ = _domain("M03_periodic", (16, 8))
    m03 = pops.Model("M03_authored_stiffened_Euler", frame=frame)
    _euler(m03, frame, gamma=1.4, pressure_offset=.2, signed_wave_speeds=True)
    m16 = _m16_model()[0]
    m03_state = next(iter(build_m03(signed_wave_speeds=True).initial_values.values()))
    m16_state = next(iter(build_m16().initial_values.values()))
    for name, model, state in (("M03", m03, m03_state), ("M16", m16, m16_state)):
        pairs = _declared_pairs(model, state)
        for lower, upper in pairs.values():
            assert lower.max() < 0 < upper.min(), name
            assert np.max(np.abs(lower + upper)) > 1e-3, name
        assert np.max(np.abs(pairs["x"][0] - pairs["y"][0])) > 1e-3, name
