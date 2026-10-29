"""N3a: the Rusanov, upwind and HLL faces of pops.lib.numerics are Python bodies, not native routes.

Source-level checks only; the native parity runs in
tests/python/integration/runtime/test_api_v02_python_faces_native.py.
"""
from types import SimpleNamespace

import pytest

import pops
from pops.codegen.user_riemann_lowering import emit_user_face_policy
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.lib.numerics import hll, rusanov, spectral_radius, upwind
from pops.numerics import reconstruction, riemann
from pops.numerics.riemann.user import authenticated_user_face


def _transport(width=1):
    frame = Rectangle("n3a_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("n3a_transport", frame=frame)
    U = model.state("U", components=tuple("q%d" % k for k in range(width)))
    speeds = (.7, .3)
    flux = model.flux("transport", state=U, frame=frame,
        components={axis: tuple(a*q for q in U)
                    for axis, a in zip(frame.axes, speeds, strict=True)},
        waves={axis: (a,)*width for axis, a in zip(frame.axes, speeds, strict=True)})
    return U, flux


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
    _same_face(default, rusanov(flux, U, smax=spectral_radius).riemann)
    doubled = rusanov(flux, U, smax=lambda uL, uR, fL, fR, s: 2*s).riemann
    assert doubled.options["source_identity"] != default.options["source_identity"]
    with pytest.raises(TypeError, match="speed function"):
        rusanov(flux, U, smax=1.)


def test_rusanov_keeps_a_given_reconstruction():
    U, flux = _transport()
    method = rusanov(flux, U, reconstruction=reconstruction.MUSCL(reconstruction.Minmod()))
    assert method.reconstruction == reconstruction.MUSCL(reconstruction.Minmod())


def test_upwind_without_speed_is_rusanov_with_the_spectral_radius():
    """For a flux linear in u, Rusanov with the R9 speed is the upwind flux (section 4.4)."""
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


def test_hll_needs_signed_wave_speeds_that_the_user_face_does_not_receive():
    """CF15 failing graph: the derived HLL speeds need a face input the frozen engine lacks.

    ``riemann.User`` gives the body the spectral radius only. HLL needs the slowest and fastest
    signed speeds of each trace along the face normal (R9), which depend on the axis of the
    face. M03 (Euler) and M16 (HyQMOM 15 moments, native HLL from the Jacobian) both need them.
    When the face input lands, this test becomes the parity test of the derived default.
    """
    U, flux = _transport()
    with pytest.raises(NotImplementedError, match="signed wave speeds"):
        hll(flux, U)
    with pytest.raises(TypeError, match="speeds"):
        hll(flux, U, speeds=(0., 1.))
    with pytest.raises(TypeError, match="pair"):
        hll(flux, U, speeds=lambda uL, uR, fL, fR, s: s)


def test_m03_and_m16_have_no_python_hll_without_the_face_input():
    """The two structures of CF15 for the signed-speed face input (N3a report).

    M03 is the stiffened-gas Euler system (waves u.n - c, u.n + c); M16 is the HyQMOM 15-moment
    B.1 transport, whose native HLL takes its signed speeds from the flux Jacobian. Both
    builders accept the Python HLL and both refuse it for the same missing face input.
    """
    from examples.api_v02.models_M01_M14 import build_m03
    from examples.api_v02.models_M15_M18 import build_m16

    for build in (build_m03, build_m16):
        with pytest.raises(NotImplementedError, match="signed wave speeds"):
            build(scheme=hll)
