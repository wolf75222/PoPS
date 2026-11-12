"""N3a, ROM-99: the b6 Rusanov, upwind and HLL faces (examples/api_v02/b6_numerics.py,
the parity oracles of N3 lot 1) are Python bodies.

Source-level checks only; the native parity runs in
tests/python/integration/runtime/test_api_v02_python_faces_native.py.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import pops
from pops._report import DiagnosticError
from pops.codegen.user_riemann_lowering import emit_user_face_policy, prepare_user_face_carrier
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from examples.api_v02.b6_numerics import hll, rusanov, upwind
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
    """Section 4.4: ``smax(uL, uR, n)``, here the speed of the transport along the face normal."""
    U, flux = _transport()
    default = rusanov(flux, U).riemann
    _same_face(default, rusanov(flux, U, smax=None).riemann)

    def along(uL, uR, n):
        return abs(.7*n[0] + .3*n[1])

    stated = authenticated_user_face(rusanov(flux, U, smax=along).riemann)
    _same_face(stated, riemann.User(
        body=lambda left, right, fl, fr, s, *, n: tuple(
            .5*(a + b) - .5*along(left, right, n)*(r - l)
            for l, r, a, b in zip(left, right, fl, fr, strict=True)),
        state=U, stability=lambda left, right, fl, fr, s, *, n: along(left, right, n)))
    assert stated.options["stability"] != default.options["stability"]
    assert stated.options["stability"] == riemann.User(
        body=lambda left, right, fl, fr, s: tuple(fl), state=U,
        stability=lambda left, right, fl, fr, s, *, n: along(left, right, n)).options["stability"]
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


def _generated_body(policy):
    """The generated body and recorded speed: from the density to the speed's refusal."""
    return policy[policy.index("State density{};"):policy.index("if (!std::isfinite(numerical_speed)")]


def _upwind_body(speed, left, right):
    """The row of 2.6.9: F* = s uL where s > 0, else s uR, component by component."""
    return tuple(pops.math.where(speed > 0, lambda l=l: speed*l, lambda r=r: speed*r)
                 for l, r in zip(left, right, strict=True))


def test_upwind_with_a_stated_speed_is_the_face_a_case_would_write_by_hand():
    """Section 4.4 and 2.6.9: ``speed(face)``, here the velocity of the case dotted with the face
    normal; F* = s uL where s > 0, else s uR, and the flux records |s| (adjudication section 20)."""
    U, flux = _transport(width=2)
    face = authenticated_user_face(upwind(
        flux, U, speed=lambda face: .7*face.normal[0] + .3*face.normal[1]).riemann)
    _same_face(face, riemann.User(
        body=lambda left, right, fl, fr, s, *, n: _upwind_body(.7*n[0] + .3*n[1], left, right),
        state=U, stability=lambda left, right, fl, fr, s, *, n: abs(.7*n[0] + .3*n[1])))
    # The face also gives the two traces of the realized state, and only those.
    by_trace = authenticated_user_face(upwind(
        flux, U, speed=lambda face: face.right(U)[0] - face.left(U)[0]).riemann)
    _same_face(by_trace, riemann.User(
        body=lambda left, right, fl, fr, s: _upwind_body(right[0] - left[0], left, right),
        state=U, stability=lambda left, right, fl, fr, s: abs(right[0] - left[0])))
    other = _transport(width=2)[0]
    with pytest.raises(TypeError, match="traces of the realized state"):
        upwind(flux, U, speed=lambda face: face.left(other)[0])
    with pytest.raises(TypeError, match="speed function"):
        upwind(flux, U, speed=.7)


def test_upwind_with_a_stated_speed_records_the_stated_speed_not_the_spectral_radius():
    """SD44, SD75: the step rule of ``upwind(speed=f)`` reads |f(face)|, never R9's radius."""
    U, flux = _transport()
    policy = _policy(authenticated_user_face(upwind(
        flux, U, speed=lambda face: -.5*face.normal[0] + .25*face.normal[1]).riemann))
    body = _generated_body(policy)
    assert "bound.value" not in body and "std::fabs(" in body
    # The traces are read, the physical fluxes F(uL).n and F(uR).n are not.
    assert "left.state[0]" in body and "right.state[0]" in body
    assert "left_density.value" not in body and "right_density.value" not in body


def _hll_body(lower, upper, left, right, fl, fr):
    return tuple(pops.math.where(lower >= 0, lambda a=a: a, lambda a=a, b=b, l=l, r=r:
                 pops.math.where(upper <= 0, lambda: b, lambda: (
                     upper*a - lower*b + lower*upper*(r - l))*(1./(upper - lower))))
                 for l, r, a, b in zip(left, right, fl, fr, strict=True))


def test_hll_takes_the_pair_of_r9_only():
    """2.6.9 states one way to give the pair of ``hll``: the declared speeds of R9 (I22, CF16).
    The pilot's ``speeds=`` option is gone; the parameter is refused, not ignored."""
    U, flux = _transport()
    with pytest.raises(TypeError, match="speeds"):
        hll(flux, U, speeds=lambda uL, uR, n: (-n[0], n[0]))
    with pytest.raises(TypeError):
        hll(flux, U, None)


def test_derived_hll_reads_the_signed_speeds_of_the_face():
    """``hll(F, Uh)``: (sL, sR) is the face input ``speeds`` (ROM-99), so the face needs the
    signed wave speeds of the model, and the native face evaluates them on both traces."""
    U, flux = _transport(width=2)
    face = authenticated_user_face(hll(flux, U).riemann)
    assert face.requirements["capabilities"] == (
        "physical_flux", "provider_pack", "stability_bound", "wave_speeds")
    _same_face(face, riemann.User(
        body=lambda left, right, fl, fr, s, *, speeds: _hll_body(*speeds, left, right, fl, fr),
        state=U, stability=lambda left, right, fl, fr, s, *, speeds: pops.math.maximum(
            abs(speeds[0]), abs(speeds[1]))))
    # The flux records max(|sL|, |sR|) of its pair (adjudication section 20), not the input s.
    policy = _policy(face)
    assert "bound.value" not in _generated_body(policy)
    assert policy.count("physical.signed_wave_speeds(") == 2
    assert "physical.signed_wave_speeds(left, face," in policy
    assert "physical.signed_wave_speeds(right, face," in policy
    # Read first and refused when a trace's interval is not finite or not ordered, in the order
    # of the native two-wave face: before the stability bound, the physical fluxes and the body.
    speeds, refusal = policy.index("signed_wave_speeds("), policy.index(
        "right_slowest <= right_fastest))\n      return pops::FluxEvaluation<State>::reject(\n"
        "          pops::RiemannFailureCause::kUserNonFiniteFlux);")
    body = policy.index("State density{};")
    assert speeds < refusal < policy.index("max_normal_stability_bound(") < \
        policy.index("physical_pair_failed(") < body
    assert policy.find("= wave_slowest;", body) > body < policy.find("= wave_fastest;", body)
    assert "face.normal[" not in policy


def test_derived_hll_is_refused_before_compile_when_the_model_declares_no_signed_speeds():
    """The refusal comes when the case is validated, not from the C++ compiler (ROM-99)."""
    from examples.api_v02.models_M01_M14 import build_m01

    unsigned = build_m01(scheme=hll)
    with pytest.raises(DiagnosticError, match="numerics.invalid_discretization_plan: FiniteVolume: "
                                              ".*reads the signed wave speeds.*declares none"):
        pops.validate(unsigned.case)
    assert pops.validate(build_m01(scheme=hll, signed_wave_speeds=True).case)
    # Rusanov reads no signed speed, so the same model needs none.
    assert pops.validate(build_m01(scheme=rusanov).case)
    # The lowering refuses too, on the model a package is generated from.
    model, U, flux = _transport_model()
    emitted = model.__pops_compiler_lowering__().emit_model
    numerics = SimpleNamespace(rates=(SimpleNamespace(method=hll(flux, U)),))
    with pytest.raises(ValueError, match="pops.compile: .*declares none"):
        prepare_user_face_carrier(emitted, numerics)
    assert getattr(getattr(emitted, "_m", emitted), "_user_face", None) is None


def test_the_cf15_graph_builds_and_reads_the_face_normal_only_when_asked():
    """The CF15 graph of ROM-99, committed failing with N3a, builds now.

    Section 4.4 writes the face speed ``smax(uL, uR, n)``, a stated upwind speed as the velocity
    dotted with the face normal, and the derived HLL from the declared speeds along n. A body and
    a stability bound now receive ``n`` (and ``speeds``) when they name it, and the generated face
    binds the normal of the native face context. The derived ``hll(F, Uh)`` is bitwise equal to
    ``riemann.HLL()`` on M03 and M16 in test_api_v02_python_faces_native.py.
    """
    U, flux = _transport()

    def upwind_along_the_normal(uL, uR, fL, fR, s, n):
        speed = .7*n[0] + .3*n[1]
        return tuple(pops.math.where(speed > 0, lambda a=a: a, lambda b=b: b)
                     for a, b in zip(fL, fR, strict=True))

    face = riemann.User(body=upwind_along_the_normal, state=U,
                        stability=lambda uL, uR, fL, fR, s, n: abs(.7*n[0] + .3*n[1]))
    policy = _policy(authenticated_user_face(face))
    assert "face.normal[0]" in policy and "face.normal[1]" in policy
    assert "face.normal[2]" not in policy and "signed_wave_speeds" not in policy
    # Asking for an input that no expression reads changes nothing: same identity, same C++.
    plain = riemann.User(body=lambda uL, uR, fL, fR, s: tuple(fL), state=U,
                         stability=lambda uL, uR, fL, fR, s: s)
    _same_face(plain, riemann.User(body=lambda uL, uR, fL, fR, s, *, n, speeds: tuple(fL),
                                   state=U, stability=lambda uL, uR, fL, fR, s, *, n: s))
    assert "face.normal[" not in _policy(plain)


def _declared_pairs(model, state):
    """The declared signed pair (sL, sR) along each canonical normal at every cell (R9)."""
    emitted = model.__pops_compiler_lowering__().emit_model
    samples = np.asarray(state, dtype=float).reshape(len(state), -1)
    return {axis: tuple(np.asarray(value, dtype=float)
                        for value in emitted.eval_wave_speeds(samples, {}, axis))
            for axis in ("x", "y")}


def test_m03_and_m16_declare_signed_speeds_that_the_spectral_radius_does_not_give():
    """The two structures of CF15 (ROM-99), against the scalar transport of M01.

    Without ``speeds``, the face of ``riemann.User`` receives one speed, the spectral radius
    s = max(|sL|, |sR|) of R9, and evaluates one body on both axes. On the transport, the declared
    pair along each normal is (s, s), so a body could state it from s. On M03 (v.n - c, v.n + c)
    and on M16 (the eigenvalues of the flux Jacobian along n, which have no closed form), at each
    case's own initial state, sL < 0 < sR on every cell, sL + sR is not zero, and the pair differs
    between the two normals: neither (s, s) nor (-s, s) gives it. This is why the derived
    ``hll(F, Uh)`` reads the face input ``speeds``, the declared pair along n united over the two
    traces, which only the native face can evaluate on M16.
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


def _burgers_case(scheme):
    """A case whose flux u^2/2 is not of the form F(1) u; returns the case, its flux and state."""
    frame = Rectangle("n3_burgers_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("n3_burgers", frame=frame)
    U = model.state("U", components=("u",))
    flux = model.flux("burgers", state=U, frame=frame,
        components={axis: (U[0]*U[0]/2,) for axis in frame.axes},
        waves={axis: (U[0],) for axis in frame.axes})
    rate = model.rate("balance", equation=pops.math.ddt(U) == -pops.math.div(flux))
    case = pops.Case("n3_burgers_case")
    block = case.block("fluid", model)
    plan = pops.numerics.DiscretizationPlan()
    plan.rates.add(rate, scheme(flux, U))
    case.numerics(plan, block=block)
    return case, flux, U


def test_upwind_without_speed_still_accepts_a_flux_that_is_not_linear_rom_106():
    """The failing graph of the CF15 card ROM-106, pinned.

    Section 2.6.9, row ``upwind``: without ``speed=``, a flux that is not F(1) u is refused
    (``RealizationMismatch``, naming ``rusanov()`` and ``hll()``). The face of ``riemann.User``
    has no input F(1).n and the library cannot read the form of its key, so ``upwind(F, Uh)`` is
    still section 4.4's ``rusanov()`` and validates on Burgers. When ROM-106 lands, this test
    becomes the refusal: ``pops.validate`` raises, with both names in its message.
    """
    case, flux, U = _burgers_case(upwind)
    assert pops.validate(case)
    _same_face(upwind(flux, U).riemann, rusanov(flux, U).riemann)
