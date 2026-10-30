"""C12 face bodies retain a typed vector Expr graph and exact source identity."""
import pops
import pytest

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.numerics import riemann
from pops.representations import Conservative
from pops.spaces import CellState


def state(width=2):
    frame = Rectangle("domain", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("face_owner", frame=frame)
    components = tuple("q%d" % index for index in range(width))
    return model, model.state("U", components=components, representation=Conservative(),
                              space=CellState(frame=frame))


def test_vector_rusanov_body_is_authored_as_common_expr_algebra():
    _, U = state(3)
    def face(left, right, flux_left, flux_right, speed):
        return .5 * (flux_left + flux_right) - .5 * speed * (right - left)

    descriptor = riemann.User(body=face, state=U, stability=lambda l, r, fl, fr, s: s)
    from pops.numerics.riemann.user import authenticated_user_face

    assert authenticated_user_face(descriptor) is descriptor
    assert len(descriptor.expression) == 4
    assert descriptor.options["width"] == 3
    assert descriptor.options["source_identity"] == riemann.User(
        body=face, state=U, stability=lambda l, r, fl, fr, s: s).options[
        "source_identity"]


def test_cross_component_body_keeps_shape_and_identity():
    _, U = state(2)
    first = riemann.User(
        body=lambda left, right, flux_left, flux_right, speed:
            (flux_left[0] + right[1], flux_right[1] - left[0]), state=U,
        stability=lambda l, r, fl, fr, s: s)
    second = riemann.User(
        body=lambda left, right, flux_left, flux_right, speed:
            (flux_left[0] + right[0], flux_right[1] - left[0]), state=U,
        stability=lambda l, r, fl, fr, s: s)
    assert first.options["source_identity"] != second.options["source_identity"]


def test_scalar_state_still_uses_indexed_vector_and_tuple_return():
    _, U = state(1)
    descriptor = riemann.User(
        body=lambda left, right, flux_left, flux_right, speed: (flux_left[0],), state=U,
        stability=lambda l, r, fl, fr, s: s)
    assert descriptor.options["width"] == 1


def test_source_face_route_and_emitted_policy_share_body_identity():
    from types import SimpleNamespace
    from pops.codegen.user_riemann_lowering import emit_user_face_policy
    from pops.runtime._bricks_scheme import Spatial

    _, U = state(2)
    descriptor = riemann.User(
        body=lambda left, right, flux_left, flux_right, speed:
            .5 * (flux_left + flux_right) - .5 * speed * (right - left), state=U,
        stability=lambda l, r, fl, fr, s: s)
    route = Spatial(flux=descriptor)
    identity = descriptor.options["source_identity"]
    assert str(route.flux) == "source_face:" + identity
    emitted = emit_user_face_policy(SimpleNamespace(_user_face=descriptor))
    assert identity in emitted
    assert "pops::StabilityBound{numerical_speed" in emitted
    assert "pops::detail::physical_pair_failed" in emitted


def test_face_body_rejects_wrong_shape_and_foreign_variable():
    _, U = state(2)
    with pytest.raises(TypeError, match="tuple.*state component"):
        riemann.User(body=lambda l, r, fl, fr, s: (fl[0],), state=U,
                     stability=lambda l, r, fl, fr, s: s)
    from pops._ir.expr import Var
    with pytest.raises(ValueError, match="foreign variable"):
        riemann.User(body=lambda l, r, fl, fr, s: (Var("foreign", "face_input"), fr[1]),
                     state=U, stability=lambda l, r, fl, fr, s: s)


def test_authored_face_requires_numerical_stability_and_tracks_its_identity():
    _, U = state(1)
    body = lambda l, r, fl, fr, s: (fl[0],)
    with pytest.raises(TypeError, match="stability="):
        riemann.User(body=body, state=U)
    one = riemann.User(body=body, state=U, stability=lambda l, r, fl, fr, s: s)
    two = riemann.User(body=body, state=U, stability=lambda l, r, fl, fr, s: 2 * s)
    assert one.options["source_identity"] != two.options["source_identity"]
    with pytest.raises(TypeError, match="one scalar Expr"):
        riemann.User(body=body, state=U,
                     stability=lambda l, r, fl, fr, s: (s,))


def test_named_face_inputs_reach_only_a_body_that_names_them():
    """ROM-99: ``n`` and ``speeds`` are passed by name after the five face inputs (CF15)."""
    from types import SimpleNamespace
    from pops.codegen.user_riemann_lowering import emit_user_face_policy
    from pops.numerics.riemann.user import reads_signed_wave_speeds

    _, U = state(2)
    stability = lambda l, r, fl, fr, s: s
    normal = riemann.User(body=lambda l, r, fl, fr, s, *, n: tuple(n[0] * f for f in fl),
                          state=U, stability=stability)
    policy = emit_user_face_policy(SimpleNamespace(_user_face=normal))
    assert "face.normal[0]" in policy and "face.normal[1]" not in policy
    assert not reads_signed_wave_speeds(normal) and "signed_wave_speeds" not in policy
    # The fifth parameter is the speed, whatever its name.
    fifth = riemann.User(body=lambda l, r, fl, fr, n: tuple(n * f for f in fl), state=U,
                         stability=stability)
    assert fifth.options["source_identity"] == riemann.User(
        body=lambda l, r, fl, fr, s: tuple(s * f for f in fl), state=U,
        stability=stability).options["source_identity"]
    # An input named and not read changes neither the identity nor the policy.
    unread = riemann.User(body=lambda l, r, fl, fr, s, n, speeds: fl, state=U,
                          stability=stability)
    plain = riemann.User(body=lambda l, r, fl, fr, s: fl, state=U, stability=stability)
    assert unread.options["source_identity"] == plain.options["source_identity"]
    assert unread.requirements == plain.requirements
    assert emit_user_face_policy(SimpleNamespace(_user_face=unread)) == \
        emit_user_face_policy(SimpleNamespace(_user_face=plain))


def test_a_face_that_reads_the_signed_speeds_requires_their_declaration():
    """ROM-99: reading ``speeds`` adds the model capability wave_speeds, kept authenticated."""
    from copy import copy
    from pops.numerics.riemann.user import (
        authenticated_user_face, reads_signed_wave_speeds, require_declared_wave_speeds,
    )

    model, U = state(1)
    face = riemann.User(
        body=lambda l, r, fl, fr, s, *, speeds: (speeds[1] * fl[0] - speeds[0] * fr[0],),
        state=U, stability=lambda l, r, fl, fr, s: s)
    bound = riemann.User(body=lambda l, r, fl, fr, s: (fl[0],), state=U,
                         stability=lambda l, r, fl, fr, s, *, speeds: speeds[1] - speeds[0])
    for value in (face, bound):
        assert reads_signed_wave_speeds(value)
        assert value.requirements["capabilities"] == (
            "physical_flux", "provider_pack", "stability_bound", "wave_speeds")
        with pytest.raises(ValueError, match="here: .*reads the signed wave speeds.*declares none"):
            require_declared_wave_speeds(value, model, where="here")
        altered = copy(value)
        object.__setattr__(altered, "requirements", {
            "capabilities": ("physical_flux", "provider_pack", "stability_bound"),
            "source_compiled": True})
        with pytest.raises(ValueError, match="capabilities changed"):
            authenticated_user_face(altered)
