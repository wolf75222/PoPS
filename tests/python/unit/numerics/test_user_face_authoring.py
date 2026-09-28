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

    descriptor = riemann.User(body=face, state=U)
    from pops.numerics.riemann.user import authenticated_user_face

    assert authenticated_user_face(descriptor) is descriptor
    assert len(descriptor.expression) == 3
    assert descriptor.options["width"] == 3
    assert descriptor.options["source_identity"] == riemann.User(body=face, state=U).options[
        "source_identity"]


def test_cross_component_body_keeps_shape_and_identity():
    _, U = state(2)
    first = riemann.User(
        body=lambda left, right, flux_left, flux_right, speed:
            (flux_left[0] + right[1], flux_right[1] - left[0]), state=U)
    second = riemann.User(
        body=lambda left, right, flux_left, flux_right, speed:
            (flux_left[0] + right[0], flux_right[1] - left[0]), state=U)
    assert first.options["source_identity"] != second.options["source_identity"]


def test_scalar_state_still_uses_indexed_vector_and_tuple_return():
    _, U = state(1)
    descriptor = riemann.User(
        body=lambda left, right, flux_left, flux_right, speed: (flux_left[0],), state=U)
    assert descriptor.options["width"] == 1


def test_source_face_route_and_emitted_policy_share_body_identity():
    from types import SimpleNamespace
    from pops.codegen.user_riemann_lowering import emit_user_face_policy
    from pops.runtime._bricks_scheme import Spatial

    _, U = state(2)
    descriptor = riemann.User(
        body=lambda left, right, flux_left, flux_right, speed:
            .5 * (flux_left + flux_right) - .5 * speed * (right - left), state=U)
    route = Spatial(flux=descriptor)
    identity = descriptor.options["source_identity"]
    assert str(route.flux) == "source_face:" + identity
    emitted = emit_user_face_policy(SimpleNamespace(_user_face=descriptor))
    assert identity in emitted
    assert "FluxEvaluation<State>::ok(density, bound)" in emitted
    assert "pops::detail::physical_pair_failed" in emitted


def test_face_body_rejects_wrong_shape_and_foreign_variable():
    _, U = state(2)
    with pytest.raises(TypeError, match="tuple.*state component"):
        riemann.User(body=lambda l, r, fl, fr, s: (fl[0],), state=U)
    from pops._ir.expr import Var
    with pytest.raises(ValueError, match="foreign variable"):
        riemann.User(body=lambda l, r, fl, fr, s: (Var("foreign", "face_input"), fr[1]),
                     state=U)
