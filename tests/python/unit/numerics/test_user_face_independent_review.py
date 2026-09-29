"""Independent adversarial checks for the authored numerical face contract."""
from copy import copy
from types import SimpleNamespace

import pytest

from pops.codegen.user_riemann_lowering import emit_user_face_policy
from pops.numerics import riemann
from pops.numerics.riemann.user import authenticated_user_face
from tests.python.unit.numerics.test_user_face_authoring import state


@pytest.mark.parametrize("width", (1, 2, 5, 17, 257))
def test_common_vector_body_has_no_small_component_catalogue(width):
    _, quantity = state(width)
    descriptor = riemann.User(
        body=lambda left, right, fl, fr, speed: .5 * (fl + fr) - .5 * speed * (right - left),
        stability=lambda left, right, fl, fr, speed: speed,
        state=quantity)
    assert authenticated_user_face(descriptor) is descriptor
    source = emit_user_face_policy(SimpleNamespace(_user_face=descriptor))
    assert "density[%d] =" % (width - 1) in source
    assert "canonical_evaluation(*this, physical, left, right, face)" in source
    assert "physical_pair_failed(" in source


@pytest.mark.parametrize("field,value", (
    ("capabilities", {"vector_face": True, "conservative_shared_face": False}),
    ("requirements", {"capabilities": (), "source_compiled": True}),
))
def test_changed_face_permissions_are_rejected_before_native_emission(field, value):
    _, quantity = state(2)
    descriptor = riemann.User(body=lambda left, right, fl, fr, speed: fl,
                              stability=lambda left, right, fl, fr, speed: speed,
                              state=quantity)
    altered = copy(descriptor)
    object.__setattr__(altered, field, value)
    with pytest.raises(ValueError, match="capabilities changed"):
        authenticated_user_face(altered)


def test_lazy_inactive_face_leaf_stays_inside_selected_common_ir_branch():
    from pops.math import where

    _, quantity = state(1)
    descriptor = riemann.User(
        body=lambda left, right, fl, fr, speed:
            (where(left[0] > 0, lambda: 1 / (right[0] - right[0]), lambda: fl[0]),),
        stability=lambda left, right, fl, fr, speed: speed,
        state=quantity)
    source = emit_user_face_policy(SimpleNamespace(_user_face=descriptor))
    body = source[source.index("State density{};"):]
    assert body.index("?") < body.index("right.state[0]")
    assert "std::isfinite" in body


def test_dissipative_body_can_author_its_numerical_stability_separately():
    """The physical speed alone is not a CFL certificate for arbitrary bodies."""
    _, quantity = state(1)
    descriptor = riemann.User(
        body=lambda left, right, fl, fr, speed: .5 * (fl + fr) - 10 * speed * (right - left),
        stability=lambda left, right, fl, fr, speed: 20 * speed,
        state=quantity)
    source = emit_user_face_policy(SimpleNamespace(_user_face=descriptor))
    assert "20" in source
    # With F=0 and physical speed=1, the flux is -10*(R-L).
    # At dt/dx=.5 a unit isolated peak becomes 1 - 2*10*.5 = -9,
    # whereas the authored bound 20 rejects this step (frequency*dt=10).
    dx, dt, diffusion = 1., .5, 10.
    left_face, right_face = -diffusion * (1. - 0.), -diffusion * (0. - 1.)
    assert 1. - dt / dx * (right_face - left_face) == -9.
    assert dt * 1. / dx <= 1.
    assert dt * 20. / dx > 1.
