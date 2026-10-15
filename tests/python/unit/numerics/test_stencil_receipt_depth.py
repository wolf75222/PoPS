"""Deep public SSA compositions retain the stencil receipt boundary exactly."""

import pytest

from pops.codegen.program_emit_stencil_flux import accepted_stencil_quadrature
from pops.numerics.stencil_flux import StencilFlux
from examples.api_v02.methods_T01_T12 import _storage


def _deep_commit(*, faces, fresh=False):
    _, _, _, _, _, p, q, _ = _storage(
        "deep_receipt", ("q",), (1.0,), cells=(8, 4), sampling="cell_average"
    )
    value = q.n
    if faces:
        flux = StencilFlux(
            dimension=2,
            components=1,
            radius=1,
            body=lambda sample, spacing, axis: (
                (sample(tuple(-int(d == axis) for d in range(2)), 0),),
                1.0 / spacing[axis],
            ),
        )
        rate = p.stencil_flux_divergence("root_faces", value, flux=flux)
        value = p.value("candidate", value + p.dt * rate, at=q.next.point)
    for _ in range(1500):
        value = p.product(payload=value)["payload"]
    if fresh:
        rate = p.rate("fresh_after_deep_predictor", (value[0] ** 2,), of=value)
        value = p.value("fresh_candidate", q.n + p.dt * rate, at=q.next.point)
    elif not faces:
        value = p.value("copy_candidate", value, at=q.next.point)
    p.commit(q.next, value)
    return p


def test_deep_nonstencil_ancestry_is_not_recursion_limited():
    assert accepted_stencil_quadrature(_deep_commit(faces=False)) == ()


def test_deep_projection_cannot_hide_a_stencil_transfer():
    with pytest.raises(ValueError, match="hides stencil faces outside affine quadrature"):
        accepted_stencil_quadrature(_deep_commit(faces=True))


def test_fresh_rate_cannot_absorb_spatial_ancestry_hidden_by_deep_projections():
    # Projection ancestry does not itself provide the authoritative stage boundary.
    with pytest.raises(ValueError, match="cannot absorb a spatial or unclassified rate"):
        _deep_commit(faces=True, fresh=True)
