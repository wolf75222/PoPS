"""Exact value identity of Spatial / FiniteVolume authoring selections."""
from decimal import Decimal
from fractions import Fraction

import pytest

import pops
from pops.numerics.reconstruction import WENO5
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import HLL
from pops.numerics.riemann.waves import ExplicitPair
from pops.numerics.variables import Primitive
from pops.problem._detached import detached_frozen


def test_spatial_identity_covers_every_route_and_control_exactly():
    spatial = pops.FiniteVolume(
        limiter=WENO5(epsilon=Decimal("1e-40")),
        riemann=HLL(waves=ExplicitPair()),
        variables=Primitive(),
        positivity_floor=Fraction(1, 10**30),
        wave_speed_cache=True,
    )

    assert spatial.to_data() == {
        "schema_version": 1,
        "family": "finite_volume",
        "reconstruction": "weno5",
        "riemann": {"route": "hll", "external_id": None},
        "variables": "primitive",
        "positivity_floor": {
            "kind": "rational", "numerator": "1", "denominator": str(10**30),
        },
        "wave_speed_cache": True,
        "waves_provider": "explicit_pair",
        "weno_epsilon": {"kind": "decimal", "value": "1E-40"},
    }
    assert spatial.weno_epsilon == Decimal("1e-40")

    same = pops.FiniteVolume(
        limiter=WENO5(epsilon=Decimal("1e-40")),
        riemann=HLL(waves=ExplicitPair()),
        variables=Primitive(),
        positivity_floor=Fraction(1, 10**30),
        wave_speed_cache=True,
    )
    assert same == spatial
    assert same.identity() == spatial.identity()


def test_spatial_identity_distinguishes_routes_and_exact_numeric_domains():
    rational = pops.Spatial(limiter=Minmod(), positivity_floor=Fraction(1, 10))
    decimal = pops.Spatial(limiter=Minmod(), positivity_floor=Decimal("0.1"))
    binary64 = pops.Spatial(limiter=Minmod(), positivity_floor=0.1)

    assert len({rational.identity(), decimal.identity(), binary64.identity()}) == 3
    assert rational != pops.Spatial(limiter=Minmod(), flux=HLL(),
                                    positivity_floor=Fraction(1, 10))


def test_external_riemann_identity_includes_the_registered_brick_id():
    from pops.descriptors import BrickDescriptor

    def external(brick_id):
        return BrickDescriptor(
            brick_id, "external_cpp", category="riemann",
            native_id="pops_external_flux", scheme="user",
        )

    left = pops.Spatial(flux=external("acme.hll.v1"))
    right = pops.Spatial(flux=external("acme.hll.v2"))

    assert left.to_data()["riemann"] == {
        "route": "user", "external_id": "acme.hll.v1",
    }
    assert left.identity() != right.identity()


def test_detach_and_freeze_preserve_spatial_identity_and_seal_all_controls():
    authored = pops.FiniteVolume(
        reconstruction=WENO5(epsilon=Fraction(1, 10**18)),
        riemann=HLL(waves=ExplicitPair()),
        positivity_floor=Decimal("1e-24"),
        wave_speed_cache=True,
    )
    expected_data = authored.to_data()
    expected_identity = authored.identity()

    detached = detached_frozen(authored)

    assert detached is not authored
    assert detached.to_data() == expected_data
    assert detached.identity() == expected_identity
    assert detached == authored
    with pytest.raises(RuntimeError, match="frozen by AuthoringSnapshot"):
        detached.weno_epsilon = Decimal("2e-24")


def test_internal_token_lowering_preserves_optional_identity_controls():
    from pops.runtime._bricks_scheme import Spatial

    spatial = Spatial._from_tokens(
        "weno5", "hll", "primitive",
        positivity_floor=Fraction(1, 100),
        wave_speed_cache=True,
        waves_provider="explicit_pair",
        weno_epsilon=Decimal("1e-20"),
    )

    assert spatial.to_data()["waves_provider"] == "explicit_pair"
    assert spatial.to_data()["weno_epsilon"] == {"kind": "decimal", "value": "1E-20"}
    with pytest.raises(TypeError, match="wave_speed_cache"):
        Spatial._from_tokens("none", "rusanov", "conservative", wave_speed_cache=1)
