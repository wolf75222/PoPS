"""Exact value identity of Spatial / FiniteVolume authoring selections."""
from decimal import Decimal
from fractions import Fraction

import pytest

import pops.runtime._engine_descriptors as engine
from pops.numerics.reconstruction import WENO5
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import HLL
from pops.numerics.riemann.waves import ExplicitPair
from pops.numerics.variables import Primitive
from pops.problem._detached import detached_frozen


def test_spatial_identity_covers_every_route_and_control_exactly():
    spatial = engine.Spatial(
        limiter=WENO5(epsilon=Decimal("1e-40")),
        flux=HLL(waves=ExplicitPair()),
        recon=Primitive(),
        positivity_floor=Fraction(1, 10**30),
        wave_speed_cache=True,
    )

    assert spatial.to_data() == {
        "schema_version": 1,
        "family": "finite_volume",
        "reconstruction": "weno5",
        "riemann": {
            "route": "hll",
            "external_id": None,
            "capability_contract": {
                "required_capabilities": [
                    "physical_flux", "provider_pack", "stability_bound", "wave_speeds",
                ],
                "wave_speed_provider": "explicit_pair",
            },
        },
        "variables": "primitive",
        "positivity_floor": {
            "kind": "rational", "numerator": "1", "denominator": str(10**30),
        },
        "wave_speed_cache": True,
        "waves_provider": "explicit_pair",
        "weno_epsilon": {"kind": "decimal", "value": "1E-40"},
    }
    assert spatial.weno_epsilon == Decimal("1e-40")

    same = engine.Spatial(
        limiter=WENO5(epsilon=Decimal("1e-40")),
        flux=HLL(waves=ExplicitPair()),
        recon=Primitive(),
        positivity_floor=Fraction(1, 10**30),
        wave_speed_cache=True,
    )
    assert same == spatial
    assert same.identity() == spatial.identity()


def test_spatial_identity_distinguishes_routes_and_exact_numeric_domains():
    rational = engine.Spatial(limiter=Minmod(), positivity_floor=Fraction(1, 10))
    decimal = engine.Spatial(limiter=Minmod(), positivity_floor=Decimal("0.1"))
    binary64 = engine.Spatial(limiter=Minmod(), positivity_floor=0.1)

    assert len({rational.identity(), decimal.identity(), binary64.identity()}) == 3
    assert rational != engine.Spatial(limiter=Minmod(), flux=HLL(),
                                    positivity_floor=Fraction(1, 10))


def test_external_riemann_identity_includes_the_registered_brick_id():
    from pops.descriptors import BrickDescriptor

    def external(brick_id):
        return BrickDescriptor(
            brick_id, "external_cpp", category="riemann",
            native_id="pops_external_flux", scheme="user",
            options={
                "library_path": "/tmp/external-riemann.so",
                "library_sha256": "0" * 64,
                "abi_version": 2,
                "abi_key": "pops.external-riemann/v2;scalar=f64;index=i32;periodicity=xy",
                "native_abi_key": "host-native-abi",
                "supported_layouts": ("uniform", "amr"),
                "model_identity": "compiled-model-hash",
            },
        )

    left = engine.Spatial(flux=external("acme.hll.v1"))
    right = engine.Spatial(flux=external("acme.hll.v2"))

    assert left.to_data()["riemann"] == {
        "route": "user", "external_id": "acme.hll.v1",
        "external_library_sha256": "0" * 64,
        "external_abi_key": "pops.external-riemann/v2;scalar=f64;index=i32;periodicity=xy",
        "external_native_abi_key": "host-native-abi",
        "external_model_identity": "compiled-model-hash",
        "capability_contract": {
            "required_capabilities": [], "wave_speed_provider": None,
        },
    }
    assert left.identity() != right.identity()


def test_detach_and_freeze_preserve_spatial_identity_and_seal_all_controls():
    authored = engine.Spatial(
        reconstruction=WENO5(epsilon=Fraction(1, 10**18)),
        flux=HLL(waves=ExplicitPair()),
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
