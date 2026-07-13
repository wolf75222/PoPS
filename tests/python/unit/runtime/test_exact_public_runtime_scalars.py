"""ADC-652: public runtime descriptors retain exact scalars until native lowering."""
from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

import pops
from pops.ir import ScalarLiteral, scalar_to_native
from pops.runtime._bricks_model import _native_to_brick


def test_root_model_bricks_retain_fraction_and_decimal_authoring_values():
    state = pops.FluidState.isothermal(
        cs2=Fraction(7, 10), vacuum_floor=Decimal("1e-40"))
    transport = pops.ExB(B0=Fraction(5, 2))
    source = pops.PotentialForce(charge=Decimal("-1.0000000000000000000001"))
    elliptic = pops.BackgroundDensity(alpha=Fraction(1, 3), n0=Decimal("0.125"))

    assert state.cs2 == Fraction(7, 10) and isinstance(state.cs2, Fraction)
    assert state.vacuum_floor == Decimal("1e-40")
    assert transport.B0 == Fraction(5, 2)
    assert source.charge == Decimal("-1.0000000000000000000001")
    assert elliptic.alpha == Fraction(1, 3)
    assert elliptic.n0 == Decimal("0.125")


def test_modelspec_is_the_explicit_native_real_boundary():
    model = pops.Model(
        state=pops.FluidState.isothermal(cs2=Fraction(7, 10)),
        transport=pops.IsothermalFlux(),
        source=pops.PotentialForce(charge=Fraction(-1, 3)),
        elliptic=pops.BackgroundDensity(alpha=Decimal("0.2"), n0=Fraction(1, 8)),
    )

    assert model.cs2 == float(Fraction(7, 10))
    assert model.qom == float(Fraction(-1, 3))
    assert model.alpha == float(Decimal("0.2"))
    assert model.n0 == float(Fraction(1, 8))


def test_hybrid_native_brick_codegen_keeps_exact_payload_until_cpp():
    value = Decimal("1.123456789012345678901234567890123456789")
    with localcontext() as context:
        context.prec = 5
        brick = _native_to_brick(pops.ExB(B0=value), "hyperbolic")
        source = brick.emit("ExactExB")

    assert brick.fields["B0"] == value
    assert "1.123456789012345678901234567890123456789" in source


def test_root_time_and_spatial_descriptors_retain_exact_real_controls():
    imex = pops.IMEX(
        newton_rel_tol=Fraction(1, 10**12), newton_abs_tol=Decimal("1e-40"),
        newton_fd_eps=Fraction(1, 10**7), newton_damping=Decimal("0.875"))
    spatial = pops.Spatial(positivity_floor=Fraction(1, 10**20))

    assert imex.newton_rel_tol == Fraction(1, 10**12)
    assert imex.newton_abs_tol == Decimal("1e-40")
    assert imex.newton_fd_eps == Fraction(1, 10**7)
    assert imex.newton_damping == Decimal("0.875")
    assert spatial.positivity_floor == Fraction(1, 10**20)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: pops.FluidState(gamma=True),
        lambda: pops.ExB(B0=True),
        lambda: pops.Spatial(positivity_floor=True),
        lambda: pops.Spatial(wave_speed_cache=1),
        lambda: pops.Explicit(substeps=True),
        lambda: pops.Explicit(stride=1.0),
        lambda: pops.Explicit(ssprk3=1),
        lambda: pops.IMEX(substeps=True),
        lambda: pops.IMEX(newton_max_iters=True),
        lambda: pops.IMEX(newton_diagnostics=1),
    ],
)
def test_public_numeric_descriptors_refuse_bool_and_lossful_integer_coercions(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), Decimal("NaN")])
def test_public_runtime_real_controls_refuse_non_finite_values(bad):
    for factory in (
        lambda: pops.ExB(B0=bad),
        lambda: pops.FluidState(cs2=bad),
        lambda: pops.IMEX(newton_rel_tol=bad),
        lambda: pops.Spatial(positivity_floor=bad),
    ):
        with pytest.raises((TypeError, ValueError)):
            factory()


def test_scalar_to_native_refuses_units_and_foreign_targets():
    unit = ScalarLiteral.from_value(Fraction(1, 3), unit="m/s")
    foreign = ScalarLiteral.from_value(Fraction(1, 3), target="Real128")

    assert scalar_to_native(Fraction(1, 3), where="test") == float(Fraction(1, 3))
    with pytest.raises(TypeError, match="cannot lower unit"):
        scalar_to_native(unit, where="test")
    with pytest.raises(TypeError, match="not the native pops::Real ABI"):
        scalar_to_native(foreign, where="test")
