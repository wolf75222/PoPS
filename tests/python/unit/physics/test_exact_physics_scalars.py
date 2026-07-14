"""ADC-652 exact scalar contracts across physics authoring and native lowering."""
from __future__ import annotations

import json
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from pops.codegen._compile_emit import model_hash
from pops.ir import ScalarLiteral, scalar_literal
from pops.model.manifest import coupling_operator_manifest
from pops.physics.coupling_presets import (
    ContractedCoupling,
    coupling_operator_args,
    thermal_exchange_preset,
)
from pops.physics._facade import Model
from pops.physics.multispecies import CoupledSource
from pops.physics._scalars import canonical_scalar_data, physics_scalar_cpp


def _constant_source(*values):
    src = CoupledSource("exact_constants")
    terms = [src.param("p%d" % index, value) for index, value in enumerate(values)]
    expr = terms[0]
    for term in terms[1:]:
        expr = expr + term
    src.add("gas", role="density", expr=expr)
    return src


def _fd_model(fd_eps, im_tol=None):
    model = Model("exact_wave_speed")
    q1, q2 = model.conservative_vars("q1", "q2")
    model.flux(
        x=[Fraction(1, 2) * q1 * q1, Fraction(1, 2) * q2 * q2],
        y=[Fraction(1, 2) * q2 * q2, Fraction(1, 2) * q1 * q1],
    )
    model.wave_speeds_from_jacobian(
        eig="fd", fd_eps=fd_eps, eig_max_iter=17, im_tol=im_tol)
    model.roe_from_jacobian()
    model.primitive_vars(q1, q2)
    model.conservative_from([q1, q2])
    return model


def test_coupled_constants_retain_domain_and_json_shape():
    decimal = Decimal("0.333333333333333333333333333333333333333333333333")
    compiled = _constant_source(Fraction(1, 3), Fraction(1, 3), decimal).frequency(
        Fraction(2, 3)).compile()

    # Equal rationals deduplicate; a numerically close Decimal remains a distinct authoring scalar.
    assert compiled.consts == [Fraction(1, 3), decimal]
    assert isinstance(compiled.consts[0], Fraction)
    assert isinstance(compiled.consts[1], Decimal)
    assert compiled.frequency == Fraction(2, 3)
    data = compiled.to_data()
    assert data["constants"] == [
        {"kind": "rational", "numerator": "1", "denominator": "3"},
        {"kind": "decimal", "value": str(decimal)},
    ]
    assert data["frequency"]["constant"] == {
        "kind": "rational", "numerator": "2", "denominator": "3"}
    json.dumps(data)

    native_args = coupling_operator_args(compiled)
    assert native_args[2] == [float(Fraction(1, 3)), float(decimal)]
    assert native_args[8] == float(Fraction(2, 3))
    # The explicit native conversion must not mutate the inspectable exact descriptor.
    assert compiled.consts == [Fraction(1, 3), decimal]
    assert compiled.frequency == Fraction(2, 3)
    reference = compiled.reference_terms({})[0][2]
    assert reference == pytest.approx(
        float(Fraction(1, 3)) + float(Fraction(1, 3)) + float(decimal))

    field_source = CoupledSource("decimal_field")
    density = field_source.block("gas").role("density")
    coefficient = field_source.param("coefficient", decimal)
    field_source.add("gas", role="density", expr=coefficient * density)
    field_value = field_source.compile().reference_terms(
        {("gas", "density"): np.array([2.0])})[0][2]
    np.testing.assert_allclose(field_value, np.array([2.0 * float(decimal)]))

    manifest = coupling_operator_manifest(compiled)
    assert manifest["frequency"]["constant_mu"] == {
        "kind": "rational", "numerator": "2", "denominator": "3"}
    decimal_manifest = coupling_operator_manifest(compiled, frequency=decimal)
    assert decimal_manifest["frequency"]["constant_mu"] == {
        "kind": "decimal", "value": str(decimal)}
    json.dumps(manifest)
    json.dumps(decimal_manifest)


def test_numerically_equal_domains_are_not_float_deduplicated():
    compiled = _constant_source(Fraction(1, 2), Decimal("0.5"), 0.5).compile()
    assert compiled.consts == [Fraction(1, 2), Decimal("0.5"), 0.5]
    assert [row["kind"] for row in compiled.to_data()["constants"]] == [
        "rational", "decimal", "binary64"]


def test_thermal_preset_keeps_rate_gamma_and_half_exact():
    preset = thermal_exchange_preset(
        "a",
        "b",
        Fraction(1, 7),
        Decimal("1.333333333333333333333333333333333333"),
        Fraction(5, 3),
    )
    compiled = preset.source.compile()
    assert compiled.consts == [
        Fraction(1, 7),
        Decimal("0.333333333333333333333333333333333333"),
        Fraction(1, 2),
        Fraction(2, 3),
    ]
    contracted = ContractedCoupling(preset.source, conserved=["energy"], frequency=Fraction(3, 5))
    assert contracted.frequency == Fraction(3, 5)
    assert "'kind': 'rational'" in repr(contracted)


def test_wave_speed_knobs_emit_and_hash_exact_literals():
    im_tol = Decimal("0.00000000000000000012345678901234567890123456789")
    model = _fd_model(Fraction(1, 3), im_tol)
    ws = model._m._ws_jacobian
    assert ws["fd_eps"] == Fraction(1, 3)
    assert ws["im_tol"] == im_tol
    source = model._m.emit_cpp_brick()
    assert "pops::Real(1) / pops::Real(3)" in source
    assert str(im_tol) in source

    same = _fd_model(Fraction(1, 3), im_tol)
    decimal_domain = _fd_model(Decimal("0.333333333333333333333333333333333333"), im_tol)
    assert model_hash(model._m) == model_hash(same._m)
    assert model_hash(model._m) != model_hash(decimal_domain._m)


def test_gamma_and_native_lowering_helpers_preserve_exact_scalars():
    gamma = Decimal("1.4000000000000000000000000000000000001")
    model = Model("exact_gamma")
    model.gamma(gamma)
    assert model._m.gamma == gamma
    assert str(gamma) in model._m._emit_metadata("ExactModel")

    algebraic = ScalarLiteral.algebraic(
        "sqrt(2)", cpp="std::sqrt(pops::Real(2))")
    assert physics_scalar_cpp(Fraction(1, 3), where="native.ratio") == \
        "(pops::Real(1) / pops::Real(3))"
    assert physics_scalar_cpp(algebraic, where="native.root") == "std::sqrt(pops::Real(2))"
    assert str(gamma) in physics_scalar_cpp(gamma, where="native.decimal")
    json.dumps({
        "ratio": canonical_scalar_data(Fraction(1, 3), where="native.ratio"),
        "root": canonical_scalar_data(algebraic, where="native.root"),
        "decimal": canonical_scalar_data(gamma, where="native.decimal"),
    })


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), Decimal("NaN")])
def test_non_real_or_nonfinite_constants_fail_at_declaration(value):
    src = CoupledSource("invalid")
    with pytest.raises((TypeError, ValueError), match="CoupledSource.param"):
        src.param("bad", value)
    with pytest.raises((TypeError, ValueError), match="CoupledSource.frequency"):
        src.frequency(value)
    with pytest.raises((TypeError, ValueError), match="set_gamma"):
        Model("invalid_gamma").gamma(value)


def test_units_targets_and_unsupported_algebraic_values_are_never_erased():
    unit = scalar_literal(Fraction(1, 3), unit="1/s")
    target = scalar_literal(Fraction(1, 3), target="custom::Real")
    algebraic = ScalarLiteral.algebraic(
        "sqrt(2)", cpp="std::sqrt(pops::Real(2))")
    src = CoupledSource("annotated")
    with pytest.raises(TypeError, match="unit annotation"):
        src.param("unit", unit)
    with pytest.raises(TypeError, match="target annotation"):
        src.param("target", target)
    with pytest.raises(TypeError, match="algebraic/custom"):
        src.param("algebraic", algebraic)
    with pytest.raises(TypeError, match="algebraic/custom"):
        src.add("gas", role="density", expr=algebraic)
    density = src.block("gas").role("density")
    with pytest.raises(TypeError, match="unit annotation"):
        src.add("gas", role="density", expr=density * unit)


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf")])
def test_positive_wave_speed_knobs_reject_bool_zero_negative_and_nonfinite(value):
    model = Model("invalid_wave_speed")
    q = model.conservative_vars("q")[0]
    model.flux(x=[q], y=[q])
    with pytest.raises((TypeError, ValueError), match="fd_eps"):
        model.wave_speeds_from_jacobian(eig="fd", fd_eps=value)

    model2 = Model("invalid_im_tol")
    q2 = model2.conservative_vars("q")[0]
    model2.flux(x=[q2], y=[q2])
    with pytest.raises((TypeError, ValueError), match="im_tol"):
        model2.wave_speeds_from_jacobian(im_tol=value)


def test_positive_wave_speed_knobs_reject_annotations_and_algebraic_values():
    values = [
        scalar_literal(Fraction(1, 3), unit="dimensionless"),
        scalar_literal(Fraction(1, 3), target="custom::Real"),
        ScalarLiteral.algebraic("sqrt(2)", cpp="std::sqrt(pops::Real(2))"),
    ]
    for index, value in enumerate(values):
        model = Model("invalid_annotated_wave_%d" % index)
        q = model.conservative_vars("q")[0]
        model.flux(x=[q], y=[q])
        with pytest.raises(TypeError):
            model.wave_speeds_from_jacobian(eig="fd", fd_eps=value)
