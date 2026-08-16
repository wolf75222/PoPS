"""Exact provider and entropy-policy identity for the one HLLC/Roe pipeline."""
from __future__ import annotations

from fractions import Fraction

import pytest

from pops.codegen._compile_emit import model_hash
from pops.codegen.component_provider_packs import resolve_component_provider_packs
from pops.numerics.riemann import Harten, NoEntropyFix
from pops.numerics.riemann.providers import (
    ENTROPY_HARTEN,
    ENTROPY_NONE,
    ENTROPY_PROVIDER_OWNED,
    HLLC_FLUID_ROLES,
    ROE_DIRECT_ACTION,
    ROE_FLUID_ROLES,
    ROE_FLUX_JACOBIAN,
    authoring_provider_evidence,
)
from pops.physics import Custom, Density, Momentum
from pops.physics._facade import Model
from tests.python.support.physics_roles import X_AXIS, Y_AXIS


def _fluid_model(name: str) -> Model:
    model = Model(name)
    rho, mx, my = model.conservative_vars(
        "rho",
        "mx",
        "my",
        roles=[Density(), Momentum(X_AXIS), Momentum(Y_AXIS)],
    )
    u = model.primitive("u", mx / rho)
    v = model.primitive("v", my / rho)
    p = model.primitive("p", rho)
    model.flux(
        x=[mx, mx * u + p, mx * v],
        y=[my, my * u, my * v + p],
    )
    model.eigenvalues(x=[u - 1, u, u + 1], y=[v - 1, v, v + 1])
    model.primitive_vars(rho, u, v)
    model.conservative_from([rho, rho * u, rho * v])
    model.__pops_bind_component_provider_packs__(
        resolve_component_provider_packs(model.module)
    )
    return model


def _scalar_model(name: str) -> tuple[Model, object]:
    model = Model(name)
    (q,) = model.conservative_vars("q")
    model.flux(x=[q], y=[q])
    model.eigenvalues(x=[1], y=[1])
    model.primitive_vars(q)
    model.conservative_from([q])
    return model, q


def test_state_role_authoring_is_typed_and_lowers_once_to_exact_tokens() -> None:
    rejected = Model("legacy_string_roles")
    with pytest.raises(TypeError, match="typed ComponentRole"):
        rejected.conservative_vars("rho", roles=["Density"])
    assert rejected._m.cons_names == []
    assert rejected._m.cons_roles is None

    model = _fluid_model("exact_typed_roles")
    assert [type(role).__name__ for role in model._m.cons_roles] == [
        "Density",
        "Momentum",
        "Momentum",
    ]
    assert tuple(model.state_space().roles.values()) == (
        "density",
        "momentum:0",
        "momentum:1",
    )

    custom = Model("exact_custom_roles")
    custom.conservative_vars(
        "first", "second", roles=[Custom("q1"), Custom("q2")]
    )
    assert tuple(custom.state_space().roles.values()) == ("q1", "q2")


@pytest.mark.parametrize(
    "label", ("custom", "density", "momentum:0", "q,1", "q\x00", " q")
)
def test_public_custom_role_refuses_ambiguous_or_unserializable_labels(label: str) -> None:
    with pytest.raises(ValueError):
        Custom(label)


def test_custom_labels_do_not_satisfy_required_physical_metadata() -> None:
    model = Model("custom_metadata")
    model.conservative_vars("q1", "q2")
    model._m.set_gamma(1.4)

    with pytest.raises(ValueError, match="does not provide physical roles"):
        model._m._check_require_metadata(True, "production")


def test_hllc_and_role_roe_carry_exact_provider_and_typed_policy() -> None:
    model = _fluid_model("typed_role_roe")
    model.enable_hllc()
    model.enable_roe(entropy_fix=Harten(Fraction(1, 7)))

    evidence = authoring_provider_evidence(model)
    assert evidence.hllc_provider == HLLC_FLUID_ROLES
    assert evidence.roe_provider == ROE_FLUID_ROLES
    assert evidence.roe_entropy_policy == ENTROPY_HARTEN
    assert evidence.roe_entropy_delta == (
        '{"denominator":"7","kind":"rational","numerator":"1"}'
    )
    source = model._m.emit_cpp_brick()
    assert "const pops::HartenEntropyFix entropy_fix" in source
    assert "pops::Real(1) / pops::Real(7)" in source


def test_role_roe_policy_changes_emission_and_model_identity() -> None:
    default = _fluid_model("same_role_roe")
    default.enable_roe()
    no_fix = _fluid_model("same_role_roe")
    no_fix.enable_roe(entropy_fix=NoEntropyFix())

    assert model_hash(default._m) != model_hash(no_fix._m)
    default_source = default._m.emit_cpp_brick()
    no_fix_source = no_fix._m.emit_cpp_brick()
    assert "HartenEntropyFix" in default_source
    assert "HartenEntropyFix" not in no_fix_source
    assert authoring_provider_evidence(no_fix).roe_entropy_policy == ENTROPY_NONE


def test_direct_and_flux_jacobian_providers_remain_distinct_evidence() -> None:
    direct, q_direct = _scalar_model("direct_roe")
    direct.roe_dissipation(
        x=[direct.right(q_direct) - direct.left(q_direct)],
        y=[direct.right(q_direct) - direct.left(q_direct)],
    )
    direct_evidence = authoring_provider_evidence(direct)
    assert direct_evidence.roe_provider == ROE_DIRECT_ACTION
    assert direct_evidence.roe_entropy_policy == ENTROPY_PROVIDER_OWNED

    jacobian, _ = _scalar_model("jacobian_roe")
    jacobian.roe_from_jacobian(entropy_fix=NoEntropyFix())
    jacobian_evidence = authoring_provider_evidence(jacobian)
    assert jacobian_evidence.roe_provider == ROE_FLUX_JACOBIAN
    assert jacobian_evidence.roe_entropy_policy == ENTROPY_NONE
    assert direct_evidence != jacobian_evidence


def test_entropy_policy_refuses_untyped_magic_scalars() -> None:
    role = _fluid_model("untyped_role_entropy")
    with pytest.raises(TypeError, match="riemann.Harten"):
        role.enable_roe(entropy_fix=0.2)

    jacobian, _ = _scalar_model("untyped_jacobian_entropy")
    with pytest.raises(TypeError, match="riemann.Harten"):
        jacobian.roe_from_jacobian(entropy_fix=1.0e-6)
