"""Auxiliary codegen consumes only ProviderPack metadata."""
from __future__ import annotations

import pytest

from pops.codegen.component_provider_packs import resolve_component_provider_packs
from pops.physics._facade import Model


def _model() -> Model:
    model = Model("generic_aux")
    (density,) = model.conservative_vars("density")
    zero = 0.0 * density
    model.flux(x=[zero])
    model.eigenvalues(x=[zero])
    model.primitive_vars(density=density)
    model.conservative_from([density])
    coefficient = model.aux("collision_rate")
    model.source([-(coefficient * density)])
    return model


def test_emit_rejects_an_unbound_auxiliary_provider_pack() -> None:
    model = _model()
    with pytest.raises(ValueError, match="ProviderPack is absent"):
        model._m.emit_cpp_source(name="GenericAuxSource")


def test_standalone_flux_refuses_an_unbound_provider_consumer() -> None:
    model = Model("standalone_aux_flux")
    (density,) = model.conservative_vars("density")
    coefficient = model.aux("collision_rate")
    model.flux(x=[coefficient * density])
    model.eigenvalues(x=[density])

    with pytest.raises(ValueError, match="canonical Module"):
        model._m.emit_cpp()


def test_provider_pack_assigns_compact_slots_and_emits_consumer_local_reads() -> None:
    model = _model()
    packs = resolve_component_provider_packs(model.module)
    model.__pops_bind_component_provider_packs__(packs)

    assert packs.auxiliary.capacity == 1
    assert model._m._total_n_aux() == 1
    source = model._m.emit_cpp_source(name="GenericAuxSource")

    assert "static constexpr int n_aux = 1;" in source
    assert "flux_provider<0>()" in source
    assert "B_z" not in source
    assert "T_e" not in source
    assert "AUX_NAMED_BASE" not in source


def test_native_package_registers_routes_without_sealing_the_global_registry() -> None:
    model = _model()
    source = model.__pops_native_loader_source__(name="GenericAux", target="system")

    assert "pops_register_provider_routes" in source
    assert "install_prepared_auxiliary_provider" in source
    assert "install_auxiliary_consumer_plan" in source
    assert "seal_auxiliary_providers" not in source
    assert "pops_compiled_aux_provider_pack" in source
    assert "pops_compiled_aux_consumer_plans" in source


def test_amr_native_package_registers_routes_through_its_typed_hook() -> None:
    model = _model()
    source = model.__pops_native_loader_source__(name="GenericAuxAmr", target="amr_system")

    assert "pops_register_provider_routes_amr" in source
    assert "install_auxiliary_consumer_plan" in source
    assert "pops::AmrSystem<pops::kNativeDimension>* sys" in source
    assert "seal_auxiliary_providers" not in source


def test_native_package_accepts_an_empty_provider_pack() -> None:
    model = Model("no_aux")
    (density,) = model.conservative_vars("density")
    model.flux(x=[density])
    model.eigenvalues(x=[density])
    model.primitive_vars(density=density)
    model.conservative_from([density])

    source = model.__pops_native_loader_source__(name="NoAux", target="system")

    assert "static constexpr int n_aux" not in source
    assert "pops_register_provider_routes" in source
    assert "install_auxiliary_consumer_plan" in source
