"""Typed InputAux/DerivedAux ProviderPack and native-launcher contracts."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pops._ir import ValueExpr
from pops.codegen._compile_emit import _emit_auxiliary_route_registration
from pops.codegen.component_provider_packs import (
    ComponentProviderPacks,
    compact_auxiliary_provider_pack,
    consumer_provider_plan,
    resolve_component_provider_packs,
)
from pops.fields import AuxiliaryBoundary, DerivedAux, InputAux
from pops.model.provider_pack import ComponentContract, ComponentKey, ProviderEntry, ProviderPack
from pops.model import Module


def _module() -> Module:
    module = Module("auxiliary_producers")
    imposed = module.aux_field("imposed")
    temperature = module.aux_field("temperature")
    module.aux_provider(InputAux(
        module.aux_handle(imposed),
        boundary=AuxiliaryBoundary(width=2, kind="foextrap"),
    ))
    module.aux_provider(DerivedAux(
        module.aux_handle(temperature),
        2.0 * ValueExpr(module.aux_handle(imposed)),
        boundary=AuxiliaryBoundary(width=2, kind="foextrap"),
    ))
    return module


def test_input_and_derived_aux_routes_are_exact_and_emit_native_launcher() -> None:
    module = _module()
    packs = resolve_component_provider_packs(module)
    route_rows = list(packs.auxiliary_routes.values())
    assert [row["kind"] for row in route_rows] == ["input", "derived"]
    derived = route_rows[1]
    assert [key.component for key in derived["dependencies"]] == ["imposed"]
    assert [packs.auxiliary.declared_entry(key).slot for key in packs.auxiliary] == [0, 1]

    carrier = SimpleNamespace(owner_path=module.owner_path)
    packs.attach(carrier)
    source = _emit_auxiliary_route_registration(carrier)
    assert "AuxiliaryProviderKind::derived" in source
    assert "Provider::launcher_type::trusted_extension" in source
    assert ".address.group" in source
    assert ".address.component" in source
    assert "candidate->find" in source
    assert "Kokkos::parallel_for" in source
    assert '"imposed"' in source
    assert "halo[axis] = 2" in source
    assert "BoundaryKind::first_order_extrapolation" in source
    assert "std::vector<Output>{{" in source
    assert ", 0}}" not in source  # the package-local slot never crosses the DSO ABI


def test_aux_producer_rejects_duplicate_target_and_foreign_handle() -> None:
    module = Module("auxiliary_provider_failures")
    field = module.aux_field("coefficient")
    handle = module.aux_handle(field)
    module.aux_provider(InputAux(handle))
    with pytest.raises(ValueError, match="one producer"):
        module.aux_provider(DerivedAux(handle, 1.0 * ValueExpr(handle)))

    other = Module("other_auxiliary_provider")
    foreign = other.aux_handle(other.aux_field("coefficient"))
    with pytest.raises(ValueError, match="another Module registry"):
        module.aux_provider(InputAux(foreign))


def test_derived_aux_cycle_is_refused_before_codegen() -> None:
    module = Module("cyclic_auxiliary_producers")
    left = module.aux_handle(module.aux_field("left"))
    right = module.aux_handle(module.aux_field("right"))
    module.aux_provider(DerivedAux(left, ValueExpr(right)))
    module.aux_provider(DerivedAux(right, ValueExpr(left)))

    with pytest.raises(ValueError, match="contains a cycle"):
        resolve_component_provider_packs(module)


def test_auxiliary_boundary_rejects_ambiguous_physical_policy() -> None:
    with pytest.raises(ValueError, match="requires a value"):
        AuxiliaryBoundary(kind="dirichlet")
    with pytest.raises(ValueError, match="valid only for dirichlet"):
        AuxiliaryBoundary(kind="inherit", value=0.0)


def test_native_route_registers_only_owned_outputs_but_consumes_foreign_keys() -> None:
    module = Module("auxiliary_consumer")
    own = str(module.owner_path.canonical())
    foreign = "model/foreign_producer"
    contract = ComponentContract("auxiliary", "cell", None, "cell", "cell_scalar")
    own_key = ComponentKey(own, "aux", "material", "local_input")
    foreign_key = ComponentKey(foreign, "aux", "material", "remote_input")
    complete = ProviderPack((
        (own_key, contract, ProviderEntry("runtime_input", True, 0)),
        (foreign_key, contract, ProviderEntry("runtime_input", True, 0)),
    ))
    plan = consumer_provider_plan(complete)
    packs = ComponentProviderPacks(
        complete=complete,
        by_operator={},
        physical_flux=complete,
        auxiliary=compact_auxiliary_provider_pack(complete),
        auxiliary_routes={},
        auxiliary_route_metadata=(),
        consumer_plans={"consumer": plan},
        physical_flux_plan=plan,
    )
    carrier = SimpleNamespace(owner_path=module.owner_path)
    packs.attach(carrier)
    source = _emit_auxiliary_route_registration(carrier)

    assert source.count("install_prepared_auxiliary_provider(Provider{") == 1
    assert '"remote_input"' in source
    assert "ConsumerValue" in source
