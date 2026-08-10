"""Typed InputAux/DerivedAux ProviderPack and native-launcher contracts."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pops._ir import ValueExpr
from pops._ir.expr import Var
from pops.codegen._artifact_freeze import seal_attributes
from pops.codegen._compile_emit import _emit_auxiliary_route_registration
from pops.codegen.program_emit_kernels import ProgramProviderPlans
from pops.codegen.component_provider_packs import (
    ComponentProviderPacks,
    auxiliary_component_slot,
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


def test_provider_pack_reattachment_normalizes_only_artifact_container_freezing() -> None:
    """A list-to-tuple seal is storage-neutral, but changed metadata remains a conflict."""
    packs = resolve_component_provider_packs(_module())
    carrier = SimpleNamespace()
    packs.attach(carrier)
    seal_attributes(carrier)

    # Artifact sealing recursively turns ProviderPack JSON lists into immutable tuples.  The same
    # exact logical authority must remain idempotently attachable later in the compile lifecycle.
    assert isinstance(carrier._component_provider_metadata["entries"], tuple)
    packs.attach(carrier)

    tampered = packs.complete.to_data()
    tampered["entries"][0]["provider"]["producer"] = "different/provider"
    object.__setattr__(carrier, "_component_provider_metadata", tampered)
    with pytest.raises(ValueError, match="conflicting component-provider pack"):
        packs.attach(carrier)


def test_amr_auxiliary_hook_is_typed_and_distinct_from_the_system_hook() -> None:
    module = _module()
    packs = resolve_component_provider_packs(module)
    carrier = SimpleNamespace(owner_path=module.owner_path)
    packs.attach(carrier)

    source = _emit_auxiliary_route_registration(carrier, target="amr_system")

    assert "pops_register_auxiliary_routes_amr" in source
    assert "pops::AmrSystem<pops::kNativeDimension>* sys" in source
    assert "pops::System<pops::kNativeDimension>* sys" not in source


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


def test_auxiliary_boundary_and_derived_freshness_are_part_of_the_typed_route() -> None:
    """Halo policy and freshness belong to the typed provider, never to a named setter."""
    module = Module("fresh_derived_auxiliary")
    imposed = module.aux_handle(module.aux_field("imposed"))
    nonlinear = module.aux_handle(module.aux_field("nonlinear"))
    module.aux_provider(InputAux(
        imposed,
        boundary=AuxiliaryBoundary(width=3, kind="dirichlet", value=2.0),
    ))
    module.aux_provider(DerivedAux(nonlinear, ValueExpr(imposed) * ValueExpr(imposed)))
    packs = resolve_component_provider_packs(module)
    carrier = SimpleNamespace(owner_path=module.owner_path)
    packs.attach(carrier)
    source = _emit_auxiliary_route_registration(carrier)

    assert 'halo[axis] = 3' in source
    assert 'BoundaryKind::dirichlet, std::optional<pops::Real>{2.0}' in source
    assert 'AuxiliaryEvaluationEvent::before_residual' in source
    assert 'AuxiliaryFreshness::evaluation' in source
    assert 'Kokkos::parallel_for' in source


def test_provider_pack_failure_does_not_publish_a_partial_route() -> None:
    """A rejected duplicate cannot alter a previously built immutable provider authority."""
    owner = "model/transactional_auxiliary"
    key = ComponentKey(owner, "aux", "material", "coefficient")
    contract = ComponentContract("auxiliary", "cell", None, "cell", "cell_scalar")
    original = ProviderPack(((key, contract, ProviderEntry("runtime_input", True, 0)),))

    with pytest.raises(ValueError, match="duplicate component provider"):
        ProviderPack((
            (key, contract, ProviderEntry("runtime_input", True, 0)),
            (key, contract, ProviderEntry("derived:coefficient", True, 1)),
        ))

    assert original.lookup(key).to_data() == {
        "producer": "runtime_input", "availability": True, "slot": 0,
    }


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


def test_program_consumer_plan_is_first_use_local_and_owner_qualified() -> None:
    """A Program node never bakes a provider storage slot or reuses an operator plan."""
    contract = ComponentContract("auxiliary", "cell", None, "cell", "cell_scalar")
    left = ComponentKey("model/left", "aux", "material", "electric_x")
    right = ComponentKey("model/right", "field", "electrostatic", "electric_y")
    pack = ProviderPack((
        (left, contract, ProviderEntry("runtime_input", True, 7)),
        (right, contract, ProviderEntry("field/electrostatic", True, 2)),
    ))
    impl = SimpleNamespace(
        _auxiliary_provider_pack=pack,
        _provider_components=("electric_x", "electric_y"),
    )
    plans = ProgramProviderPlans()
    binding = plans.bind(
        impl,
        (Var("electric_y", "aux") + Var("electric_x", "aux"),),
        "case/program/17",
    )

    assert binding == {
        "qid": "case/program/17",
        "count": 2,
        "slots": {"electric_y": 0, "electric_x": 1},
    }
    cpp = plans.cpp_install("system")
    assert 'ConsumerPlan{"case/program/17"' in cpp
    assert 'Key{"model/right", "field", "electrostatic", "electric_y"}' in cpp
    assert 'Key{"model/left", "aux", "material", "electric_x"}' in cpp
    assert "ConsumerValue" in cpp
    assert "ProviderEntry" not in cpp
    assert "}}}, 7}" not in cpp  # package-local producer slots never cross the ABI


def test_provider_pack_compacts_auxiliary_consumers_without_state_or_parameter_carriers() -> None:
    owner = "model/demo"
    contract = ComponentContract("auxiliary", "cell", None, "cell", "cell_scalar")
    complete = ProviderPack((
        (ComponentKey(owner, "state", "U", "density"), contract,
         ProviderEntry("initial_state", True, 0)),
        (ComponentKey(owner, "param", "material", "gamma"), contract,
         ProviderEntry("runtime_parameter", True, 0)),
        (ComponentKey(owner, "aux", "material", "collision_rate"), contract,
         ProviderEntry("runtime_input", True, 9)),
        (ComponentKey(owner, "field", "electrostatic", "potential"), contract,
         ProviderEntry("field_solver", True, 17)),
    ))

    compact = compact_auxiliary_provider_pack(complete)
    assert [entry.slot for entry in (compact.declared_entry(key) for key in compact)] == [0, 1]
    assert auxiliary_component_slot(compact, owner_qid=owner, name="collision_rate") == 0
    assert auxiliary_component_slot(compact, owner_qid=owner, name="potential") == 1
    plan = consumer_provider_plan(complete)
    assert [row["key"]["space_kind"] for row in plan] == ["aux", "field"]
    assert [row["consumer_slot"] for row in plan] == [0, 1]


def test_provider_pack_keeps_homonymous_owner_qualified_components_distinct() -> None:
    contract = ComponentContract("auxiliary", "cell", None, "cell", "cell_scalar")
    electron = "model/electron"
    ion = "model/ion"
    pack = ProviderPack((
        (ComponentKey(electron, "aux", "material", "temperature"), contract,
         ProviderEntry("runtime_input", True, 0)),
        (ComponentKey(ion, "field", "thermodynamic", "temperature"), contract,
         ProviderEntry("field_solver", True, 0)),
    ))

    compact = compact_auxiliary_provider_pack(pack)
    assert len(compact) == 2
    assert auxiliary_component_slot(compact, owner_qid=electron, name="temperature") == 0
    assert auxiliary_component_slot(compact, owner_qid=ion, name="temperature") == 1
