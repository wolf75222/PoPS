"""bind_emitter_provider_packs reattaches the exact Module pack, never two attrs."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pops.codegen.component_provider_packs import (
    EMITTER_CARRIER_ATTRS,
    bind_emitter_provider_packs,
    emitter_carrier_snapshot,
    require_emitter_provider_carrier,
    resolve_component_provider_packs,
)
from pops.codegen.program_emit_kernels import _model_impl
from pops.fields import AuxiliaryBoundary, InputAux
from pops.model import Module
from pops.model.provider_pack import ProviderPack
from pops.physics._facade import Model
from pops.time import Program
from tests.python.unit.runtime._typed_program import (
    _source_model_for_owner,
    typed_program_state,
)


_CARRIER_ATTRS = EMITTER_CARRIER_ATTRS


class _ModuleEmitter:
    """Minimal compiler emitter that binds packs onto itself."""

    def __init__(self, module: Module) -> None:
        self.module = module

    def __pops_bind_component_provider_packs__(self, packs: object) -> None:
        packs.attach(self)


def _routed_module() -> Module:
    module = Module("emitter_pack_routes")
    module.state_space("U", ("density",))
    imposed = module.aux_field("imposed")
    module.aux_provider(InputAux(
        module.aux_handle(imposed), boundary=AuxiliaryBoundary(width=1, kind="foextrap")))
    return module


def _aux_model() -> Model:
    model = Model("emitter_pack_bind")
    (density,) = model.conservative_vars("density")
    rate = model.aux("collision_rate")
    model.flux(x=[density])
    model.eigenvalues(x=[density])
    model.primitive_vars(density=density)
    model.conservative_from([density])
    model.source([-(rate * density)])
    return model


def test_partial_two_attr_carrier_is_completed_by_exact_module_reattachment() -> None:
    model = _aux_model()
    packs = resolve_component_provider_packs(model.module)
    object.__setattr__(model, "_auxiliary_provider_metadata", packs.auxiliary.to_data())
    object.__setattr__(model, "_component_flux_consumer_plan", packs.physical_flux_plan)

    assert getattr(model, "_auxiliary_provider_pack", None) is None
    bind_emitter_provider_packs(model)

    for name in _CARRIER_ATTRS:
        assert getattr(model, name, None) is not None, name
        assert getattr(model._m, name, None) is not None, name
    assert type(model._auxiliary_provider_pack) is ProviderPack
    assert type(model._m._auxiliary_provider_pack) is ProviderPack


def test_one_altered_member_is_a_transactional_conflict() -> None:
    model = _aux_model()
    bind_emitter_provider_packs(model)
    forged = {"entries": []}
    object.__setattr__(model, "_auxiliary_provider_metadata", forged)

    with pytest.raises(ValueError, match="conflicting component-provider pack"):
        bind_emitter_provider_packs(model)
    assert model._auxiliary_provider_metadata is forged


def test_one_altered_route_is_a_transactional_conflict() -> None:
    carrier = _ModuleEmitter(_routed_module())
    bind_emitter_provider_packs(carrier)
    assert carrier._auxiliary_provider_routes
    object.__setattr__(carrier, "_auxiliary_provider_routes", {})

    with pytest.raises(ValueError, match="conflicting component-provider pack"):
        bind_emitter_provider_packs(carrier)
    assert carrier._auxiliary_provider_routes == {}


def test_exact_reattachment_is_idempotent() -> None:
    model = _aux_model()
    bind_emitter_provider_packs(model)
    first = emitter_carrier_snapshot(model)

    bind_emitter_provider_packs(model)
    bind_emitter_provider_packs(model)

    assert emitter_carrier_snapshot(model) == first
    assert emitter_carrier_snapshot(model._m) == first
    assert set(first) == set(_CARRIER_ATTRS)
    assert all(first[name] is not None for name in _CARRIER_ATTRS)


def test_model_impl_does_not_skip_reattachment_when_a_pack_already_exists() -> None:
    model = _aux_model()
    bind_emitter_provider_packs(model)
    forged = {"entries": []}
    object.__setattr__(model._m, "_auxiliary_provider_metadata", forged)

    with pytest.raises(ValueError, match="conflicting component-provider pack"):
        _model_impl(model)
    assert model._m._auxiliary_provider_metadata is forged


def test_bind_emitter_without_module_authority_does_not_invent_a_pack() -> None:
    carrier = SimpleNamespace()
    bind_emitter_provider_packs(carrier)
    assert not hasattr(carrier, "_auxiliary_provider_pack")
    assert not hasattr(carrier, "_component_flux_consumer_plan")


def test_model_hash_refuses_auxiliary_only_private_carrier() -> None:
    model = _aux_model()
    packs = resolve_component_provider_packs(model.module)
    object.__setattr__(model._m, "_auxiliary_provider_pack", packs.auxiliary)
    object.__setattr__(model._m, "_auxiliary_provider_metadata", packs.auxiliary.to_data())

    with pytest.raises(ValueError, match="complete exact ProviderPack carrier"):
        model._m._model_hash()


def test_native_loader_refuses_five_of_twelve_partial_carrier() -> None:
    model = _aux_model()
    packs = resolve_component_provider_packs(model.module)
    carrier = model._m
    object.__setattr__(carrier, "_component_provider_pack", packs.complete)
    object.__setattr__(carrier, "_component_provider_metadata", packs.complete.to_data())
    object.__setattr__(carrier, "_auxiliary_provider_pack", packs.auxiliary)
    object.__setattr__(carrier, "_auxiliary_provider_metadata", packs.auxiliary.to_data())
    object.__setattr__(carrier, "_component_flux_consumer_plan", packs.physical_flux_plan)
    assert sum(1 for name in _CARRIER_ATTRS if getattr(carrier, name, None) is not None) == 5

    with pytest.raises(ValueError, match="complete exact ProviderPack carrier"):
        carrier.emit_cpp_native_loader(name="PartialCarrier", target="system")


def test_model_impl_refuses_unbound_private_implementation() -> None:
    model = _aux_model()
    with pytest.raises(ValueError, match="complete exact ProviderPack carrier"):
        _model_impl(model._m)


def test_serialized_mapping_is_refused_in_provider_pack_slot() -> None:
    model = _aux_model()
    bind_emitter_provider_packs(model)
    object.__setattr__(model, "_component_provider_pack", model._component_provider_metadata)

    with pytest.raises(TypeError, match="exact ProviderPack"):
        bind_emitter_provider_packs(model)
    assert isinstance(model._component_provider_pack, dict)


def _stale_but_complete_private_carrier() -> Model:
    """Bind exactly, then replace one member with valid-shaped stale metadata."""
    model = _aux_model()
    bind_emitter_provider_packs(model)
    stale = model._m._component_provider_metadata
    object.__setattr__(model._m, "_auxiliary_provider_metadata", stale)
    return model


def test_model_hash_refuses_stale_but_complete_private_carrier() -> None:
    model = _stale_but_complete_private_carrier()
    with pytest.raises(ValueError, match="attachment witness"):
        model._m._model_hash()


def test_model_impl_refuses_stale_but_complete_private_carrier() -> None:
    model = _stale_but_complete_private_carrier()
    with pytest.raises(ValueError, match="attachment witness"):
        _model_impl(model._m)


def test_native_loader_refuses_stale_but_complete_private_carrier() -> None:
    model = _stale_but_complete_private_carrier()
    with pytest.raises(ValueError, match="attachment witness"):
        model._m.emit_cpp_native_loader(name="StaleCarrier", target="system")


def test_witness_refuses_bool_metadata_replaced_by_equal_int() -> None:
    model = _aux_model()
    bind_emitter_provider_packs(model)
    provider = model._m._component_provider_metadata["entries"][0]["provider"]
    assert provider["availability"] is True
    provider["availability"] = 1

    with pytest.raises(ValueError, match="attachment witness"):
        model._m._model_hash()


def test_witness_refuses_int_metadata_replaced_by_equal_float() -> None:
    model = _aux_model()
    bind_emitter_provider_packs(model)
    provider = model._m._component_provider_metadata["entries"][0]["provider"]
    assert provider["slot"] == 0
    provider["slot"] = 0.0

    with pytest.raises(ValueError, match="attachment witness"):
        model._m._model_hash()


def test_public_shaped_witness_mapping_is_not_an_attachment_witness() -> None:
    model = _aux_model()
    bind_emitter_provider_packs(model)
    object.__setattr__(
        model._m,
        "_pops_emitter_attachment_witness",
        emitter_carrier_snapshot(model._m),
    )

    with pytest.raises(ValueError, match="authenticated exact attachment witness"):
        require_emitter_provider_carrier(model._m)


def test_attachment_witness_is_deeply_immutable() -> None:
    model = _aux_model()
    bind_emitter_provider_packs(model)
    witness = model._m._pops_emitter_attachment_witness
    metadata = witness.snapshot()["_auxiliary_provider_metadata"]

    with pytest.raises(TypeError):
        metadata["entries"] = ()
    with pytest.raises(AttributeError, match="attachment witness is immutable"):
        witness._snapshot = emitter_carrier_snapshot(model._m)
    with pytest.raises(AttributeError, match="attachment witness is immutable"):
        del witness._snapshot

    object.__setattr__(
        model._m,
        "_auxiliary_provider_metadata",
        model._m._component_provider_metadata,
    )
    with pytest.raises(ValueError, match="attachment witness"):
        model._m._model_hash()


def test_witness_refuses_mutated_or_deleted_typed_route_boundary() -> None:
    carrier = _ModuleEmitter(_routed_module())
    bind_emitter_provider_packs(carrier)
    route = next(iter(carrier._auxiliary_provider_routes.values()))
    boundary = route["boundary"]

    object.__setattr__(boundary, "width", 2)
    with pytest.raises(ValueError, match="does not match"):
        require_emitter_provider_carrier(carrier)

    carrier = _ModuleEmitter(_routed_module())
    bind_emitter_provider_packs(carrier)
    route = next(iter(carrier._auxiliary_provider_routes.values()))
    del route["boundary"].width
    with pytest.raises(ValueError, match="cannot be projected"):
        require_emitter_provider_carrier(carrier)


def test_genuine_witness_cannot_authenticate_a_copied_carrier() -> None:
    source = _ModuleEmitter(_routed_module())
    bind_emitter_provider_packs(source)
    copied = SimpleNamespace()
    for name in _CARRIER_ATTRS:
        object.__setattr__(copied, name, getattr(source, name))
    object.__setattr__(
        copied,
        "_pops_emitter_attachment_witness",
        source._pops_emitter_attachment_witness,
    )

    with pytest.raises(ValueError, match="belongs to a different emitter"):
        require_emitter_provider_carrier(copied)


def test_source_model_for_owner_refuses_program_owner_mismatch() -> None:
    program_a, module_a, _case_a, _block_a, _state_a, _temporal_a = typed_program_state(
        "owner_mismatch_a"
    )
    _program_b, module_b, _case_b, _block_b, _state_b, _temporal_b = typed_program_state(
        "owner_mismatch_b"
    )
    assert _source_model_for_owner(module_a.owner_path, program=program_a) is module_a
    with pytest.raises(ValueError, match="does not match queried owner"):
        _source_model_for_owner(module_b.owner_path, program=program_a)


def test_source_model_for_owner_rejects_canonical_owner_collision() -> None:
    first, module_a, _case_a, _block_a, _state_a, _temporal_a = typed_program_state(
        "owner_collision"
    )
    second, module_b, _case_b, _block_b, _state_b, _temporal_b = typed_program_state(
        "owner_collision"
    )
    owner = module_a.owner_path
    assert module_a is not module_b

    with pytest.raises(ValueError, match="ambiguous"):
        _source_model_for_owner(owner)
    assert _source_model_for_owner(owner, program=first) is module_a
    assert _source_model_for_owner(owner, program=second) is module_b


def test_source_model_for_owner_refuses_unrelated_detached_program() -> None:
    _source, module, _case, _block, _state, _temporal = typed_program_state(
        "owner_detached_source"
    )

    with pytest.raises(ValueError, match="matching the supplied Program"):
        _source_model_for_owner(module.owner_path, program=Program("unrelated_program"))
