"""Storage-only selection cannot silently acquire a hyperbolic numerical flux."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pops.codegen.loader import CompiledModel
from pops.numerics import StateStorage
from pops.runtime._state_storage import StateStorageSpatial, require_state_storage_model


def _compiled(caps):
    return CompiledModel(
        so_path="/nonexistent/storage.so", backend="production", cons_names=["u"],
        cons_roles=["scalar"], prim_names=["u"], n_vars=1, gamma=None, n_aux=0,
        params={}, caps=caps, abi_key="", model_hash="", cxx="c++", std="20",
        native_dimension=2,
    )


def test_storage_method_is_deterministic_and_has_no_flux_authority():
    method = StateStorage()
    assert method.validate()
    assert method.validate_rate_contract({"state": object(), "flux": None})
    assert method.validate_rate_contract({"state": object(), "flux": ()})
    assert method.runtime_configuration() == method.to_data()
    assert method.resolve_references(lambda value: value).to_data() == method.to_data()
    spatial = method.runtime_spatial()
    assert type(spatial) is StateStorageSpatial
    assert spatial == method.runtime_spatial()
    assert spatial.limiter == "state_storage"
    assert spatial.flux == "unavailable"
    assert not hasattr(spatial, "riemann_capability_contract")
    assert spatial.identity() == method.runtime_spatial().identity()
    with pytest.raises((FrozenInstanceError, TypeError, AttributeError)):
        spatial.flux = "rusanov"


def test_storage_refuses_hyperbolic_rates_and_insufficient_halos():
    with pytest.raises(ValueError, match="hyperbolic flux"):
        StateStorage().validate_rate_contract({"state": object(), "flux": object()})
    with pytest.raises(TypeError, match="evolved state"):
        StateStorage().validate_rate_contract({"flux": None})
    with pytest.raises(ValueError, match="ghost cell"):
        StateStorageSpatial().validate(ghost_depth=0)
    assert StateStorageSpatial().validate(ghost_depth=1)


@pytest.mark.parametrize("fact", [None, False, 1, "true"])
def test_storage_admission_requires_an_exact_compiled_capability(fact):
    with pytest.raises(ValueError, match="authenticated Program-only"):
        require_state_storage_model(_compiled({"program_only_storage": fact}),
                                    StateStorageSpatial(), where="test")


def test_storage_admission_keeps_other_spatial_methods_on_their_existing_guard():
    assert require_state_storage_model(_compiled({"program_only_storage": True}),
                                       StateStorageSpatial(), where="test")
    assert not require_state_storage_model(object(), object(), where="test")
