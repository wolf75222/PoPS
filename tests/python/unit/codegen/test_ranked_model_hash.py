"""The native cache identity observes every axis of the authored rank."""
from __future__ import annotations

from pops.codegen._compile_emit import model_hash
from pops.codegen.module_lowering import lower_and_validate
from pops.model import ProviderPack
from pops.physics._facade import Model


def _advection(*, dimension: int, z_scale: float = 1.0) -> Model:
    model = Model("ranked_hash")
    (state,) = model.conservative_vars("state")
    fluxes = {"x": [state]}
    if dimension >= 2:
        fluxes["y"] = [2.0 * state]
    if dimension >= 3:
        fluxes["z"] = [z_scale * state]
    model.flux(**fluxes)
    model.eigenvalues(**{
        axis: [values[0] * 0.0 + 1.0] for axis, values in fluxes.items()
    })
    return model


def _canonical_hash(model: Model) -> str:
    """Hash only the formula carrier authenticated by its canonical Module."""
    emit_model, source_module = lower_and_validate(model, facade=model)

    assert emit_model is model
    assert source_module is model.module
    assert type(emit_model._m._auxiliary_provider_pack) is ProviderPack
    return model_hash(emit_model._m)


def test_model_hash_accepts_every_exact_rank() -> None:
    hashes = {
        _canonical_hash(_advection(dimension=dimension))
        for dimension in (1, 2, 3)
    }
    assert len(hashes) == 3


def test_model_hash_observes_the_z_flux() -> None:
    assert _canonical_hash(_advection(dimension=3, z_scale=1.0)) != _canonical_hash(
        _advection(dimension=3, z_scale=3.0)
    )
