"""The native cache identity observes every axis of the authored rank."""
from __future__ import annotations

from pops.codegen._compile_emit import model_hash
from pops.physics._model import HyperbolicModel


def _advection(*, dimension: int, z_scale: float = 1.0) -> HyperbolicModel:
    model = HyperbolicModel("ranked_hash")
    (state,) = model.conservative_vars("state")
    fluxes = {"x": [state]}
    if dimension >= 2:
        fluxes["y"] = [2.0 * state]
    if dimension >= 3:
        fluxes["z"] = [z_scale * state]
    model.set_flux(**fluxes)
    model.set_eigenvalues(**{
        axis: [values[0] * 0.0 + 1.0] for axis, values in fluxes.items()
    })
    return model


def test_model_hash_accepts_every_exact_rank() -> None:
    hashes = {model_hash(_advection(dimension=dimension)) for dimension in (1, 2, 3)}
    assert len(hashes) == 3


def test_model_hash_observes_the_z_flux() -> None:
    assert model_hash(_advection(dimension=3, z_scale=1.0)) != model_hash(
        _advection(dimension=3, z_scale=3.0)
    )
