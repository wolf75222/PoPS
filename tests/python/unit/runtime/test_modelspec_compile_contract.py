"""Legacy ModelSpec compilation must preserve requested physics or reject before mutation."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import pops.runtime._engine_descriptors as engine
from pops.physics import _facade
from pops.runtime._amr_system_equation import _AmrSystemEquation
from pops.runtime._modelspec_compile import compile_modelspec_package
from pops.runtime._system_install import _SystemInstall


def _request(physics):
    if physics == "exb":
        return engine.Model(
            engine.Scalar(), engine.ExB(), engine.NoSource(), engine.ChargeDensity(charge=1.0))
    source = {
        "potential": engine.PotentialForce,
        "gravity": engine.GravityForce,
        "magnetic": engine.MagneticLorentzForce,
        "potential_magnetic": engine.PotentialMagneticForce,
    }[physics]()
    return engine.Model(
        engine.FluidState("compressible", gamma=1.4), engine.CompressibleFlux(), source,
        engine.BackgroundDensity(alpha=1.0, n0=1.0))


@pytest.mark.parametrize("target", ("system", "amr_system"))
@pytest.mark.parametrize("physics", ("exb", "potential", "gravity", "magnetic", "potential_magnetic"))
def test_missing_modelspec_field_authority_refuses_before_authoring_or_native_mutation(
    monkeypatch, target, physics,
):
    request = _request(physics)

    def unexpected_model(*_args, **_kwargs):
        raise AssertionError("unsupported ModelSpec reached formula authoring or compilation")

    monkeypatch.setattr(_facade, "Model", unexpected_model)
    message = "ModelSpec ExB.*field-output provider plan" if physics == "exb" else (
        "ModelSpec source.*explicit field/auxiliary provider bindings")
    with pytest.raises(ValueError, match=message):
        compile_modelspec_package(request, name="rejected", target=target)

    class NativeMutationSentinel:
        def __getattr__(self, name):
            raise AssertionError("unsupported ModelSpec reached native access %s" % name)

    host = SimpleNamespace(
        _lifecycle="assembling", _s=NativeMutationSentinel(),
        _lower_spatial=lambda spatial: spatial,
        _authored_block_options={"existing": {"source": "none"}},
    )
    before = dict(vars(host))
    add_equation = (
        _SystemInstall.add_equation if target == "system" else _AmrSystemEquation.add_equation)
    with pytest.raises(ValueError, match=message):
        add_equation(host, "rejected", request, spatial=engine.Spatial())
    assert vars(host) == before
