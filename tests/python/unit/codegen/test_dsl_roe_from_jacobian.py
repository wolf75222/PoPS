"""Compiled generic Roe regression for the final Gaussian-moment preset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pops.runtime._engine_descriptors as engine
from pops.lib.models.moments import Gaussian
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Roe
from pops.numerics.riemann.waves import FromJacobian, provider_of
from pops.physics import Model
from pops.runtime._system import System
from tests.python.support.requirements import (
    default_cxx,
    missing_aot_requirement,
    repo_include,
)


INCLUDE = repo_include()

pytestmark = (
    pytest.mark.compiler,
    pytest.mark.kokkos,
    pytest.mark.native_loader,
    pytest.mark.regression,
)


def _compile_native(model: Model, path: Path):
    lowering = model.__pops_compiler_lowering__()
    assert lowering.facade is model
    assert lowering.source_module is model.module
    return lowering.emit_model.compile(str(path), INCLUDE, backend="production")


def _smooth_density(n: int) -> np.ndarray:
    points = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(points, points, indexing="xy")
    return 1.0 + 0.4 * np.exp(-60.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))


def test_final_gaussian_moments_run_generic_roe_from_jacobian(tmp_path: Path) -> None:
    reason = missing_aot_requirement(INCLUDE, default_cxx())
    if reason:
        pytest.skip(reason)

    model = Gaussian.transport(
        order=2,
        name="final_gaussian_roe",
        robust=True,
        exact_speeds=True,
        roe=True,
    )
    assert isinstance(model, Model)
    provider = provider_of(model)
    assert provider is not None
    assert provider.kind == FromJacobian(eig="numeric").kind

    # The generic route cannot have accidentally fallen back to the fluid-role implementation.
    state_space = model.module.state_spaces()["U"]
    roles = set(state_space.roles.values())
    assert "Density" in roles
    assert not roles.intersection({"MomentumX", "MomentumY", "Energy"})

    compiled = _compile_native(model, tmp_path / "gaussian_roe.so")
    assert compiled.has_roe
    assert compiled.has_wave_speeds

    n = 24
    # Realizable centered Maxwellian moments (u=v=0, covariance=I), modulated by density.
    maxwellian = np.array((1.0, 0.0, 1.0, 0.0, 0.0, 1.0))
    initial = maxwellian[:, None, None] * _smooth_density(n)[None, :, :]
    simulation = System(n=n, L=1.0, periodic=True)
    simulation.add_equation(
        "moments",
        model=compiled,
        spatial=engine.Spatial(limiter=FirstOrder(), flux=Roe()),
        time=engine.Explicit(),
    )
    simulation.set_state("moments", initial)
    for _ in range(10):
        simulation.step(5.0e-4)
    final = np.asarray(simulation.get_state("moments"))
    assert np.isfinite(final).all()
    np.testing.assert_allclose(final[0].sum(), initial[0].sum(), rtol=1.0e-12, atol=1.0e-12)
