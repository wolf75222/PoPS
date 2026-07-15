"""Compiled generic Roe regression through the final public lifecycle."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pops
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.models.moments import Gaussian
from pops.lib.time import ForwardEuler
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.riemann import FromJacobian, provider_of
from pops.numerics.spatial import FiniteVolume
from pops.physics import Model
from pops.time import FixedDt


ROOT = Path(__file__).resolve().parents[4]
N = 24
DT = 5.0e-4
STEPS = 10

pytestmark = [
    pytest.mark.compiler,
    pytest.mark.kokkos,
    pytest.mark.native_loader,
    pytest.mark.regression,
]


def _smooth_density(n: int) -> np.ndarray:
    points = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(points, points, indexing="xy")
    return 1.0 + 0.4 * np.exp(-60.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))


def test_final_gaussian_moments_run_generic_roe_from_jacobian(
    isolated_native_cache, native_cxx, kokkos_root
) -> None:
    del isolated_native_cache, kokkos_root
    frame = Rectangle(
        "final-gaussian-roe-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    model = Gaussian.transport(
        order=2,
        name="final_gaussian_roe",
        robust=True,
        exact_speeds=True,
        roe=True,
        frame=frame,
    )
    assert isinstance(model, Model)
    provider = provider_of(model)
    assert provider is not None
    assert provider.kind == FromJacobian(eig="numeric").kind

    # The generic route cannot have accidentally fallen back to the fluid-role implementation.
    state = model.states["U"]
    roles = set(state.space.roles.values())
    assert "Density" in roles
    assert not roles.intersection({"MomentumX", "MomentumY", "Energy"})

    flux = model.fluxes["transport"]
    rate = model.operators["transport"]
    case = pops.Case("final_gaussian_roe_case")
    block = case.block("moments", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Roe(),
        ),
    )
    case.numerics(numerics, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(DT))
    case.program(program)

    layout = Uniform(
        CartesianGrid(
            frame=model.frame,
            cells=(N, N),
            periodic=PeriodicAxes(model.frame.axes),
        )
    )
    resolved = pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include"), "cxx": native_cxx},
    )
    artifact = pops.compile(resolved)
    artifact.verify()
    assert len(artifact.blocks) == 1
    compiled_model = artifact.blocks[0].model
    assert compiled_model.has_roe
    assert compiled_model.has_wave_speeds

    # Realizable centered Maxwellian moments (u=v=0, covariance=I), modulated by density.
    maxwellian = np.array((1.0, 0.0, 1.0, 0.0, 0.0, 1.0))
    initial = maxwellian[:, None, None] * _smooth_density(N)[None, :, :]
    simulation = pops.bind(
        artifact,
        initial_state={"moments": np.ascontiguousarray(initial)},
    )
    report = pops.run(simulation, t_end=STEPS * DT, max_steps=STEPS)
    assert report.accepted_steps == STEPS
    final = np.asarray(simulation.get_state("moments"), dtype=np.float64).reshape(
        initial.shape
    )
    assert np.isfinite(final).all()
    assert not np.array_equal(final, initial)
    np.testing.assert_allclose(final[0].sum(), initial[0].sum(), rtol=1.0e-12, atol=1.0e-12)
