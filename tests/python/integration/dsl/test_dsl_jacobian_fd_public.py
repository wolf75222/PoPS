"""Partitioned finite-difference Jacobian speeds through the public HLL lifecycle."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pops
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.riemann import FromJacobian, provider_of
from pops.numerics.spatial import FiniteVolume
from pops.physics import Model
from pops.time import FixedDt


ROOT = Path(__file__).resolve().parents[4]
N = 16
DT = 2.0e-4

pytestmark = [
    pytest.mark.compiler,
    pytest.mark.kokkos,
    pytest.mark.native_loader,
    pytest.mark.regression,
]


def _model() -> tuple[Model, object, object, object, dict[str, list[list[int]]]]:
    frame = Rectangle(
        "partitioned-fd-hll-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model("partitioned_fd_hll", frame=frame)
    state = model.state("U", components=("q1", "q2"))
    q1, q2 = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={
            # Two independent characteristic blocks. Reversing the y partition proves that block
            # identity is direction-local instead of inferred from list position.
            x_axis: (1.25 * q1, -0.75 * q2),
            y_axis: (-0.5 * q1, 1.5 * q2),
        },
    )
    blocks = {"x": [[0], [1]], "y": [[1], [0]]}
    model.wave_speeds_from_jacobian(eig="fd", blocks=blocks)
    rate = model.rate("transport", equation=ddt(state) == -div(flux))
    return model, state, flux, rate, blocks


def _initial_state() -> np.ndarray:
    points = (np.arange(N) + 0.5) / N
    x, y = np.meshgrid(points, points, indexing="xy")
    return np.stack(
        (
            1.0 + 0.2 * np.sin(2.0 * np.pi * x) * np.cos(4.0 * np.pi * y),
            -0.4 + 0.15 * np.cos(4.0 * np.pi * x) * np.sin(2.0 * np.pi * y),
        )
    )


def test_final_case_consumes_partitioned_fd_jacobian_speeds_in_hll(
    isolated_native_cache, native_cxx, kokkos_root,
) -> None:
    del isolated_native_cache, kokkos_root
    model, state, flux, rate, blocks = _model()
    provider = provider_of(model)
    requested = FromJacobian(eig="fd", blocks=blocks)
    assert provider is not None
    assert provider.kind == requested.kind
    assert provider.options()["eig"] == "fd"
    assert requested.options()["blocks"] == blocks

    case = pops.Case("partitioned_fd_hll_case")
    block = case.block("transport", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.HLL(waves=requested),
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
    assert compiled_model.has_wave_speeds

    initial = np.ascontiguousarray(_initial_state())
    simulation = pops.bind(artifact, initial_state={"transport": initial})
    report = pops.run(simulation, t_end=DT, max_steps=1)
    assert report.accepted_steps == 1
    final = np.asarray(simulation.get_state("transport"), dtype=np.float64).reshape(
        initial.shape
    )
    assert np.isfinite(final).all()
    assert not np.array_equal(final, initial)
    np.testing.assert_allclose(
        final.sum(axis=(1, 2)), initial.sum(axis=(1, 2)), rtol=0.0, atol=1.0e-12
    )
