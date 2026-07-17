"""Authenticated Uniform restart refuses a missing required Program-history ring."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pops
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.physics import Model
from pops.runtime._checkpoint_manifest import (
    IDENTITY_KEY,
    MANIFEST_KEY,
    seal_checkpoint_payload,
)
from pops.time import FixedDt


ROOT = Path(__file__).resolve().parents[4]
N = 4
DT = 0.01


def _bound_history_runtime(native_cxx: str):
    frame = Rectangle(
        "restart-history-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model("restart-history-model", frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    zero = 0.0 * rho
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (zero,), y_axis: (zero,)},
        waves={x_axis: (zero,), y_axis: (zero,)},
    )
    source = model.source("growth", on=state, value=(0.5 * rho,))
    rate = model.rate(
        "source-rate", equation=ddt(state) == -div(flux) + source)

    case = pops.Case("restart-history-case")
    block = case.block("blk", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(numerics, block=block)
    from pops.lib.time import AdamsBashforth

    program = AdamsBashforth(block[state], rate=rate, order=2)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    layout = Uniform(CartesianGrid(
        frame=frame,
        cells=(N, N),
        periodic=PeriodicAxes(frame.axes),
    ))
    resolved = pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include"), "cxx": native_cxx},
    )
    artifact = pops.compile(resolved)
    artifact.verify()
    return pops.bind(
        artifact,
        initial_state={"blk": np.ones((1, N, N), dtype=np.float64)},
    )


@pytest.mark.compiler
@pytest.mark.native_loader
def test_authenticated_restart_missing_required_history_fails_at_exact_guard(
    tmp_path, isolated_native_cache, native_cxx, kokkos_root,
):
    del isolated_native_cache, kokkos_root
    runtime = _bound_history_runtime(native_cxx)
    report = pops.run(runtime, t_end=DT, max_steps=1)
    assert report.accepted_steps == 1
    assert runtime.history_names(), "the accepted runtime must own its required history ring"

    valid_path = runtime.checkpoint(tmp_path / "with_history.npz")
    with np.load(valid_path, allow_pickle=False) as stored:
        payload = {
            name: np.asarray(stored[name]).copy()
            for name in stored.files
            if name not in {MANIFEST_KEY, IDENTITY_KEY}
        }
    required_histories = tuple(str(name) for name in payload["history_names"])
    assert required_histories, "the authentic checkpoint must contain its required history"
    for name in tuple(payload):
        if name.startswith("history_"):
            del payload[name]
    payload["history_names"] = np.array([], dtype="U1")
    seal_checkpoint_payload(runtime, payload, runtime_kind="uniform")
    path = tmp_path / "missing_history.npz"
    np.savez_compressed(path, **payload)

    with pytest.raises(
        RuntimeError,
        match="checkpoint does not contain required Program history",
    ) as caught:
        runtime.restart(path)
    assert required_histories[0] in str(caught.value)
