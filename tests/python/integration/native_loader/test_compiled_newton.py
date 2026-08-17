"""Compiled-package Newton options actually run the Program primitive."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.python.integration.native_loader.test_equation_cadence import (
    _HOLD_CELLS,
    _compile_block,
    _compile_include,
    _ranked_axes,
    _require_native,
    _select_installed_native_dimension,
    _state_handle,
)
from tests.python.support.requirements import default_cxx


_REACTION = 2.0
_DT = 0.05
_U0 = 1.2


def _quadratic_model(name: str, dimension: int) -> Any:
    from pops.physics._facade import Model

    model = Model(name)
    (u,) = model.conservative_vars("u")
    model.primitive_vars(u)
    model.conservative_from([u])
    zero = [0.0 * u]
    model.flux(**{axis: zero for axis in _ranked_axes(dimension)})
    model.eigenvalues(**{axis: zero for axis in _ranked_axes(dimension)})
    model.source([-_REACTION * u * u])
    return model


def _implicit_euler_root(u0: float, dt: float) -> float:
    return (-1.0 + np.sqrt(1.0 + 4.0 * dt * _REACTION * u0)) / (2.0 * dt * _REACTION)


def _run_compiled_newton(tmp_path: Path, *, dimension: int, target: str, max_iters: int,
                         diagnostics: bool = True) -> tuple[Any, np.ndarray]:
    import pops.runtime._engine_descriptors as engine
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.problem import Case
    from pops.runtime._system import AmrSystem, AmrSystemConfig, System

    include = _compile_include()
    model = _quadratic_model("compiled-newton-%s" % target, dimension)
    case = Case("compiled-newton-%s" % target)
    handle = _state_handle(case, "fluid", model)
    compiled = _compile_block(
        model, include, owner="compiled-newton-%s/fluid" % target, target=target
    )
    time = engine.IMEX(
        newton_max_iters=max_iters,
        newton_rel_tol=1.0e-12,
        newton_abs_tol=1.0e-14,
        newton_diagnostics=diagnostics,
    )
    spatial = engine.Spatial(limiter=FirstOrder(), flux=Rusanov())
    if target == "system":
        simulation = System(
            shape=(_HOLD_CELLS,) * dimension,
            lower=(0.0,) * dimension,
            upper=(1.0,) * dimension,
            periodicity=(True,) * dimension,
        )
        simulation.add_equation("fluid", compiled, spatial=spatial, time=time)
        initial = np.full((1,) + (_HOLD_CELLS,) * dimension, _U0)
        simulation.set_state("fluid", initial)
    else:
        config = AmrSystemConfig()
        config.shape = (_HOLD_CELLS, _HOLD_CELLS)
        config.lower = (0.0, 0.0)
        config.upper = (1.0, 1.0)
        config.periodicity = (True, True)
        config.regrid_every = 0
        simulation = AmrSystem(config)
        simulation.set_temporal_relations([2], [1], ["integral_only"])
        simulation.add_equation("fluid", compiled, spatial=spatial, time=time)
        initial = np.full((_HOLD_CELLS, _HOLD_CELLS), _U0)
        simulation.set_density("fluid", initial)
    simulation.install_equation_cadence(
        {"fluid": handle},
        model=model,
        include=include,
        cxx=default_cxx(),
        so_path=str(tmp_path / ("compiled_newton_%s_%d.so" % (target, max_iters))),
        native_dimension=dimension,
        target=target,
    )
    simulation.step(_DT)
    if target == "system":
        state = np.asarray(simulation.get_state("fluid"), dtype=np.float64)
    else:
        state = np.asarray(simulation.density("fluid"), dtype=np.float64)
    return simulation, state


def test_compiled_uniform_newton_runs_and_reports(tmp_path: Path) -> None:
    _require_native()
    dimension = _select_installed_native_dimension()
    simulation, state = _run_compiled_newton(
        tmp_path, dimension=dimension, target="system", max_iters=25
    )
    expected = _implicit_euler_root(_U0, _DT)
    np.testing.assert_allclose(state, expected, rtol=0.0, atol=1.0e-10)
    report = simulation.newton_report()
    assert report["enabled"] is True
    assert report["converged"] is True
    assert int(report["iterations"]) >= 1
    assert float(report["residual"]) < 1.0e-10
    inspect_newton = simulation.inspect().to_dict()["options"]["blocks"][0]["newton"]
    assert inspect_newton["max_iters"] == 25
    assert inspect_newton["diagnostics"] is True

    with pytest.raises(RuntimeError, match="prepared solve failed|iteration"):
        _run_compiled_newton(tmp_path, dimension=dimension, target="system", max_iters=1)


def test_compiled_amr_newton_runs_and_reports(tmp_path: Path) -> None:
    _require_native()
    dimension = _select_installed_native_dimension()
    if dimension != 2:
        pytest.skip("compiled AMR Newton proof uses the installed rank-2 artifact")
    simulation, state = _run_compiled_newton(
        tmp_path, dimension=dimension, target="amr_system", max_iters=25
    )
    expected = _implicit_euler_root(_U0, _DT)
    np.testing.assert_allclose(state, expected, rtol=0.0, atol=1.0e-10)
    report = simulation.newton_report()
    assert report["enabled"] is True
    assert report["converged"] is True
    assert int(report["iterations"]) >= 1
    assert float(report["residual"]) < 1.0e-10
    inspect_newton = simulation.inspect().to_dict()["options"]["blocks"][0]["newton"]
    assert inspect_newton["max_iters"] == 25
    assert inspect_newton["diagnostics"] is True
