"""Production cadence: Explicit(stride), AMR stride, partial IMEX, step_adaptive."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pops.physics._facade import Model
from pops.problem import Case
from pops.time import adaptive_strides
from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
    require_native_or_skip,
)


_FAST_RATE = -0.5
_SLOW_RATE = -0.25
_HOLD_STRIDE = 4
_HOLD_DT = 0.1
_HOLD_CELLS = 4
_EXPLICIT_SRC = 1.0
_IMPLICIT_SRC = 2.0


def _require_native() -> None:
    missing = missing_native_compile_requirement(repo_include(), default_cxx())
    if missing:
        require_native_or_skip(missing, optional_skip=pytest.skip)
    try:
        import pops.runtime._engine_descriptors  # noqa: F401
        import pops.runtime._system  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        require_native_or_skip(
            "cadence runtime unavailable: %s" % exc,
            optional_skip=pytest.skip,
        )


def _select_installed_native_dimension() -> int:
    from pops._native_selector import select_native_dimension, selected_native_dimension

    selected = selected_native_dimension()
    if selected is not None:
        return selected
    configured = os.environ.get("POPS_NATIVE_DIM")
    if configured in {"1", "2", "3"}:
        return select_native_dimension(int(configured)).__native_dimension__
    from pops._native_manifest import native_manifest_path

    manifest = native_manifest_path()
    if manifest is None or not manifest.is_file():
        require_native_or_skip(
            "need POPS_NATIVE_DIM or an installed native variant",
            optional_skip=pytest.skip,
        )
        raise AssertionError("require_native_or_skip must not return")
    import json

    rows = json.loads(manifest.read_text(encoding="utf-8")).get("variants", [])
    dims = sorted({int(row["dimension"]) for row in rows})
    if len(dims) != 1:
        require_native_or_skip(
            "need POPS_NATIVE_DIM or exactly one installed native variant",
            optional_skip=pytest.skip,
        )
        raise AssertionError("require_native_or_skip must not return")
    return select_native_dimension(dims[0]).__native_dimension__


def _ranked_axes(dimension: int) -> tuple[str, ...]:
    return ("x", "y", "z")[:dimension]


def _decay_model(name: str, dimension: int) -> Any:
    model = Model(name)
    (rho,) = model.conservative_vars("rho")
    model.primitive_vars(rho)
    model.conservative_from([rho])
    zero = [0.0 * rho]
    model.flux(**{axis: zero for axis in _ranked_axes(dimension)})
    model.eigenvalues(**{axis: zero for axis in _ranked_axes(dimension)})
    return model


def _imex_model(name: str, dimension: int) -> Any:
    model = Model(name)
    rho, q = model.conservative_vars("rho", "q")
    model.primitive_vars(rho, q)
    model.conservative_from([rho, q])
    zero = [0.0 * rho, 0.0 * q]
    model.flux(**{axis: zero for axis in _ranked_axes(dimension)})
    model.eigenvalues(**{axis: [0.0 * rho, 0.0 * q] for axis in _ranked_axes(dimension)})
    model.source([_EXPLICIT_SRC + 0.0 * rho, _IMPLICIT_SRC + 0.0 * q])
    return model


def _state_handle(case: Any, block_name: str, model: Any) -> Any:
    block = case.block(block_name, model)
    declaration = next(
        record for record in model.declaration_index().records() if record.kind == "state"
    )
    return block[declaration]


def _compile_include() -> Any:
    from pops.codegen.abi import module_header_signature
    from pops.codegen.toolchain import pops_header_signature, pops_include

    include = pops_include()
    baked = module_header_signature()
    if baked is not None and pops_header_signature(include) != baked:
        require_native_or_skip(
            "installed _pops header signature does not match pops_include(); "
            "rebuild the native module against the current headers",
            optional_skip=pytest.skip,
        )
    return include


def _compile_block(model: Any, include: Any, *, owner: str, target: str = "system") -> Any:
    return model.compile(
        backend="production",
        include=include,
        cxx=default_cxx(),
        consumer_owner_qid=owner,
        target=target,
    )


def _uniform_system(dimension: int, compiled_fast: Any, compiled_slow: Any, *,
                    slow_stride: int = 1, time_kind: str = "explicit"):
    import pops.runtime._engine_descriptors as engine
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.runtime._system import System

    simulation = System(
        shape=(_HOLD_CELLS,) * dimension,
        lower=(0.0,) * dimension,
        upper=(1.0,) * dimension,
        periodicity=(True,) * dimension,
    )
    spatial = engine.Spatial(limiter=FirstOrder(), flux=Rusanov())
    fast_time = engine.Explicit(method="euler")
    slow_time = (
        engine.IMEX(stride=slow_stride)
        if time_kind == "imex"
        else engine.Explicit(method="euler", stride=slow_stride)
    )
    simulation._batch_native_packages = True
    try:
        simulation.add_equation("fast", compiled_fast, spatial=spatial, time=fast_time)
        simulation.add_equation("slow", compiled_slow, spatial=spatial, time=slow_time)
    finally:
        simulation._batch_native_packages = False
    simulation._commit_pending_native_packages()
    return simulation


def test_adaptive_strides_match_the_oracle_formula() -> None:
    assert adaptive_strides({"fast": 4.0, "slow": 1.0}) == {"fast": 1, "slow": 4}
    assert adaptive_strides({"a": 3.0, "b": 2.0, "c": 1.0}) == {"a": 1, "b": 1, "c": 3}


def test_explicit_stride_schedules_without_a_handwritten_subcycle(tmp_path: Path) -> None:
    _require_native()
    dimension = _select_installed_native_dimension()
    include = _compile_include()
    model = _decay_model("equation-cadence-model", dimension)
    case = Case("equation-cadence-case")
    fast_handle = _state_handle(case, "fast", model)
    slow_handle = _state_handle(case, "slow", model)
    compiled_fast = _compile_block(model, include, owner="equation-cadence-case/fast")
    compiled_slow = _compile_block(model, include, owner="equation-cadence-case/slow")
    simulation = _uniform_system(dimension, compiled_fast, compiled_slow, slow_stride=_HOLD_STRIDE)
    simulation.install_equation_cadence(
        {"fast": fast_handle, "slow": slow_handle},
        model=model,
        include=include,
        cxx=default_cxx(),
        so_path=str(tmp_path / "equation_cadence.so"),
        native_dimension=dimension,
        linear_rates={"fast": _FAST_RATE, "slow": _SLOW_RATE},
    )
    initial = np.full((1,) + (_HOLD_CELLS,) * dimension, 2.0)
    window = _HOLD_STRIDE * _HOLD_DT
    expected_fast = initial * (1.0 + _FAST_RATE * _HOLD_DT) ** _HOLD_STRIDE
    expected_slow = initial * (1.0 + _SLOW_RATE * window)
    not_slow_subcycled = initial * (1.0 + _SLOW_RATE * _HOLD_DT) ** _HOLD_STRIDE
    simulation.set_state("fast", initial)
    simulation.set_state("slow", initial)
    for _ in range(_HOLD_STRIDE - 1):
        simulation.step(_HOLD_DT)
        np.testing.assert_allclose(np.asarray(simulation.get_state("fast")), initial, atol=0.0)
        np.testing.assert_allclose(np.asarray(simulation.get_state("slow")), initial, atol=0.0)
    simulation.step(_HOLD_DT)
    np.testing.assert_allclose(
        np.asarray(simulation.get_state("fast")), expected_fast, rtol=0.0, atol=1.0e-13
    )
    np.testing.assert_allclose(
        np.asarray(simulation.get_state("slow")), expected_slow, rtol=0.0, atol=1.0e-13
    )
    assert float(np.max(np.abs(np.asarray(simulation.get_state("slow")) - not_slow_subcycled))) > 1.0e-6
    assert simulation.macro_step() == _HOLD_STRIDE


def test_step_adaptive_installs_oracle_strides(tmp_path: Path) -> None:
    _require_native()
    dimension = _select_installed_native_dimension()
    include = _compile_include()
    model = _decay_model("step-adaptive-model", dimension)
    case = Case("step-adaptive-case")
    fast_handle = _state_handle(case, "fast", model)
    slow_handle = _state_handle(case, "slow", model)
    compiled_fast = _compile_block(model, include, owner="step-adaptive-case/fast")
    compiled_slow = _compile_block(model, include, owner="step-adaptive-case/slow")
    simulation = _uniform_system(dimension, compiled_fast, compiled_slow)
    speeds = {"fast": 4.0, "slow": 1.0}
    assert adaptive_strides(speeds) == {"fast": 1, "slow": 4}
    program = simulation.install_step_adaptive(
        {"fast": fast_handle, "slow": slow_handle},
        speeds,
        model=model,
        include=include,
        cxx=default_cxx(),
        so_path=str(tmp_path / "step_adaptive.so"),
        native_dimension=dimension,
        linear_rates={"fast": _FAST_RATE, "slow": _SLOW_RATE},
    )
    assert program.cadence_contract().stride == 4
    initial = np.full((1,) + (_HOLD_CELLS,) * dimension, 2.0)
    window = 4 * _HOLD_DT
    simulation.set_state("fast", initial)
    simulation.set_state("slow", initial)
    dt = simulation.step_adaptive(
        0.4, wave_speeds=speeds, min_cell_spacing=_HOLD_DT * 4.0 / 0.4
    )
    assert dt == pytest.approx(_HOLD_DT, rel=0.0, abs=1.0e-15)
    # One oracle macro-step holds both until the window closes only when cadence is used
    # across four steps. A single step_adaptive call runs the Program once with dt=macro.
    # Use four macro-steps of dt so stride-4 catch-up matches the two-clock numbers.
    held = _uniform_system(dimension, compiled_fast, compiled_slow)
    held.install_step_adaptive(
        {"fast": fast_handle, "slow": slow_handle},
        speeds,
        model=model,
        include=include,
        cxx=default_cxx(),
        so_path=str(tmp_path / "step_adaptive_held.so"),
        native_dimension=dimension,
        linear_rates={"fast": _FAST_RATE, "slow": _SLOW_RATE},
    )
    held.set_state("fast", initial)
    held.set_state("slow", initial)
    for _ in range(3):
        held.step_adaptive(0.4, wave_speeds=speeds, min_cell_spacing=_HOLD_DT * 4.0 / 0.4)
        np.testing.assert_allclose(np.asarray(held.get_state("fast")), initial, atol=0.0)
        np.testing.assert_allclose(np.asarray(held.get_state("slow")), initial, atol=0.0)
    held.step_adaptive(0.4, wave_speeds=speeds, min_cell_spacing=_HOLD_DT * 4.0 / 0.4)
    expected_fast = initial * (1.0 + _FAST_RATE * _HOLD_DT) ** 4
    expected_slow = initial * (1.0 + _SLOW_RATE * window)
    np.testing.assert_allclose(np.asarray(held.get_state("fast")), expected_fast, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(held.get_state("slow")), expected_slow, atol=1.0e-13)


def test_partial_imex_mask_runs_on_uniform(tmp_path: Path) -> None:
    _require_native()

    dimension = _select_installed_native_dimension()
    include = _compile_include()
    model = _imex_model("partial-imex-model", dimension)
    case = Case("partial-imex-case")
    handle = _state_handle(case, "fluid", model)
    compiled = _compile_block(model, include, owner="partial-imex-case/fluid")
    import pops.runtime._engine_descriptors as engine
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.runtime._system import System

    simulation = System(
        shape=(_HOLD_CELLS,) * dimension,
        lower=(0.0,) * dimension,
        upper=(1.0,) * dimension,
        periodicity=(True,) * dimension,
    )
    simulation.add_equation(
        "fluid",
        compiled,
        spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
        time=engine.IMEX(implicit_vars=["q"]),
    )
    simulation.install_equation_cadence(
        {"fluid": handle},
        model=model,
        include=include,
        cxx=default_cxx(),
        so_path=str(tmp_path / "partial_imex.so"),
        native_dimension=dimension,
    )
    initial = np.full((2,) + (_HOLD_CELLS,) * dimension, 0.0)
    initial[0] = 3.0
    initial[1] = 5.0
    simulation.set_state("fluid", initial)
    dt = 0.2
    simulation.step(dt)
    got = np.asarray(simulation.get_state("fluid"))
    expected = np.array(initial, copy=True)
    expected[0] = 3.0 + dt * _EXPLICIT_SRC
    expected[1] = 5.0 + dt * _IMPLICIT_SRC
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1.0e-13)


def test_amr_compiled_stride_holds_then_catches_up(tmp_path: Path) -> None:
    _require_native()
    import pops.runtime._engine_descriptors as engine
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.runtime._system import AmrSystem, AmrSystemConfig

    dimension = _select_installed_native_dimension()
    if dimension != 2:
        pytest.skip("compiled AMR stride proof uses the installed rank-2 artifact")
    include = _compile_include()
    model = _decay_model("amr-stride-model", dimension)
    case = Case("amr-stride-case")
    slow_handle = _state_handle(case, "slow", model)
    compiled_slow = _compile_block(
        model, include, owner="amr-stride-case/slow", target="amr_system"
    )
    config = AmrSystemConfig()
    config.shape = (_HOLD_CELLS, _HOLD_CELLS)
    config.lower = (0.0, 0.0)
    config.upper = (1.0, 1.0)
    config.periodicity = (True, True)
    config.regrid_every = 0
    simulation = AmrSystem(config)
    simulation.set_temporal_relations([2], [1], ["integral_only"])
    simulation.add_equation(
        "slow",
        compiled_slow,
        spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
        time=engine.Explicit(method="euler", stride=_HOLD_STRIDE),
    )
    initial = np.full((_HOLD_CELLS, _HOLD_CELLS), 2.0)
    window = _HOLD_STRIDE * _HOLD_DT
    expected_slow = initial * (1.0 + _SLOW_RATE * window)
    silent_stride_one = initial * (1.0 + _SLOW_RATE * _HOLD_DT) ** _HOLD_STRIDE
    simulation.set_density("slow", initial)
    simulation.install_equation_cadence(
        {"slow": slow_handle},
        model=model,
        include=include,
        cxx=default_cxx(),
        so_path=str(tmp_path / "amr_stride.so"),
        native_dimension=dimension,
        linear_rates={"slow": _SLOW_RATE},
        target="amr_system",
    )
    for _ in range(_HOLD_STRIDE - 1):
        simulation.step(_HOLD_DT)
        np.testing.assert_allclose(np.asarray(simulation.density("slow")), initial, atol=0.0)
    simulation.step(_HOLD_DT)
    got = np.asarray(simulation.density("slow"))
    np.testing.assert_allclose(got, expected_slow, atol=1.0e-13)
    assert float(np.max(np.abs(got - silent_stride_one))) > 1.0e-6


def test_partial_imex_mask_runs_on_amr(tmp_path: Path) -> None:
    _require_native()
    import pops.runtime._engine_descriptors as engine
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.runtime._system import AmrSystem, AmrSystemConfig

    dimension = _select_installed_native_dimension()
    if dimension != 2:
        pytest.skip("compiled AMR IMEX proof uses the installed rank-2 artifact")
    include = _compile_include()
    model = _imex_model("amr-partial-imex-model", dimension)
    case = Case("amr-partial-imex-case")
    handle = _state_handle(case, "fluid", model)
    compiled = _compile_block(
        model, include, owner="amr-partial-imex-case/fluid", target="amr_system"
    )
    config = AmrSystemConfig()
    config.shape = (_HOLD_CELLS, _HOLD_CELLS)
    config.lower = (0.0, 0.0)
    config.upper = (1.0, 1.0)
    config.periodicity = (True, True)
    config.regrid_every = 0
    simulation = AmrSystem(config)
    simulation.set_temporal_relations([2], [1], ["integral_only"])
    simulation.add_equation(
        "fluid",
        compiled,
        spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
        time=engine.IMEX(implicit_vars=["q"]),
    )
    initial = np.zeros((2, _HOLD_CELLS, _HOLD_CELLS), dtype=np.float64)
    initial[0] = 3.0
    initial[1] = 5.0
    simulation.set_conservative_state("fluid", initial)
    simulation.install_equation_cadence(
        {"fluid": handle},
        model=model,
        include=include,
        cxx=default_cxx(),
        so_path=str(tmp_path / "amr_partial_imex.so"),
        native_dimension=dimension,
        target="amr_system",
    )
    dt = 0.2
    simulation.step(dt)
    got = np.asarray(simulation.block_level_state_global("fluid", 0), dtype=np.float64).reshape(
        initial.shape
    )
    expected = np.array(initial, copy=True)
    expected[0] = 3.0 + dt * _EXPLICIT_SRC
    expected[1] = 5.0 + dt * _IMPLICIT_SRC
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1.0e-13)
