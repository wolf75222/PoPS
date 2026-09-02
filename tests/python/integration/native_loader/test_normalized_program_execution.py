"""ADC-668: manual and preset methods execute through one native ProgramGraph pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pytest

import pops.lib.time as libtime
from pops.codegen.program_graph_lowering import emit_program_graph
from pops.codegen.program_persistent_plan import (
    get_program_resource_plan,
    persistent_slot_token,
)
from pops.physics._facade import Model
from pops.problem import Case
from pops.solvers import DenseLU
from pops.time import (
    Clock,
    FailRun,
    LocalLinear,
    Program,
    ProgramGraph,
    SampleAndHold,
    StagePoint,
    TimePoint,
)
from pops.time._program.detach import detach_compiled_program
from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
    require_native_or_skip,
)


_EXPLICIT_RATE = -0.75
_IMPLICIT_RATE = -1.25


def _require_native() -> None:
    missing = missing_native_compile_requirement(repo_include(), default_cxx())
    if missing:
        require_native_or_skip(missing, optional_skip=pytest.skip)
    try:
        import pops.runtime._engine_descriptors  # noqa: F401
        import pops.runtime._system  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        require_native_or_skip(
            "normalized Program execution runtime unavailable: %s" % exc,
            optional_skip=pytest.skip,
        )


def _authoring() -> tuple[Any, Any, Any, Any]:
    model = Model("normalized-execution-model")
    (rho,) = model.conservative_vars("rho")
    model.primitive_vars(rho)
    model.conservative_from([rho])
    model.flux(x=[0.0 * rho], y=[0.0 * rho])
    model.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    model.source([_EXPLICIT_RATE * rho])
    explicit = model.rate("explicit_decay", flux=False, sources=("default",))
    implicit = model.local_linear_map("implicit_decay", [[_IMPLICIT_RATE]])
    block = Case("normalized-execution-case").block("fluid", model)
    declaration = next(
        record for record in model.declaration_index().records() if record.kind == "state"
    )
    return model, block[declaration], explicit, implicit


def _manual_ssprk2(state: Any, rate: Any) -> Program:
    program = Program("SSPRK2")
    temporal = program.state(state)
    u0 = temporal.n
    point0 = StagePoint("ssprk2_stage_0", {"main": TimePoint(program.clock, 0)})
    k0 = program.value("ssprk2_k_0", rate(u0), at=point0)
    point1 = StagePoint("ssprk2_stage_1", {"main": TimePoint(program.clock, 1)})
    u1 = program.value("ssprk2_U1", u0 + program.dt * k0, at=point1)
    k1 = program.value("ssprk2_k_1", rate(u1), at=point1)
    half = Fraction(1, 2)
    out = program.value(
        "ssprk2_step",
        u0 + program.dt * half * k0 + program.dt * half * k1,
        at=temporal.next.point,
    )
    program.commit(temporal.next, out)
    return program


def _manual_imex_euler(state: Any, explicit: Any, implicit: Any) -> Program:
    program = Program("IMEX")
    temporal = program.state(state)
    u0 = temporal.n
    point = StagePoint(
        "imex-euler_stage_0",
        {
            "explicit": TimePoint(program.clock, 0),
            "implicit": TimePoint(program.clock, 1),
        },
    )
    predictor = program.value("imex-euler_predictor_0", 1 * u0, at=point)
    linear = program.value("imex-euler_L_0", program._call(implicit), at=point)
    stage = program.solve(
        LocalLinear(operator=program.I - program.dt * linear, rhs=predictor),
        solver=DenseLU(),
        name="imex-euler_stage_solve_0",
    ).consume(action=FailRun())
    stage = program.value("imex-euler_stage_0", stage, at=point)
    explicit_rate = program.value("imex-euler_k_exp_0", explicit(stage), at=point)
    implicit_rate = program.value("imex-euler_k_imp_0", program.apply(linear, stage), at=point)
    out = program.value(
        "imex-euler_step",
        u0 + program.dt * explicit_rate + program.dt * implicit_rate,
        at=temporal.next.point,
    )
    program.commit(temporal.next, out)
    return program


@dataclass(frozen=True)
class _NormalizedProgram:
    authored: Program
    detached: Program
    graph: ProgramGraph
    source: str


def _normalize_and_lower(program: Program, model: Any) -> _NormalizedProgram:
    detached = detach_compiled_program(program)
    graph = detached.to_graph()
    source = emit_program_graph(graph, lowering_program=detached, model=model)
    assert detached.to_graph().graph_hash == graph.graph_hash
    return _NormalizedProgram(
        authored=program,
        detached=detached,
        graph=graph,
        source=source,
    )


def test_normalized_program_resource_plan_keeps_only_materialized_occurrences() -> None:
    """Accepted-state aliases never become unprimed runtime-sized plan rows."""

    model, state, explicit, implicit = _authoring()
    programs = (
        _manual_ssprk2(state, explicit),
        _manual_imex_euler(state, explicit, implicit),
    )
    prepare_slots = re.compile(
        r"ctx\.prepare_(?:rhs|state|scalar)_scratch\((\d+),"
    )
    for program in programs:
        plan = get_program_resource_plan(program, target="system")
        source = _normalize_and_lower(program, model).source
        prepared = {int(slot) for slot in prepare_slots.findall(source)}
        planned = {entry.slot for entry in plan.entries}
        # These two authored values are read-only aliases: the state is owned
        # by the accepted System and an outcome component aliases the prepared
        # solve result.  Neither can be materialized as an independent slot.
        aliases = [
            value
            for value in program._values
            if value.op in {"state", "solve_outcome_component"}
        ]
        assert aliases
        for alias in aliases:
            assert alias.id not in {entry.key.value_id for entry in plan.entries}
            with pytest.raises(KeyError):
                persistent_slot_token(program, alias, target="system")
        # The remaining symbolic rows are exactly the scratch owners emitted
        # into the install-time prelude; there is no unresolved slot left for
        # host materialization to guess or allocate during a step.
        assert planned == prepared
        assert list(sorted(planned)) == list(range(len(planned)))


@dataclass(frozen=True)
class _NativeResult:
    state: np.ndarray
    installed_hash: str
    time: float
    macro_step: int


def _install_and_step(
    compiled: Any, compiled_model: Any, initial: np.ndarray, dt: float
) -> _NativeResult:
    import pops.runtime._engine_descriptors as engine
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.runtime._system import System

    simulation = System(n=4, L=1.0, periodicity=(True, True))
    simulation.add_equation(
        "fluid",
        compiled_model,
        spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
        time=engine.Explicit(method="euler"),
    )
    simulation.set_state("fluid", np.stack([initial]))
    simulation.install_program(compiled.so_path)
    assert simulation.installed_program_hash() == compiled.program_hash
    simulation.step(dt)
    return _NativeResult(
        state=np.asarray(simulation.get_state("fluid"))[0].copy(),
        installed_hash=simulation.installed_program_hash(),
        time=float(simulation.time()),
        macro_step=int(simulation.macro_step()),
    )


def test_manual_preset_ssprk2_and_imex_execute_through_same_program_graph_pipeline(
    tmp_path: Path,
) -> None:
    _require_native()
    from pops.codegen._compile_drivers import compile_problem

    model, state, explicit, implicit = _authoring()
    authored = {
        "ssprk2_manual": _manual_ssprk2(state, explicit),
        "ssprk2_preset": libtime.SSPRK2(state, rate=explicit),
        "imex_manual": _manual_imex_euler(state, explicit, implicit),
        "imex_preset": libtime.IMEX(
            state,
            explicit_operator=explicit,
            implicit_operator=implicit,
        ),
    }
    normalized = {name: _normalize_and_lower(program, model) for name, program in authored.items()}

    for value in normalized.values():
        assert value.detached._compiled_detached is True
        assert not value.detached._operator_registries
        assert type(value.graph) is ProgramGraph
        assert value.source.count("state->step = [ctx_owner = state->ctx_owner](double dt)") == 1
        assert "ctx.install(" not in value.source
        assert value.source.count("ctx.begin_step(dt)") == 1
        assert value.source.count("ctx.commit_many(") == 1
        assert (
            value.source.index("state->step = [ctx_owner = state->ctx_owner](double dt)")
            < value.source.index("ctx.begin_step(dt)")
            < value.source.index("ctx.commit_many(")
        )

    for manual, preset in (
        ("ssprk2_manual", "ssprk2_preset"),
        ("imex_manual", "imex_preset"),
    ):
        assert normalized[manual].graph.to_data() == normalized[preset].graph.to_data()
        assert normalized[manual].graph.graph_hash == normalized[preset].graph.graph_hash
        assert normalized[manual].source == normalized[preset].source
    assert "pops::SolveOutcome" in normalized["imex_manual"].source
    assert normalized["imex_manual"].source.index("pops::SolveOutcome") < normalized[
        "imex_manual"
    ].source.index("ctx.commit_many(")

    compiled = {
        name: compile_problem(
            so_path=str(tmp_path / ("%s.so" % name)),
            model=model,
            time=value.authored,
            include=repo_include(),
            cxx=default_cxx(),
        )
        for name, value in normalized.items()
    }
    for name, value in compiled.items():
        assert Path(value.so_path).is_file()
        assert value.program.to_graph().graph_hash == normalized[name].graph.graph_hash
    assert compiled["ssprk2_manual"].program_hash == compiled["ssprk2_preset"].program_hash
    assert compiled["imex_manual"].program_hash == compiled["imex_preset"].program_hash

    compiled_model = model.compile(
        backend="production",
        include=repo_include(),
        cxx=default_cxx(),
    )
    axis = (np.arange(4, dtype=float) + 0.5) / 4.0
    x, y = np.meshgrid(axis, axis, indexing="ij")
    initial = 1.0 + 0.2 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    dt = 0.2
    results = {
        name: _install_and_step(value, compiled_model, initial, dt)
        for name, value in compiled.items()
    }

    z = dt * _EXPLICIT_RATE
    expected_ssprk2 = initial * (1.0 + z + 0.5 * z * z)
    implicit_stage = initial / (1.0 - dt * _IMPLICIT_RATE)
    expected_imex = initial + dt * (_EXPLICIT_RATE + _IMPLICIT_RATE) * implicit_stage
    for name in ("ssprk2_manual", "ssprk2_preset"):
        np.testing.assert_allclose(results[name].state, expected_ssprk2, rtol=0.0, atol=1.0e-13)
    for name in ("imex_manual", "imex_preset"):
        np.testing.assert_allclose(results[name].state, expected_imex, rtol=0.0, atol=1.0e-13)
    np.testing.assert_array_equal(
        results["ssprk2_manual"].state,
        results["ssprk2_preset"].state,
    )
    np.testing.assert_array_equal(
        results["imex_manual"].state,
        results["imex_preset"].state,
    )
    for name, result in results.items():
        assert result.installed_hash == compiled[name].program_hash
        assert result.time == pytest.approx(dt, rel=0.0, abs=1.0e-15)
        assert result.macro_step == 1
        assert np.isfinite(result.state).all()


_FAST_RATE = -0.5
_SLOW_RATE = -0.25
_HOLD_STRIDE = 4
_HOLD_DT = 0.1
_HOLD_CELLS = 4


def _select_installed_native_dimension() -> int:
    from pops._native_selector import select_native_dimension, selected_native_dimension

    selected = selected_native_dimension()
    if selected is not None:
        return selected
    configured = os.environ.get("POPS_NATIVE_DIM")
    if configured in {"1", "2", "3"}:
        return select_native_dimension(int(configured)).__native_dimension__
    import json

    import pops

    manifest = Path(pops.__file__).resolve().parent / "_native" / "variants.json"
    if not manifest.is_file():
        require_native_or_skip(
            "no installed native variant manifest", optional_skip=pytest.skip)
        raise AssertionError("require_native_or_skip must not return")
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


def _state_handle(case: Any, block_name: str, model: Any) -> Any:
    block = case.block(block_name, model)
    declaration = next(
        record for record in model.declaration_index().records() if record.kind == "state"
    )
    return block[declaration]


def _hold_catchup_program(fast_handle: Any, slow_handle: Any, *, count: int) -> Program:
    program = Program("two_species_hold_catchup")
    fast = program.state(fast_handle)
    slow = program.state(slow_handle)
    child = Clock("fast", owner=program.owner_path)
    child_state = program.state(fast_handle, clock=child)
    fast_on_child = program.synchronize(
        fast.n, at=TimePoint(child), relation=SampleAndHold(), name="fast_to_child")
    program.synchronize(
        slow.n, at=TimePoint(child), relation=SampleAndHold(), name="slow_held_on_fast")

    def _fast_tick(builder, value):
        return builder.value(
            "fast_tick",
            value + builder.dt * (_FAST_RATE * value),
            at=child_state.next.point,
        )

    advanced = program.subcycle(
        fast_on_child,
        clock=child,
        within=program.clock,
        count=count,
        body_fn=_fast_tick,
        name="fast_ticks",
    )
    program.commit(
        fast.next,
        program.synchronize(
            advanced, at=fast.next.point, relation=SampleAndHold(), name="fast_to_macro"),
    )
    program.commit(
        slow.next,
        program.value(
            "slow_catchup",
            slow.n + program.dt * (_SLOW_RATE * slow.n),
            at=slow.next.point,
        ),
    )
    return program


def _uniform_system(dimension: int, compiled_fast: Any, compiled_slow: Any):
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
    time = engine.Explicit(method="euler")
    simulation._batch_native_packages = True
    try:
        simulation.add_equation("fast", compiled_fast, spatial=spatial, time=time)
        simulation.add_equation("slow", compiled_slow, spatial=spatial, time=time)
    finally:
        simulation._batch_native_packages = False
    simulation._commit_pending_native_packages()
    return simulation


def test_two_clock_hold_then_catchup_executes_on_uniform_system(tmp_path: Path) -> None:
    """Prove frozen two-species hold-then-catch-up via Program clocks + subcycle + SampleAndHold."""
    _require_native()
    from pops.codegen._compile_drivers import compile_problem

    dimension = _select_installed_native_dimension()
    model = _decay_model("hold-catchup-model", dimension)
    case = Case("hold-catchup-case")
    fast_handle = _state_handle(case, "fast", model)
    slow_handle = _state_handle(case, "slow", model)
    program = _hold_catchup_program(fast_handle, slow_handle, count=_HOLD_STRIDE)

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
    compiled = compile_problem(
        so_path=str(tmp_path / "hold_catchup.so"),
        model=model,
        time=program,
        include=include,
        cxx=default_cxx(),
        native_dimension=dimension,
    )
    compiled_fast = model.compile(
        backend="production",
        include=include,
        cxx=default_cxx(),
        consumer_owner_qid="hold-catchup-case/fast",
    )
    compiled_slow = model.compile(
        backend="production",
        include=include,
        cxx=default_cxx(),
        consumer_owner_qid="hold-catchup-case/slow",
    )
    initial = np.full((1,) + (_HOLD_CELLS,) * dimension, 2.0)
    window = _HOLD_STRIDE * _HOLD_DT
    expected_fast = initial * (1.0 + _FAST_RATE * _HOLD_DT) ** _HOLD_STRIDE
    expected_slow = initial * (1.0 + _SLOW_RATE * window)
    not_slow_subcycled = initial * (1.0 + _SLOW_RATE * _HOLD_DT) ** _HOLD_STRIDE

    windowed = _uniform_system(dimension, compiled_fast, compiled_slow)
    windowed.set_state("fast", initial)
    windowed.set_state("slow", initial)
    windowed.install_program(compiled.so_path)
    windowed.step(window)
    got_fast = np.asarray(windowed.get_state("fast"))
    got_slow = np.asarray(windowed.get_state("slow"))
    np.testing.assert_allclose(got_fast, expected_fast, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(got_slow, expected_slow, rtol=0.0, atol=1.0e-13)
    assert float(np.max(np.abs(got_slow - not_slow_subcycled))) > 1.0e-6
    assert windowed.macro_step() == 1
    assert windowed.time() == pytest.approx(window, rel=0.0, abs=1.0e-15)

    held = _uniform_system(dimension, compiled_fast, compiled_slow)
    held.set_state("fast", initial)
    held.set_state("slow", initial)
    held.set_program_cadence(1, _HOLD_STRIDE)
    held.install_program(compiled.so_path)
    for _ in range(_HOLD_STRIDE - 1):
        held.step(_HOLD_DT)
        np.testing.assert_allclose(np.asarray(held.get_state("fast")), initial, atol=0.0)
        np.testing.assert_allclose(np.asarray(held.get_state("slow")), initial, atol=0.0)
    held.step(_HOLD_DT)
    np.testing.assert_allclose(np.asarray(held.get_state("fast")), expected_fast, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(held.get_state("slow")), expected_slow, rtol=0.0, atol=1.0e-13)
    assert held.macro_step() == _HOLD_STRIDE
    assert held.time() == pytest.approx(window, rel=0.0, abs=1.0e-15)
