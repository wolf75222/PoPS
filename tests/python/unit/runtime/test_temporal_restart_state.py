"""ADC-667 strict Uniform next-attempt checkpoint state."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from pops._bootstrap import StepAttemptRejected
from pops.runtime._native_step_target import native_step_target
from pops.runtime._program_cadence_checkpoint import (
    capture_program_cadence,
    prepare_program_cadence,
    restore_program_cadence,
)
from pops.runtime._step_strategy import (
    resolve_run_strategy,
    run_control_payload,
    run_step_attempt,
)
from pops.runtime._temporal_restart import TemporalRestartState
from pops.runtime._uniform_restart_preflight import preflight_uniform_restart
from pops.time import Clock, ErrorControlledDt, FixedDt, TimePoint


ROOT = Path(__file__).resolve().parents[4]


class _Native:
    def __init__(self, *, reject=False):
        self.t = 0.0
        self.cursor = 0
        self.reject = reject

    def time(self):
        return self.t

    def macro_step(self):
        return self.cursor

    def step(self, dt):
        if self.reject:
            raise StepAttemptRejected("rejected")
        self.t += dt
        self.cursor += 1


class _Engine:
    def __init__(self, native, state):
        self._s = native
        self._temporal_restart_state = state


def _bound_state(strategy=None):
    state = TemporalRestartState()
    state.begin_run(
        strategy or run_control_payload(FixedDt(0.125)),
        time=0.0,
        macro_step=0,
    )
    return state


def _bound_uniform_runtime(native_cxx, *, attempt_policy):
    """Compile and bind a real Uniform runtime with the requested native attempt policy."""
    if attempt_policy not in {"forced_reject", "error_retry"}:
        raise ValueError("attempt_policy must be 'forced_reject' or 'error_retry'")
    import pops
    from pops.codegen import Production
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.layouts import Uniform
    from pops.math import ddt, div
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.numerics.terms import Flux, SourceTerm
    from pops.physics import Model
    from pops.time import GuardRole, Program, RejectAttempt

    n = 4
    frame = Rectangle("temporal-rejection-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D()
    )
    x_axis, y_axis = frame.axes
    model = Model("temporal-rejection-model", frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
        waves={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
    )
    source_rate = 0.5
    source = model.source("forcing", on=state, value=(source_rate + 0.0 * rho,))
    source_operator = model.module.operator_handle("forcing")
    rate = model.rate("transport-rate", equation=ddt(state) == -div(flux) + source)
    case = pops.Case("temporal-rejection-case")
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
    program = Program("temporal_native_%s" % attempt_policy)
    temporal = program.state(block[state])
    rhs = program.rhs(state=temporal.n, terms=[Flux(), SourceTerm(source_operator)])
    candidate = program.value("candidate", temporal.n + program.dt * rhs, at=temporal.next.point)
    if attempt_policy == "forced_reject":
        candidate = program.guard(
            "forced_native_rejection",
            candidate,
            program.norm_inf(candidate) < 0.0,
            action=RejectAttempt(),
        )
        strategy = FixedDt(0.125)
    else:
        strategy = ErrorControlledDt(
            dt_init=0.125,
            rtol=1.0e-3,
            atol=1.0e-8,
            dt_min=0.01,
            dt_max=0.25,
            max_rejections=2,
            shrink=0.5,
            growth=1.25,
        )
        increment = program.value(
            "candidate_increment",
            candidate - temporal.n,
            at=temporal.next.point,
        )
        candidate = program.guard(
            "dt_dependent_error_estimate",
            candidate,
            program.norm_inf(increment) <= source_rate * strategy.dt_init * strategy.shrink,
            action=RejectAttempt(),
            role=GuardRole.ERROR_ESTIMATE,
        )
    program.commit(temporal.next, candidate)
    program.step_strategy(strategy)
    case.program(program)
    layout = Uniform(
        CartesianGrid(
            frame=frame,
            cells=(n, n),
            periodic=PeriodicAxes(frame.axes),
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
    initial = np.ones((1, n, n), dtype=np.float64)
    return pops.bind(artifact, initial_state={"blk": initial})


def _nested_schedule():
    macro = Clock("macro")
    child = Clock("chemistry")
    return macro, child, {
        "schema_version": 1,
        "kind": "pops.temporal-program-schedule",
        "primary_clock": macro.qualified_id,
        "clocks": [
            {"id": macro.qualified_id, "descriptor": macro.to_data(), "ticks_per_macro": 1},
            {"id": child.qualified_id, "descriptor": child.to_data(), "ticks_per_macro": 3},
        ],
        "subcycles": [{
            "node_id": 7, "parent_clock": macro.qualified_id,
            "child_clock": child.qualified_id, "count": 3,
        }],
        "synchronizations": [{
            "node_id": 8, "source_clock": macro.qualified_id,
            "target_clock": child.qualified_id,
            "relation": {
                "kind": "sample_and_hold",
                "schema_version": 1,
                "provider": {"kind": "latest_accepted_sample", "schema_version": 1},
            },
            "point": TimePoint(child).to_data(),
        }],
        "schedules": [],
        "histories": [],
    }


def _typed_history_schedule():
    macro, child, schedule = _nested_schedule()
    state = {
        "kind": "state",
        "qualified_id": "case/fluid/U",
        "block_ref": {"kind": "block", "qualified_id": "case/fluid"},
    }
    space = {"kind": "state", "name": "U", "components": ["rho"]}
    interpolation = {
        "kind": "linear",
        "schema_version": 1,
        "minimum_samples": 2,
    }
    validity = {
        "schema_version": 1,
        "oldest": TimePoint(macro, step=-2).to_data(),
        "newest": TimePoint(macro).to_data(),
    }
    contract = {
        "schema_version": 1,
        "owner": macro.to_data()["owner"],
        "state": state,
        "space": space,
        "clock": macro.to_data(),
        "validity": validity,
        "interpolation": interpolation,
        "depth": 2,
    }
    schedule["synchronizations"][0]["relation"] = {
        "kind": "history_interpolation",
        "schema_version": 1,
        "provider": {
            "kind": "typed_history",
            "schema_version": 1,
            "contract": deepcopy(contract),
        },
        "interpolation": deepcopy(interpolation),
    }
    schedule["histories"] = [{
        "name": "fluid.U",
        "owner": state["block_ref"],
        "state": deepcopy(state),
        "space": deepcopy(space),
        "clock": macro.qualified_id,
        "depth": 2,
        "ring_slots": 3,
        "ncomp": None,
        "validity": deepcopy(validity),
        "interpolation": deepcopy(interpolation),
        "checkpoint_policy": None,
    }]
    return macro, child, schedule


def test_accepted_attempt_advances_cursor_and_round_trips_exact_controller_state():
    native = _Native()
    state = _bound_state()
    run_step_attempt(_Engine(native, state), native, FixedDt(0.125), t_end=1.0)

    payload = state.checkpoint_json(time=native.time(), macro_step=native.macro_step())
    restored = TemporalRestartState.from_json(
        np.array(payload), time=native.time(), macro_step=native.macro_step()
    )
    data = restored.to_data()
    assert data["schedule_cursors"] == {
        "macro_step": {"macro_step": 1, "phase": "accepted"},
    }
    assert data["controller_state"]["last_accepted_dt"] == (0.125).hex()
    assert data["transaction_stats"] == {"accepted": 1, "rejected": 0, "failed": 0}
    restored.begin_run(run_control_payload(FixedDt(0.125)), time=0.125, macro_step=1)
    with pytest.raises(RuntimeError, match="checkpointed step strategy"):
        restored.begin_run(run_control_payload(FixedDt(0.25)), time=0.125, macro_step=1)


def test_system_direct_step_publishes_one_synchronized_fixed_dt_restart_envelope():
    """The real low-level System seam reports the accepted direct step without private reads."""
    import pops.runtime._engine_descriptors as engine
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.runtime._system import System
    from tests.python.support.explicit_program import install_forward_euler_program

    n = 8
    dt = 0.01
    system = System(n=n, L=1.0, periodicity=(True, True))
    system.add_equation(
        "scalar",
        engine.Model(
            state=engine.FluidState("isothermal", cs2=0.5),
            transport=engine.IsothermalFlux(),
            source=engine.NoSource(),
            elliptic=engine.BackgroundDensity(alpha=0.0, n0=0.0),
        ),
        spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
        time=engine.Explicit(method="euler"),
    )
    coordinates = (np.arange(n, dtype=np.float64) + 0.5) / n
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    rho = 1.0 + 0.2 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    initial = np.stack((rho, 0.3 * rho, -0.1 * rho))
    system.set_state("scalar", initial)
    install_forward_euler_program(system)

    system.step(dt)

    assert system.macro_step() == 1
    assert system.time() == pytest.approx(dt, rel=0.0, abs=1e-15)
    assert not np.array_equal(np.asarray(system.get_state("scalar")), initial)
    temporal = system.program_report().temporal
    assert temporal["strategy"] == run_control_payload(FixedDt(dt))
    assert temporal["clock"] == {"time": float(dt).hex(), "macro_step": 1}
    assert temporal["schedule_cursors"] == {
        "macro_step": {"macro_step": 1, "phase": "accepted"},
    }
    assert temporal["controller_state"] == {"last_accepted_dt": float(dt).hex()}
    assert temporal["transaction_stats"] == {"accepted": 1, "rejected": 0, "failed": 0}
    assert temporal["status"] == "accepted"
    assert temporal["synchronized"] is True


def test_nested_clock_cursors_round_trip_at_only_the_accepted_boundary():
    macro, child, schedule = _nested_schedule()
    state = TemporalRestartState()
    state.configure_program(schedule, time=0.0, macro_step=0)
    state.begin_run(run_control_payload(FixedDt(0.125)), time=0.0, macro_step=0)
    state.accept(before_time=0.0, before_step=0, time=0.125, macro_step=1)

    assert state.cursor_for_clock(macro)["tick"] == 1
    assert state.cursor_for_clock(child)["tick"] == 3
    assert state.schedule_cursors["subcycle:7"]["next_iteration"] == 0
    assert state.synchronization_cursors["8"] == {
        "macro_step": 1,
        "source_tick": 1,
        "target_tick": 3,
        "phase": "accepted",
    }

    payload = state.checkpoint_json(time=0.125, macro_step=1)
    restored = TemporalRestartState.from_json(
        payload, time=0.125, macro_step=1, program_schedule=schedule
    )
    assert restored.cursor_for_clock(child) == state.cursor_for_clock(child)
    with pytest.raises(RuntimeError, match="no cursor for qualified clock"):
        restored.cursor_for_clock(Clock("unrelated"))


def test_restart_rejects_a_different_installed_nested_clock_schedule():
    _, _, schedule = _nested_schedule()
    state = TemporalRestartState()
    state.configure_program(schedule, time=0.0, macro_step=0)
    state.begin_run(run_control_payload(FixedDt(0.125)), time=0.0, macro_step=0)
    payload = state.checkpoint_json(time=0.0, macro_step=0)
    changed = json.loads(json.dumps(schedule))
    changed["subcycles"][0]["count"] = 2
    changed["clocks"][1]["ticks_per_macro"] = 2
    with pytest.raises(ValueError, match="differs from installed program"):
        TemporalRestartState.from_json(payload, time=0.0, macro_step=0, program_schedule=changed)


def test_temporal_schedule_refuses_a_providerless_cross_clock_relation():
    _, _, schedule = _nested_schedule()
    schedule["synchronizations"][0]["relation"].pop("provider")

    with pytest.raises(ValueError, match="explicit provider"):
        TemporalRestartState().configure_program(schedule, time=0.0, macro_step=0)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("state", {
            "kind": "state",
            "qualified_id": "case/other/V",
            "block_ref": {"kind": "block", "qualified_id": "case/other"},
        }),
        ("space", {"kind": "state", "name": "V", "components": ["energy"]}),
        ("depth", 3),
        ("validity", {
            "schema_version": 1,
            "oldest": TimePoint(Clock("macro"), step=-1).to_data(),
            "newest": TimePoint(Clock("macro")).to_data(),
        }),
        ("interpolation", {"kind": "dense_output", "schema_version": 1, "order": 2}),
    ],
)
def test_temporal_schedule_rejects_history_provider_registry_drift(
        field, replacement):
    _, _, schedule = _typed_history_schedule()
    contract = schedule["synchronizations"][0]["relation"]["provider"]["contract"]
    contract[field] = replacement

    with pytest.raises(ValueError, match="provider"):
        TemporalRestartState().configure_program(schedule, time=0.0, macro_step=0)


def test_temporal_schedule_rejects_history_provider_source_clock_drift():
    _, child, schedule = _typed_history_schedule()
    relation = schedule["synchronizations"][0]["relation"]
    relation["provider"]["contract"]["clock"] = child.to_data()
    relation["provider"]["contract"]["owner"] = child.to_data()["owner"]
    relation["provider"]["contract"]["validity"] = {
        "schema_version": 1,
        "oldest": TimePoint(child, step=-2).to_data(),
        "newest": TimePoint(child).to_data(),
    }

    with pytest.raises(ValueError, match="source clock"):
        TemporalRestartState().configure_program(schedule, time=0.0, macro_step=0)


@pytest.mark.compiler
@pytest.mark.native_loader
def test_rejection_preserves_native_cursor_and_makes_checkpoint_ineligible(
    tmp_path,
    isolated_native_cache,
    native_cxx,
    kokkos_root,
):
    del isolated_native_cache, kokkos_root
    runtime = _bound_uniform_runtime(native_cxx, attempt_policy="forced_reject")
    engine = runtime._executor
    native = native_step_target(engine)
    initial = np.asarray(runtime.state_global("blk"), dtype=np.float64).copy()
    with pytest.raises(StepAttemptRejected):
        run_step_attempt(engine, native, FixedDt(0.125), t_end=0.125)

    assert (runtime.time(), runtime.macro_step()) == (0.0, 0)
    assert np.array_equal(np.asarray(runtime.state_global("blk"), dtype=np.float64), initial), (
        "the rejected native attempt must roll back the complete state"
    )
    temporal = runtime.program_report().temporal
    assert temporal["transaction_stats"] == {
        "accepted": 0,
        "rejected": 1,
        "failed": 0,
    }
    assert temporal["status"] == "rejected"
    assert temporal["synchronized"] is False
    target = tmp_path / "must_not_exist.npz"
    with pytest.raises(RuntimeError, match="accepted synchronized"):
        runtime.checkpoint(target)
    assert not target.exists()

    retrying = _bound_uniform_runtime(native_cxx, attempt_policy="error_retry")
    retrying_engine = retrying._executor
    retrying_initial = np.asarray(retrying.state_global("blk"), dtype=np.float64).copy()
    report = run_step_attempt(
        retrying_engine,
        native_step_target(retrying_engine),
        resolve_run_strategy(retrying_engine),
        t_end=0.125,
    )
    assert report.status == "accepted"
    assert report.attempts == 2
    assert retrying.time() == pytest.approx(0.0625, rel=0.0, abs=1.0e-15)
    assert retrying.macro_step() == 1
    assert np.allclose(
        np.asarray(retrying.state_global("blk"), dtype=np.float64),
        retrying_initial + 0.5 * 0.0625,
        rtol=0.0,
        atol=1.0e-14,
    ), "only the accepted retry may update the runtime state"
    retrying_temporal = retrying.program_report().temporal
    assert retrying_temporal["transaction_stats"] == {
        "accepted": 1,
        "rejected": 1,
        "failed": 0,
    }
    assert retrying_temporal["controller_state"] == {
        "last_accepted_dt": (0.0625).hex(),
    }
    assert retrying_temporal["status"] == "accepted"
    assert retrying_temporal["synchronized"] is True


def test_strict_temporal_manifest_refuses_missing_or_unsynchronized_state():
    state = _bound_state()
    payload = json.loads(state.checkpoint_json(time=0.0, macro_step=0))
    payload.pop("event_queue")
    with pytest.raises(ValueError, match="incomplete strict manifest"):
        TemporalRestartState.from_json(json.dumps(payload), time=0.0, macro_step=0)

    payload = json.loads(state.checkpoint_json(time=0.0, macro_step=0))
    payload["synchronized"] = False
    payload["status"] = "rejected"
    with pytest.raises(ValueError, match="not an accepted synchronized point"):
        TemporalRestartState.from_json(json.dumps(payload), time=0.0, macro_step=0)


@pytest.mark.parametrize(
    ("section", "value", "message"),
    [
        ("strategy", {**run_control_payload(FixedDt(0.1)), "extra": True}, "strategy"),
        ("controller_state", {"last_accepted_dt": "0x1p-3", "extra": 0}, "controller"),
        ("event_queue", [{"kind": "output"}], "event"),
        ("transaction_stats", {"accepted": 0, "rejected": -1, "failed": 0}, "statistics"),
    ],
)
def test_strict_temporal_sections_reject_extra_keys_and_invalid_values(section, value, message):
    state = _bound_state()
    payload = json.loads(state.checkpoint_json(time=0.0, macro_step=0))
    payload[section] = value
    with pytest.raises((TypeError, ValueError), match=message):
        TemporalRestartState.from_json(np.array(json.dumps(payload)), time=0.0, macro_step=0)


class _Payload(dict):
    @property
    def files(self):
        return list(self)


def test_uniform_preflight_rejects_incomplete_dynamic_indexes_before_native_restore():
    payload = _Payload(
        {
            "program_hash": np.array("ab" * 32),
            "history_names": np.array([], dtype="U1"),
            "cache_nodes": np.array([], dtype=np.int64),
            "cache_names": np.array([], dtype="U1"),
            "temporal_restart_state": np.array("{}"),
            "program_cadence_substeps": np.array(1, dtype=np.int64),
            "program_cadence_stride": np.array(1, dtype=np.int64),
            "program_cadence_window_steps": np.array(0, dtype=np.int64),
            "program_cadence_window_dt": np.array(0.0, dtype=np.float64),
            "program_cadence_window_start_time": np.array(0.0, dtype=np.float64),
            "program_last_dt": np.array(0.0, dtype=np.float64),
        }
    )
    preflight_uniform_restart(payload)

    payload["history_names"] = np.array(["rhs"])
    with pytest.raises(ValueError, match="history 'rhs'.*incomplete strict manifest"):
        preflight_uniform_restart(payload)


class _CadenceEngine:
    def __init__(
        self, *, substeps=2, stride=3, held_steps=2, dt=0.3, start=4.0, last_dt=0.07
    ):
        self.substeps = substeps
        self.stride = stride
        self.held_steps = held_steps
        self.dt = dt
        self.start = start
        self.last_dt = last_dt
        self.restored = None

    def program_substeps(self):
        return self.substeps

    def time(self):
        return 4.3

    def program_stride(self):
        return self.stride

    def program_cadence_window_steps(self):
        return self.held_steps

    def program_cadence_window_dt(self):
        return self.dt

    def program_cadence_window_start_time(self):
        return self.start

    def program_last_dt(self):
        return self.last_dt

    def restore_program_cadence_window(
        self,
        accumulated_dt,
        held_steps,
        window_start_time,
        accepted_last_dt,
        accepted_time,
        macro_step,
    ):
        self.restored = (
            accumulated_dt,
            held_steps,
            window_start_time,
            accepted_last_dt,
            accepted_time,
            macro_step,
        )


def test_program_cadence_checkpoint_preserves_exact_variable_dt_window():
    engine = _CadenceEngine()
    captured = capture_program_cadence(engine, macro_step=5)
    payload = _Payload({key: np.array(value) for key, value in captured.to_payload().items()})

    prepared = prepare_program_cadence(engine, payload, macro_step=5, accepted_time=4.3)
    assert prepared == captured
    assert prepared.to_data()["accumulated_dt"] == (0.3).hex()
    assert prepared.to_data()["window_start_time"] == (4.0).hex()
    assert prepared.to_data()["last_dt"] == (0.07).hex()

    restore_program_cadence(engine, prepared, macro_step=5, accepted_time=4.3)
    assert engine.restored == (0.3, 2, 4.0, 0.07, 4.3, 5)


def test_uniform_restart_restores_clock_before_selective_history_replay(monkeypatch):
    from pops.runtime._program_cadence_checkpoint import ProgramCadenceCheckpointState
    from pops.runtime._system_io import _PreparedUniformRestart, _SystemIO
    import pops.runtime._system_io_history as history_io

    events = []

    class Native:
        def __init__(self):
            self.clock = (0.0, 0)
            self.staged = None

        def set_state(self, block, values):
            events.append(("state", block))

        def set_potential(self, values):
            events.append(("potential", len(values)))

        def set_clock(self, time, macro_step):
            assert self.staged == (0.125, time, macro_step)
            self.clock = (time, macro_step)
            events.append(("clock", time, macro_step))

        def restore_program_cadence_window(
            self,
            accumulated_dt,
            held_steps,
            window_start_time,
            accepted_last_dt,
            accepted_time,
            macro_step,
        ):
            assert (accumulated_dt, held_steps, window_start_time) == (0.0, 0, 0.0)
            self.staged = (accepted_last_dt, accepted_time, macro_step)
            events.append(("cadence", accepted_time, macro_step))

    native = Native()

    def assert_checkpoint_cursor_before_replay(sim, payload):
        assert sim is native
        assert sim.clock == (1.5, 3)
        events.append(("history", sim.clock))
        return "replay-report"

    monkeypatch.setattr(history_io, "restore_histories", assert_checkpoint_cursor_before_replay)
    payload = {
        "blocks": np.array(["blk"]),
        "state_blk": np.array([1.0], dtype=np.float64),
        "phi": np.array([0.0], dtype=np.float64),
        "field_provider_slots": np.array([], dtype="U1"),
        "history_names": np.array(["blk.state"]),
        "cache_names": np.array([], dtype="U1"),
        "cache_nodes": np.array([], dtype=np.int64),
        "t": np.array(1.5, dtype=np.float64),
        "macro_step": np.array(3, dtype=np.int64),
    }
    prepared = _PreparedUniformRestart(
        payload=payload,
        restart_identity="restart-id",
        temporal_state="temporal-state",
        cadence_state=ProgramCadenceCheckpointState(
            substeps=1,
            stride=1,
            held_steps=0,
            accumulated_dt=0.0,
            window_start_time=0.0,
            last_dt=0.125,
        ),
    )

    class Owner:
        _s = native
        _temporal_restart_state = "old-temporal-state"
        _step_controller = object()

    owner = Owner()
    assert _SystemIO._apply_checkpoint_restart(owner, prepared) == "restart-id"
    assert owner._last_restart_report == "replay-report"
    assert events.index(("clock", 1.5, 3)) < events.index(("history", (1.5, 3)))


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda payload: payload.pop("program_cadence_window_dt"), "missing"),
        (
            lambda payload: payload.__setitem__("program_cadence_window_steps", np.array(2.0)),
            "exact integer",
        ),
        (
            lambda payload: payload.__setitem__(
                "program_cadence_window_steps", np.array(1, dtype=np.int64)
            ),
            "macro-step phase",
        ),
        (
            lambda payload: payload.__setitem__(
                "program_cadence_window_dt", np.array(float("inf"))
            ),
            "finite",
        ),
        (
            lambda payload: payload.__setitem__(
                "program_cadence_window_dt", np.array(0.3, dtype=np.float32)
            ),
            "binary64",
        ),
        (
            lambda payload: payload.__setitem__(
                "program_last_dt", np.array(-0.01, dtype=np.float64)
            ),
            "last dt",
        ),
        (
            lambda payload: payload.__setitem__(
                "program_last_dt", np.array(0.07, dtype=np.float32)
            ),
            "binary64",
        ),
        (
            lambda payload: payload.__setitem__(
                "program_cadence_stride", np.array(4, dtype=np.int64)
            ),
            "macro-step phase",
        ),
    ],
)
def test_program_cadence_checkpoint_rejects_incomplete_or_inconsistent_state(mutation, error):
    engine = _CadenceEngine()
    payload = _Payload(
        {
            "program_cadence_substeps": np.array(2, dtype=np.int64),
            "program_cadence_stride": np.array(3, dtype=np.int64),
            "program_cadence_window_steps": np.array(2, dtype=np.int64),
            "program_cadence_window_dt": np.array(0.3, dtype=np.float64),
            "program_cadence_window_start_time": np.array(4.0, dtype=np.float64),
            "program_last_dt": np.array(0.07, dtype=np.float64),
        }
    )
    mutation(payload)
    with pytest.raises((TypeError, ValueError), match=error):
        prepare_program_cadence(engine, payload, macro_step=5, accepted_time=4.3)


def test_program_cadence_checkpoint_rejects_installed_configuration_mismatch():
    engine = _CadenceEngine(stride=4, held_steps=1)
    payload = _Payload(
        {
            "program_cadence_substeps": np.array(2, dtype=np.int64),
            "program_cadence_stride": np.array(3, dtype=np.int64),
            "program_cadence_window_steps": np.array(2, dtype=np.int64),
            "program_cadence_window_dt": np.array(0.3, dtype=np.float64),
            "program_cadence_window_start_time": np.array(4.0, dtype=np.float64),
            "program_last_dt": np.array(0.07, dtype=np.float64),
        }
    )
    with pytest.raises(ValueError, match="differs from the installed cadence"):
        prepare_program_cadence(engine, payload, macro_step=5, accepted_time=4.3)
