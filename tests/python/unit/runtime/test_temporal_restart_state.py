"""ADC-667 strict Uniform next-attempt checkpoint state."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pops._bootstrap import StepAttemptRejected
from pops.runtime._native_step_target import native_step_target
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
        time=0.0, macro_step=0,
    )
    return state


def _uniform_artifact(native_cxx, *, attempt_policy):
    """Compile a real Uniform artifact with the requested native attempt policy."""
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
    from pops.time import (
        AcceptedStep,
        Every,
        GuardRole,
        Hold,
        Program,
        RejectAttempt,
        Schedule,
    )

    n = 4
    frame = Rectangle(
        "temporal-rejection-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
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
    source_value = (
        source_rate + 0.0 * rho
        if attempt_policy == "forced_reject"
        else source_rate * rho
    )
    source = model.source("forcing", on=state, value=(source_value,))
    rate = model.rate(
        "transport-rate", equation=ddt(state) == -div(flux) + source)
    held_copy = model.local_transform("held_copy", (rho,))
    if attempt_policy == "error_retry":
        model.module.operator_capabilities("held_copy", cacheable=True)
    source_operator = model.module.operator_handle("forcing")
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
    if attempt_policy == "forced_reject":
        rhs = program.rhs(
            state=temporal.n, terms=[Flux(), SourceTerm(source_operator)])
        candidate = program.value(
            "candidate", temporal.n + program.dt * rhs, at=temporal.next.point)
        candidate = program.guard(
            "forced_native_rejection",
            candidate,
            program.norm_inf(candidate) < 0.0,
            action=RejectAttempt(),
        )
        strategy = FixedDt(0.125)
    else:
        schedule = Schedule(
            Every(AcceptedStep(program.clock), 2),
            off=Hold(),
        )
        held = held_copy(
            temporal.n,
            name="held_state",
            schedule=schedule,
        )
        rhs = program.rhs(
            state=temporal.n, terms=[Flux(), SourceTerm(source_operator)])
        program.store_history("blk.held_state", held)
        previous = program.history(
            "blk.held_state",
            lag=1,
            space=held.space,
            block=temporal.block,
            state_ref=temporal.state,
        )
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
        candidate = program.value(
            "candidate",
            held + program.dt * rhs + 0.125 * (held - previous),
            at=temporal.next.point,
        )
        increment = program.value(
            "candidate_increment",
            candidate - temporal.n,
            at=temporal.next.point,
        )
        candidate = program.guard(
            "dt_dependent_error_estimate",
            candidate,
            program.norm_inf(increment)
            <= source_rate * strategy.dt_init * strategy.shrink,
            action=RejectAttempt(),
            role=GuardRole.ERROR_ESTIMATE,
        )
    program.commit(temporal.next, candidate)
    program.step_strategy(strategy)
    case.program(program)
    layout = Uniform(CartesianGrid(
        frame=frame,
        cells=(n, n),
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
    return artifact


def _bind_uniform_artifact(artifact):
    import pops

    n = 4
    initial = np.ones((1, n, n), dtype=np.float64)
    return pops.bind(artifact, initial_state={"blk": initial})


def _bound_uniform_runtime(native_cxx, *, attempt_policy):
    """Compile and bind a real Uniform runtime with the requested native attempt policy."""
    return _bind_uniform_artifact(
        _uniform_artifact(native_cxx, attempt_policy=attempt_policy))


def _write_historical_uniform_v2(current, owner, destination):
    """Project one current test checkpoint onto the exact historical Uniform v2 schema."""
    from pops.runtime._checkpoint_manifest import (
        IDENTITY_KEY,
        MANIFEST_KEY,
        seal_checkpoint_payload,
    )
    from pops.time._history.persistence import HistoryPersistence

    with np.load(current, allow_pickle=False) as stored:
        legacy = {
            name: np.array(stored[name], copy=True)
            for name in stored.files
            if name not in {
                MANIFEST_KEY,
                IDENTITY_KEY,
                "temporal_restart_state",
                "field_provider_slots",
                "runtime_consumer_graph",
                "runtime_consumer_cursors",
                "runtime_consumer_diagnostics",
            }
            and not name.startswith("history_fill_count_")
            and not name.startswith("history_requested_stored_slots_")
            and not name.startswith("history_storage_mode_")
        }
    legacy["pops_checkpoint_version"] = np.asarray(2, dtype=np.int64)
    for name in (str(value) for value in legacy["history_names"]):
        policy = HistoryPersistence.from_json(str(legacy["history_policy_" + name]))
        old_manifest = {"kind": policy.kind, **policy.options()}
        legacy["history_policy_" + name] = np.asarray(json.dumps(
            old_manifest, sort_keys=True, separators=(",", ":")))
    seal_checkpoint_payload(owner, legacy, runtime_kind="uniform")
    with open(destination, "wb") as stream:
        np.savez_compressed(stream, **legacy)
    return destination


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
            "relation": {"kind": "sample_and_hold", "schema_version": 1},
            "point": TimePoint(child).to_data(),
        }],
        "schedules": [],
        "histories": [],
    }


def test_accepted_attempt_advances_cursor_and_round_trips_exact_controller_state():
    native = _Native()
    state = _bound_state()
    run_step_attempt(_Engine(native, state), native, FixedDt(0.125), t_end=1.0)

    payload = state.checkpoint_json(time=native.time(), macro_step=native.macro_step())
    restored = TemporalRestartState.from_json(
        np.array(payload), time=native.time(), macro_step=native.macro_step())
    data = restored.to_data()
    assert data["schedule_cursors"] == {
        "macro_step": {"macro_step": 1, "phase": "accepted"},
    }
    assert data["controller_state"]["last_accepted_dt"] == (0.125).hex()
    assert data["transaction_stats"] == {"accepted": 1, "rejected": 0, "failed": 0}
    restored.begin_run(
        run_control_payload(FixedDt(0.125)), time=0.125, macro_step=1)
    with pytest.raises(RuntimeError, match="checkpointed step strategy"):
        restored.begin_run(
            run_control_payload(FixedDt(0.25)), time=0.125, macro_step=1)


def test_system_direct_step_publishes_one_synchronized_fixed_dt_restart_envelope():
    """The real low-level System seam reports the accepted direct step without private reads."""
    import pops.runtime._engine_descriptors as engine
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.runtime._system import System

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
        "macro_step": 1, "source_tick": 1, "target_tick": 3, "phase": "accepted",
    }

    payload = state.checkpoint_json(time=0.125, macro_step=1)
    restored = TemporalRestartState.from_json(
        payload, time=0.125, macro_step=1, program_schedule=schedule)
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
        TemporalRestartState.from_json(
            payload, time=0.0, macro_step=0, program_schedule=changed)


@pytest.mark.compiler
@pytest.mark.native_loader
def test_rejection_preserves_native_cursor_and_makes_checkpoint_ineligible(
    tmp_path, monkeypatch, isolated_native_cache, native_cxx, kokkos_root,
):
    del isolated_native_cache, kokkos_root
    runtime = _bound_uniform_runtime(native_cxx, attempt_policy="forced_reject")
    engine = runtime._executor
    native = native_step_target(engine)
    initial = np.asarray(runtime.state_global("blk"), dtype=np.float64).copy()
    with pytest.raises(StepAttemptRejected):
        run_step_attempt(engine, native, FixedDt(0.125), t_end=0.125)

    assert (runtime.time(), runtime.macro_step()) == (0.0, 0)
    assert np.array_equal(
        np.asarray(runtime.state_global("blk"), dtype=np.float64), initial
    ), "the rejected native attempt must roll back the complete state"
    temporal = runtime.program_report().temporal
    assert temporal["transaction_stats"] == {
        "accepted": 0, "rejected": 1, "failed": 0,
    }
    assert temporal["status"] == "rejected"
    assert temporal["synchronized"] is False
    target = tmp_path / "must_not_exist.npz"
    with pytest.raises(RuntimeError, match="accepted synchronized"):
        runtime.checkpoint(target)
    assert not target.exists()

    retrying = _bound_uniform_runtime(native_cxx, attempt_policy="error_retry")
    retrying_engine = retrying._executor
    retrying_initial = np.asarray(
        retrying.state_global("blk"), dtype=np.float64).copy()
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
        retrying_initial * (1.0 + 0.5 * 0.0625),
        rtol=0.0,
        atol=1.0e-14,
    ), "only the accepted retry may update the runtime state"
    retrying_temporal = retrying.program_report().temporal
    assert retrying_temporal["transaction_stats"] == {
        "accepted": 1, "rejected": 1, "failed": 0,
    }
    assert retrying_temporal["controller_state"] == {
        "last_accepted_dt": (0.0625).hex(),
    }
    assert retrying_temporal["status"] == "accepted"
    assert retrying_temporal["synchronized"] is True

    # Public lifecycle proof: adaptive retry + schedule cache + AB2 history survive an accepted
    # checkpoint.  The first post-restart accepted dt and every later state are exact.
    import pops
    from pops.codegen.checkpoint_migration import (
        UniformCheckpointMigrationState,
        migrate_uniform_checkpoint,
    )

    artifact = retrying._install_plan.artifact

    def fresh():
        return _bind_uniform_artifact(artifact)

    total_steps = 6
    checkpoint_step = 3

    def accepted_deadline(count):
        probe = fresh()
        with pytest.raises(RuntimeError, match="max_steps exhausted before t_end"):
            pops.run(probe, t_end=10.0, max_steps=count, console=False)
        assert probe.macro_step() == count
        return probe.time()

    checkpoint_deadline = accepted_deadline(checkpoint_step)
    next_deadline = accepted_deadline(checkpoint_step + 1)
    final_deadline = accepted_deadline(total_steps)

    reference = fresh()
    pops.run(
        reference, t_end=final_deadline, max_steps=total_steps, console=False)

    split = fresh()
    pops.run(
        split, t_end=checkpoint_deadline, max_steps=checkpoint_step, console=False)
    checkpoint = Path(split.checkpoint(tmp_path / "adaptive-current"))
    with np.load(checkpoint, allow_pickle=False) as stored:
        checkpoint_time = float(stored["t"])
        assert checkpoint_time == checkpoint_deadline
        checkpoint_temporal = json.loads(str(stored["temporal_restart_state"]))
        history_names = tuple(str(value) for value in stored["history_names"])
        cache_nodes = tuple(int(value) for value in stored["cache_nodes"])
        assert history_names
        assert cache_nodes
        assert checkpoint_temporal["schedule_cursors"]
        assert checkpoint_temporal["history_cursors"]
        assert checkpoint_temporal["cache_cursors"]
        history_values = {
            (name, slot): np.array(stored["history_%s_%d" % (name, slot)])
            for name in history_names
            for slot in (int(value) for value in stored[
                "history_stored_slots_" + name
            ])
        }
        assert all("cache_value_%d" % node in stored.files for node in cache_nodes)

    resumed = fresh()
    resumed.restart(checkpoint)
    assert resumed.program_report().temporal == checkpoint_temporal
    for (name, slot), values in history_values.items():
        assert np.array_equal(
            np.asarray(resumed.history_global(name, slot)), values)

    # Build an exact historical fixture, then exercise the public offline migration.  The runtime
    # loader itself continues to reject v2; only the migrated current payload is restarted.
    historical = _write_historical_uniform_v2(
        checkpoint, split, tmp_path / "adaptive-v2.npz")
    descriptor = ErrorControlledDt.from_data(
        checkpoint_temporal["strategy"]["strategy"])
    controls = descriptor.restore_runtime_controls(
        checkpoint_temporal["strategy"]["controls"])
    migration_state = UniformCheckpointMigrationState(
        step_strategy=descriptor,
        controls=controls,
        last_accepted_dt=float.fromhex(
            checkpoint_temporal["controller_state"]["last_accepted_dt"]),
        transaction_stats=checkpoint_temporal["transaction_stats"],
    )

    # An incomplete historical envelope and a refusal by the exact current preflight are both
    # publication failures: neither may leave even a destination archive behind.
    with np.load(historical, allow_pickle=False) as stored:
        incomplete_payload = {
            name: np.array(stored[name], copy=True)
            for name in stored.files
            if name != "state_blk"
        }
    incomplete = tmp_path / "adaptive-v2-incomplete.npz"
    np.savez_compressed(incomplete, **incomplete_payload)
    incomplete_destination = tmp_path / "must-not-publish-incomplete"
    with pytest.raises(ValueError, match="NPZ keys differ|digest mismatch"):
        migrate_uniform_checkpoint(
            incomplete,
            incomplete_destination,
            runtime=fresh(),
            state=migration_state,
        )
    assert not incomplete_destination.with_suffix(".npz").exists()

    refused_destination = tmp_path / "must-not-publish-preflight"
    preflight_runtime = fresh()
    with monkeypatch.context() as patch:
        def refuse_current_preflight(self, payload):
            del self, payload
            raise ValueError("forced current-format preflight refusal")

        patch.setattr(
            type(preflight_runtime),
            "_inspect_checkpoint_payload",
            refuse_current_preflight,
        )
        with pytest.raises(ValueError, match="forced current-format preflight refusal"):
            migrate_uniform_checkpoint(
                historical,
                refused_destination,
                runtime=preflight_runtime,
                state=migration_state,
            )
    assert not refused_destination.with_suffix(".npz").exists()

    migrated_runtime = fresh()
    migration = migrate_uniform_checkpoint(
        historical.with_suffix(""),
        tmp_path / "adaptive-migrated",
        runtime=migrated_runtime,
        state=migration_state,
    )
    assert migration.from_payload_version == 2
    assert migration.to_payload_version == 3
    assert migration.source_restart_identity != migration.migrated_restart_identity
    assert Path(migration.destination) == tmp_path / "adaptive-migrated.npz"
    assert Path(migration.destination).is_file()
    migrated_runtime.restart(migration.destination)
    assert migrated_runtime.program_report().temporal == checkpoint_temporal

    # The next accepted dt is itself part of the proof, not merely the final solution.
    for runtime in (split, resumed, migrated_runtime):
        pops.run(runtime, t_end=next_deadline, max_steps=1, console=False)
    next_dt = split.time() - checkpoint_time
    assert next_dt > 0.0
    assert resumed.time() - checkpoint_time == next_dt
    assert migrated_runtime.time() - checkpoint_time == next_dt
    assert np.array_equal(
        np.asarray(resumed.state_global("blk")),
        np.asarray(split.state_global("blk")),
    )
    assert np.array_equal(
        np.asarray(migrated_runtime.state_global("blk")),
        np.asarray(split.state_global("blk")),
    )
    assert resumed.program_report().temporal == split.program_report().temporal
    assert migrated_runtime.program_report().temporal == split.program_report().temporal

    remaining = total_steps - checkpoint_step - 1
    for runtime in (split, resumed, migrated_runtime):
        pops.run(
            runtime, t_end=final_deadline, max_steps=remaining, console=False)
    expected = np.asarray(reference.state_global("blk"))
    for runtime in (split, resumed, migrated_runtime):
        assert runtime.macro_step() == reference.macro_step() == total_steps
        assert runtime.time() == reference.time()
        assert np.array_equal(np.asarray(runtime.state_global("blk")), expected)


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
    payload = _Payload({
        "program_hash": np.array("ab" * 32),
        "history_names": np.array([], dtype="U1"),
        "cache_nodes": np.array([], dtype=np.int64),
        "cache_names": np.array([], dtype="U1"),
        "temporal_restart_state": np.array("{}"),
    })
    preflight_uniform_restart(payload)

    payload["history_names"] = np.array(["rhs"])
    with pytest.raises(ValueError, match="history 'rhs'.*incomplete strict manifest"):
        preflight_uniform_restart(payload)
