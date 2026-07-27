"""ADC-667 strict Uniform next-attempt checkpoint state."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
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
FROZEN_UNIFORM_V2_B64 = (
    ROOT / "tests/data/adc667/uniform_v2_ab2_98b7ffe6.npz.b64"
)
FROZEN_UNIFORM_V2_SHA256 = (
    "82490ddc97dbf37e6431c3c0ddb61c30439bdf4df9166f659146634d27766226"
)


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
    source_value = (
        source_rate + 0.0 * rho
        if attempt_policy == "forced_reject"
        else source_rate * rho
    )
    source = model.source("forcing", on=state, value=(source_value,))
    source_operator = model.module.operator_handle("forcing")
    rate = model.rate("transport-rate", equation=ddt(state) == -div(flux) + source)
    held_copy = model.local_transform("held_copy", (rho,))
    if attempt_policy == "error_retry":
        model.module.operator_capabilities("held_copy", cacheable=True)
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
            state=temporal.n,
            terms=[Flux(), SourceTerm(source_operator)],
        )
        candidate = program.value(
            "candidate",
            temporal.n + program.dt * rhs,
            at=temporal.next.point,
        )
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
            state=temporal.n,
            terms=[Flux(), SourceTerm(source_operator)],
        )
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
    return artifact


def _linear_history_artifact(native_cxx, *, child_owned=False):
    """Compile a Uniform Program whose solution depends on the native interpolated slots."""
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
        Clock,
        InterpolateHistory,
        LinearInterpolation,
        Program,
        SampleAndHold,
    )

    n = 4
    frame = Rectangle(
        "linear-history-domain",
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model("linear-history-model", frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "zero-flux",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
        waves={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
    )
    source = model.source("unit-source", on=state, value=(1.0 + 0.0 * rho,))
    rate = model.rate("rate", equation=ddt(state) == -div(flux) + source)
    source_operator = model.module.operator_handle("unit_source")
    case = pops.Case("linear-history-case")
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

    program = Program(
        "native_child_linear_history" if child_owned else "native_linear_history"
    )
    temporal = program.state(block[state])
    fast = Clock("fast", owner=program.owner_path)
    if child_owned:
        child = program.state(block[state], clock=fast)
        program.keep_history(child, depth=2, interpolation=LinearInterpolation())
        fast_value = program.subcycle(
            child.n,
            clock=fast,
            within=program.clock,
            count=2,
            body_fn=lambda P, value: P.value(
                "fast_advance",
                value + P.dt * value,
                at=child.next.point,
            ),
        )
        # Executed native interpolation from a child-owned ring at -3/2 child ticks. The result is
        # deliberately not the committed state: this isolates the ledger capability from the simple
        # subcycled evolution while still making a missing/invalid child timestamp fail the step.
        program.synchronize(
            child.prev,
            at=TimePoint(program.clock, Fraction(1, 4), step=-1),
            relation=InterpolateHistory(child.prev),
            name="child_history_at_macro_coordinate",
        )
        candidate = program.synchronize(
            fast_value,
            at=temporal.next.point,
            relation=SampleAndHold(),
        )
    else:
        program.keep_history(
            temporal,
            depth=2,
            interpolation=LinearInterpolation(),
        )
        interpolated = program.synchronize(
            temporal.prev,
            at=TimePoint(fast, step=-1),
            relation=InterpolateHistory(temporal.prev),
            name="half_previous_interval",
        )
        fast_value = program.subcycle(
            interpolated,
            clock=fast,
            within=program.clock,
            count=2,
            body_fn=lambda P, value: P.value("fast_copy", 1 * value),
        )
        returned = program.synchronize(
            fast_value,
            at=temporal.next.point,
            relation=SampleAndHold(),
        )
        rhs = program.rhs(
            state=temporal.n,
            terms=[Flux(), SourceTerm(source_operator)],
        )
        candidate = program.value(
            "candidate",
            returned + program.dt * rhs,
            at=temporal.next.point,
        )
    program.commit(temporal.next, candidate)
    program.step_strategy(FixedDt(0.1))
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
    return artifact


def _bind_uniform_artifact(artifact):
    import pops

    n = 4
    initial = np.ones((1, n, n), dtype=np.float64)
    return pops.bind(artifact, initial_state={"blk": initial})


def _bound_uniform_runtime(native_cxx, *, attempt_policy):
    """Compile and bind a real Uniform runtime with the requested native attempt policy."""
    return _bind_uniform_artifact(
        _uniform_artifact(native_cxx, attempt_policy=attempt_policy)
    )


def _write_resealed_uniform_payload(payload, owner, destination, replacements):
    """Write an authenticated test payload after one intentional semantic mutation."""
    from pops.runtime._checkpoint_manifest import (
        IDENTITY_KEY,
        MANIFEST_KEY,
        seal_checkpoint_payload,
    )

    candidate = {
        name: np.array(value, copy=True)
        for name, value in payload.items()
        if name not in {MANIFEST_KEY, IDENTITY_KEY}
    }
    candidate.update(
        {name: np.array(value, copy=True) for name, value in replacements.items()}
    )
    seal_checkpoint_payload(owner, candidate, runtime_kind="uniform")
    with open(destination, "wb") as stream:
        np.savez_compressed(stream, **candidate)
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
    monkeypatch,
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
        retrying_initial * (1.0 + 0.5 * 0.0625),
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

    # Public lifecycle proof: adaptive retry + schedule cache + retained history survive one
    # accepted current-format checkpoint. The queued next proposal and every later state are exact.
    import pops

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
        reference,
        t_end=final_deadline,
        max_steps=total_steps,
        console=False,
    )

    split = fresh()
    pops.run(
        split,
        t_end=checkpoint_deadline,
        max_steps=checkpoint_step,
        console=False,
    )
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
        assert checkpoint_temporal["event_queue"]
        assert checkpoint_temporal["event_queue"][0]["kind"] \
            == "error_controlled_dt.proposal"
        history_values = {
            (name, slot): np.array(stored["history_%s_%d" % (name, slot)])
            for name in history_names
            for slot in (
                int(value)
                for value in stored["history_stored_slots_" + name]
            )
        }
        assert all("cache_value_%d" % node in stored.files for node in cache_nodes)
        cache_values = {
            node: np.array(stored["cache_value_%d" % node])
            for node in cache_nodes
        }

    resumed = fresh()
    resumed.restart(checkpoint)
    assert resumed.program_report().temporal == checkpoint_temporal
    for (name, slot), values in history_values.items():
        assert np.array_equal(
            np.asarray(resumed.history_global(name, slot)),
            values,
        )
    restored_native = resumed._executor._s
    assert tuple(int(node) for node in restored_native.program_cache_nodes()) \
        == cache_nodes
    for node, values in cache_values.items():
        assert np.array_equal(
            np.asarray(restored_native.program_cache_global(node)),
            values,
        )

    # Queue, retained history, and off-schedule cache are authenticated continuation state. Any
    # corruption must fail before a fresh runtime can be mutated.
    with np.load(checkpoint, allow_pickle=False) as stored:
        pristine = {
            name: np.array(stored[name], copy=True)
            for name in stored.files
        }
    corruptions = {
        "queue": {
            "temporal_restart_state": np.asarray(
                str(pristine["temporal_restart_state"]).replace(
                    "error_controlled_dt.proposal",
                    "error_controlled_dt.corrupt",
                )
            ),
        },
        "history": {
            "history_%s_%d" % next(iter(history_values)): (
                next(iter(history_values.values())) + 1.0
            ),
        },
        "cache": {
            "cache_value_%d" % next(iter(cache_values)): (
                next(iter(cache_values.values())) + 1.0
            ),
        },
    }
    for label, replacement in corruptions.items():
        damaged = dict(pristine)
        damaged.update(replacement)
        path = tmp_path / ("adaptive-corrupt-%s.npz" % label)
        np.savez_compressed(path, **damaged)
        refused = fresh()
        before = np.asarray(refused.state_global("blk")).copy()
        with pytest.raises(ValueError, match="digest|identity|integrity"):
            refused.restart(path)
        assert np.array_equal(np.asarray(refused.state_global("blk")), before)

    # Integrity alone is not semantic validity: re-seal malformed continuation metadata with the
    # current runtime identities and prove the strict preflight refuses it before beginning the
    # native restore transaction or mutating state/history/cache/clock.
    first_history = history_names[0]
    first_cache = cache_nodes[0]
    invalid_metadata = {
        "history-slot-dt-nan": (
            {
                "history_slot_dt_" + first_history: np.full(
                    len(pristine["history_slot_dt_" + first_history]),
                    np.nan,
                    dtype=np.float64,
                )
            },
            "outgoing-dt ledger must be finite",
        ),
        "history-fill-count": (
            {"history_fill_count_" + first_history: np.asarray(-1, dtype=np.int64)},
            "history_fill_count_.*must be >= 0",
        ),
        "cache-node": (
            {"cache_nodes": np.asarray([-1], dtype=np.int64)},
            "cache index is inconsistent",
        ),
        "cache-future-update": (
            {
                "cache_last_update_%d" % first_cache: np.asarray(
                    int(pristine["macro_step"]), dtype=np.int64
                )
            },
            "last update is not an accepted prior step",
        ),
        "cache-accumulated-dt-nan": (
            {
                "cache_accum_dt_%d" % first_cache: np.asarray(
                    np.nan, dtype=np.float64
                )
            },
            "cache_accum_dt_.*must be finite",
        ),
        "cache-empty-name": (
            {"cache_names": np.asarray([""], dtype="U1")},
            "must contain non-empty text",
        ),
    }
    for label, (replacement, message) in invalid_metadata.items():
        path = _write_resealed_uniform_payload(
            pristine,
            split,
            tmp_path / ("adaptive-resealed-invalid-%s.npz" % label),
            replacement,
        )
        refused = fresh()
        engine = refused._executor
        native = engine._s
        before_state = np.asarray(refused.state_global("blk")).copy()
        before_clock = (refused.time(), refused.macro_step())
        before_histories = {
            (name, slot): np.asarray(native.history_global(name, slot)).copy()
            for name in native.history_names()
            for slot in range(native.history_depth(name))
        }
        before_cache_nodes = tuple(native.program_cache_nodes())

        def transaction_must_not_begin(self):
            del self
            raise AssertionError("semantic preflight reached _begin_checkpoint_restart")

        with monkeypatch.context() as patch:
            patch.setattr(
                type(engine),
                "_begin_checkpoint_restart",
                transaction_must_not_begin,
            )
            with pytest.raises((TypeError, ValueError, RuntimeError), match=message):
                refused.restart(path)
        assert "_checkpoint_restart_python_snapshot" not in engine.__dict__
        assert (refused.time(), refused.macro_step()) == before_clock
        assert np.array_equal(np.asarray(refused.state_global("blk")), before_state)
        assert tuple(native.program_cache_nodes()) == before_cache_nodes
        for (name, slot), values in before_histories.items():
            assert np.array_equal(np.asarray(native.history_global(name, slot)), values)

    # The first accepted dt after restart is itself part of the contract, not merely the final
    # field. Both runtimes must also publish the same following queued proposal.
    for runtime in (split, resumed):
        pops.run(runtime, t_end=next_deadline, max_steps=1, console=False)
    next_dt = split.time() - checkpoint_time
    assert next_dt > 0.0
    assert resumed.time() - checkpoint_time == next_dt
    assert np.array_equal(
        np.asarray(resumed.state_global("blk")),
        np.asarray(split.state_global("blk")),
    )
    assert resumed.program_report().temporal == split.program_report().temporal
    assert resumed.program_report().temporal["event_queue"] \
        == split.program_report().temporal["event_queue"]

    remaining = total_steps - checkpoint_step - 1
    for runtime in (split, resumed):
        pops.run(
            runtime,
            t_end=final_deadline,
            max_steps=remaining,
            console=False,
        )
    expected = np.asarray(reference.state_global("blk"))
    for runtime in (split, resumed):
        assert runtime.macro_step() == reference.macro_step() == total_steps
        assert runtime.time() == reference.time()
        assert np.array_equal(np.asarray(runtime.state_global("blk")), expected)


@pytest.mark.compiler
@pytest.mark.native_loader
def test_uniform_native_linear_history_uses_bracketing_slots_and_restarts_exactly(
    native_cxx,
    tmp_path,
):
    import pops

    artifact = _linear_history_artifact(native_cxx)

    def fresh():
        return _bind_uniform_artifact(artifact)

    reference = fresh()
    pops.run(reference, t_end=0.3, max_steps=3, console=False)
    expected = np.full((1, 4, 4), 1.225, dtype=np.float64)
    assert np.allclose(
        np.asarray(reference.state_global("blk")),
        expected,
        rtol=0.0,
        atol=4.0e-15,
    )

    split = fresh()
    pops.run(split, t_end=0.2, max_steps=2, console=False)
    checkpoint = Path(split.checkpoint(tmp_path / "linear-history"))
    with np.load(checkpoint, allow_pickle=False) as stored:
        name = str(stored["history_names"][0])
        # The accepted ring carries U^1 and U^0. The next native step first stores live U^2
        # into slot 0, then interpolates that slot with slot 1.
        assert np.allclose(
            stored["history_%s_1" % name],
            1.1,
            rtol=0.0,
            atol=4.0e-15,
        )
        assert np.allclose(
            stored["history_%s_2" % name],
            1.0,
            rtol=0.0,
            atol=4.0e-15,
        )
        assert float(stored["history_slot_dt_%s" % name][1]) \
            == pytest.approx(0.1)

    resumed = fresh()
    resumed.restart(checkpoint)
    pops.run(split, t_end=0.3, max_steps=1, console=False)
    pops.run(resumed, t_end=0.3, max_steps=1, console=False)
    assert np.array_equal(
        np.asarray(resumed.state_global("blk")),
        np.asarray(split.state_global("blk")),
    )
    assert np.array_equal(
        np.asarray(resumed.state_global("blk")),
        np.asarray(reference.state_global("blk")),
    )


@pytest.mark.compiler
@pytest.mark.native_loader
def test_uniform_child_clock_history_owns_exact_slot_ledger_across_restart(
    native_cxx,
    tmp_path,
):
    import pops

    artifact = _linear_history_artifact(native_cxx, child_owned=True)

    def fresh():
        return _bind_uniform_artifact(artifact)

    reference = fresh()
    pops.run(reference, t_end=0.3, max_steps=3, console=False)

    split = fresh()
    pops.run(split, t_end=0.2, max_steps=2, console=False)
    checkpoint = Path(split.checkpoint(tmp_path / "child-linear-history"))
    with np.load(checkpoint, allow_pickle=False) as stored:
        name = str(stored["history_names"][0])
        temporal = json.loads(str(stored["temporal_restart_state"]))
        (history,) = temporal["program_schedule"]["histories"]
        assert history["name"] == name
        assert history["clock"] != temporal["program_schedule"]["primary_clock"]
        assert np.array_equal(
            stored["history_slot_dt_%s" % name],
            np.full(3, 0.05, dtype=np.float64),
        )

    resumed = fresh()
    resumed.restart(checkpoint)
    pops.run(split, t_end=0.3, max_steps=1, console=False)
    pops.run(resumed, t_end=0.3, max_steps=1, console=False)
    assert np.array_equal(
        np.asarray(resumed.state_global("blk")),
        np.asarray(split.state_global("blk")),
    )
    assert np.array_equal(
        np.asarray(resumed.state_global("blk")),
        np.asarray(reference.state_global("blk")),
    )


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


def test_frozen_release_v2_fixture_is_refused_offline_and_at_runtime_boundary():
    import base64
    import hashlib
    from io import BytesIO

    from pops.runtime._checkpoint_manifest import (
        authenticate_checkpoint_payload,
        inspect_checkpoint_payload_integrity,
    )

    encoded = "".join(FROZEN_UNIFORM_V2_B64.read_text().splitlines())
    raw = base64.b64decode(encoded, validate=True)
    assert hashlib.sha256(raw).hexdigest() == FROZEN_UNIFORM_V2_SHA256

    with np.load(BytesIO(raw), allow_pickle=False) as stored:
        assert int(stored["pops_checkpoint_version"]) == 2
        assert str(stored["program_hash"]) == (
            "d1880e66a6b39e4d56aafe1f817591e5dec9212705b430c338bde8836e448215"
        )
        assert {
            "pops_checkpoint_manifest",
            "pops_restart_identity",
            "temporal_restart_state",
            "runtime_consumer_cursors",
            "field_provider_slots",
        }.isdisjoint(stored.files)

        message = "no canonical manifest/restart identity"
        with pytest.raises(ValueError, match=message):
            inspect_checkpoint_payload_integrity(stored, runtime_kind="uniform")
        with pytest.raises(ValueError, match=message):
            authenticate_checkpoint_payload(
                object(),
                stored,
                runtime_kind="uniform",
            )


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
            "t": np.array(0.0, dtype=np.float64),
            "macro_step": np.array(0, dtype=np.int64),
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
