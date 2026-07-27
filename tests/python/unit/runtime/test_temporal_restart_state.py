"""ADC-667 strict Uniform next-attempt checkpoint state."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
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
FROZEN_UNIFORM_V2 = ROOT / "tests/data/adc667/uniform_v2_ab2_98b7ffe6.npz"
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
        "linear-history-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
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
        "native_child_linear_history" if child_owned else "native_linear_history")
    temporal = program.state(block[state])
    fast = Clock("fast", owner=program.owner_path)
    if child_owned:
        child = program.state(block[state], clock=fast)
        program.keep_history(
            child, depth=2, interpolation=LinearInterpolation())
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
            temporal, depth=2, interpolation=LinearInterpolation())
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


def _write_compatible_uniform_v2_projection(current, owner, destination):
    """Project current data onto the exact v2 schema within the documented identity subset."""
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


def _write_resealed_uniform_payload(
    payload, owner, destination, replacements, *, semantic_identity=None,
):
    """Write an authenticated test payload after an intentional semantic mutation."""
    from pops.runtime._checkpoint_manifest import (
        IDENTITY_KEY,
        MANIFEST_KEY,
        _seal_checkpoint_payload_with_identities,
        seal_checkpoint_payload,
    )

    candidate = {
        name: np.array(value, copy=True)
        for name, value in payload.items()
        if name not in {MANIFEST_KEY, IDENTITY_KEY}
    }
    candidate.update({
        name: np.array(value, copy=True) for name, value in replacements.items()
    })
    if semantic_identity is None:
        seal_checkpoint_payload(owner, candidate, runtime_kind="uniform")
    else:
        _semantic, artifact, bind = owner._checkpoint_identities()
        _seal_checkpoint_payload_with_identities(
            candidate,
            runtime_kind="uniform",
            semantic=semantic_identity,
            artifact=artifact,
            bind=bind,
            run=owner.last_run_identity,
        )
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


def test_frozen_release_v2_fixture_is_authentic_unsealed_legacy_evidence():
    from pops.runtime._checkpoint_manifest import inspect_checkpoint_payload_integrity

    assert hashlib.sha256(FROZEN_UNIFORM_V2.read_bytes()).hexdigest() \
        == FROZEN_UNIFORM_V2_SHA256
    with np.load(FROZEN_UNIFORM_V2, allow_pickle=False) as stored:
        assert int(stored["pops_checkpoint_version"]) == 2
        assert str(stored["program_hash"]) \
            == "d1880e66a6b39e4d56aafe1f817591e5dec9212705b430c338bde8836e448215"
        assert {
            "pops_checkpoint_manifest",
            "pops_restart_identity",
            "temporal_restart_state",
            "runtime_consumer_cursors",
            "field_provider_slots",
        }.isdisjoint(stored.files)
        with pytest.raises(
            ValueError,
            match="no canonical manifest/restart identity; historical formats are refused",
        ):
            inspect_checkpoint_payload_integrity(stored, runtime_kind="uniform")


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
        assert checkpoint_temporal["event_queue"]
        assert checkpoint_temporal["event_queue"][0]["kind"] \
            == "error_controlled_dt.proposal"
        history_values = {
            (name, slot): np.array(stored["history_%s_%d" % (name, slot)])
            for name in history_names
            for slot in (int(value) for value in stored[
                "history_stored_slots_" + name
            ])
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
            np.asarray(resumed.history_global(name, slot)), values)
    restored_native = resumed._executor._s
    assert tuple(int(node) for node in restored_native.program_cache_nodes()) \
        == cache_nodes
    for node, values in cache_values.items():
        assert np.array_equal(
            np.asarray(restored_native.program_cache_global(node)), values)

    # The sealed archive rejects corruption of every stateful continuation authority before
    # mutating a fresh runtime: queue, retained history, and off-schedule cache.
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
            {"history_slot_dt_" + first_history: np.full(
                len(pristine["history_slot_dt_" + first_history]),
                np.nan,
                dtype=np.float64,
            )},
            "slot dt values must be finite",
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
            {"cache_last_update_%d" % first_cache: np.asarray(
                int(pristine["macro_step"]), dtype=np.int64)},
            "last update is not an accepted prior step",
        ),
        "cache-accumulated-dt-nan": (
            {"cache_accum_dt_%d" % first_cache: np.asarray(
                np.nan, dtype=np.float64)},
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
            assert np.array_equal(
                np.asarray(native.history_global(name, slot)), values)

    # Build the exact authenticated v2 compatibility projection, then exercise the public offline
    # transformer. This proves its plumbing without misrepresenting the unsealed artifact produced
    # by the actual historical writer as a semantically equivalent current Program.
    historical = _write_compatible_uniform_v2_projection(
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

    # Runtime restart is current-format-only and never dispatches to the explicit offline
    # transformer. A fully current sealed envelope carrying codec version 2 reaches the exact
    # version guard and is refused before a native restore transaction can begin.
    version_only_v2 = _write_resealed_uniform_payload(
        pristine,
        split,
        tmp_path / "adaptive-current-envelope-v2-codec.npz",
        {"pops_checkpoint_version": np.asarray(2, dtype=np.int64)},
    )
    migration_was_called = False

    def migration_must_not_be_called(*args, **kwargs):
        del args, kwargs
        nonlocal migration_was_called
        migration_was_called = True
        raise AssertionError("runtime restart attempted implicit checkpoint migration")

    direct_refusal = fresh()
    direct_before = np.asarray(direct_refusal.state_global("blk")).copy()
    with monkeypatch.context() as patch:
        patch.setattr(
            "pops.codegen.checkpoint_migration.migrate_uniform_checkpoint",
            migration_must_not_be_called,
        )
        with pytest.raises(
            ValueError,
            match=r"payload version 2 unsupported; expected exactly 3",
        ):
            direct_refusal.restart(version_only_v2)
    assert migration_was_called is False
    assert direct_refusal.macro_step() == 0
    assert direct_refusal.time() == 0.0
    assert np.array_equal(
        np.asarray(direct_refusal.state_global("blk")), direct_before)

    # The true frozen release-v2 file is also refused directly and by the narrow offline
    # transformer. It lacks a canonical envelope, and its compiled Program hash differs from the
    # current one. No output is published until an explicit version map proves those missing facts.
    frozen_destination = tmp_path / "must-not-publish-frozen-v2"
    frozen_runtime = fresh()
    assert frozen_runtime.installed_program_hash() \
        != "d1880e66a6b39e4d56aafe1f817591e5dec9212705b430c338bde8836e448215"
    with pytest.raises(
        ValueError,
        match="checkpoint has no canonical manifest",
    ):
        frozen_runtime.restart(FROZEN_UNIFORM_V2)
    with pytest.raises(
        ValueError,
        match="checkpoint has no canonical manifest",
    ):
        migrate_uniform_checkpoint(
            FROZEN_UNIFORM_V2,
            frozen_destination,
            runtime=frozen_runtime,
            state=migration_state,
        )
    assert not frozen_destination.with_suffix(".npz").exists()

    # A pre-contract semantic identity cannot be proven equivalent to the current Program. The
    # compatibility transformer refuses it rather than inventing a mapping.
    from pops.identity import make_identity

    with np.load(historical, allow_pickle=False) as stored:
        historical_payload = {
            name: np.array(stored[name], copy=True) for name in stored.files
        }
    unmapped_historical = _write_resealed_uniform_payload(
        historical_payload,
        split,
        tmp_path / "adaptive-v2-unmapped-semantic.npz",
        {},
        semantic_identity=make_identity(
            "semantic",
            {"schema": "pre-history-contract", "version": 2},
        ),
    )
    unmapped_destination = tmp_path / "must-not-publish-unmapped-semantic"
    with pytest.raises(NotImplementedError, match="compatibility subset.*semantic identity"):
        migrate_uniform_checkpoint(
            unmapped_historical,
            unmapped_destination,
            runtime=fresh(),
            state=migration_state,
        )
    assert not unmapped_destination.with_suffix(".npz").exists()

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
    assert resumed.program_report().temporal["event_queue"] \
        == split.program_report().temporal["event_queue"]

    remaining = total_steps - checkpoint_step - 1
    for runtime in (split, resumed, migrated_runtime):
        pops.run(
            runtime, t_end=final_deadline, max_steps=remaining, console=False)
    expected = np.asarray(reference.state_global("blk"))
    for runtime in (split, resumed, migrated_runtime):
        assert runtime.macro_step() == reference.macro_step() == total_steps
        assert runtime.time() == reference.time()
        assert np.array_equal(np.asarray(runtime.state_global("blk")), expected)


@pytest.mark.compiler
@pytest.mark.native_loader
def test_uniform_native_linear_history_uses_bracketing_slots_and_restarts_exactly(
    native_cxx, tmp_path,
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
            stored["history_%s_1" % name], 1.1, rtol=0.0, atol=4.0e-15)
        assert np.allclose(
            stored["history_%s_2" % name], 1.0, rtol=0.0, atol=4.0e-15)
        assert float(stored["history_slot_dt_%s" % name][1]) == pytest.approx(0.1)

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
    native_cxx, tmp_path,
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
        history, = temporal["program_schedule"]["histories"]
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
        "t": np.array(0.0, dtype=np.float64),
        "macro_step": np.array(0, dtype=np.int64),
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
