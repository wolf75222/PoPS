#!/usr/bin/env python3
"""Checkpoint/restart of compiled-Program histories + the program-hash guard (epic ADC-399 / ADC-406b).

A compiled `Program` with multistep histories (e.g. Adams-Bashforth 2) carries System-owned ring
buffers across macro-steps (the previous RHS R_{n-1}, ...). For a checkpoint/restart to be correct the
rings MUST survive the checkpoint, so a CONTINUOUS run is bit-for-bit identical to a (run, checkpoint,
restart, continue) run -- without the rings the post-restart AB2 would cold-start again and diverge.
The typed semantic/artifact/bind/run identities and every payload digest are recorded too: restarting
under a different composition is rejected before any state mutation.

(A) NPZ facade keys (pure Python / numpy, always runs when numpy is present): the checkpoint key naming
    scheme (program_hash, history_names, history_depth_<n>, history_<n>_<k>, history_init_<n>) and the
    hash-mismatch comparison round-trip through numpy.savez/load with the exact dtypes the facade uses.
    This pins the serialization contract independently of the engine.

(B) Spec 45 + 39 (continuous == restart, history preserved): run an AB2 program N macro-steps
    continuously -> final state A. A fresh system runs N/2 steps, checkpoints; a SECOND fresh system
    (re-added block, re-installed program) restarts and runs N/2 more -> final state B. Assert A == B to
    machine precision (this exercises the history surviving the checkpoint) and that the clock matches.

(C) Spec 46 (hash mismatch): checkpoint an AB2 program, then restart a DIFFERENT compiled Program
    (Forward Euler, different IR hash) from that checkpoint -> RuntimeError containing
    "checkpoint was created with a different compiled Program hash".

Sections (B)/(C) explicitly skip only when numpy, the native extension, or the native toolchain is
absent. Once those prerequisites are present, compile and install failures are test failures.
"""
import json
import os
import tempfile

from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
    require_native_or_skip,
)

try:
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.runtime._system import System  # ADC-545 advanced runtime seam
except Exception as exc:  # noqa: BLE001 -- optional outside a native-capable checkout
    require_native_or_skip(
        "test_time_history_checkpoint cannot import its runtime: %s" % exc)


def _pops_time():
    global lt  # ready schemes live in pops.lib.time (Spec 4)
    try:
        import pops.time as t
        import pops.lib.time as lt  # ready schemes live in pops.lib.time (Spec 4)
    except Exception as exc:  # pops not importable here -> skip, never fake
        require_native_or_skip(
            "test_time_history_checkpoint pops.time unavailable: %s" % exc)
    return t


def _require_native_toolchain(section):
    missing = missing_native_compile_requirement(repo_include(), default_cxx())
    if missing:
        require_native_or_skip(
            "test_time_history_checkpoint %s: %s" % (section, missing))


_C = 0.75  # source coefficient: S(rho) = _C*rho (a linear ODE rho' = c rho; R changes every step)
_DT = 0.01
_NSTEPS = 6  # even, so N/2 is a whole number of macro-steps


def test_collective_failure_preserves_one_scientific_error_family():
    """MPI transport retains typed causes instead of flattening them to RuntimeError."""
    from pops.output._checkpoint_collective import (
        _error_record,
        _raise_collective_failure,
    )

    failures = (
        (0, _error_record(ValueError("manifest digest mismatch"))),
        (1, _error_record(ValueError("history depth is invalid"))),
    )
    try:
        _raise_collective_failure("restart-validation", failures)
    except ValueError as error:
        message = str(error)
        assert "rank 0: builtins.ValueError: manifest digest mismatch" in message
        assert "rank 1: builtins.ValueError: history depth is invalid" in message
    else:
        raise AssertionError("homogeneous collective ValueError was not reconstructed")

    mixed = (
        (0, _error_record(ValueError("invalid manifest"))),
        (1, _error_record(TypeError("invalid schema"))),
    )
    try:
        _raise_collective_failure("restart-validation", mixed)
    except RuntimeError as error:
        message = str(error)
        assert "rank 0: builtins.ValueError: invalid manifest" in message
        assert "rank 1: builtins.TypeError: invalid schema" in message
    else:
        raise AssertionError("mixed collective error families did not fail closed")

    class BrokenMessage(ValueError):
        def __str__(self):
            raise RuntimeError("formatting must not escape before the collective")

    record = _error_record(BrokenMessage())
    assert record["family"] == "ValueError"
    assert record["message"] == "<exception message unavailable>"


def _authorize_identity_runtime(sim, compiled):
    """Attach the exact identity boundary to this deliberately low-level integration engine."""
    from pops.identity import make_identity
    from pops.model.bind_schema import BindSchema
    from pops.runtime._bound_snapshot import BoundSnapshot
    from tests.python.support.native_execution_context import (
        compiled_problem_execution_context,
    )

    component = compiled.program
    authored = getattr(component, "program", component)
    context = compiled_problem_execution_context(compiled, target="system")
    sim._execution_context = context
    sim._step_strategy = authored._step_strategy
    sim._step_transaction_plan = authored.transaction_plan()
    snapshot = BoundSnapshot(
        semantic_identity=compiled.semantic_identity,
        artifact_identity=compiled.artifact_identity,
        layout={"kind": "uniform"}, blocks=[{"name": "blk"}], field_plans={},
        step_transaction=sim._step_transaction_plan.to_data(),
        params=[], aux_evidence={}, initial_evidence={},
        bind_schema_identity=make_identity("bind-schema", BindSchema().to_dict()),
        execution_context=context.to_data(),
    )
    sim._finalize_bind(snapshot)
    return snapshot


# ---- (A) Current strict NPZ envelope: pure numpy, always runs when numpy is present ----
def test_current_checkpoint_envelope_roundtrips():
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001 -- numpy unavailable in this interpreter
        require_native_or_skip('-- (A) skipped: numpy unavailable: %s --' % exc)
        return
    from types import SimpleNamespace
    from pops.identity import make_identity
    from pops.runtime._bound_snapshot import BoundSnapshot
    from pops.runtime._checkpoint_manifest import (
        authenticate_checkpoint_payload, seal_checkpoint_payload)
    from pops.runtime._run_manifest import RunManifest
    from pops.runtime._step_strategy import run_control_payload
    from pops.runtime._temporal_restart import TemporalRestartState
    from pops.runtime._engine_descriptors import abi_key
    from pops.time import FixedDt

    snapshot = BoundSnapshot(
        semantic_identity=make_identity("semantic", {"test": "npz-envelope"}),
        artifact_identity=make_identity("artifact", {"binary": "npz-envelope"}),
        layout={"kind": "uniform"}, blocks=[{"name": "blk"}], field_plans={},
        step_transaction={},
        params=[], aux_evidence={}, initial_evidence={},
        bind_schema_identity=make_identity("bind-schema", {"slots": []}),
    )
    run = RunManifest(
        bind_identity=snapshot.bind_identity, start_time=0.0, start_macro_step=0,
        controls={"t_end": 0.1, "step_transaction": run_control_payload(FixedDt(0.05)),
                  "max_steps": 10,
                  "output_mode": "current-directory"})
    owner = SimpleNamespace(
        _checkpoint_identities=lambda: (
            snapshot.semantic_identity, snapshot.artifact_identity, snapshot.bind_identity),
        last_run_identity=run.run_identity)
    temporal_state = TemporalRestartState()
    temporal_state.begin_run(
        run_control_payload(FixedDt(0.05)), time=0.0, macro_step=0)
    temporal_state.accept(before_time=0.0, before_step=0, time=0.05, macro_step=1)
    temporal_state.accept(before_time=0.05, before_step=1, time=0.1, macro_step=2)
    temporal = temporal_state.to_data()
    out = {
        "pops_checkpoint_version": 3, "t": 0.1, "macro_step": 2,
        "abi_key": abi_key(), "program_hash": "deadbeef" * 8,
        "temporal_restart_state": np.array(json.dumps(temporal, sort_keys=True)),
        "history_names": np.array([], dtype="U1"),
        "cache_nodes": np.array([], dtype=np.int64),
        "cache_names": np.array([], dtype="U1"),
        "state_blk": np.arange(16, dtype=np.float64),
    }
    expected = seal_checkpoint_payload(owner, out, runtime_kind="uniform")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.npz")
        with open(path, "wb") as f:
            np.savez_compressed(f, **out)
        d = np.load(path, allow_pickle=False)
        assert authenticate_checkpoint_payload(
            owner, d, runtime_kind="uniform") == expected


# ---- (A2) Selective-persistence key scheme + strict reader dispatch (pure numpy) ----
def test_history_persistence_key_scheme():
    """The checkpoint records requested and effective slots plus the resolved storage mode.

    Without an in-window regrid they coincide with the policy selection.  The reader dispatches on
    the manifest and refuses any resolved-plan mismatch. Pure numpy (no engine): a fake System
    captures restore_history / restore_history_slot_dt / rebuild_history_slots calls.
    """
    try:
        import json

        import numpy as np
    except Exception as exc:  # noqa: BLE001
        require_native_or_skip('-- (A2) skipped: numpy unavailable: %s --' % exc)
        return
    from pops.runtime._system_io_history import (
        prepare_history_capture,
        restore_histories,
        serialize_histories,
    )
    from pops.time._history.persistence import Revolve

    hname = "blk.state"
    depth = 5
    ncomp, ny, nx = 1, 4, 4
    # A distinct per-slot global buffer so a slot mixup is caught.
    full = {k: np.full(ncomp * ny * nx, float(k + 1)) for k in range(depth)}

    class FakeSystem:
        def __init__(self, present_slots):
            self._present = present_slots
            self.restored = {}
            self.restored_dt = {}
            self.rebuilt = None
            self.init = None
            self.fill_count = None

        # --- writer side ---
        def history_names(self):
            return [hname]

        def history_depth(self, name):
            return depth

        def history_ncomp(self, name):
            return ncomp

        def history_initialized(self, name):
            return True

        def history_fill_count(self, name):
            return depth

        def history_global(self, name, slot):
            return full[slot]

        def history_slot_dt(self, name, slot):
            return 0.01 * (slot + 1)

        # --- reader side ---
        def restore_history(self, name, slot, values):
            self.restored[slot] = np.asarray(values)

        def restore_history_slot_dt(self, name, slot, dt):
            self.restored_dt[slot] = dt

        def set_history_initialized(self, name, initialized):
            self.init = initialized

        def restore_history_fill_count(self, name, fill_count):
            self.fill_count = int(fill_count)

        def rebuild_history_slots(self, name, stored_slots):
            self.rebuilt = list(stored_slots)
            return depth - len(stored_slots)

    # WRITER: Revolve(3) on depth 5 stores {0, 2, 4}; only those slot buffers are emitted.
    out = {}
    writer = FakeSystem(present_slots=None)
    serialize_histories(writer, {hname: Revolve(3)}, out)
    stored = sorted(int(s) for s in out["history_stored_slots_" + hname])
    assert stored == [0, 2, 4], stored
    assert list(out["history_requested_stored_slots_" + hname]) == stored
    assert str(out["history_storage_mode_" + hname]) == "policy"
    for k in stored:
        assert ("history_%s_%d" % (hname, k)) in out
    for k in (1, 3):
        assert ("history_%s_%d" % (hname, k)) not in out, "a recomputed slot is not written"
    policy_wire = json.loads(str(out["history_policy_" + hname]))
    assert policy_wire["payload"]["policy"] == "revolve"
    assert len(out["history_slot_dt_" + hname]) == depth

    promoted = prepare_history_capture(
        writer, {hname: Revolve(3)}, macro_step=6, regrid_every=4)
    promoted_ring = promoted.rings[0]
    assert promoted_ring.requested_stored_slots == (0, 2, 4)
    assert promoted_ring.stored_slots == tuple(range(depth))
    assert promoted_ring.storage_mode == "dense_regrid_safety"
    assert promoted_ring.regrid_steps == (4,)

    class FillAgeSystem(FakeSystem):
        def __init__(self, fill_count):
            super().__init__(present_slots=None)
            self._fill_count = int(fill_count)

        def history_initialized(self, name):
            return self._fill_count > 0

        def history_fill_count(self, name):
            return self._fill_count

    # Cold registration, first accepted store and depth-1 accepted stores still contain at least
    # one startup broadcast copy. They persist densely. Only a fully warm ring may use its selective
    # anchors.
    for fill_count in (0, 1, depth - 1, depth):
        fill_plan = prepare_history_capture(
            FillAgeSystem(fill_count),
            {hname: Revolve(3)},
            macro_step=9,
            regrid_every=0,
        ).rings[0]
        assert fill_plan.fill_count == fill_count
        if fill_count < depth:
            assert fill_plan.storage_mode == "dense_cold_start_safety"
            assert fill_plan.stored_slots == tuple(range(depth))
        else:
            assert fill_plan.storage_mode == "policy"
            assert fill_plan.stored_slots == (0, 2, 4)

    # READER: round-trip through numpy so the dtypes match, then restore + replay.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "v2.npz")
        payload = dict(out)
        payload["history_names"] = np.array([hname])
        payload["history_depth_" + hname] = depth
        payload["history_init_" + hname] = True
        with open(path, "wb") as f:
            np.savez_compressed(f, **payload)
        d = np.load(path, allow_pickle=False)
        reader = FakeSystem(present_slots=stored)
        report = restore_histories(reader, d)
    assert sorted(reader.restored) == [0, 2, 4], "only the stored slots are restored"
    assert reader.rebuilt == [0, 2, 4], "replay is driven from the stored slots"
    assert len(reader.restored_dt) == depth, "every slot's dt is restored"
    assert reader.fill_count == depth, "the authentic ring age is restored"
    row = report.histories[0]
    assert row["policy_kind"] == "revolve" and row["stored_slots"] == 3 and row["recomputed_slots"] == 2

    # POLICY-COMPAT GUARD: a stored-slots array that disagrees with the policy is refused verbatim.
    bad = dict(payload)
    bad["history_stored_slots_" + hname] = np.asarray([0, 1, 4], dtype=np.int64)  # not Revolve(3)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.npz")
        with open(path, "wb") as f:
            np.savez_compressed(f, **bad)
        d = np.load(path, allow_pickle=False)
        raised = False
        try:
            restore_histories(FakeSystem(stored), d)
        except RuntimeError as exc:
            raised = "resolved storage plan" in str(exc)
        assert raised, "an effective stored-slots / resolved-plan mismatch must be refused"

    # STRICT FORMAT: a checkpoint without the persistence manifest is refused; conversion belongs
    # outside the runtime core.
    v1 = {"history_names": np.array([hname]),
          "history_depth_" + hname: depth,
          "history_init_" + hname: True}
    for k in range(depth):
        v1["history_%s_%d" % (hname, k)] = full[k]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "v1.npz")
        with open(path, "wb") as f:
            np.savez_compressed(f, **v1)
        d = np.load(path, allow_pickle=False)
        reader = FakeSystem(present_slots=list(range(depth)))
        raised = False
        try:
            restore_histories(reader, d)
        except RuntimeError as exc:
            raised = "persistence manifest" in str(exc)
    assert raised, "a checkpoint without the current persistence manifest must be refused"

    # UNKNOWN kind at restart fails loud (a checkpoint written by a newer pops).
    unknown = dict(payload)
    unknown_policy = Revolve(3).to_manifest()
    unknown_policy["payload"]["policy"] = "brand_new"
    unknown["history_policy_" + hname] = np.array(json.dumps(unknown_policy))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "unknown.npz")
        with open(path, "wb") as f:
            np.savez_compressed(f, **unknown)
        d = np.load(path, allow_pickle=False)
        raised = False
        try:
            restore_histories(FakeSystem(stored), d)
        except ValueError as exc:
            raised = "unknown" in str(exc)
        assert raised, "an unknown policy kind must fail loud at restart"


def test_restore_histories_installs_every_ring_before_replay():
    """Checkpoint key order cannot expose a partially restored Program history image."""
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        require_native_or_skip(
            "-- two-phase history restore skipped: numpy unavailable: %s --" % exc
        )
        return
    from pops.runtime._system_io_history import restore_histories
    from pops.time._history.persistence import Revolve

    names = ("second.state", "first.state")
    depth = 5
    stored = (0, 2, 4)
    policy = Revolve(3)
    payload = {
        "history_names": np.asarray(names),
        "macro_step": 9,
        "regrid_every": 0,
    }
    for index, name in enumerate(names):
        payload["history_depth_" + name] = depth
        payload["history_policy_" + name] = np.array(
            json.dumps(policy.to_manifest(), sort_keys=True, separators=(",", ":"))
        )
        payload["history_requested_stored_slots_" + name] = np.asarray(
            stored, dtype=np.int64
        )
        payload["history_stored_slots_" + name] = np.asarray(stored, dtype=np.int64)
        payload["history_storage_mode_" + name] = np.array("policy")
        payload["history_slot_dt_" + name] = np.asarray(
            [0.01 * (slot + 1) for slot in range(depth)], dtype=np.float64
        )
        payload["history_init_" + name] = True
        payload["history_fill_count_" + name] = depth
        for slot in stored:
            payload["history_%s_%d" % (name, slot)] = np.asarray(
                [100.0 * index + slot], dtype=np.float64
            )

    class CoupledReplayRuntime:
        def __init__(self):
            self.anchors = {name: set() for name in names}
            self.slot_dt = {name: set() for name in names}
            self.initialized = {name: False for name in names}
            self.events = []

        def restore_history(self, name, slot, _values):
            self.anchors[name].add(int(slot))
            self.events.append(("anchor", name, int(slot)))

        def restore_history_slot_dt(self, name, slot, _dt):
            self.slot_dt[name].add(int(slot))

        def set_history_initialized(self, name, initialized):
            self.initialized[name] = bool(initialized)

        def restore_history_fill_count(self, name, fill_count):
            assert int(fill_count) == depth

        def rebuild_history_slots(self, name, stored_slots):
            assert tuple(stored_slots) == stored
            assert all(self.anchors[ring] == set(stored) for ring in names)
            assert all(self.slot_dt[ring] == set(range(depth)) for ring in names)
            assert all(self.initialized.values())
            self.events.append(("replay", name))
            return depth - len(stored)

    runtime = CoupledReplayRuntime()
    report = restore_histories(runtime, payload)
    first_replay = next(i for i, event in enumerate(runtime.events) if event[0] == "replay")
    assert all(event[0] == "anchor" for event in runtime.events[:first_replay])
    assert report.total_recomputed == 4
    assert report.total_replay_steps == 4

    invalid = dict(payload)
    invalid["history_stored_slots_first.state"] = np.asarray(
        [0, 1, 4], dtype=np.int64
    )
    untouched = CoupledReplayRuntime()
    try:
        restore_histories(untouched, invalid)
    except RuntimeError as exc:
        assert "resolved storage plan" in str(exc)
    else:
        raise AssertionError("invalid second ring storage plan was accepted")
    assert untouched.events == []
    assert all(not anchors for anchors in untouched.anchors.values())


# ---- shared engine setup for (B)/(C) ----
def _passive_source_model(name):
    """A 1-variable model (rho), ZERO flux, default LINEAR source S(rho) = _C*rho (R = c*rho changes
    every step). A complete compilable block (flux + primitive + eigenvalue + source)."""
    from pops.physics._facade import Model
    m = Model(name)
    (rho,) = m.conservative_vars("rho")
    u = m.primitive("u", 0.0 * rho)
    m.primitive_vars(rho=rho, u=u)
    m.conservative_from([rho])
    m.flux(x=[0.0 * rho], y=[0.0 * rho])
    m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    m.source([_C * rho])
    return m


def _build_system(pops, np, n):
    """A fresh n x n periodic System with the compiled passive-source block added; (sim, has_engine)."""
    import pops.runtime._engine_descriptors as engine

    sim = System(n=n, L=1.0, periodicity=(True, True))
    if not hasattr(sim, "install_program") or not hasattr(sim, "history_names"):
        require_native_or_skip(
            "test_time_history_checkpoint requires install_program/history_names bindings")
    compiled_model = _passive_source_model("ckpt_block").compile(backend="production")
    sim.add_equation("blk", compiled_model,
                     spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                     time=engine.Explicit(method="euler"))
    return sim, True


def _rho0(np, n):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)


def _compile_program(pops, t, builder, builder_options, prog_name, model_name):
    """Compile a ready scheme built from exact final-API Case/Model handles."""
    from pops.problem import Case

    model = _passive_source_model(model_name)
    rate = model.rate("%s_rate" % prog_name, flux=False, sources=("default",))
    case = Case(name="%s-case" % prog_name)
    block = case.block("blk", model.module)
    spaces = tuple(model.module.state_spaces().values())
    assert len(spaces) == 1, "the passive checkpoint model has exactly one state"
    declaration = model.module.state_handle(spaces[0])
    P = builder(block[declaration], rate=rate, **builder_options)
    P.step_strategy(t.FixedDt(_DT))
    from pops.codegen._compile_drivers import compile_problem
    return compile_problem(model=model, time=P)


# ---- (B) spec 45 + 39: continuous == (run, checkpoint, restart, continue), bit-for-bit ----
def _run_section_b(t):
    _require_native_toolchain("section B")
    try:
        import numpy as np

        import pops
    except Exception as exc:  # noqa: BLE001 -- numpy / _pops unavailable
        require_native_or_skip(
            "test_time_history_checkpoint section B imports unavailable: %s" % exc)

    n = 16
    sim_cont, has_engine = _build_system(pops, np, n)

    compiled = _compile_program(
        pops, t, lt.AdamsBashforth, {"order": 2}, "ab2_ckpt", "ab2_prog_b")

    rho0 = _rho0(np, n)
    half = _NSTEPS // 2

    # (1) CONTINUOUS run: N steps in one go -> final state A.
    sim_cont.set_state("blk", np.stack([rho0]))
    sim_cont.install_program(compiled.so_path)
    for _ in range(_NSTEPS):
        sim_cont.step(_DT)
    state_a = np.array(sim_cont.get_state("blk"))[0]

    # (2) RUN N/2, CHECKPOINT.
    sim1, _ = _build_system(pops, np, n)
    sim1.set_state("blk", np.stack([rho0]))
    sim1.install_program(compiled.so_path)
    _authorize_identity_runtime(sim1, compiled)
    sim1.run(t_end=half * _DT, max_steps=half)
    from pops.time._history.persistence import Dense
    sim1.set_history_persistence({name: Dense() for name in sim1._s.history_names()})
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = sim1.checkpoint(os.path.join(tmp, "ab2"))

        # (3) FRESH system, re-add block, re-install the SAME program, RESTART, run N/2 more -> B.
        sim2, _ = _build_system(pops, np, n)
        sim2.install_program(compiled.so_path)  # the hash guard needs the program installed first
        _authorize_identity_runtime(sim2, compiled)
        sim2.restart(ckpt)
        assert sim2.macro_step() == half, \
            "restart restores macro_step (%d != %d)" % (sim2.macro_step(), half)
        sim2.run(t_end=_NSTEPS * _DT, max_steps=_NSTEPS - half)
    state_b = np.array(sim2.get_state("blk"))[0]

    err = float(np.abs(state_a - state_b).max())
    assert sim2.macro_step() == sim_cont.macro_step(), \
        "the clock must match after the restart run (%d != %d)" % (
            sim2.macro_step(), sim_cont.macro_step())
    assert abs(sim2.time() - sim_cont.time()) <= 1e-12, "t must match after restart"

    # Cross-check that the history actually mattered. A restart that DROPPED the rings would cold-start
    # AB2 at the resume step and diverge. The offline cold-restart reference (AB2, then a fresh FE cold
    # start at the resume step, then AB2) is what a pre-406b ring-less restart would produce; it must
    # DIFFER from the continuous run, so the bit-exact match above is a non-trivial result.
    ref_cold = _offline_ab2_cold_restart(rho0, _DT, _NSTEPS, half)
    cold = float(np.abs(state_a - ref_cold).max())

    print("  AB2 ckpt/restart: max|continuous - restart| = %.2e  "
          "max|continuous - ring-less restart| = %.2e (N=%d, half=%d)" % (err, cold, _NSTEPS, half))
    assert err <= 1e-12, \
        "continuous == (run, ckpt, restart, continue) to machine precision (max|d| = %.2e)" % err
    assert cold > 1e-9, \
        "the history must matter: a ring-less restart diverges (max|d| = %.2e)" % cold
    return err


def _offline_ab2_cold_restart(rho0, dt, nsteps, resume):
    """The AB2 trajectory if the history were LOST at the @p resume checkpoint: AB2 for the first
    @p resume steps, then a fresh FE cold start at @p resume, then AB2 again. This is the WRONG result a
    pre-406b (ring-less) restart would produce -- used only to prove the correct restart is non-trivial."""
    rho = rho0.copy()
    r_prev = _C * rho
    for k in range(nsteps):
        if k == resume:  # cold restart: forget R_{n-1}, refill from the current R
            r_prev = _C * rho
        r_n = _C * rho
        rho = rho + dt * (1.5 * r_n - 0.5 * r_prev)
        r_prev = r_n
    return rho


# ---- (C) spec 46: restart a DIFFERENT compiled Program -> hash-mismatch RuntimeError ----
def _run_section_c(t):
    _require_native_toolchain("section C")
    try:
        import numpy as np

        import pops
    except Exception as exc:  # noqa: BLE001
        require_native_or_skip(
            "test_time_history_checkpoint section C imports unavailable: %s" % exc)

    n = 8
    sim, has_engine = _build_system(pops, np, n)

    ab2 = _compile_program(
        pops, t, lt.AdamsBashforth, {"order": 2}, "ab2_c", "ab2_prog_c")
    fe = _compile_program(pops, t, lt.ForwardEuler, {}, "fe_c", "fe_prog_c")
    assert ab2.program_hash != fe.program_hash, \
        "AB2 and Forward Euler must have different IR hashes (else the test is vacuous)"

    sim.set_state("blk", np.stack([np.ones((n, n))]))
    sim.install_program(ab2.so_path)
    _authorize_identity_runtime(sim, ab2)
    sim.run(t_end=2 * _DT, max_steps=2)
    from pops.time._history.persistence import Dense
    sim.set_history_persistence({name: Dense() for name in sim._s.history_names()})
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = sim.checkpoint(os.path.join(tmp, "ab2_for_mismatch"))

        # A fresh system that installs the WRONG (Forward Euler) program, then restarts the AB2 ckpt.
        sim2, _ = _build_system(pops, np, n)
        sim2.install_program(fe.so_path)
        _authorize_identity_runtime(sim2, fe)
        try:
            sim2.restart(ckpt)
        except ValueError as exc:
            msg = str(exc)
            assert "identity" in msg and "bound runtime" in msg, \
                "the canonical identity mismatch must fail before mutation; got: %s" % msg
            print("  hash mismatch raised as expected: %s" % msg.splitlines()[0][:120])
            return True
    raise AssertionError("restarting a different compiled Program must raise (spec test 46)")


def test_uniform_ab2_history_checkpoint_restart_is_bit_identical():
    """Collect the Uniform AB2 checkpoint/restart acceptance as an ordinary test."""
    _run_section_b(_pops_time())


def test_uniform_restart_refuses_a_different_compiled_program():
    """Collect the authenticated Uniform Program-identity guard as an ordinary test."""
    _run_section_c(_pops_time())


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print("PASS test_time_history_checkpoint (%d checks)" % len(fns))


if __name__ == "__main__":
    _run()
