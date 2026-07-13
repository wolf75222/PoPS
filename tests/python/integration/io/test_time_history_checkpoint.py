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

Sections (B)/(C) self-skip (never fake the engine) without numpy / _pops / install_program / a compiler /
a visible Kokkos, exactly like test_time_history.py.
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
import json
import os
import sys
import tempfile
from pops.runtime.system import System  # ADC-545 advanced runtime seam


def _pops_time():
    global lt  # ready schemes live in pops.lib.time (Spec 4)
    try:
        import pops.time as t
        import pops.lib.time as lt  # ready schemes live in pops.lib.time (Spec 4)
    except Exception as exc:  # pops not importable here -> skip, never fake
        print("skip test_time_history_checkpoint (pops.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


_C = 0.75  # source coefficient: S(rho) = _C*rho (a linear ODE rho' = c rho; R changes every step)
_DT = 0.01
_NSTEPS = 6  # even, so N/2 is a whole number of macro-steps


def _authorize_identity_runtime(sim, compiled):
    """Attach the exact identity boundary to this deliberately low-level integration engine."""
    from pops.identity import make_identity
    from pops.runtime._bound_snapshot import BoundSnapshot

    component = compiled.program
    authored = getattr(component, "program", component)
    sim._step_strategy = authored._step_strategy
    sim._step_transaction_plan = authored.transaction_plan()
    snapshot = BoundSnapshot(
        semantic_identity=compiled.semantic_identity,
        artifact_identity=compiled.artifact_identity,
        layout={"kind": "uniform"}, blocks=[{"name": "blk"}], solvers={},
        step_transaction=sim._step_transaction_plan.to_data(),
        params=[], aux_evidence={}, initial_evidence={},
        bind_schema_identity=make_identity("bind-schema", {"slots": []}),
    )
    sim._finalize_bind(snapshot)
    return snapshot


# ---- (A) Current strict NPZ envelope: pure numpy, always runs when numpy is present ----
def test_current_checkpoint_envelope_roundtrips(_t):
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001 -- numpy unavailable in this interpreter
        print("-- (A) skipped: numpy unavailable: %s --" % exc)
        return
    from types import SimpleNamespace
    from pops.identity import make_identity
    from pops.runtime._bound_snapshot import BoundSnapshot
    from pops.runtime._checkpoint_manifest import (
        authenticate_checkpoint_payload, seal_checkpoint_payload)
    from pops.runtime._run_manifest import RunManifest
    from pops.runtime._step_strategy import run_control_payload
    from pops.runtime._temporal_restart import TemporalRestartState
    from pops.runtime.bricks import abi_key
    from pops.time import FixedDt

    snapshot = BoundSnapshot(
        semantic_identity=make_identity("semantic", {"test": "npz-envelope"}),
        artifact_identity=make_identity("artifact", {"binary": "npz-envelope"}),
        layout={"kind": "uniform"}, blocks=[{"name": "blk"}], solvers={},
        step_transaction={},
        params=[], aux_evidence={}, initial_evidence={},
        bind_schema_identity=make_identity("bind-schema", {"slots": []}),
    )
    run = RunManifest(
        bind_identity=snapshot.bind_identity, start_time=0.0, start_macro_step=0,
        controls={"t_end": 0.1, "step_transaction": run_control_payload(FixedDt(0.05)),
                  "max_steps": 10,
                  "output_mode": "current-directory"})
    owner = SimpleNamespace(bound_snapshot=snapshot, last_run_identity=run.run_identity)
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
def test_history_persistence_key_scheme(_t):
    """The current checkpoint stores only policy-selected slots + policy manifest + per-slot dt;
    the restore_histories reader dispatches on the manifest, restores the stored slots, and the
    policy-compat guard refuses a stored-slots / policy mismatch verbatim. Pure numpy (no engine): a
    fake System captures restore_history / restore_history_slot_dt / rebuild_history_slots calls."""
    try:
        import json

        import numpy as np
    except Exception as exc:  # noqa: BLE001
        print("-- (A2) skipped: numpy unavailable: %s --" % exc)
        return
    from pops.runtime._system_io_history import restore_histories, serialize_histories
    from pops.time.history_persistence import Revolve

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

        # --- writer side ---
        def history_names(self):
            return [hname]

        def history_depth(self, name):
            return depth

        def history_ncomp(self, name):
            return ncomp

        def history_initialized(self, name):
            return True

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

        def rebuild_history_slots(self, name, stored_slots):
            self.rebuilt = list(stored_slots)
            return depth - len(stored_slots)

    # WRITER: Revolve(3) on depth 5 stores {0, 2, 4}; only those slot buffers are emitted.
    out = {}
    writer = FakeSystem(present_slots=None)
    serialize_histories(writer, {hname: Revolve(3)}, out)
    stored = sorted(int(s) for s in out["history_stored_slots_" + hname])
    assert stored == [0, 2, 4], stored
    for k in stored:
        assert ("history_%s_%d" % (hname, k)) in out
    for k in (1, 3):
        assert ("history_%s_%d" % (hname, k)) not in out, "a recomputed slot is not written"
    policy_wire = json.loads(str(out["history_policy_" + hname]))
    assert policy_wire["payload"]["policy"] == "revolve"
    assert len(out["history_slot_dt_" + hname]) == depth

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
            raised = "stored slots" in str(exc) and "policy" in str(exc)
        assert raised, "a stored-slots / policy mismatch must be refused verbatim"

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
    sim = System(n=n, L=1.0, periodic=True)
    if not hasattr(sim, "install_program") or not hasattr(sim, "history_names"):
        return None, None
    try:
        compiled_model = _passive_source_model("ckpt_block").compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible
        print("-- skipped: model compile could not build the .so: %s --" % str(exc)[:160])
        return None, None
    sim.add_equation("blk", compiled_model,
                     spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                     time=pops.Explicit(method="euler"))
    return sim, True


def _rho0(np, n):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)


def _compile_program(pops, t, builder, prog_name, model_name):
    """compile_problem for the program built by @p builder (e.g. lt.adams_bashforth2). Returns the
    handle or None if the toolchain is absent."""
    from tests.python.support.typed_program import program_states, synthetic_module

    P = t.Program(prog_name)
    module = synthetic_module("%s_state" % prog_name, components=("rho",))
    _case, states = program_states(P, module, ("blk",))
    builder(P, states["blk"])
    P.step_strategy(t.FixedDt(_DT))
    try:
        from pops.codegen._compile_drivers import compile_problem
        return compile_problem(model=_passive_source_model(model_name), time=P)
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        print("-- skipped: compile_problem could not build the .so: %s --" % str(exc)[:160])
        return None


# ---- (B) spec 45 + 39: continuous == (run, checkpoint, restart, continue), bit-for-bit ----
def _run_section_b(t):
    try:
        import numpy as np

        import pops
    except Exception as exc:  # noqa: BLE001 -- numpy / _pops unavailable
        print("-- (B) skipped: pops/numpy unavailable: %s --" % exc)
        return None

    n = 16
    sim_cont, has_engine = _build_system(pops, np, n)
    if sim_cont is None:
        print("-- (B) skipped: _pops lacks the install_program/history bindings (rebuild _pops) --")
        return None

    compiled = _compile_program(pops, t, lt.adams_bashforth2, "ab2_ckpt", "ab2_prog_b")
    if compiled is None:
        return None

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
    from pops.time.history_persistence import Dense
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
    try:
        import numpy as np

        import pops
    except Exception as exc:  # noqa: BLE001
        print("-- (C) skipped: pops/numpy unavailable: %s --" % exc)
        return None

    n = 8
    sim, has_engine = _build_system(pops, np, n)
    if sim is None:
        print("-- (C) skipped: _pops lacks the install_program/history bindings (rebuild _pops) --")
        return None

    ab2 = _compile_program(pops, t, lt.adams_bashforth2, "ab2_c", "ab2_prog_c")
    fe = _compile_program(pops, t, lt.forward_euler, "fe_c", "fe_prog_c")
    if ab2 is None or fe is None:
        return None
    assert ab2.program_hash != fe.program_hash, \
        "AB2 and Forward Euler must have different IR hashes (else the test is vacuous)"

    sim.set_state("blk", np.stack([np.ones((n, n))]))
    sim.install_program(ab2.so_path)
    _authorize_identity_runtime(sim, ab2)
    sim.run(t_end=2 * _DT, max_steps=2)
    from pops.time.history_persistence import Dense
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


def _run():
    t = _pops_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_history_checkpoint (A: %d checks)" % len(fns))
    _run_section_b(t)
    _run_section_c(t)


if __name__ == "__main__":
    _run()
