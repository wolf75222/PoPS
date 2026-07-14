#!/usr/bin/env python3
"""ADC-631 (d) / ADC-635: AMR multistep history under MPI -- distributed == replicated.

The v3 history payload gathers each ring slot with the collective ``_global`` accessor (all ranks call
``history_global``), and the native selective-persistence replay re-steps the installed Program whose
regrids are collective, so the deterministic reconstruction is per-rank identical. This lane runs the
checkpoint/restart scenario under np=2 with a DISTRIBUTED coarse and with a REPLICATED coarse and asserts
the gathered continuation is identical -- the AMR MPI-parity contract. ADC-635 adds a case whose replay
window straddles a regrid, so the in-window regrid remap replays collectively across ranks too.

Runs the real comparison only under ``mpirun -np 2`` (``pops._pops.n_ranks() == 2``); under a single
rank it self-skips (exit 0), so the serial CI lane is a no-op and the MPI lane exercises it. Self-skips
without pops / a built _pops / a compiler / a visible Kokkos.
"""
import os
import sys
import tempfile

try:
    import numpy as np

    import pops
    import pops.runtime._engine_descriptors as engine
    import pops.lib.time as lt
    from pops.codegen._compile_drivers import compile_problem
    from pops import _pops
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.problem import Case
    from pops.runtime._system import AmrSystem
    from tests.python.integration._final_field_program import (
        compile_block_model,
        passive_source_model,
    )
    from tests.python.support.typed_program import program_states, state_handle
except Exception as exc:  # noqa: BLE001
    print("skip test_amr_history_mpi (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)

N = 16
DT = 2.0e-3
NSTEPS = 6
HALF = 3
_fails = 0


def chk(cond, label):
    global _fails
    if _pops.my_rank() == 0:
        print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        _fails += 1


def _passive_source_model(name):
    return passive_source_model(name, coefficient=0.6)


def _ab2_program(model, name):
    module = model.module
    case = Case("%s-case" % name)
    state = case.block("blk", module)[state_handle(module)]
    return lt.AdamsBashforth(
        state, rate=module.operator_handle("source_rate"), order=2)


def _state_ring_program(model, name):
    """ADC-635: a depth-3 STATE ring (Interval(2) -> stores {0,2}, replays slot 1). A single-step
    Markov recurrence so the single-seed replay reconstructs the gap bit-for-bit."""
    from pops.time.history_persistence import Interval
    P = pops.Program(name)
    _case, states = program_states(P, model, ("blk",))
    U = states["blk"]
    P.keep_history(U, depth=3, checkpoint_policy=Interval(2))
    nxt = P.value(
        "Un", U.n + P.dt * (0.6 * U.n) + 0.0 * U.prev(2), at=U.next.point)
    P.commit(U.next, nxt)
    return P


def _blob():
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.4 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (0.12 ** 2))


def _build(distribute_coarse):
    try:
        amr = AmrSystem(n=N, L=1.0, regrid_every=2, distribute_coarse=distribute_coarse)
    except TypeError:
        amr = AmrSystem(n=N, L=1.0, regrid_every=2)  # older signature (replicated only)
    if not hasattr(amr, "install_program") or not hasattr(amr, "history_names"):
        return None, None
    try:
        model = _passive_source_model("mpi_blk_%d" % distribute_coarse)
        compiled = compile_problem(
            model=model,
            time=_ab2_program(model, "mpi_ab2_%d" % distribute_coarse),
            target="amr_system",
        )
        block_cm = compile_block_model(model, target="amr_system")
    except RuntimeError as exc:
        return None, "compile: %s" % str(exc)[:180]
    try:
        amr.add_equation("blk", block_cm,
                         spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                         time=engine.Explicit(method="ssprk2"))
        amr.set_refinement(1.1)
        amr.set_density("blk", _blob())
        amr.install_program(compiled.so_path)
    except RuntimeError as exc:
        return None, "install: %s" % str(exc)[:240]
    return amr, None


def _run_ckpt_restart(distribute_coarse, tmp):
    amr, err = _build(distribute_coarse)
    if amr is None:
        return None, err
    for _ in range(HALF):
        amr.step(DT)
    ckpt = amr.checkpoint(os.path.join(tmp, "mpi_%d" % distribute_coarse))
    fresh, _ = _build(distribute_coarse)
    fresh.restart(ckpt)
    for _ in range(NSTEPS - HALF):
        fresh.step(DT)
    return np.asarray(fresh.density("blk")), None  # density() is a collective global gather


def _build_state_ring(distribute_coarse, regrid_every=4):
    try:
        amr = AmrSystem(n=N, L=1.0, regrid_every=regrid_every, distribute_coarse=distribute_coarse)
    except TypeError:
        amr = AmrSystem(n=N, L=1.0, regrid_every=regrid_every)
    if not hasattr(amr, "install_program") or not hasattr(amr, "history_names"):
        return None, None
    try:
        model = _passive_source_model("mpi635_blk_%d" % distribute_coarse)
        compiled = compile_problem(
            model=model,
            time=_state_ring_program(model, "mpi635_ring_%d" % distribute_coarse),
            target="amr_system",
        )
        block_cm = compile_block_model(model, target="amr_system")
        bg_model = _passive_source_model("mpi635_bg_%d" % distribute_coarse)
        bg_cm = compile_block_model(bg_model, target="amr_system")
    except RuntimeError as exc:
        return None, "compile: %s" % str(exc)[:180]
    try:
        amr.add_equation("blk", block_cm,
                         spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                         time=engine.Explicit(method="ssprk2"))
        # FROZEN background block (ADC-635): >= 2 blocks keeps the 2-level seed (a single-block Program
        # is coarse-only, where regrid() is a structural no-op), so the in-window regrid COMPLETES and
        # the replay exercises the collective remap. The Program never commits "bg" (stays frozen).
        amr.add_equation("bg", bg_cm,
                         spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                         time=engine.Explicit(method="ssprk2"))
        amr.set_refinement(1.1)
        amr.set_density("blk", _blob())
        amr.set_density("bg", np.full((N, N), 0.5))  # flat, below threshold: tags nothing, ever
        amr.install_program(compiled.so_path)
        persistence = getattr(getattr(compiled, "program", None), "_history_persistence", None)
        if persistence:
            amr.set_history_persistence(
                {name: policy for name, (_depth, policy) in persistence.items()})
    except RuntimeError as exc:
        return None, "install: %s" % str(exc)[:240]
    return amr, None


def _run_state_ring(distribute_coarse, tmp, half=6, nsteps=10):
    """Checkpoint a non-Dense ring at a step whose replay window contains an in-window regrid (m=6,
    regrid_every=4 -> a regrid at cursor 4 inside the replay window), fresh restart, continue."""
    amr, err = _build_state_ring(distribute_coarse)
    if amr is None:
        return None, err
    for _ in range(half):
        amr.step(DT)
    ckpt = amr.checkpoint(os.path.join(tmp, "mpi635_%d" % distribute_coarse))
    fresh, _ = _build_state_ring(distribute_coarse)
    fresh.restart(ckpt)
    for _ in range(nsteps - half):
        fresh.step(DT)
    return np.asarray(fresh.density("blk")), None  # collective global gather


def test_amr_history_mpi_distributed_equals_replicated():
    if _pops.n_ranks() < 2:
        print("skip test_amr_history_mpi (needs mpirun -np>=2; n_ranks=%d)" % _pops.n_ranks())
        return
    with tempfile.TemporaryDirectory() as tmp:
        repl, err = _run_ckpt_restart(0, tmp)
        if repl is None:
            print("skip (%s)" % err)
            return
        dist, err2 = _run_ckpt_restart(1, tmp)
        if dist is None:
            print("skip (%s)" % err2)
            return
    d = float(np.abs(repl - dist).max())
    chk(np.array_equal(repl, dist),
        "np=2 distributed == replicated after history checkpoint/restart (max|d| = %.3e)" % d)


def test_amr_history_mpi_in_window_regrid_replay():
    """ADC-635: a non-Dense ring whose replay window straddles a regrid restarts bit-identically under
    np=2 (distributed == replicated), so the in-window-regrid replay reproduces the collective remap."""
    if _pops.n_ranks() < 2:
        print("skip test_amr_history_mpi_in_window_regrid (needs mpirun -np>=2)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        repl, err = _run_state_ring(0, tmp)
        if repl is None:
            print("skip (%s)" % err)
            return
        dist, err2 = _run_state_ring(1, tmp)
        if dist is None:
            print("skip (%s)" % err2)
            return
    d = float(np.abs(repl - dist).max())
    chk(np.array_equal(repl, dist),
        "np=2 in-window-regrid ring replay: distributed == replicated (max|d| = %.3e)" % d)


if __name__ == "__main__":
    test_amr_history_mpi_distributed_equals_replicated()
    test_amr_history_mpi_in_window_regrid_replay()
    if _pops.my_rank() == 0:
        print("FAILURES:", _fails)
    sys.exit(1 if _fails else 0)
