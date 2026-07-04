#!/usr/bin/env python3
"""ADC-631 (d): AMR multistep history under MPI -- distributed == replicated (collective + replay).

The v3 history payload gathers each ring slot with the collective ``_global`` accessor (all ranks call
``history_global``), and the native selective-persistence replay re-steps the installed Program whose
regrids are collective, so the deterministic reconstruction is per-rank identical. This lane runs the
same checkpoint/restart scenario under np=2 with a DISTRIBUTED coarse and with a REPLICATED coarse, and
asserts the gathered continuation is identical to the mono-rank result -- the AMR MPI-parity contract.

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
    import pops.lib.time as lt
    from pops import _pops
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.physics.facade import Model
    from pops.runtime.system import AmrSystem
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
    m = Model(name)
    (rho,) = m.conservative_vars("rho")
    u = m.primitive("u", 0.0 * rho)
    m.primitive_vars(rho=rho, u=u)
    m.conservative_from([rho])
    m.flux(x=[0.0 * rho], y=[0.0 * rho])
    m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    m.source([0.6 * rho])
    m.elliptic_rhs(rho)
    return m


def _ab2_program(name):
    P = pops.time.Program(name)
    lt.adams_bashforth2(P, "blk")
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
        compiled = pops.codegen.compile_problem(
            model=_passive_source_model("mpi_prog_%d" % distribute_coarse),
            time=_ab2_program("mpi_ab2_%d" % distribute_coarse), target="amr_system")
        block_cm = _passive_source_model("mpi_blk_%d" % distribute_coarse).compile(
            backend="production", target="amr_system")
    except RuntimeError as exc:
        return None, "compile: %s" % str(exc)[:180]
    try:
        amr.add_equation("blk", block_cm,
                         spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                         time=pops.Explicit(method="ssprk2"))
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


if __name__ == "__main__":
    test_amr_history_mpi_distributed_equals_replicated()
    if _pops.my_rank() == 0:
        print("FAILURES:", _fails)
    sys.exit(1 if _fails else 0)
