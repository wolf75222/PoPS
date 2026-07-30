#!/usr/bin/env python3
"""ADC-702: one Program history contract exercised by the real Uniform and AMR runtimes.

The same Adams-Bashforth 2 Program contract is checked independently on ``System`` and on a flat
``AmrSystem``: registration creates a retained ring, the first store cold-starts every lag, reads
observe the accepted preceding rates, and rotation advances the recurrence exactly once per accepted
step. The oracle is the analytical AB2 recurrence for ``du/dt = c*u``; this is deliberately not a
byte-parity comparison between two context implementations.

Missing native prerequisites are explicit local skips and required-lane failures, exactly like
test_amr_program_parity. Pytest + __main__ guard (CI runs ``python3 <file>``).
"""
import sys

from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
    require_native_or_skip,
)


_native_missing = missing_native_compile_requirement(repo_include(), default_cxx())
if _native_missing:
    require_native_or_skip("test_program_history_contract: %s" % _native_missing)

try:
    import numpy as np

    import pops.runtime._engine_descriptors as engine
    import pops.lib.time as lt
    from pops.codegen._compile_drivers import compile_problem
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.runtime._system import AmrSystem, System
    from tests.python.integration._final_field_program import (
        compile_block_model,
        passive_field_model,
        resolve_periodic_field_program,
    )
except Exception as exc:  # noqa: BLE001 -- pops/numpy unavailable in this interpreter
    require_native_or_skip(
        "test_program_history_contract cannot import pops/numpy: %s" % exc)

N = 16
NSTEPS = 5
DT = 5.0e-3
_C = 0.6  # linear source S(rho) = _C*rho: R changes every step, so the AB2 ring MATTERS

_fails = 0


def chk(cond, label):
    global _fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        _fails += 1


def _passive_source_model(name):
    """Final scalar/source model; the resolved Case also installs its periodic field solve."""
    return passive_field_model(name, coefficient=_C)


def _ab2_plan(model, *, target, name="adc631_ab2"):
    return resolve_periodic_field_program(
        model,
        lambda state, rate, _fields: lt.AdamsBashforth(
            state,
            rate=rate,
            order=2,
        ),
        name=name,
        block_name="blk",
        target=target,
        n=N,
    )


def _rho0():
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)


def _ring_slots(sim):
    """Every ring's every stored slot as flat float64 buffers (concatenated, order-stable)."""
    out = {}
    for hname in sim.history_names():
        d = int(sim.history_depth(hname))
        out[hname] = [np.asarray(sim.history_global(hname, k), dtype=np.float64).ravel()
                      for k in range(d)]
    return out


def _system_run(u0):
    sim = System(n=N, L=1.0, periodicity=(True, True))
    if not hasattr(sim, "install_program") or not hasattr(sim, "history_names"):
        require_native_or_skip(
            "test_program_history_contract requires System install_program/history_names bindings")
    model = _passive_source_model("blkS")
    plan = _ab2_plan(model, target="system")
    block_cm = compile_block_model(model, target="system")
    compiled = compile_problem(
        model=model,
        time=plan.time,
        field_plans=plan.field_plans,
        problem_snapshot=plan.snapshot,
    )
    sim.add_equation("blk", block_cm,
                     spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                     time=engine.Explicit(method="ssprk2"))
    sim.set_state("blk", np.stack([u0]))
    sim.install_program(compiled.so_path)
    for _ in range(NSTEPS):
        sim.step(DT)
    return (np.array(sim.get_state("blk"))[0], _ring_slots(sim)), None


def _amr_run(u0):
    amr = AmrSystem(n=N, L=1.0, regrid_every=0)  # FLAT: no refinement -> nlev=1 (coarse-only)
    if not hasattr(amr, "install_program") or not hasattr(amr, "history_names"):
        require_native_or_skip(
            "test_program_history_contract requires AmrSystem install_program/history_names bindings")
    model = _passive_source_model("blkA")
    plan = _ab2_plan(model, target="amr_system")
    compiled = compile_problem(
        model=model,
        time=plan.time,
        target="amr_system",
        field_plans=plan.field_plans,
        problem_snapshot=plan.snapshot,
    )
    block_cm = compile_block_model(model, target="amr_system")
    amr.add_equation("blk", block_cm,
                     spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                     time=engine.Explicit(method="ssprk2"))
    amr.set_density("blk", u0)
    amr.install_program(compiled.so_path)
    for _ in range(NSTEPS):
        amr.step(DT)
    return (np.array(amr.density("blk")), _ring_slots(amr), int(amr.n_levels())), None


def _ab2_factor(steps):
    """Exact scalar recurrence used by the compiled AB2 Program, including Euler cold start."""
    previous = 1.0
    current = 1.0 + DT * _C
    if steps == 0:
        return previous
    for _ in range(1, steps):
        previous, current = current, current + DT * _C * (
            1.5 * current - 0.5 * previous
        )
    return current


def _check_history_contract(label, rho, rings, initial):
    expected = _ab2_factor(NSTEPS) * initial
    error = float(np.abs(rho - expected).max())
    chk(np.all(np.isfinite(rho)), "%s produces a finite accepted state" % label)
    chk(np.all(rho > 0.0), "%s preserves positivity for the linear-growth contract" % label)
    chk(error < 5.0e-13,
        "%s follows the analytical AB2 recurrence (max|error| = %.3e)" % (label, error))
    chk(bool(rings), "%s registers at least one native history ring" % label)
    chk(all(len(slots) >= 2 for slots in rings.values()),
        "%s retains current and lag-one slots" % label)
    chk(all(np.all(np.isfinite(slot)) for slots in rings.values() for slot in slots),
        "%s exposes finite retained slot buffers" % label)


def test_uniform_and_amr_run_the_same_program_history_contract():
    print("== shared Program history contract on real Uniform and AMR providers ==")
    u0 = _rho0()
    sys_out, sys_err = _system_run(u0)
    assert sys_out is not None, sys_err
    amr_out, amr_err = _amr_run(u0)
    assert amr_out is not None, amr_err
    sys_rho, sys_rings = sys_out
    amr_rho, amr_rings, nlev = amr_out

    chk(nlev == 1, "the AMR contract fixture is flat and has one storage level")
    _check_history_contract("System", sys_rho, sys_rings, u0)
    _check_history_contract("AmrSystem", amr_rho, amr_rings, u0)
    assert _fails == 0


def main():
    test_uniform_and_amr_run_the_same_program_history_contract()
    print("FAILURES:", _fails)
    sys.exit(1 if _fails else 0)


if __name__ == "__main__":
    main()
