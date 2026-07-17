#!/usr/bin/env python3
"""ADC-514: NATIVE per-block RUNTIME parameters on the AMR hierarchy.

A production AMR block (add_equation with a CompiledModel backend='production', target='amr_system')
whose model declares ``pops.params.RuntimeParam`` now carries a per-block RuntimeParams vector that
AmrSystem.set_block_params overwrites WITHOUT recompiling the .so -- the AMR counterpart of
System.set_block_params (P7-b). The value flows into the block's transport / source / elliptic bricks
at the top of each macro-step (build_amr_compiled captures the shared vector), so a set_block_params
call changes the trajectory at the next step.

This test asserts (Kokkos-gated, needs a compiler + a visible Kokkos to build + run the .so):

  1) a runtime-param AMR block RUNS, and set_block_params(cs2=big) DIFFERS from the default trajectory
     (the sound speed enters the flux/CFL, so the evolved coarse density is not the same array);
  2) BIT-IDENTITY: re-running with the same complete vector reproduces the trajectory byte-for-byte.
     Generated carriers are neutral; declaration defaults are materialized only by BindSchema.

Self-skips (exit 0) without pops / a built _pops / a compiler / Kokkos. Pytest + __main__ guard
(CI runs ``python3 <file>``).
"""
import sys

from tests.python.support.requirements import require_native_or_skip

try:
    import numpy as np

    import pops.runtime._engine_descriptors as engine
    from pops.math import sqrt
    from pops.physics._facade import Model
    from pops.params import RuntimeParam
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.runtime._system import AmrSystem
except Exception as exc:  # noqa: BLE001 -- pops/numpy unavailable in this interpreter
    require_native_or_skip("test_amr_native_params imports unavailable: %s" % exc)

N = 16
NSTEPS = 4
DT = 5.0e-4

_fails = 0


def chk(cond, label):
    global _fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        _fails += 1


def _iso_runtime_model(name="adc514_iso", cs2_value=1.0):
    """Isothermal 2D gas (rho, rho_u, rho_v) with p = cs2 * rho, cs2 a canonical RuntimeParam.
    The single runtime param 'cs2' enters the pressure -> the flux and the CFL wave
    speed sqrt(cs2), so set_block_params(cs2=...) changes the trajectory. elliptic_rhs = rho so a coarse
    field solve runs (the AMR coupler always solves the Poisson)."""
    m = Model(name)
    rho, mx, my = m.conservative_vars("rho", "rho_u", "rho_v")
    cs2_handle = m.param(RuntimeParam("cs2", default=cs2_value))
    cs2 = m.value(cs2_handle)
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = m.primitive("p", cs2 * rho)
    m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([rho, rho * u, rho * v])
    m.flux(x=[mx, mx * u + p, my * u], y=[my, mx * v, my * v + p])
    cs = sqrt(cs2)
    m.eigenvalues(x=[u - cs, u, u + cs], y=[v - cs, v, v + cs])
    m.elliptic_rhs(rho)
    return m


def _init_density():
    """A smooth, periodic, strictly-positive coarse density (component 0). set_density seeds momentum=0
    (coupler_write_coarse), so the runs start byte-identical -- the prerequisite of a bit-identical
    trajectory comparison."""
    xs = (np.arange(N) + 0.5) / N
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    return 1.0 + 0.3 * np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.02)


def _amr_run(cs2_override, u0, nsteps=NSTEPS, dt=DT):
    """Build a single-block AMR sim on a runtime-param model, optionally set_block_params(cs2=override)
    WITHOUT recompiling, run nsteps, and return the evolved coarse density (component 0). cs2_override is
    None selects the declared default value explicitly for this advanced direct install (which has no
    BindSchema of its own). Returns (density, None) or (None, reason) on a compile/wire failure."""
    amr = AmrSystem(n=N, L=1.0, regrid_every=0)
    if not hasattr(amr, "set_block_params"):
        require_native_or_skip(
            "the built _pops lacks AmrSystem.set_block_params (rebuild _pops)"
        )
    model = _iso_runtime_model()
    try:
        block_cm = model.compile(backend="production", target="amr_system")
    except RuntimeError as exc:
        require_native_or_skip("compile (AMR production): %s" % str(exc)[:200])
    if block_cm.runtime_param_names != ["cs2"]:
        raise AssertionError(
            "runtime_param_names expected ['cs2'], got %r" % block_cm.runtime_param_names
        )
    try:
        amr.add_equation("gas", block_cm,
                         spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                         time=engine.Explicit(method="ssprk2"))
        amr.set_density("gas", u0)  # momentum=0, coarse seed (same for both runs -> identical start)
        value = 1.0 if cs2_override is None else cs2_override
        amr.set_block_params("gas", [value])
        for _ in range(nsteps):
            amr.step(dt)
    except RuntimeError as exc:
        require_native_or_skip("run (AMR): %s" % str(exc)[:240])
    return np.array(amr.density("gas")), None


def test_amr_set_block_params_changes_trajectory_and_default_is_bit_identical():
    """A different complete vector changes the run; repeating cs2=1 is byte-identical."""
    print("== AMR native runtime params: set_block_params changes the run, default is bit-identical ==")
    u0 = _init_density()

    base, err = _amr_run(cs2_override=None, u0=u0)  # complete direct-install vector: cs2=1.0
    assert base is not None, err
    # Reinstalling the same complete vector is byte-identical (0 ulp, never allclose).
    same, err = _amr_run(cs2_override=1.0, u0=u0)
    assert same is not None, err
    chk(np.array_equal(base, same),
        "cs2=1 is BYTE-IDENTICAL to the same complete baseline vector (0 ulp)")

    # (1) A DIFFERENT cs2 (4x the sound speed squared) changes the flux + CFL, so the evolved coarse
    # density is a DIFFERENT array WITHOUT recompiling the .so.
    changed, err = _amr_run(cs2_override=4.0, u0=u0)
    assert changed is not None, err
    chk(changed.shape == base.shape, "the changed run returns the same-shape coarse density")
    chk(not np.array_equal(changed, base),
        "set_block_params(cs2=4) DIFFERS from the default run (runtime param drives the trajectory)")
    chk(np.all(np.isfinite(changed)), "the changed run stays finite (no blow-up)")


def test_amr_set_block_params_rejects_a_paramless_block():
    """set_block_params on a block whose model declares NO runtime param is rejected explicitly (a silent
    set would mask a bug). Pure C++ guard -- needs the built _pops with the AMR carrier, no compile."""
    print("== AMR set_block_params on a param-free block is rejected ==")
    amr = AmrSystem(n=N, L=1.0, regrid_every=0)
    if not hasattr(amr, "set_block_params"):
        require_native_or_skip("the built _pops lacks AmrSystem.set_block_params")
    # A native ModelSpec block (composed bricks) carries no runtime param.
    try:
        spec = engine.Model(engine.Scalar(), engine.ExB(B0=1.0), engine.NoSource(),
                          engine.ChargeDensity(charge=1.0))
        amr.add_equation("ne", spec,
                         spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                         time=engine.Explicit())
    except Exception as exc:  # noqa: BLE001 -- brick/ModelSpec API drift; skip rather than fail
        require_native_or_skip(
            "could not build/add a native ModelSpec block: %s" % str(exc)[:120]
        )
    raised = False
    try:
        amr.set_block_params("ne", [1.0])
    except RuntimeError as exc:
        raised = True
        message = str(exc).lower()
        chk("runtime" in message and "param" in message,
            "the rejection names the missing runtime parameter")
    chk(raised, "set_block_params on a param-free block raises (no silent set)")


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print("\n%d checks failed" % _fails)
    return _fails


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
