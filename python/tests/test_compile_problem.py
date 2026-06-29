#!/usr/bin/env python3
"""pops.compile_problem + sim.install + compiled time Program, end to end.

(A) Validation (pure Python, always runs): compile_problem rejects missing models,
    missing Programs, and non-Production typed backends; CompiledProgramCadence wires
    substeps/stride > 1, numeric cfl, and cfl='program' dt-bound routing.

(B) End-to-end parity (skips cleanly unless the full toolchain is present): build an
    isothermal transport module + a Forward-Euler Program, compile_problem -> problem.so
    (compiled WITH Kokkos, so its ABI key matches _pops and it loads in-process),
    sim.install(compiled, ...), sim.step(dt), and check parity against the reference
    one-step U0 + dt * eval_rhs (after solve_fields) -- the same primitives the Program
    drives, so bit/near parity. Runs in CI (gate-python rebuilds _pops with the
    install_program binding) and locally once _pops is rebuilt; skips if _pops lacks the
    install seam, numpy/_pops is absent, no compiler/Kokkos is visible, or the .so compile
    fails -- never faking the engine.
"""
import os
import sys
import tempfile
from pathlib import Path


def _skip(msg):
    print("skip test_compile_problem (%s)" % msg)
    if "pytest" in sys.modules:
        import pytest
        pytest.skip("test_compile_problem (%s)" % msg, allow_module_level=True)
    sys.exit(0)


np = None
pops = None
physics = None
adctime = None
AOT = None
ddt = None
div = None
grad = None
laplacian = None
sqrt = None
spatial_catalog = None
FirstOrder = None
Rusanov = None
Explicit = None
CompiledProgramCadence = None
GeometricMG = None


def _repo_include():
    include = Path(__file__).resolve().parents[2] / "include"
    return str(include) if include.is_dir() else None


def _compile_kwargs(**kwargs):
    include = _repo_include()
    if include is not None:
        kwargs.setdefault("include", include)
    return kwargs


def _ensure_kokkos_root():
    if os.environ.get("POPS_KOKKOS_ROOT") or os.environ.get("Kokkos_ROOT"):
        return
    prefix = Path(sys.prefix)
    if (prefix / "include" / "Kokkos_Core.hpp").is_file():
        os.environ["POPS_KOKKOS_ROOT"] = str(prefix)


def _ensure_repo_include():
    if os.environ.get("POPS_INCLUDE"):
        return
    include = _repo_include()
    if include is not None:
        os.environ["POPS_INCLUDE"] = include


def _load_deps():
    global np, pops, physics, adctime, AOT, ddt, div, grad, laplacian, sqrt
    global spatial_catalog, FirstOrder, Rusanov, Explicit
    global CompiledProgramCadence, GeometricMG
    if pops is not None:
        return
    try:
        import numpy as _np

        import pops as _pops
        from pops import physics as _physics
        from pops import time as _adctime
        from pops.codegen import AOT as _AOT
        from pops.math import ddt as _ddt
        from pops.math import div as _div
        from pops.math import grad as _grad
        from pops.math import laplacian as _laplacian
        from pops.math import sqrt as _sqrt
        from pops.numerics import spatial as _spatial_catalog
        from pops.numerics.reconstruction import FirstOrder as _FirstOrder
        from pops.numerics.riemann import Rusanov as _Rusanov
        from pops.runtime.bricks import Explicit as _Explicit
        from pops.runtime._compiled_cadence import CompiledProgramCadence as _CompiledProgramCadence
        from pops.solvers import GeometricMG as _GeometricMG
    except Exception as exc:  # noqa: BLE001  -- numpy or _pops unavailable in this interpreter
        _skip("pops/numpy unavailable: %s" % exc)

    np = _np
    pops = _pops
    physics = _physics
    adctime = _adctime
    AOT = _AOT
    ddt = _ddt
    div = _div
    grad = _grad
    laplacian = _laplacian
    sqrt = _sqrt
    spatial_catalog = _spatial_catalog
    FirstOrder = _FirstOrder
    Rusanov = _Rusanov
    Explicit = _Explicit
    CompiledProgramCadence = _CompiledProgramCadence
    GeometricMG = _GeometricMG
    _ensure_repo_include()
    _ensure_kokkos_root()

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def raises(exc_types, fn):
    try:
        fn()
    except exc_types:
        return True
    except Exception:  # noqa: BLE001  -- wrong exception type is a failure, not a pass
        return False
    return False


def _transport_problem_module(name="transport_problem"):
    """Public physics authoring model lowered to the operator-first Module compile_problem consumes."""
    m = physics.Model(name)
    U = m.state("U", components=["rho", "mx", "my"],
                roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"})
    rho, mx, my = U
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    cs2 = 0.5
    c = sqrt(cs2)
    p = m.scalar("p", cs2 * rho)
    flux = m.flux(
        "F",
        on=U,
        x=[mx, mx * u + p, mx * v],
        y=[my, mx * v, my * v + p],
        waves={"x": [u - c, u, u + c], "y": [v - c, v, v + c]},
    )
    phi = m.field("phi")
    m.solve_field(
        "fields_from_state",
        equation=(-laplacian(phi) == rho),
        outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
        solver=GeometricMG(),
    )
    m.rate("explicit_rate", ddt(U) == -div(flux))
    return m.to_module()


def _runtime_block_model(name="transport_block"):
    """Runtime block Module for System.install; no legacy physics facade."""
    return _transport_problem_module(name)


def _fe_program(module, name="forward_euler_parity", coeff=1.0):
    P = adctime.Program(name)
    P.bind_operators(module)
    states = module.state_spaces()
    operators = module.operator_registry()
    dt = P.dt
    U = P.state("ions", space=states["U"])
    fields = P.call(operators.get("fields_from_state"), U, name="fields")
    R = P.call(operators.get("explicit_rate"), U, fields, name="R")
    P.commit("ions", P.linear_combine("U1", U + coeff * dt * R))
    return P


def _compile_problem(**overrides):
    module = _transport_problem_module()
    kwargs = {"model": module, "time": _fe_program(module)}
    kwargs.update(overrides)
    return pops.compile_problem(**_compile_kwargs(**kwargs))


def _fv():
    return spatial_catalog.FiniteVolume(reconstruction=FirstOrder(), riemann=Rusanov())


def _initial_state(n):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    return np.stack([rho, 0.4 * rho, -0.2 * rho])


def make_sim(initial, *, compiled=None, model_name="transport_block"):
    sim = pops.System(n=initial.shape[-1], L=1.0, periodic=True)
    sim.install(
        compiled,
        instances={
            "ions": {
                "model": _runtime_block_model(model_name),
                "initial": initial,
                "spatial": _fv(),
                "time": Explicit.euler(),
            }
        },
        solvers={"phi": GeometricMG()},
    )
    return sim


def _fe_scaled(name, a):
    """A Forward-Euler Program U <- U + a*dt*R (the dt coefficient varies with a)."""
    module = _transport_problem_module("transport_" + name)
    return module, _fe_program(module, name=name, coeff=a)


def main():
    global fails
    _load_deps()
    fails = 0

    # ---- (A) validation: pure Python, always runs ----
    print("== (A) compile_problem / CompiledProgramCadence validation ==")
    chk(raises(ValueError, lambda: _compile_problem(time=None)),
        "compile_problem without a Program rejected")
    chk(raises(ValueError, lambda: pops.compile_problem(time=_fe_program(_transport_problem_module()))),
        "compile_problem without a physical model rejected")
    chk(raises(ValueError, lambda: _compile_problem(backend=AOT())),
        "compile_problem accepts only Production() for compiled problems")
    # substeps>1 / stride>1 are wired now (ADC-411): they store the cadence
    # (System.set_program_cadence applies it around the program closure) instead of being rejected.
    chk(CompiledProgramCadence(substeps=2).substeps == 2,
        "CompiledProgramCadence substeps>1 accepted (wired, ADC-411)")
    chk(CompiledProgramCadence(stride=2).stride == 2,
        "CompiledProgramCadence stride>1 accepted (wired, ADC-411)")
    chk(CompiledProgramCadence(cfl="program").cfl == "program",
        "CompiledProgramCadence cfl='program' accepted (routed to Program dt_bound)")
    chk(CompiledProgramCadence().kind == "compiled",
        "CompiledProgramCadence() default ok (kind 'compiled')")

    # ---- (B) end-to-end parity: skips unless the full toolchain is present ----
    if not hasattr(pops.System(n=8, L=1.0, periodic=True), "_install_program_so"):
        print("-- (B) skipped: _pops lacks the compiled Program install seam (rebuild _pops) --")
        print("%s test_compile_problem (A only)" % ("FAIL" if fails else "PASS"))
        return 1 if fails else 0

    print("== (B) end-to-end: compiled Program vs reference one-step ==")

    dt = 2e-3
    n = 24
    u0_init = _initial_state(n)

    problem_module = _transport_problem_module()
    try:
        compiled = pops.compile_problem(
            **_compile_kwargs(model=problem_module, time=_fe_program(problem_module)))
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        _skip("compile_problem could not build the .so: %s" % str(exc)[:160])

    chk(compiled.program_name == "forward_euler_parity", "handle carries the program name")
    chk(bool(compiled.program_hash), "handle carries the IR hash")

    # Reference: one Forward-Euler step via the same primitives the Program drives.
    try:
        ref = make_sim(u0_init, compiled=None, model_name="transport_ref")
    except RuntimeError as exc:
        _skip("runtime block could not build/install the native model: %s" % str(exc)[:160])

    U0 = np.array(ref._get_state("ions"))
    ref.solve_fields()
    R0 = np.array(ref._eval_rhs("ions"))
    U_ref = U0 + dt * R0

    # Compiled-Program path: lower the FE IR -> problem.so, install it, step once.
    prog = make_sim(u0_init, compiled=compiled, model_name="transport_prog")
    step0 = prog.macro_step()
    prog.step(dt)  # SystemStepper dispatches to the installed Program
    U_prog = np.array(prog._get_state("ions"))

    emax = float(np.abs(U_prog - U_ref).max())
    change = float(np.abs(U_prog - U0).max())
    chk(emax < 1e-12, "compiled FE Program == reference one-step (max|d| = %.2e)" % emax)
    chk(prog.macro_step() == step0 + 1,
        "macro_step advanced (%d -> %d)" % (step0, prog.macro_step()))
    chk(change > 1e-9, "the step actually changed the state (change = %.2e)" % change)

    # ---- (C) cache + debug dump (reaching here means the toolchain compiled the .so in section B) ----
    print("== (C) compile_problem cache + debug dump ==")

    # Cache HIT: compiling the same Program twice (no explicit so_path) returns the SAME cached .so.
    cache_module = _transport_problem_module("transport_cache")
    c1 = pops.compile_problem(
        **_compile_kwargs(model=cache_module, time=_fe_program(cache_module, "cache_probe")))
    cache_module_2 = _transport_problem_module("transport_cache")
    c2 = pops.compile_problem(
        **_compile_kwargs(model=cache_module_2, time=_fe_program(cache_module_2, "cache_probe")))
    chk(c1.so_path == c2.so_path and os.path.isfile(c1.so_path),
        "cache HIT: identical Program -> same cached .so")

    # Cache MISS: a different dt coefficient is a different IR -> different generated source ->
    # different cache key -> different .so.
    module_a, prog_a = _fe_scaled("cache_coeff", 1.0)
    module_b, prog_b = _fe_scaled("cache_coeff", 2.0)
    c_a = pops.compile_problem(**_compile_kwargs(model=module_a, time=prog_a))
    c_b = pops.compile_problem(**_compile_kwargs(model=module_b, time=prog_b))
    chk(c_a.so_path != c_b.so_path, "cache MISS: a changed dt coefficient invalidates the cache")

    # debug=True writes the generated .cpp next to the .so for inspection.
    dbg_so = os.path.join(tempfile.mkdtemp(), "dbg_problem.so")
    debug_module = _transport_problem_module("transport_debug")
    pops.compile_problem(dbg_so, **_compile_kwargs(
        model=debug_module, time=_fe_program(debug_module, "debug_probe"), debug=True))
    dbg_cpp = os.path.splitext(dbg_so)[0] + ".cpp"
    chk(os.path.isfile(dbg_cpp), "debug=True writes the generated .cpp next to the .so")
    if os.path.isfile(dbg_cpp):
        with open(dbg_cpp) as _f:
            dumped = _f.read()
        chk("pops_install_program" in dumped and "ProgramContext" in dumped,
            "the dumped .cpp contains the ProgramContext closure")

    print("%s test_compile_problem" % ("FAIL (%d)" % fails if fails else "PASS"))
    return 1 if fails else 0


def test_compile_problem_script():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
