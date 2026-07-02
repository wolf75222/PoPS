#!/usr/bin/env python3
"""Spec 5 sec.12.1 (criteria #40-41): the RUNTIME / COMPILED headline objects print readably.

The inert architecture gate ``tests/python/architecture/test_print_readability.py`` asserts the print
contract for the inert authoring objects, but it deliberately CANNOT cover the three headline
objects that are not inert:

  * ``pops.System`` / ``pops.AmrSystem`` -- constructing one allocates Fabs and initialises Kokkos
    (it needs the compiled ``_pops`` extension), so it has no place in an import-only inert gate;
    that gate also runs source-only in CI (before ``_pops`` is built), so it would only skip them.
  * ``pops.codegen.CompiledProblem`` -- normally the result of a Kokkos-gated ``compile_problem``.

All three already define a short ``__str__`` (``System`` / ``AmrSystem`` since ADC-505,
``CompiledProblem`` since ADC-509), but nothing ASSERTED it. This test closes that gap. It runs in
the ``gate-python`` step, which executes ``tests/python/**/test_*.py`` AFTER the ``_pops`` module is
built, so it can construct a real (empty) ``System`` / ``AmrSystem``; the ``CompiledProblem`` is
SYNTHETIC (a real in-memory ``Program`` + a real ``CompiledModel`` carrier, NO compile, mirroring
``test_compiled_introspection``). For each object it asserts ``str(obj)`` is

  * short         -- under 800 characters (a one-line summary, never a Fab / ``.so`` dump);
  * deterministic -- ``str(x) == str(x)`` (no memory address, no run-dependent ordering);
  * array-free    -- no ``array(`` / ``ndarray`` substring;
  * not the default repr -- no ``object at 0x...`` address leak.

Without the compiled ``_pops`` extension the whole module skips (``System`` cannot be built).

Pytest + __main__ guard (CI runs ``python3 <file>``).
"""
import sys

try:
    import pops._pops  # noqa: F401  -- System / AmrSystem need the native runtime
    import pops
    from pops import time as adctime
    from pops.codegen.loader import CompiledModel, CompiledProblem
except Exception as exc:  # noqa: BLE001 -- _pops not built in this interpreter
    print("skip test_print_readability_runtime (_pops unavailable: %s)" % exc)
    sys.exit(0)

_MAX_PRINT_LEN = 800


def _assert_readable(label, obj):
    """The four print-contract properties of ``str(obj)`` (mirrors the inert gate)."""
    text = str(obj)
    assert text, "str(%s) is empty" % label
    assert len(text) < _MAX_PRINT_LEN, (
        "str(%s) is %d chars (>= %d): print(obj) must be a short summary, not a dump"
        % (label, len(text), _MAX_PRINT_LEN))
    assert str(obj) == text, (
        "str(%s) is not deterministic (a memory address or run-dependent ordering leaked)" % label)
    assert "array(" not in text and "ndarray" not in text, (
        "str(%s) contains a raw array dump; print(obj) must stay numerics-free" % label)
    assert "object at 0x" not in text, (
        "str(%s) is the default object repr (leaks a memory address, unreadable): %r"
        % (label, text))


def _synthetic_compiled():
    """A SYNTHETIC ``CompiledProblem``: a real lowered ``Program`` + a real ``CompiledModel``, NO
    compile (same inert construction as ``test_compiled_introspection``). Nothing here compiles a
    ``.so`` or touches the runtime; only ``CompiledProblem``'s pure-Python metadata wrapper runs."""
    P = adctime.Program("demo")
    dt = P.dt
    U = P.state("plasma")
    f = P.solve_fields("phi", U)
    R = P._rhs_legacy(state=U, fields=f, flux=True, sources=["default"])
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    m = CompiledModel(
        so_path="/nonexistent/problem.so", backend="production", adder="add_native_block",
        cons_names=["rho", "mx", "my"], cons_roles=["Density", "MomentumX", "MomentumY"],
        prim_names=["rho", "mx", "my"], n_vars=3, gamma=1.4, n_aux=0, params={},
        caps={"cpu": True}, abi_key="SIG|c++|c++23", model_hash="modelhash", cxx="c++", std="c++23")
    return CompiledProblem("/tmp/pops-cache/problem.so", P, m, "SIG|c++|c++23", "c++", "c++23",
                           problem_hash="deadbeefcafe", cache_key="0badc0de")


def test_system_print_is_readable():
    """str(System) is a short block-registry summary, never a Fab dump or a bare address."""
    print("== System prints readably ==")
    sim = pops.System(n=8, L=1.0, periodic=True)
    _assert_readable("System", sim)


def test_amr_system_print_is_readable():
    """str(AmrSystem) is a short block-registry summary on the AMR hierarchy (frozen, no regrid)."""
    print("== AmrSystem prints readably ==")
    sim = pops.AmrSystem(n=16, L=1.0, periodic=True, regrid_every=0)
    _assert_readable("AmrSystem", sim)


def test_compiled_problem_print_is_readable():
    """str(CompiledProblem) is a short name/hash/backend summary, never the .so contents."""
    print("== CompiledProblem prints readably ==")
    cp = _synthetic_compiled()
    _assert_readable("CompiledProblem", cp)


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print("FAIL %s: %s" % (fn.__name__, exc))
    print("\n%d/%d test functions passed" % (len(fns) - failed, len(fns)))
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
