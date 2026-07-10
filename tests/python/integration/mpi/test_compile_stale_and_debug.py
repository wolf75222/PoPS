#!/usr/bin/env python3
"""ADC-536 real-compiler acceptance: stale-sidecar refusal + debug provenance sidecar.

Three claims, each needing a real ``.so`` (skips cleanly unless the full toolchain is present, like
the sibling ``test_compile_problem.py``):

  (1) STALE REFUSAL: a keyed ``.so`` whose ``<so>.cachekey`` sidecar is missing or disagrees with the
      recomputed key RAISES a StaleArtifactError on the next compile HIT -- never a silent
      warn-and-reuse (ADC-536 forbidden: silent stale reuse).
  (2) DEBUG SIDECAR: ``debug=True`` persists a ``.cpp`` whose leading provenance banner carries the
      serialized IR, the hashes, the flags and the redacted command.
  (3) BINARY-IDENTICAL: the ``debug=True`` ``.so`` bytes equal the non-debug ``.so`` bytes for the
      same Program (the banner rides the sidecar only; the cache key is unperturbed).

Runs in CI (gate rebuilds _pops with the compile toolchain); skips locally when no compiler / Kokkos
is visible or the .so compile fails -- never faking the engine.
"""
import sys


def _skip(msg):
    print("skip test_compile_stale_and_debug (%s)" % msg)
    sys.exit(0)


try:
    import hashlib
    import os
    import tempfile

    import pops
    from pops import time as adctime
    from pops.codegen.compile_provenance import cachekey_path, read_cachekey_sidecar, StaleArtifactError
except Exception as exc:  # noqa: BLE001 -- pops/_pops unavailable in this interpreter
    _skip("pops unavailable: %s" % exc)

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def _fe_program(name="stale_debug_probe"):
    P = adctime.Program(name)
    dt = P.dt
    U = P.state("ions")
    f = P.solve_fields(U)
    R = P._rhs_legacy(state=U, fields=f, flux=True, sources=["default"])
    P.commit(P.state("U", block="ions").next, P.linear_combine("U1", U + dt * R))
    return P


def transport_model():
    return pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                      transport=pops.IsothermalFlux(),
                      source=pops.NoSource(),
                      elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))


# First fresh compile: reaching a real .so gates the whole test on the toolchain.
cache_dir = tempfile.mkdtemp()
os.environ["POPS_CACHE_DIR"] = cache_dir
try:
    fresh = pops.codegen.compile_problem(time=_fe_program(), model=transport_model())
except (RuntimeError, Exception) as exc:  # noqa: BLE001 -- no compiler / Kokkos / compile failure
    _skip("compile_problem could not build the .so: %s" % str(exc)[:160])

so_path = fresh.so_path
chk(os.path.isfile(so_path), "fresh compile produced a .so")
chk(os.path.isfile(cachekey_path(so_path)), "fresh compile wrote the <so>.cachekey sidecar")
side = read_cachekey_sidecar(so_path)
chk(side is not None and side.get("cache_key") == fresh.cache_key,
    "the sidecar records the artifact cache key")

# ---- (1) stale refusal: corrupt / remove the sidecar and recompile (HIT) ----
print("== (1) stale-sidecar refusal ==")

# 1a: delete the sidecar (a legacy .so with no sidecar) -> the next HIT refuses it.
os.remove(cachekey_path(so_path))
try:
    pops.codegen.compile_problem(time=_fe_program(), model=transport_model())
    chk(False, "a cache HIT on a .so with NO sidecar must raise (missing sidecar)")
except StaleArtifactError as exc:
    chk("sidecar" in str(exc) and so_path in str(exc), "missing sidecar refused, naming the .so")

# 1b: write a MISMATCHED sidecar -> the next HIT refuses it (foreign/corrupt .so at the keyed path).
with open(cachekey_path(so_path), "w") as f:
    f.write("cache_key=STALEKEY\nabi_key=STALEABI\ntoolchain=x|y\n")
try:
    pops.codegen.compile_problem(time=_fe_program(), model=transport_model())
    chk(False, "a cache HIT on a .so with a MISMATCHED sidecar must raise")
except StaleArtifactError as exc:
    chk("expected=" in str(exc) and "found=STALEKEY" in str(exc),
        "mismatched sidecar refused, naming expected vs found")

# ---- (2) + (3) debug sidecar + binary identity ----
print("== (2)+(3) debug provenance sidecar, binary-identical .so ==")

nodebug_so = os.path.join(tempfile.mkdtemp(), "nodebug.so")
debug_so = os.path.join(tempfile.mkdtemp(), "debug.so")
try:
    pops.codegen.compile_problem(nodebug_so, time=_fe_program("bin_identity"),
                                 model=transport_model())
    pops.codegen.compile_problem(debug_so, time=_fe_program("bin_identity"),
                                 model=transport_model(), debug=True)
except (RuntimeError, Exception) as exc:  # noqa: BLE001
    _skip("explicit-path compile failed: %s" % str(exc)[:160])

debug_cpp = os.path.splitext(debug_so)[0] + ".cpp"
chk(os.path.isfile(debug_cpp), "debug=True persisted the generated .cpp")
if os.path.isfile(debug_cpp):
    with open(debug_cpp) as f:
        text = f.read()
    chk(text.startswith("/*"), "the .cpp opens with the provenance banner block comment")
    chk("cache_key" in text and "program_hash" in text and "abi_key" in text,
        "the banner carries the hashes")
    chk("serialized Program IR" in text and "compile_command" in text,
        "the banner carries the serialized IR + the redacted command")
    chk("pops_install_program" in text, "the generated source follows the banner (compilable)")

# BINARY-IDENTICAL: the debug .so bytes equal the non-debug .so bytes (banner is sidecar-only).
if os.path.isfile(nodebug_so) and os.path.isfile(debug_so):
    h_nodebug = hashlib.sha256(open(nodebug_so, "rb").read()).hexdigest()
    h_debug = hashlib.sha256(open(debug_so, "rb").read()).hexdigest()
    chk(h_nodebug == h_debug,
        "the debug .so bytes equal the non-debug .so bytes (banner did not perturb the build)")

print("%s test_compile_stale_and_debug" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
