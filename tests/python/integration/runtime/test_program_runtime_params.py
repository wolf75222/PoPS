#!/usr/bin/env python3
"""Compiled time PROGRAM runtime parameters, end to end (epic ADC-479 / ADC-510, Spec 5 C5).

A compiled time Program whose physics reads a ``dsl.Param(..., kind="runtime")`` carries the value in
a per-PROGRAM-block ``pops::RuntimeParams`` owned by the System (not the .so closure), so the value can
be CHANGED at run time WITHOUT recompiling -- the SAME no-recompile contract as the AOT-native
``set_block_params`` (P7-b), mirrored for the Program. The lowered ``source`` / ``linear_source``
kernel reads the CURRENT value via ``ctx.program_params(block).get(index)``.

(A) HOST-testable (pure Python, always runs): the codegen REPLACES the old "a later phase" reject with
    a real emission -- a runtime-param source lowers ``ctx.program_params(0)`` + ``params.get(0)`` and
    exports the ``pops_program_param_*`` metadata round-tripping the name + default; a const param
    stays inlined (count 0). The pure routing core maps name -> per-block vector and rejects an unknown
    name. The install-time validation rejects a params= name no Program kernel reads.

(B) END-TO-END (skips cleanly unless the full Kokkos toolchain is present): compile a Program whose
    source S = k * rho reads the runtime param k, bind it, step, record the result; re-set k to a
    DIFFERENT value WITHOUT recompiling, step from the same state, and assert the result DIFFERS as
    predicted (the source contribution scales LINEARLY in k). Runs in CI (gate-python rebuilds _pops);
    skips if numpy/_pops/compiler/Kokkos is absent or the .so compile fails -- never faking the engine.
"""
import sys
from pops.runtime.system import System  # ADC-545 advanced runtime seam


def _skip(msg):
    print("skip test_program_runtime_params (%s)" % msg)
    sys.exit(0)


try:
    import numpy as np

    import pops
    from pops import time as adctime
    from pops.physics import RuntimeParam, ConstParam
    from pops.physics.facade import Model
except Exception as exc:  # noqa: BLE001  -- numpy or _pops unavailable in this interpreter
    _skip("pops/numpy unavailable: %s" % exc)

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


def _decay_model(kind="runtime", k_value=2.0):
    """Scalar density (rho) with NO transport and a NAMED source S = k * rho (k runtime or const). The
    Program lowers the source kernel itself, so a runtime k reaches it via ctx.program_params (C5)."""
    m = Model("decay")
    (rho,) = m.conservative_vars("rho")
    k = m.param({"runtime": RuntimeParam, "const": ConstParam}[kind]("k", k_value))
    m.primitive_vars(rho=rho)
    m.conservative_from([rho])
    m.flux(x=[rho * 0.0], y=[rho * 0.0])      # no transport: isolate the source contribution
    m.eigenvalues(x=[rho * 0.0], y=[rho * 0.0])
    m.source_term("decay", [k * rho])         # S = k * rho reads the runtime param k
    return m


def _decay_program(name="decay_runtime"):
    """U <- U + dt * S, S = the named 'decay' source (the directly-lowered source kernel path)."""
    P = adctime.Program(name)
    U = P.state("gas")
    S = P.source("decay", state=U)
    P.commit("gas", P.linear_combine("U1", U + P.dt * S))
    return P


# ---- (A) host-testable: codegen emission + metadata + routing + install validation ----
print("== (A) compiled-Program runtime-param codegen + routing (pure Python) ==")
P = _decay_program()
src_rt = P.emit_cpp_program(model=_decay_model("runtime", 2.0))
chk("ctx.program_params(0)" in src_rt, "runtime source binds ctx.program_params(0)")
chk("params.get(0)" in src_rt, "runtime source reads params.get(0) (not inlined)")
chk("pops_program_param_count() { return 1; }" in src_rt, "metadata exports 1 runtime param")
chk('"k"' in src_rt and "pops_program_param_name" in src_rt, "metadata exports the param NAME 'k'")
chk("a later phase" not in src_rt, "the old 'a later phase' reject text is gone")

src_const = _decay_program().emit_cpp_program(model=_decay_model("const", 2.0))
chk("params.get(" not in src_const, "const param stays INLINE (no params.get read)")
chk("ctx.program_params(" not in src_const, "const param -> no per-block RuntimeParams binding")
chk("pops_program_param_count() { return 0; }" in src_const, "const-only -> 0 runtime params")

from pops.codegen.program_emit_params import program_param_entries, program_param_routes  # noqa: E402
chk(program_param_entries(P, _decay_model("runtime", 2.0)) == [(0, "k", 0, 2.0)],
    "_program_param_entries routes k -> (block 0, index 0, default 2.0)")
per_block, defaults = program_param_routes(P, _decay_model("runtime", 2.0))
chk(per_block == {0: ["k"]} and defaults == {"k": 2.0}, "routes -> {0: ['k']}, default 2.0")

# Pure routing core: supplied value, declaration default, unknown-name rejection.
from pops.runtime._install_param_routing import route_program_params  # noqa: E402
pb, unknown = route_program_params({0: ["k"]}, {"k": 2.0}, {"k": 7.0})
chk(pb == {0: [7.0]} and unknown == [], "route supplied k=7.0 (no unknown)")
pb2, _ = route_program_params({0: ["k"]}, {"k": 2.0}, {})
chk(pb2 == {0: [2.0]}, "route falls back to the declaration default 2.0")
_, unk3 = route_program_params({0: ["k"]}, {"k": 2.0}, {"nope": 1.0})
chk(unk3 == ["nope"], "an unknown param name is flagged (no silent drop)")

# ---- (B) end-to-end: skips unless the full Kokkos toolchain is present ----
if not hasattr(System(n=8, L=1.0, periodic=True), "install_program"):
    print("-- (B) skipped: _pops lacks the install_program binding (rebuild _pops) --")
    print("%s test_program_runtime_params (A only)" % ("FAIL" if fails else "PASS"))
    sys.exit(1 if fails else 0)
if not hasattr(System(n=8, L=1.0, periodic=True), "set_program_params"):
    print("-- (B) skipped: _pops lacks the set_program_params binding (rebuild _pops) --")
    print("%s test_program_runtime_params (A only)" % ("FAIL" if fails else "PASS"))
    sys.exit(1 if fails else 0)

print("== (B) end-to-end: a different k yields a different step, no recompile ==")
from pops.numerics.reconstruction import FirstOrder  # noqa: E402
from pops.numerics.riemann import Rusanov  # noqa: E402

n = 24
x = (np.arange(n) + 0.5) / n
X, Y = np.meshgrid(x, x, indexing="ij")
rho0 = (1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)).reshape(-1)
dt = 1e-2


def make_sim(model):
    # The block is registered with its production-compiled model (the native brick path:
    # add_equation, not add_block which takes a _pops.ModelSpec); install_program then OVERLAYS the
    # whole-system Program, whose source reads k via ctx.program_params -> set_program_params changes
    # it. No set_poisson: _decay_program has no solve_fields, so install_program needs no solver.
    sim = System(n=n, L=1.0, periodic=True)
    try:
        cm = model.compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        _skip("block model compile could not build the .so: %s" % str(exc)[:160])
    sim.add_equation("gas", cm,
                     spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                     time=pops.Explicit(method="euler"))
    sim.set_state("gas", rho0.tolist())
    return sim


try:
    compiled = pops.codegen.compile_problem(model=_decay_model("runtime", 2.0), time=_decay_program())
except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
    _skip("compile_problem could not build the .so: %s" % str(exc)[:160])

routes_fn = getattr(compiled, "runtime_param_routes", None)
chk(callable(routes_fn) and routes_fn()[0] == {0: ["k"]},
    "the compiled handle declares the runtime route {0: ['k']}")

# Step with k = 2.0 (the declaration default the .so seeds at install).
sim2 = make_sim(_decay_model("runtime", 2.0))
sim2.install_program(compiled.so_path)
U0 = np.array(sim2.get_state("gas"))
sim2.step(dt)
U2 = np.array(sim2.get_state("gas"))
d2 = U2 - U0  # the per-step increment dt * (k=2) * rho

# Re-bind a FRESH sim on the SAME .so (no recompile), set k = 6.0, step from the same state.
sim6 = make_sim(_decay_model("runtime", 2.0))
sim6.install_program(compiled.so_path)        # same cached .so -> no recompile
sim6.set_program_params(0, [6.0])             # change the runtime param: effect at the next step
U0b = np.array(sim6.get_state("gas"))
sim6.step(dt)
U6 = np.array(sim6.get_state("gas"))
d6 = U6 - U0b  # the per-step increment dt * (k=6) * rho

chk(float(np.abs(d2).max()) > 1e-6, "the k=2 step actually changed the state (source non-trivial)")
# S = k*rho, no flux: the increment is dt*k*rho, LINEAR in k -> d6 == 3*d2 (k 2 -> 6) to round-off.
chk(np.allclose(d6, 3.0 * d2, rtol=1e-9, atol=1e-12),
    "a DIFFERENT k (2 -> 6) yields a DIFFERENT step scaling x3, WITHOUT recompiling (max|d6-3 d2| "
    "= %.2e)" % float(np.abs(d6 - 3.0 * d2).max()))

# The cache is HIT on a second compile of the same Program -> the runtime change needed no recompile.
c2 = pops.codegen.compile_problem(model=_decay_model("runtime", 2.0), time=_decay_program())
chk(c2.so_path == compiled.so_path, "cache HIT: same Program -> same .so (the k change recompiled nothing)")

# An unknown params= name routed at bind raises a clear ValueError (no silent drop).
sim_bad = make_sim(_decay_model("runtime", 2.0))
sim_bad.install_program(compiled.so_path)
chk(raises(ValueError, lambda: sim_bad._install_program_params(compiled, {"nope": 1.0})),
    "install rejects a params= name no Program kernel reads")

print("%s test_program_runtime_params" % ("FAIL" if fails else "PASS"))
sys.exit(1 if fails else 0)


if __name__ == "__main__":
    pass
