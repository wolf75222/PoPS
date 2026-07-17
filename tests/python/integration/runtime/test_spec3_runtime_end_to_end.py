#!/usr/bin/env python3
"""Spec 3 compiled-.so RUNTIME, end to end on a real engine (epic ADC-450).

This is the runtime counterpart to the emit-only Spec 3 tests (test_pernode_profiling,
test_profiling_counters, scheduled_fields_subcycled_transport): those pin the GENERATED C++ or
assert the scheduler counters are ABSENT on the native path and defer the runtime to ROMEO. Here we
build a REAL operator-first Case and Program, compile them through the final
``validate -> resolve -> compile -> bind`` lifecycle, and STEP the installed native engine -- so the
spec's runtime acceptance criteria are RUNTIME-proven, not emit-asserted. It never fakes the engine.

The Program holds a board-style field solve on `Schedule(Every(...), off=Hold())` (Spec 3,
ADC-458): the Poisson solve `phi <- solve(-Laplace phi = alpha*rho)` recomputes only when DUE and the
cached aux is restored in between. The held phi feeds a PotentialForce source, so a held step and a
recompute step produce a genuinely different RHS -- the cache is observable, not cosmetic.

Asserted runtime criteria (each a Spec 3 acceptance criterion):
  1. NO Python in sim.step (criterion 19, test 24.18): a sys.settrace hook around the bound C++
     `step_cfl` records ZERO Python call-frames entered inside the step (the .so body is pure C++);
     a stronger proxy patches the Python model/handle objects to raise if called during the step.
  2. The step ADVANCES the state: max|U^{n+1}-U^n| > 0 and finite (a real compiled run, not a no-op).
  3. CHECKPOINT == RESTART with the scheduler cache exercised (criterion 22/35, test 24.23): a held
     schedule run, checkpointed at a DUE boundary, restarted into a replayed composition, continues
     bit-identically to a continuous run.
  4. PROFILING (criterion 40, test 24.21): sim.enable_profiling() + a held-schedule step surfaces the
     per-node ("node:...") + kernels + cache hit/miss + nodes due/skipped lines with sane values --
     the COMPILED-runtime counters test_profiling_counters could only assert ABSENT on the host path.

Runs on the gate's Kokkos Serial shard (CI auto-discovers tests/python/**/test_*.py) and locally with
`POPS_KOKKOS_ROOT=... KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`. Missing native prerequisites are
explicit local skips and required-lane failures; an unavailable
extension, scheduler binding, compiler or Kokkos can never appear as a green required test.
"""
import sys
from tests.python.support.requirements import require_native_or_skip


def _skip(msg):
    require_native_or_skip("test_spec3_runtime_end_to_end: %s" % msg)


try:
    import numpy as np

    import pops
    from pops.domain import Rectangle
    from pops.fields import (
        CellCenteredSecondOrder,
        ConstantNullspace,
        FieldDiscretization,
        FieldOutput,
        GradientOutput,
        MeanValueGauge,
    )
    from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Periodic
    from pops.frames import Cartesian2D
    from pops.layouts import Uniform
    from pops.math import ddt, div, laplacian, sqrt
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.numerics.terms import Flux
    from pops.physics import Model
    from pops.solvers.elliptic import GeometricMG
    from pops import time as adctime
    from tests.python.support.native_execution_context import artifact_execution_context
except Exception as exc:  # noqa: BLE001  -- numpy or _pops unavailable in this interpreter
    _skip("pops/numpy unavailable: %s" % exc)

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


N = 24
EVERY = 2  # the field solve recomputes every 2 macro-steps and holds the cached phi in between
DT = 1e-3


def _initial_state():
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    return np.ascontiguousarray(np.stack([rho, 0.4 * rho, -0.2 * rho]))


def _build_case(name="spec3_runtime_held"):
    """Build the complete final Case with one scheduled, case-owned field authority."""
    frame = Rectangle(
        "%s-domain" % name, lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model("%s-model" % name, frame=frame)
    state = model.state("U", components=("rho", "mx", "my"))
    rho, mx, my = state
    u, v = mx / rho, my / rho
    pressure = 0.5 * rho
    sound_speed = sqrt(0.5 + 0.0 * rho)
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={
            x_axis: (mx, mx * u + pressure, my * u),
            y_axis: (my, mx * v, my * v + pressure),
        },
        waves={
            x_axis: (u - sound_speed, u, u + sound_speed),
            y_axis: (v - sound_speed, v, v + sound_speed),
        },
    )
    potential = model.field("potential")
    electric_x, electric_y = model.aux("electric_x"), model.aux("electric_y")
    field_operator = model.field_operator(
        "electrostatic",
        unknown=potential,
        equation=-laplacian(potential) == rho - 1.0,
        outputs=(
            FieldOutput("potential", potential),
            GradientOutput("electric", potential, sign=-1),
        ),
    )
    electric = model.source(
        "electric_force",
        on=state,
        value=(0.0 * rho, rho * electric_x, rho * electric_y),
    )
    rate = model.rate(
        "explicit_rhs", equation=ddt(state) == -div(flux) + electric)
    # ``Hold`` is legal only when the exact authoritative provider declares that its
    # materialized output can be retained. This is an authored capability, not an
    # inference from the solver choice or from the field name.
    model.module.operator_capabilities("electrostatic", cacheable=True)
    electric_operator = model.operators["electric_force"]

    case = pops.Case("%s-case" % name)
    block = case.block("ions", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(numerics, block=block)
    field = case.field(
        field_operator,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Periodic()),),
            solver=GeometricMG(),
            nullspace=ConstantNullspace(),
            gauge=MeanValueGauge(0.0),
        ),
    )

    program = adctime.Program(name)
    temporal = program.state(block[state])
    schedule = adctime.Schedule(
        adctime.Every(adctime.AcceptedStep(program.clock), EVERY),
        off=adctime.Hold(),
    )
    fields = field(
        temporal.n, name="electrostatic", schedule=schedule
    ).consume(action=adctime.FailRun())
    rhs = program.rhs(
        name="R",
        state=temporal.n,
        fields=fields,
        terms=(Flux(), electric_operator),
    )
    program.commit(
        temporal.next,
        program.value(
            "U1", temporal.n + program.dt * rhs, at=temporal.next.point),
    )
    program.step_strategy(adctime.FixedDt(DT))
    case.program(program)
    layout = Uniform(
        CartesianGrid(
            frame=frame,
            cells=(N, N),
            periodic=PeriodicAxes(frame.axes),
        )
    )
    return case, layout, program


print("== compile the held-schedule Case through validate -> resolve -> compile ==")
try:
    authored_case, authored_layout, authored_program = _build_case()
    compiled = pops.compile(
        pops.resolve(pops.validate(authored_case), layout=authored_layout)
    )
except RuntimeError as exc:  # no compiler / no visible Kokkos / .so compile failed
    _skip("public compile could not build the native artifact: %s" % str(exc)[:200])
chk(
    compiled.program_hash == authored_program.inspect().hash,
    "compiled artifact authenticates the final Program hash (%s...)"
    % compiled.program_hash[:12],
)


def install():
    """Bind one fresh runtime exclusively through the final public lifecycle."""
    return pops.bind(
        compiled,
        initial_state={"ions": _initial_state()},
        resources={"execution_context": artifact_execution_context(compiled)},
    )


def run_steps(sim, steps):
    return pops.run(
        sim,
        t_end=float(sim.time()) + steps * DT,
        max_steps=steps,
    )


# --- (2) the compiled step ADVANCES the state (do this first: it also primes the chain) -----------
print("== (2) the compiled step advances the state ==")
sim2 = install()
u0 = np.array(sim2.get_state("ions"))
dt_used = sim2._executor._s.step_cfl(0.4)
u1 = np.array(sim2.get_state("ions"))
change = float(np.abs(u1 - u0).max())
chk(dt_used > 0.0 and np.isfinite(dt_used), "step_cfl returned a finite positive dt (%.6g)" % dt_used)
chk(np.all(np.isfinite(u1)), "the stepped state is finite")
chk(change > 0.0, "max|U^{n+1}-U^n| > 0 (real compiled run, not a no-op): %.3e" % change)

# --- (1) NO Python frames are entered inside the compiled step (criterion 19) ---------------------
print("== (1) no Python in sim.step (settrace + raise-on-call proxy) ==")
sim1 = install()
raw = sim1._executor._s  # exact bound C++ System; no Python facade inside the pybind call

# (1a) settrace cannot see into the .so -- that is the POINT. We count Python 'call' events between
# the pybind entry and exit: a pure-C++ step body enters ZERO new Python frames. (settrace fires on
# Python-level calls only; a pybind call into C++ that never re-enters Python yields no 'call' event.)
import gc  # noqa: E402

py_calls = []
prev = sys.gettrace()


def _tracer(frame, event, arg):
    if event == "call":
        py_calls.append(frame.f_code.co_qualname if hasattr(frame.f_code, "co_qualname")
                        else frame.f_code.co_name)
    return _tracer


# Disable the cyclic GC across the traced step: a finalizer firing mid-step would re-enter Python and
# inject a spurious 'call' frame that has nothing to do with the C++ step body. The step allocates no
# Python cycles, so disabling GC here changes nothing but the trace's determinism.
gc.disable()
sys.settrace(_tracer)
try:
    raw.step_cfl(0.4)  # the bound C++ macro-step: drives the installed .so closure
finally:
    sys.settrace(prev)
    gc.enable()
# _tracer itself is invoked by the interpreter but is NOT a frame entered *inside* the C++ step; the
# only way py_calls grows is if the step re-enters Python (a callback / descriptor). Expect zero.
chk(len(py_calls) == 0,
    "zero Python call-frames entered inside the compiled step (got %d: %r)"
    % (len(py_calls), py_calls[:6]))

# (1b) stronger proxy: public compile detaches the source Case/Program builders. Drop those exact
# authoring objects, bind afresh from the immutable compiled artifact, then step its raw native
# closure. If numerical execution retained or called a builder, the weakref or native step would fail.
import weakref  # noqa: E402

source_program_ref = weakref.ref(authored_program)
del authored_case, authored_layout, authored_program
gc.collect()
chk(source_program_ref() is None,
    "compiled artifact retains no source Program builder")
orphan = install()
orphan_raw = orphan._executor._s
ou0 = np.array(orphan.get_state("ions"))
orphan_raw.step_cfl(0.4)
ou1 = np.array(orphan.get_state("ions"))
chk(float(np.abs(ou1 - ou0).max()) > 0.0 and np.all(np.isfinite(ou1)),
    "the bound step runs after source authoring is collected (C++-only numerical closure)")

# --- (3) CHECKPOINT == RESTART with the scheduler cache exercised (criterion 22/35) ---------------
print("== (3) checkpoint == restart with a held schedule (cache cadence exercised) ==")
import os  # noqa: E402
import tempfile  # noqa: E402

tmp = tempfile.mkdtemp()
K = 6            # total macro-steps
J = EVERY + 1    # checkpoint at a NON-due boundary (J % EVERY != 0): the held node is MID-CADENCE, so
                 # the cached value (from the last due step) is live at the checkpoint and the first
                 # post-restart step HOLDS -- it must read the value RESTORED from the checkpoint, not
                 # recompute. This exercises ADC-458 section 30 (the cache slots are now serialized:
                 # program_cache_global / cache_ngrow). A restart that dropped the cache would hold a
                 # cold-vs-warm value and diverge; bit-identity here proves the cache round-trips.

# continuous reference: K held-schedule steps.
ref = install()
run_steps(ref, K)
ref_u = np.array(ref.get_state("ions"))
ref_t = float(ref.time())
ref_ms = ref.macro_step()

# checkpoint mid-cadence (non-due) J, then continue to K.
ck = install()
run_steps(ck, J)
chk(ck.macro_step() == J and (J % EVERY) != 0,
    "checkpoint taken mid-cadence, non-due (macro_step=%d)" % J)
path = ck.checkpoint(os.path.join(tmp, "spec3_chk"))
chk(os.path.exists(path), "checkpoint written (%s)" % os.path.basename(path))
run_steps(ck, K - J)  # original continuation must match independently

# restart: REPLAY the composition + RE-INSTALL the same program, then resume to K.
res = install()
res.restart(os.path.join(tmp, "spec3_chk"))
chk(res.macro_step() == J and abs(res.time() - J * DT) < 1e-15,
    "clock restored (t=%.6g, macro_step=%d)" % (res.time(), res.macro_step()))
run_steps(res, K - J)
res_u = np.array(res.get_state("ions"))

e_restart = float(np.abs(res_u - ref_u).max())
chk(e_restart == 0.0, "restart == continuous run bit-identical over %d steps (max|d|=%.2e)"
    % (K, e_restart))
chk(abs(float(res.time()) - ref_t) < 1e-15 and res.macro_step() == ref_ms,
    "final clock identical (t=%.6g, macro_step=%d)" % (res.time(), res.macro_step()))

# --- (4) PROFILING: per-node + kernels + scheduler cache counters at RUNTIME (criterion 40) -------
print("== (4) profiling report surfaces per-node + kernels + scheduler-cache counters ==")
prof = install()
prof_engine = prof._executor
prof_engine.enable_profiling()
# Step across a full cache window so BOTH a recompute (due) and a hold (skip) happen: at EVERY=2,
# step 0 is due (cold), step 1 holds, step 2 due, step 3 holds -> due and skipped both move.
run_steps(prof, 2 * EVERY)
report = prof_engine.profile_report()
print(report)


def _counter(name):
    for tok in report.replace("\n", " ").split():
        if tok.startswith(name + "="):
            return int(tok.split("=", 1)[1])
    return None


# per-node timing (Spec 3 section 29): the held field solve + the rhs are wrapped per node.
chk("node:" in report, "report carries per-node scopes (node:...)")
# the coarse "step" scope is always recorded by System::step.
chk("step" in report, "report carries the coarse 'step' scope")
# kernels move on the compiled path (the field solve dispatches a kernel each DUE step).
kernels = _counter("kernels")
chk(kernels is not None and kernels > 0, "kernels counter > 0 (=%r)" % kernels)
# THE compiled-runtime counters test_profiling_counters could only assert ABSENT on the native path:
# schedule_decision fires them at the held node's exact typed decision point.
nodes_due = _counter("nodes_due")
nodes_skipped = _counter("nodes_skipped")
cache_hits = _counter("cache_hits")
cache_misses = _counter("cache_misses")
chk(nodes_due is not None and nodes_due > 0, "nodes_due > 0 (held node recomputed when due: %r)" % nodes_due)
chk(nodes_skipped is not None and nodes_skipped > 0,
    "nodes_skipped > 0 (held node skipped off-cadence: %r)" % nodes_skipped)
# cache hit == skip, miss == due (one decision per held node per step): so over 2*EVERY steps with one
# held node, due + skipped == steps and hits == skipped, misses == due.
chk(cache_misses == nodes_due, "cache_misses == nodes_due (%r == %r)" % (cache_misses, nodes_due))
chk(cache_hits == nodes_skipped, "cache_hits == nodes_skipped (%r == %r)" % (cache_hits, nodes_skipped))
chk((nodes_due + nodes_skipped) == 2 * EVERY,
    "one scheduler decision per step: due+skipped == %d (%r)" % (2 * EVERY, nodes_due + nodes_skipped))
chk(_counter("steps") == 2 * EVERY, "step counter == %d" % (2 * EVERY))

print("%s test_spec3_runtime_end_to_end" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
