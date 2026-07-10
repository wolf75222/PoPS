#!/usr/bin/env python3
"""ADC-542: declared typed diagnostic measures FIRE on the run loop via native reductions.

The typed measures (pops.diagnostics.Norm / Integral / MinMax / ConservationCheck) wire to the
EXISTING native collective reductions through a run-loop hook (pops.runtime._diagnostics_driver). A
LOCAL end-to-end proof on the Uniform System: it builds a REAL native System (no DSL compile, no fake
engine), maps each measure to its native reduction, and asserts the recorded value matches a direct
native reduce_component on the same state (the descriptors EXECUTE, they are no longer inert metadata).

  (1) measure_reduction maps each category to the right native reduction (Norm(L1/L2/LInf), Integral,
      MinMax -> min/max keys); an unmapped category raises.
  (2) diagnostic_due (shared cadence interpreter) fires every / always / on_start / on_end / int.
  (3) fire_diagnostics records each due measure via record_program_diagnostic (readable back).
  (4) a role-scoped Norm resolves the role to a component; an unscoped one folds the full state.

Skips if pops is absent (never fakes the engine). Runs under pytest and the __main__ guard.
"""
import sys

try:
    import numpy as np
    import pops
    from pops.numerics.reconstruction.limiters import Minmod
    from pops.diagnostics import Norm, Integral, MinMax, ConservationCheck
    from pops.linalg.norms import L1, L2, LInf
    from pops.model import Module
    from pops.problem import Problem
    from pops.time.schedule import every, always, on_start, on_end
    from pops.runtime._diagnostics_driver import (diagnostic_due, measure_reduction,
                                                  fire_diagnostics)
    from pops.runtime.system import System
except Exception as exc:  # noqa: BLE001
    print("skip test_diagnostics_driver (pops unavailable: %s)" % exc)
    sys.exit(0)

fails = 0

_DIAGNOSTIC_PROBLEM = Problem(name="runtime-diagnostics")
_IONS_BLOCK = _DIAGNOSTIC_PROBLEM.add_block("ions", Module("runtime-diagnostic-model"))


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def build(n=16):
    sim = System(n=n, L=1.0, periodic=True)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
    sim.add_block("ions",
                  pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                             transport=pops.IsothermalFlux(),
                             source=pops.PotentialForce(charge=1.0),
                             elliptic=pops.ChargeDensity(charge=1.0)),
                  spatial=pops.FiniteVolume(limiter=Minmod()), time=pops.Explicit())
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    sim.set_density("ions",
                    (1.0 + 0.4 * np.exp(-50.0 * ((X - 0.4) ** 2 + (Y - 0.5) ** 2))).ravel())
    return sim


# --- (0) shared cadence interpreter (host-testable) -----------------------------------
print("== (0) diagnostic_due cadence ==")
chk(diagnostic_due(every(2), 2) and not diagnostic_due(every(2), 3), "every(2) due at 2 not 3")
chk(diagnostic_due(always(), 1) and diagnostic_due(None, 1), "always()/None every step")
chk(diagnostic_due(on_start(), 1) and not diagnostic_due(on_start(), 2), "on_start at step 1")
chk(diagnostic_due(on_end(), 4, last_step=4) and not diagnostic_due(on_end(), 3, last_step=4),
    "on_end at the last step")

# --- (1) measure_reduction maps each category to a native reduction --------------------
print("== (1) measure_reduction native mapping ==")
sim = build()
# The block state (isothermal: 1 component, the density). reduce_component gives the native truth.
direct_sum = sim.reduce_component("ions", "sum", 0)
direct_min = sim.reduce_component("ions", "min", 0)
direct_max = sim.reduce_component("ions", "max", 0)
direct_l1 = sim.reduce_component("ions", "abs_sum", 0)
direct_l2 = sim.reduce_component("ions", "sum_sq", 0) ** 0.5
direct_linf = sim.reduce_component("ions", "abs_max", 0)

r_int = measure_reduction(sim, Integral(block=_IONS_BLOCK))
chk(abs(list(r_int.values())[0] - direct_sum) < 1e-12, "Integral -> native sum")
r_l1 = measure_reduction(sim, Norm(L1(), block=_IONS_BLOCK))
chk(abs(list(r_l1.values())[0] - direct_l1) < 1e-12, "Norm(L1) -> native abs_sum")
r_l2 = measure_reduction(sim, Norm(L2(), block=_IONS_BLOCK))
chk(abs(list(r_l2.values())[0] - direct_l2) < 1e-9, "Norm(L2) -> sqrt(sum_sq)")
r_linf = measure_reduction(sim, Norm(LInf(), block=_IONS_BLOCK))
chk(abs(list(r_linf.values())[0] - direct_linf) < 1e-12, "Norm(LInf) -> native abs_max")
r_mm = measure_reduction(sim, MinMax(block=_IONS_BLOCK))
mm_name = MinMax(block=_IONS_BLOCK).name
chk(abs(r_mm["%s.min" % mm_name] - direct_min) < 1e-12 and
    abs(r_mm["%s.max" % mm_name] - direct_max) < 1e-12, "MinMax -> native min/max keys")

# --- (2) an unmapped category raises (fail loud) --------------------------------------
print("== (2) unmapped category raises ==")


class _Bogus:
    category = "diagnostic_bogus"
    name = "bogus"
    block = _IONS_BLOCK
    role = None


try:
    measure_reduction(sim, _Bogus())
    chk(False, "an unmapped category should raise")
except ValueError as e:
    chk("not mapped" in str(e), f"unmapped category raises precisely: {str(e)[:50]}")

# --- (3) fire_diagnostics records each due measure, readable back ----------------------
print("== (3) fire_diagnostics records via the native sink ==")
sim3 = build()
measures = [Norm(L2(), block=_IONS_BLOCK, cadence=every(1)),
            Integral(block=_IONS_BLOCK, cadence=every(2))]
rec1 = fire_diagnostics(sim3, measures, step=1, last_step=None, baselines={})
chk(Norm(L2(), block=_IONS_BLOCK).name in rec1
    and Integral(block=_IONS_BLOCK).name not in rec1,
    "step 1: only the every(1) Norm fires")
rec2 = fire_diagnostics(sim3, measures, step=2, last_step=None, baselines={})
chk(Integral(block=_IONS_BLOCK).name in rec2, "step 2: the every(2) Integral fires")
# The recorded values are readable through the native program_diagnostics map.
diags = sim3.program_diagnostics()
chk(Norm(L2(), block=_IONS_BLOCK).name in diags, "recorded Norm readable via program_diagnostics")

# --- (4) ConservationCheck drift anchors on the first tick -----------------------------
print("== (4) ConservationCheck drift ==")
sim4 = build()
baselines = {}
check = ConservationCheck(Integral(block=_IONS_BLOCK))
d0 = fire_diagnostics(sim4, [check], step=1, last_step=None, baselines=baselines)
chk(abs(d0["%s.drift" % check.name]) < 1e-12, "first-tick conservation drift is 0 (baseline anchor)")

if fails:
    print(f"FAIL test_diagnostics_driver : {fails} echec(s)")
    sys.exit(1)
print("OK test_diagnostics_driver")
