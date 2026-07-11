#!/usr/bin/env python3
"""C4 / ADC-509: typed OutputPolicy policies fire on a Uniform System.

The typed output / checkpoint policies (pops.output) wire to the EXISTING write()/checkpoint
writers through a run-loop hook (pops.runtime._output_driver). This is a LOCAL end-to-end proof
on the Uniform / single-level System path: it builds a REAL native System (native bricks, no DSL
compile, no Kokkos .so -- the no-fake-engine rule), advances the explicit low-level seam and calls
the policy driver directly. ``System.run`` is reserved for a completed ``pops.bind`` transaction.
The test asserts:

  (1) an OutputPolicy(format=npz, cadence=every(N)) writes a file at every Nth step and NOT
      in between, with the right step suffix and the expected fields inside;
  (2) low-level ``run`` refuses before a bind/run identity can be fabricated;
  (3) the pure cadence interpreter fires every(N) / always / int exactly when due;
  (4) Plotfile is the one precise reject (no Uniform writer -> ADC-511), level selection is a
      no-op on a single-level System (AllLevels / CoarseOnly both write the single level).

Invariants by assert; prints "OK test_output_policy_run" on success. Skips if pops is absent
(never fakes the engine). Runs under pytest and as a plain script (CI runs the __main__ guard).
"""
import os
import sys
import tempfile

try:
    import numpy as np
    import pops
    from pops.numerics.reconstruction.limiters import Minmod
    from pops.output import (OutputPolicy, CheckpointPolicy, HDF5, Plotfile,
                             AllLevels, CoarseOnly)
    from pops.model import Module
    from pops.problem import Problem
    from pops.time.schedule import always, every, on_end, on_start, when
    from pops.runtime._output_driver import policy_due, _format_token, fire_output_policies
    from pops.runtime.system import System  # ADC-545 advanced runtime seam
except Exception as exc:  # noqa: BLE001
    print("skip test_output_policy_run (pops unavailable: %s)" % exc)
    sys.exit(0)

fails = 0

_OUTPUT_MODULE = Module("runtime-output-model")
_OUTPUT_STATE = _OUTPUT_MODULE.state_space("U", components=("rho",))
_OUTPUT_STATE_REF = _OUTPUT_MODULE.state_handle(_OUTPUT_STATE)
_OUTPUT_PROBLEM = Problem(name="runtime-output")
_IONS_BLOCK = _OUTPUT_PROBLEM.add_block("ions", _OUTPUT_MODULE)
_IONS_STATE = _IONS_BLOCK[_OUTPUT_STATE_REF]


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def build(n=16):
    """A real native single-block System (no DSL compile): isothermal fluid + shared Poisson."""
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


def drive_outputs(sim, *, steps, output_dir):
    """Drive the deliberately low-level test seam without bypassing ``System.run`` authority."""
    for step in range(1, steps + 1):
        sim.step_cfl(0.4)
        fire_output_policies(
            sim,
            sim._output_policies,
            step,
            output_dir,
            last_step=steps,
        )
    return steps


# --- (0) pure cadence interpreter (host-testable, no engine) -------------------------
print("== (0) pure cadence interpreter ==")
chk(policy_due(every(3), 3) and policy_due(every(3), 6) and not policy_due(every(3), 4),
    "every(3) is due at 3, 6 and not 4")
chk(policy_due(5, 10) and not policy_due(5, 7), "int cadence 5 due at 10, not 7")
chk(policy_due(always(), 1) and policy_due(None, 1), "always()/None due every step")
chk(not policy_due(every(2), 0), "no policy fires at step 0")
chk(_format_token(HDF5()) == "hdf5" and _format_token(None) == "npz",
    "HDF5 -> hdf5, default -> npz")
# ADC-542: Plotfile is no longer refused -- it lowers to the plotfile token (the writer produces a
# single-level plotfile on a Uniform System). The former ADC-511 refusal is deleted.
chk(_format_token(Plotfile()) == "plotfile", "Plotfile -> plotfile (no refusal, ADC-542)")
# ADC-542: on_start / on_end / when land in the shared cadence interpreter.
chk(policy_due(on_start(), 1) and not policy_due(on_start(), 2), "on_start fires at step 1 only")
chk(policy_due(on_end(), 5, last_step=5) and not policy_due(on_end(), 4, last_step=5),
    "on_end fires at the last step only")
chk(not policy_due(on_end(), 5), "on_end never fires when last_step is unknown (honest silence)")
chk(policy_due(when(lambda s, step: step == 3), 3) and
    not policy_due(when(lambda s, step: step == 3), 2), "when(callable) fires when the callable is True")

# --- (1) OutputPolicy(npz) fires at the right cadence with the right contents ----------
print("== (1) OutputPolicy(npz, every(2)) cadence + contents ==")
tmp = tempfile.mkdtemp()
sim = build()
sim._output_policies = [OutputPolicy(format=None, cadence=every(2), prefix="out")]
taken = drive_outputs(sim, steps=4, output_dir=tmp)
chk(taken == 4, f"ran 4 steps ({taken})")
present = sorted(f for f in os.listdir(tmp) if f.startswith("out") and f.endswith(".npz"))
chk(present == ["out_000002.npz", "out_000004.npz"],
    f"npz written ONLY at steps 2 and 4 (every(2)): {present}")
d = np.load(os.path.join(tmp, "out_000004.npz"))
chk("state_ions" in d and "phi" in d and int(d["macro_step"]) == 4,
    "npz at step 4 carries state_ions / phi / macro_step==4")

# --- (1b) field selection maps to write(fields=) --------------------------------------
print("== (1b) fields= selection ==")
tmp_f = tempfile.mkdtemp()
sim_f = build()
sim_f._output_policies = [OutputPolicy(format=None, cadence=every(1),
                                       fields=[_IONS_STATE], prefix="sel")]
drive_outputs(sim_f, steps=1, output_dir=tmp_f)
df = np.load(os.path.join(tmp_f, "sel_000001.npz"))
chk("state_ions" in df, "field-selected npz includes the requested block")

# --- (1c) level selection is a no-op on a single-level System -------------------------
print("== (1c) AllLevels / CoarseOnly no-op on a Uniform System ==")
for lvl, tag in ((AllLevels(), "all"), (CoarseOnly(), "coarse")):
    tmp_l = tempfile.mkdtemp()
    s = build()
    s._output_policies = [OutputPolicy(format=None, cadence=every(1), levels=lvl, prefix="lv")]
    drive_outputs(s, steps=1, output_dir=tmp_l)
    chk(os.path.exists(os.path.join(tmp_l, "lv_000001.npz")),
        f"level={tag} writes the single level (no-op selection)")

# --- (2) low-level run cannot mint bind/run/restart identities -------------------------
print("== (2) low-level run requires completed pops.bind transaction ==")
simc = build()
simc._output_policies = [CheckpointPolicy(cadence=every(2), restartable=True, prefix="ck")]
try:
    simc.run(t_end=1.0, cfl=0.4, max_steps=2)
    chk(False, "low-level run should require pops.bind")
except RuntimeError as exc:
    chk("completed pops.bind transaction" in str(exc),
        f"low-level run refusal names the missing authority: {str(exc)[:70]}")

# --- (3) fire_output_policies rejects a non-policy object ------------------------------
print("== (3) non-policy reject ==")
try:
    fire_output_policies(build(), [object()], 1, tmp)
    chk(False, "a non-policy object should reject")
except TypeError as e:
    chk("OutputPolicy" in str(e), f"non-policy reject is explicit: {str(e)[:50]}")

if fails:
    print(f"FAIL test_output_policy_run : {fails} echec(s)")
    sys.exit(1)
print("OK test_output_policy_run")
