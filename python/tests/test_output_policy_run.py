#!/usr/bin/env python3
"""C4 / ADC-509: typed OutputPolicy / CheckpointPolicy fire on the Uniform System run loop.

The typed output / checkpoint policies (pops.output) wire to the EXISTING write()/checkpoint
writers through a run-loop hook (pops.runtime._output_driver). This is a LOCAL end-to-end proof
on the Uniform / single-level System path: it builds a REAL native System (native bricks, no DSL
compile, no Kokkos .so -- the no-fake-engine rule), attaches output policies on its
``_output_policies`` (exactly what pops.bind flows from a Case), runs a few steps, and asserts:

  (1) an OutputPolicy(format=npz, cadence=every(N)) writes a file at every Nth step and NOT
      in between, with the right step suffix and the expected fields inside;
  (2) a CheckpointPolicy(cadence=every(M)) writes a restartable checkpoint, and restarting it in
      a replayed composition round-trips the state bit-identically;
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
    from pops.time.schedule import every, always
    from pops.runtime._output_driver import policy_due, _format_token, fire_output_policies
except Exception as exc:  # noqa: BLE001
    print("skip test_output_policy_run (pops unavailable: %s)" % exc)
    sys.exit(0)

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def build(n=16):
    """A real native single-block System (no DSL compile): isothermal fluid + shared Poisson."""
    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
    sim._add_block("ions",
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


# --- (0) pure cadence interpreter (host-testable, no engine) -------------------------
print("== (0) pure cadence interpreter ==")
chk(policy_due(every(3), 3) and policy_due(every(3), 6) and not policy_due(every(3), 4),
    "every(3) is due at 3, 6 and not 4")
chk(policy_due(5, 10) and not policy_due(5, 7), "int cadence 5 due at 10, not 7")
chk(policy_due(always(), 1) and policy_due(None, 1), "always()/None due every step")
chk(not policy_due(every(2), 0), "no policy fires at step 0")
chk(_format_token(HDF5()) == "hdf5" and _format_token(None) == "npz",
    "HDF5 -> hdf5, default -> npz")
try:
    _format_token(Plotfile())
    chk(False, "Plotfile should reject")
except NotImplementedError as e:
    chk("ADC-511" in str(e), f"Plotfile precise reject names ADC-511: {str(e)[:50]}")

# --- (1) OutputPolicy(npz) fires at the right cadence with the right contents ----------
print("== (1) OutputPolicy(npz, every(2)) cadence + contents ==")
tmp = tempfile.mkdtemp()
sim = build()
sim._output_policies = [OutputPolicy(format=None, cadence=every(2), prefix="out")]
taken = sim.run(t_end=1.0, cfl=0.4, max_steps=4, output_dir=tmp)
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
                                       fields=["ions"], prefix="sel")]
sim_f.run(t_end=1.0, cfl=0.4, max_steps=1, output_dir=tmp_f)
df = np.load(os.path.join(tmp_f, "sel_000001.npz"))
chk("state_ions" in df, "field-selected npz includes the requested block")

# --- (1c) level selection is a no-op on a single-level System -------------------------
print("== (1c) AllLevels / CoarseOnly no-op on a Uniform System ==")
for lvl, tag in ((AllLevels(), "all"), (CoarseOnly(), "coarse")):
    tmp_l = tempfile.mkdtemp()
    s = build()
    s._output_policies = [OutputPolicy(format=None, cadence=every(1), levels=lvl, prefix="lv")]
    s.run(t_end=1.0, cfl=0.4, max_steps=1, output_dir=tmp_l)
    chk(os.path.exists(os.path.join(tmp_l, "lv_000001.npz")),
        f"level={tag} writes the single level (no-op selection)")

# --- (2) CheckpointPolicy fires and round-trips on restart ----------------------------
print("== (2) CheckpointPolicy(every(2)) + restart round-trip ==")
tmp_c = tempfile.mkdtemp()
simc = build()
simc._output_policies = [CheckpointPolicy(cadence=every(2), restartable=True, prefix="ck")]
simc.run(t_end=1.0, cfl=0.4, max_steps=2, output_dir=tmp_c)
ckpts = sorted(f for f in os.listdir(tmp_c) if f.startswith("ck") and f.endswith(".npz"))
chk(ckpts == ["ck_000002.npz"], f"checkpoint written at step 2 (every(2)): {ckpts}")
ref = np.asarray(simc.get_state("ions"))
restored = build()  # composition REPLAYED (v1 contract)
restored.restart(os.path.join(tmp_c, "ck_000002"))
chk(restored.macro_step() == 2, f"restart restored macro_step==2 ({restored.macro_step()})")
chk(np.array_equal(np.asarray(restored.get_state("ions")), ref),
    "restart round-trips the state BIT-IDENTICALLY")

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
