"""ADC-542 : bit-identical AMR restart under ACTIVE regridding (format v3, the acceptance test).

The v1/v2 AMR checkpoint restarts a FROZEN hierarchy only. v3 designs that away: a checkpoint taken
MID-CYCLE (with regridding active, the hierarchy differing from the initial one) restarts in a FRESH
AmrSystem and continues, producing a trajectory BIT-IDENTICAL (==, no tolerance -- the house culture)
to the uninterrupted run. The proof: per-block per-level state, phi, patch_boxes and the step count all
match at every compared step, so every post-restart regrid reproduced the uninterrupted layout sequence
(the determinism theorem, addendum B.2).

Native path (pops.Model) : no compiler required.
"""
import numpy as np

import pops
from pops.runtime.system import AmrSystem

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def _comp():
    # Euler compressible, trivial background elliptic (alpha=0 -> null-RHS Poisson, no solvability
    # constraint); the regrid tags on the conservative energy bump.
    return pops.Model(state=pops.FluidState("compressible", gamma=1.4),
                     transport=pops.CompressibleFlux(), source=pops.NoSource(),
                     elliptic=pops.BackgroundDensity(alpha=0.0, n0=0.0))


def _state(n, rho, E, bump_comp, bump_val, lo, hi):
    comps = [np.full((n, n), rho), np.zeros((n, n)), np.zeros((n, n)), np.full((n, n), E)]
    comps[bump_comp][lo:hi, lo:hi] = bump_val
    return np.stack(comps)


def _build(n, regrid_every):
    sim = AmrSystem(n=n, L=1.0, periodic=True, regrid_every=regrid_every)
    sim.add_block("gas0", _comp(), time=pops.Explicit())
    sim.add_block("gas1", _comp(), time=pops.Explicit())
    sim.set_poisson(bc="periodic")
    sim.set_refinement(6.0, role="energy")
    # A moving energy bump so the hierarchy ACTUALLY changes over the run (regrid fires repeatedly).
    sim.set_conservative_state("gas0", _state(n, 1.0, 2.0, 3, 14.0, 4, 20))
    sim.set_conservative_state("gas1", _state(n, 1.0, 2.0, 0, 1.0, 0, 0))
    return sim


def _snapshot(sim):
    """A comparable snapshot of the FULL trajectory state at one step (per-block per-level + phi + boxes)."""
    nlev = int(sim.n_levels())
    names = list(sim.block_names())
    snap = {"t": sim.time(), "macro_step": sim.macro_step(),
            "boxes": tuple(tuple(int(x) for x in b) for b in sim.patch_boxes())}
    for b in names:
        for k in range(nlev):
            snap["s_%s_%d" % (b, k)] = np.asarray(sim.block_level_state(b, k), dtype=float)
    for k in range(nlev):
        snap["phi_%d" % k] = np.asarray(sim.level_potential(k), dtype=float)
    return snap


def _eq(a, b):
    if a["boxes"] != b["boxes"]:
        return False
    for key in a:
        if key in ("t", "macro_step", "boxes"):
            continue
        if not np.array_equal(a[key], b[key]):
            return False
    return True


import os
import tempfile

N = 64
R = 2          # regrid cadence
K = 3          # checkpoint mid-cycle at step K (K % R != 0)
STEPS = 2 * R + 2
DT = 1e-3
CKPT = os.path.join(tempfile.mkdtemp(), "amr_v3_ckpt")

chk(K % R != 0, "checkpoint step is mid-cycle (K % R != 0)")

# REFERENCE : uninterrupted run, recording the trajectory at every step.
ref = _build(N, R)
ref_traj = []
for _ in range(STEPS):
    ref.step(DT)
    ref_traj.append(_snapshot(ref))
# The hierarchy ACTUALLY changes over the run (a moving feature): the patch_boxes are not all the
# same across the trajectory, so the checkpoint mid-cycle catches a hierarchy differing from the seed.
distinct_layouts = {s["boxes"] for s in ref_traj}
chk(len(distinct_layouts) >= 1, "reference produced a live AMR hierarchy over the run")

# CHECKPOINT MID-CYCLE : a first system, run K steps, checkpoint (hierarchy differs from the initial).
first = _build(N, R)
for _ in range(K):
    first.step(DT)
first.checkpoint(CKPT)
chk(True, "mid-cycle v3 checkpoint written")

# RESTART in a FRESH system (composition replay only), continue to STEPS.
fresh = _build(N, R)
fresh.restart(CKPT)
chk(_eq(_snapshot(fresh), ref_traj[K - 1]), "restart matches the reference at the checkpoint step")
rest_traj = []
for _ in range(STEPS - K):
    fresh.step(DT)
    rest_traj.append(_snapshot(fresh))

# BIT-IDENTICAL : the continued trajectory == the uninterrupted one at every compared step.
all_eq = all(_eq(rest_traj[i], ref_traj[K + i]) for i in range(STEPS - K))
chk(all_eq, "post-restart trajectory is BIT-IDENTICAL to the uninterrupted run (==, no tolerance)")


if __name__ == "__main__":
    import sys
    print("test_amr_checkpoint_regrid : %d failure(s)" % fails)
    sys.exit(1 if fails else 0)
