#!/usr/bin/env python3
"""ADC-542: CheckpointPolicy honesty gate + AMR level-policy resolution (host-testable).

  (1) CheckpointPolicy declares capabilities()/requirements() like every other typed route;
      restartable is honestly amr_compatible (v3 restarts under active regridding).
  (2) validate() refuses ONLY the physically-impossible residue: require_bit_identical across a
      CHANGED rank count (IEEE754 reassociation). It never refuses restartable=True for being AMR.
  (3) the AMR output driver resolves the typed level policy (AllLevels / CoarseOnly / SelectedLevels)
      and refuses an out-of-range SelectedLevels verbatim (late-bound to the live n_levels).

Pure descriptor / driver logic -- no engine needed (a tiny fake exposes only n_levels for (3)).
Runs under pytest and the __main__ guard.
"""
import sys

try:
    from pops.output import CheckpointPolicy
    from pops.runtime._amr_output_driver import resolve_levels
    from pops.output.policies import AllLevels, CoarseOnly, SelectedLevels
except Exception as exc:  # noqa: BLE001
    print("skip test_checkpoint_honesty (pops unavailable: %s)" % exc)
    sys.exit(0)

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


# --- (1) capabilities / requirements present ------------------------------------------
print("== (1) CheckpointPolicy capabilities/requirements ==")
cp = CheckpointPolicy(restartable=True, require_bit_identical=True)
caps = cp.capabilities().to_dict()
chk(caps.get("restartable") is True and caps.get("amr_compatible") is True,
    "restartable checkpoint is honestly amr_compatible (v3 restarts under regridding)")
chk(caps.get("device_host_sync") is True and caps.get("cadence_slot") == "checkpoint",
    "declares device_host_sync + cadence_slot")
req = cp.requirements().to_dict()
chk(req.get("restartable_route") is True and req.get("bit_identical_route") is True,
    "requirements name the restartable / bit-identical route")

# --- (2) validate refuses ONLY cross-rank bit-identity --------------------------------
print("== (2) validate: cross-rank bit-identity refusal ==")
chk(CheckpointPolicy(restartable=True).validate({"amr": True, "regrid_every": 5}) is True,
    "restartable=True on an AMR regrid context is NOT refused (v3 restarts under regridding)")
chk(CheckpointPolicy(require_bit_identical=True).validate({"ranks": 4, "restart_ranks": 4}) is True,
    "require_bit_identical on the SAME rank count is fine")
try:
    CheckpointPolicy(require_bit_identical=True).validate({"ranks": 4, "restart_ranks": 2})
    chk(False, "cross-rank bit-identity should refuse")
except ValueError as e:
    chk("rank-count change" in str(e) and "IEEE754" in str(e),
        f"cross-rank bit-identity refused verbatim: {str(e)[:60]}")

# --- (3) AMR level-policy resolution + out-of-range refusal ---------------------------
print("== (3) AMR level-policy resolution ==")


class _FakeAmr:
    def __init__(self, n_levels):
        self._n = n_levels

    def n_levels(self):
        return self._n


sim = _FakeAmr(3)
chk(resolve_levels(sim, AllLevels()) == [0, 1, 2], "AllLevels -> every level")
chk(resolve_levels(sim, CoarseOnly()) == [0], "CoarseOnly -> [0]")
chk(resolve_levels(sim, SelectedLevels(0, 2)) == [0, 2], "SelectedLevels(0,2) -> the subset")
chk(resolve_levels(sim, None) == [0, 1, 2], "default (None) -> every level")
try:
    resolve_levels(sim, SelectedLevels(99))
    chk(False, "out-of-range SelectedLevels should refuse")
except ValueError as e:
    chk("out of range" in str(e) and "n_levels=3" in str(e),
        f"out-of-range level refused verbatim (names the live count): {str(e)[:60]}")

if fails:
    print(f"FAIL test_checkpoint_honesty : {fails} echec(s)")
    sys.exit(1)
print("OK test_checkpoint_honesty")
