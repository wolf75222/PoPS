# Independent M19 free-streaming audit

Frozen source base 5af964e0d5ebccecdcea56117ff923698a54bfc4, separate
PoPS-sol61-m19-offline-audit worktree. No production/Native/ENV/cache writes,
JIT, build or simulation. This is transport only, not BGK/Vlasov/GPU reception.

The actual public model declares ddt(f)=-div((0,v*f)) in a Cartesian chart
axis0=v and axis1=x. Native layout cells=(nv,nx), exported arrays=(1,nx,nv).
Axis0 no-flux boundaries and axis1 periodicity match that chart. Signed analytic
Aux is coordinate(v); flux and explicit HLL wave pair use that Aux. The
coefficient is not a Python cell array. system_aux.cpp107–151 gathers the actual
Fab values with exact-domain ownership and collective sum; it is global also
in MPI. SSPRK2 is the authored two-stage formula, not a named integrator shortcut.
Binary dt=1/(4nx), nx/2 steps to1/8, hence Courant<1/4; cases32x8 and64x12.

The oracle independently uses the signed upwind eigenvalue and SSPRK2 stability
polynomial. Its continuum integral averages over both x and v: it retains the
sinc derivative velocity moment, unlike the discrete midpoint coefficient.
Physical mass and velocity moments have dx*dv=2/(nx*nv), not cell-count
normalization. Existing guards remain3e-12, continuum .65*t/nx and nontrivial
change .005; auxiliary tolerance1e-14. Negative sign/Euler/stationary/axes/dtype/
nonfinite and signedzero cache probes reject. No thresholds changed.

The public fixture saves initial/accepted/continuous/restored/replay, authenticates
accepted/continuous/replay checkpoints against their live issuer, authenticates
restored, and checks full physical/cache/history arrays including empty entries.
It checks exact clock and accepted/rejected steps, then recompares checkpoint
pins after all observations. Generated Program C++/IR and component binaries
are retained. Model C++ can be absent and is explicitly recorded as null; that
is a provenance gap, not permission to regenerate and call it compiled source.
Initial checkpoint does not carry the same run provenance as accepted phases.

New NumPy-only `sol61_m19_freestreaming_saved_audit.py` contract
sol61.m19-freestreaming-saved-audit@1 reads hash-pinned actual receipt/phase NPZ/
checkpoints, independently recalculates Fourier and continuum errors, checks
signed Aux, phase clocks and full restart/replay payload bytes. It intentionally
reports native_authority_or_root_approval_verified=false. File hashes supplied
by the caller establish integrity, not Native authenticity or ROOT approval.
It does not inspect compiled body semantics, reconstruct restart identity, or
replace ROOT package/SDK/DSO/all-rank/JUnit/seal audit. It is a useful second
scientific/payload gate, not a qualification reader. The separate product-
reduction M19 saved reader has a different contract and is not reused silently.

Usage after ROOT creates genuine evidence (stdout to a new private output):

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops/bin/python tests/review/sol61_m19_freestreaming_saved_audit.py ACTUAL_RECEIPT.json ACTUAL_SHA256
```

Source synthetic checks:11 PASS .18s, no PoPS import, no native evidence minted.
Actual private Source import asserted; _pops absent. Affected Source plus offline
cohort32 PASS8.31s, zero skips. Native reception
remains ROOT's future execution after its immutable Halo batch closes.
