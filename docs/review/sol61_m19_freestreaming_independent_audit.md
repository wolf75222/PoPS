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

## Bounded independent corrective review

Base author6c04ca5b. Independent reading of actual public transport fixture confirms
physical v nativeaxis0, x axis1, no-flux velocity and periodic position, signed
HLL explicit(v,v), FirstOrder and SSPRK2. Fourier eigenvalue is the upwind signed
velocity symbol, growth1+dtλ+(dtλ)^2/2. Continuum expression independently averages
f0(x-vt,v); centered velocity first moment contributes the positive sine term.
CellMidpoint realization versus original continuum averaging remains distinct.
No equation, Frame, Core, FD/guard, model or native source changed.

Real reader holes corrected: duplicate JSON keys/phase records were silently
collapsed by json.loads; boolean axes and float rational controls compared equal
to integers; CP{}==CP{} could claim payload equality with no physical state.
Strict JSON/type checks, five unique receipt/state/checkpoint paths, exact phase
state members and Aux-capture flag now reject those. Current UniformCP8 scalar
type/clock/version, actual state_kinetic bytes versus saved physical NPZ and
POPSAUX2 rank/type/minimum prefix are checked; manifest must cover payload keys.
This is a structural/projection check, not a POPSAUX2 complete semantic decoder or
a restart identity/hash reconstruction. Native authority/root approval remains
explicitlyFalse in output; no qualification or real captures are synthesized.
The existing native fixture already authenticates checkpoints at capture/restored
and proves separate checkpoint paths; ROOT must run it and audit actual evidence.

Independent coherent Source command: env-uPYTHONPATH PYTHONDONTWRITEBYTECODE1
ir17python -m pytest --noconftest -pno:cacheprovider -opythonpath=python
tests/review/test_sol61_m19_freestreaming_saved_audit.py
tests/review/test_sol61_m19_freestreaming_source.py:53PASS9.25s zeroSkip. Labelled
synthetic positive/negative CP projection cases are algebra/parser checks only.
Duplicate/missing phases, signedzero payload, negativevelocity/euler/stationary,
dtype/NaN/transpose, rationaltype, empty/truncated/duplicate NPZ and corrupt CP
physical state/clock/version/Aux/manifest coverage are refused. No Native/JIT/build.

Principles→decision→file→oracle→command→status:1.1 physical equations authority→
independent Fourier/continuum signs retained→saved_audit.science→wrong sign/time
order negatives→53Source→PASS;1.3 generic mechanisms→typed JSON/phase/archive
protocol guards→saved_audit metadata/checkpoint_projection→bool/duplicate/emptyCP
counterexamples→same command→PASS;1.6 real public lifecycle→Source validates
but does not attest Native authenticity→native fixture snapshots→ROOT future
actual capture/read→NOT RUN;1.8 new extension counterexample→strict projection
and phase coverage→new reader guards→labelled corruption cases→SourcePASS.
