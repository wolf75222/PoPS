# Independent captured-D saved-state reception

This helper receives only the two installed-native witnesses declared by author
`b39f4a9906f904cd2f857c2874878da8aa84084f`:
`tests/python/integration/runtime/test_public_captured_diffusion.py` and
`tests/python/support/captured_diffusion_mms.py`. Neither file is imported or
executed by the receiver. No native output, JIT, native call, installation or
build was produced by this review. Native serial/MPI2 reception remains pending
ROOT's rebuilt package and actual archived outputs.

The receiver is `tests/review/sol61_captured_diffusion_saved_reception.py`.
It uses NumPy, Fraction and stdlib, plus the existing independent narrow
NPZ/CBOR/envelope readers in `sol61_m19_saved_reception.py` and
`sol61_integral_feedback_offline_oracle.py`. Those dependencies themselves use
`sol61_m19_product_oracle.py`; this captured-D receiver never invokes its physical
product oracle. No PoPS import or author forcing/original_action import occurs.

## Physical contract and limits

The explicit reception qualification is `saved-states-original-residual@1`.
The scalar witness has order `[0]`; the three-component witness `[2,0,1]`.
Both are Uniform Dim2, N16 on [0,1]², periodic in both physical axes. The receiver
checks the declaration's literal matrices, original cubic/reaction/diffusion
body, exact material/forcing capture, stage c=0, canonical observation order
and exterior source consumer with an AST contract. Source bytes and their
actual ROOT-owned source commit are additionally externally sealed.

D = 0.012 for the scalar. The coupled exact rational D and R are:

```
D = [[.012, .002, 0], [-.001, .014, 0], [.001, 0, 0]]
R = [[1.1, .03, -.01], [-.02, 1.3, .02], [.01, -.03, 1.5]]
```

D is singular, nonsymmetric and has signed off-diagonal entries. No inverse,
SPD assumption, eigenvalue clipping, condensation or normalization is performed.
The residual is the ORIGINAL `-div(D*(1+alpha)*grad(q))+R*q+.2*q³-forcing`.
Each periodic face is visited once, with opposite flux contributions to its
neighboring cells. The coefficient uses the explicitly selected arithmetic
mean. Physical x is NumPy's last spatial axis, y the preceding one. The unit
square's face-area/volume/gradient quotient is N². Tests compare the independent
face construction with a separate scalar per-cell stencil and a constant-D
Fourier symbol, and verify conservative telescoping.

The initial alpha and target are the declaration's CENTRE SAMPLES:
alpha=.25 sin(2πx)+.15 cos(2πy),
q_i=.15+.025 cos(2π(i+1)x)+.02 sin(2πy).
They are not antiderivative-derived continuum cell averages. Initial forcing
must match this discrete original operator within 2e-12; these fixed pre-native
roundoff guards are not native Newton stopping tolerances. Target/material
samples use 2e-14. Saved solutions/residual/consumer guards are the author's
unchanged 3e-8, controls match the seven original Newton controls, FD step 1e-6.
Forcing/material bytes must stay equal to the INITIAL captures in every phase.
The accumulated physical response must equal the saved solution times the actual
accepted duration (.01 after step1, .02 after step2). This is not a solve over a
constant q or a uniform replacement of captured D.

Checkpoint envelopes, all typed-array hashes and restart identities are
independently recomputed. Static spatial contracts assert dimensions, unit-square
bounds, shape, periodic axes and no refinement; their identities are checked.
Uniform state schemas keep physical component names/order. Native history data
must match the saved q, with depth/fill/initialized/slot0, outgoing duration .01,
and exact binary64 publication start/interval from POPSHID1. Publication ordinal
1 follows from the single declared store in each new window, as implemented by
next_history_sample (program_runtime_state.hpp). Temporal boundary
clock, last accepted dt and each persisted accepted cursor are checked.
Accepted/reloaded and continuous/replay saved arrays must match in BYTES.
Continuous/replay checkpoint payloads, including the temporal restart image,
are byte-identical, except the two checkpoint envelope members which can carry
legitimately different run identity. No independent reloaded checkpoint exists;
the accepted checkpoint is checked against the reloaded arrays.

No private PreparedFieldCapture owner/lease/point image is persisted by this
fixture. The source's exact capture point and the native history publication
point are checked separately; they are not silently relabeled as that missing
private image. Carrier/diagnostic/history comparisons originally performed in
memory are not separately saved, though persisted continuation arrays are
sealed and byte-compared. No ownership partition/empty-rank proof can be inferred
from gathered global arrays alone. There is no AMR, continuum convergence,
arbitrary-D solvability, full physical-model or CPP→DSO qualification.

## Two external seals and exact inventory

`assemble` writes only a pending `sol61.captured-d-owner-pins@1` template. It is
idempotent for identical bytes, refuses an existing different template, and
NEVER writes approval. ROOT separately supplies an approval file and two
SHA256 hashes out of band. The approval exact object is:

```json
{"schema":"sol61.captured-d-root-approval@1","approved_by":"ROOT",
 "pins_sha256":"EXTERNAL_64_LOWERCASE_HEX",
 "qualification":"saved-states-original-residual@1"}
```

The owner-pins exact keys are schema, qualification, archive_root, file_roots,
mode (`serial`/`mpi2`), ranks (1/2), owner, junit, cases. JUnit is rank-ordered,
unique, one file per rank and exactly two compiler cases per file. Exact names,
properties artifact_identity/dimension/rank/size/evidence_path and
captured_diffusion_receipt must agree. Failures/errors/skips, count disagreement,
foreign realms, duplicate tests/properties and mismatched ranks are refused.

Each case directory is CLOSED to exactly ten actual files: receipt.json,
initial.npz, accepted/continuous/reloaded/replay NPZ, accepted/continuous/replay
checkpoints, and the one retained Program CPP. Names of the checkpoint and CPP
files are read from the authentic receipt, not guessed. Files are unique and
must be direct children of that case. Receipt SHA links and external owner pins
must both match. Binary origins reside outside the phase directory and are
bounded by the separately approved canonical roots. Direct/parent symlinks and
`..` are refused before resolve, including declared alias origins. Reads use one
fd's bounded fstat extent and recheck inode/size/mtime/ctime. NPZ decompression,
member duplication and object-pickle payloads are independently bounded/refused.
The NPZ/JSON/source budget is 64 MiB; authentic extension/compiled-DSO hashes use
a separate 1 GiB bounded-read budget. These are offline reader budgets, not
production ABI or model restrictions.

ROOT supplies `sol61.captured-d-execution-owner@1` with exact keys:

```
schema
source_commit                  # actual observed author/fixture source HEAD
native_build_source_commit     # actual authenticated build SHA, or null gap
abi_key                        # actual selected SDK/native ABI string
python_package: {path, sha256}  # actual loaded Python package entry file
sdk: {path, sha256}             # actual SDK manifest file
native: {path, sha256}          # actual selected native extension
sources:
  fixture: {path, sha256}       # actual executed fixture/saved source bytes
  physical_helper: {path, sha256} # actual original declaration source bytes
cpp_dso_links: null             # no reviewed link receipt format in this helper
```

Every leaf uses an actual absolute canonical path and SHA256; file_roots must
include archive_root and the actual source/package/cache roots explicitly
approved by ROOT. Missing metadata is refused, not inferred from a neighboring
DSO or discovered by glob. `native_build_source_commit=null` is an explicit gap,
never an upcast from observed fixture source. The native origin must match the
receipt. Three block DSOs and the one Program DSO are hashed at their genuine
recorded paths. The retained Program CPP must carry the actual general-field
arithmetic template, original-residual and exact capture-layout guards. These
are byte/source-route checks, not proof of semantic equivalence or linking.
Original artifact/Program aggregate payloads are absent, so no cryptographic
aggregate reconstruction is claimed. Package/SDK/source and execution association
remain ROOT-owner-attested, even though each leaf's bytes are authenticated.

```
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python \
  tests/review/sol61_captured_diffusion_saved_reception.py assemble \
  --archive-root /ACTUAL/ARCHIVE --case /ACTUAL/SCALAR --case /ACTUAL/COUPLED \
  --junit /ACTUAL/RANK0.xml [--junit /ACTUAL/RANK1.xml] \
  --file-root /ACTUAL/ARCHIVE --file-root /ACTUAL/PACKAGE \
  --file-root /ACTUAL/SOURCE --file-root /ACTUAL/CACHE \
  --owner /ACTUAL/owner.json --output /ACTUAL/pending-pins.json

env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python \
  tests/review/sol61_captured_diffusion_saved_reception.py receive \
  --pins /ACTUAL/pending-pins.json --pins-sha256 EXTERNAL_ROOT_PINS_SHA \
  --approval /ACTUAL/root-approval.json --approval-sha256 EXTERNAL_ROOT_APPROVAL_SHA
```

The receiver returns computed saved-state metrics and an explicit gap list;
`cpp_dso_link_qualified` is always false. It performs no native execution.

## Synthetic protocol reception

`tests/review/test_sol61_captured_diffusion_saved_reception.py` creates only
labelled temporary SYNTHETIC arrays, protocol placeholders and fake identities.
No positive native NPZ, actual-native receipt or ROOT approval is manufactured.
The original initial data test uses scalar libm and an independent direct
per-cell stencil. Protocol tests exercise fully rehashed NPZ+receipt and
checkpoint typed-array/envelope payloads before scientific guards. The attacks
include harmonic/PSD/uniform operator substitution, doubled consumer duration,
missing/mutated captures, constant/permuted q, time/replay drift, stale publication
points, axes/bounds/periodicity, temporal cursors, JUnit, path aliases, unresealed
bytes and closed inventory changes. Frozen author source is read via git show as
data only; source sign/coefficient/capture-point/consumer mutations are refused.

Commands from this private worktree (no installed package use):

```
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest \
  tests/review/test_sol61_captured_diffusion_saved_reception.py -q
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m ruff check \
  tests/review/sol61_captured_diffusion_saved_reception.py \
  tests/review/test_sol61_captured_diffusion_saved_reception.py
```

Source/host reception: 62 tests passed; Ruff passed; CLI help and a fresh checker
import verified that no PoPS modules were loaded. These results are not native
positives.

Native qualification is pending genuine ROOT-selected files, owner metadata,
clean JUnit and the two external seals. This source/host reception is not CI or
native MPI/GPU validation.
