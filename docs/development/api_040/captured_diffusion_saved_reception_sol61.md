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

The explicit reception qualification is `saved-states-original-residual@2`.
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
Uniform state schemas keep physical component names/order. Version 2 requires
`fixture_schema="pops.captured-diffusion-native-fixture@2"` in the exact receipt
schema. The authoring `store_history(depth=1)` declares maximum lag, hence TWO
physical slots, not one. Each checkpoint must persist both slots [0,1], both
outgoing durations .01 and both POPSHID1 samples. Fill count is 1 after step1
(accepted/reloaded) and 2 after step2 (continuous/replay); initialized is true.

The actual store/rotation order is authoritative: `HistoryManager::prepare_sample_store`
initializes both samples on the first store and subsequently overwrites sample0;
`ProgramContext::store_history_` similarly initializes both fields, then overwrites
field0. Generated control commits, then rotates fields/durations/samples once.
Thus accepted physical slot1 contains the latest q; after step2 slot0 contains q
from the first accepted phase. After step1 both slots contain the same initial
publication. `System::history_global(name,slot)` gathers the physical slot with
no remapping. This is an accepted-boundary rule, not a general semantic claim
about every in-flight history read.

Both slot fields are checked in BYTES against their corresponding saved phase:
slot0 against first accepted q, slot1 against the current saved q. Both publication
starts after step1 are canonical +0; after step2 they are (+0,.01) in physical
slot order. Interval is .01 and kind Publication is 2 for each slot. Publication
ordinal is 1 in EACH new window, not macrostep ordinal 2: the one declared store
has no previous sample with the same start/interval. POPSHID1 name, Uniform
level=-1, physical depth=2 and exact length are required. Exact binary64 images
are compared, with no epsilon, clock relabel or acceptance of observed metadata.
Temporal boundary clock, last accepted dt and each persisted accepted cursor
are checked.

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

`assemble` writes only a pending `sol61.captured-d-owner-pins@2` template. It is
idempotent for identical bytes, refuses an existing different template, and
NEVER writes approval. ROOT separately supplies an approval file and two
SHA256 hashes out of band. The approval exact object is:

```json
{"schema":"sol61.captured-d-root-approval@2","approved_by":"ROOT",
 "pins_sha256":"EXTERNAL_64_LOWERCASE_HEX",
 "qualification":"saved-states-original-residual@2"}
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

ROOT supplies `sol61.captured-d-execution-owner@2` with exact keys:

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


## Version 2 correction and historical evidence

The parent e5457b235a0eb82253660916a1c9cdd011be79dc supplied 62 synthetic
source/protocol checks only, never a native qualification. It incorrectly
interpreted max lag 1 as physical depth 1 and checked only one history slot.
ROOT's genuine two native runs solved and ran successfully, then failed the
fixture's incorrect depth assertion. Those failures remain evidence; version 2
neither upgrades their receipts nor substitutes synthetic arrays for saved states.
Owner metadata, pending pins and external approval schema suffixes are now @2,
as is the qualification. Old @1 seals/receipts are refused, not normalized.

The correction changes only offline protocol/history validation and its tests.
The original D/R, forcing/material recipes, arithmetic face policy, source
consumer, Newton controls and physical tolerances are unchanged. New synthetic
protocol cases use distinct q1/q2 to expose wrong physical order despite the
stationary MMS, plus fully resealed metadata/value/window/ordinal/fill/slot
attacks. Synthetic baseline checkpoints are test inputs, not native evidence.

Read-only authority reviewed from ROOT's bfae73f3 source:
`python/pops/time/_program/history.py:208–211`,
`include/pops/runtime/program/program_runtime_state.hpp:183–207,238–264,299–339`,
`include/pops/runtime/program/program_context.hpp:2184–2250`,
`python/pops/codegen/program_emit_control.py:506–525`,
`src/runtime/system/system_io.cpp:195–212`, and
`python/pops/runtime/_system_io_history.py:378–402`.
Native version-2 reception remains pending genuine files and both external ROOT
seals; neither native execution nor a physical positive archive is produced here.

Version-2 local receipt: 80 synthetic source/protocol tests PASS (18.11s),
Ruff PASS, `git diff --check` PASS and standalone CLI `--help` PASS.
No installed PoPS import, native build/run, JIT or environment mutation.

## Explicit Frozen version 3 archive reception (2026-10-01)

This is a separate `saved-states-original-residual@3` qualification for fixture
`pops.captured-diffusion-native-fixture@3` from ROOT gel `766f078`
(Native source `457e0700`). The default assemble path and historical @2
qualification, receipts, gap list and 80 existing protocol/math tests remain.
No @2 receipt is promoted to @3. Candidate fixture @2 is outside this Frozen
reader's scientific qualification.

The fixture's accepted/continuous/replay checkpoint paths are distinct from one
another and from all saved observation paths. ROOT's source hashes each actual
checkpoint immediately after its capture, then rechecks those hashes when
publishing the receipt. This prevents subsequent observation NPZ writes from
silently replacing the checkpoint. The same compiled Program component exports
both its retained CPP and `dump_ir` image; it does not rebuild a second Program.
The @3 reader requires the CLOSED eleven-file inventory, including that one
actual IR, and refuses path aliasing or extra files.

The reader recomputes the carried IR10 Program hash using the actual serialization
projection: only node provenance is excluded, recursively for the declared node
regions and dt-bound nodes. Semantic attributes named provenance are retained.
It requires the same hash in the one actual CPP `pops_program_hash` export, the
IR receipt, and all three actual checkpoints. Program block order comes from
carried IR block handles and the CPP block-name registry; the observed order is
forcing/material/response. It is not rewritten to the old @2 literal order.
This is a Program IR/CPP/checkpoint link, **not** a reconstructed artifact
aggregate identity or proof of CPP-to-DSO compilation/linking.

Every @3 checkpoint must have actual Uniform payload version 8, the sealed
state/history/exchange/auxiliary images, and POPSDIA1 Program diagnostics. The
codec validates exact uint8 storage, signed-int64 rank offsets, width64, rank
ordinal/size, name lengths/order/uniqueness and no trailing bytes. Opaque names
and binary64 bits are retained independently for each rank; no artificial
rank-value equality is imposed. The five field diagnostics must be the names
emitted by the retained CPP, with finite nonnegative norms and integer counters.
Zero finite-difference JVPs is legitimate when the current solve already meets
the original residual guard. The optional native frontier-duration record is
accepted separately. Continuous/replay diagnostic images, like all continuation
payloads except the two run-origin envelope members, must agree in bytes.

The carried IR also authenticates the one macro clock and maximum lag1 history
registry. Logical history cursors contain clock/newest_tick/oldest_tick/valid_lags,
initialized/cold_start_extended; they do not contain a fabricated phase field.
The two physical ring slots remain distinct from logical lag1. The reader
checks exact schedule, controller, accepted clock, history lags, transaction
counts and absence of queued events for these two-step FixedDt witnesses.
Fully resealed cursor mutations still fail those semantic checks.

### Required ROOT pins and authentic full JUnit

`assemble --fixture-version 3` creates pending metadata only. The owner object
remains `sol61.captured-d-execution-owner@2` with the exact source/package/SDK/
native/binary fields documented above; `cpp_dso_links` remains null. ROOT must
independently authenticate its source commits, executed fixture/helper bytes,
loaded package/extension/SDK and execution association. No agent-created approval
is accepted as an owner seal.

New pending schema: `sol61.captured-d-owner-pins@3`. Exact keys:

```
schema, qualification, archive_root, file_roots, mode, ranks, owner,
junit, cases, other_junit_cases
```

`qualification` is `saved-states-original-residual@3`; `cases` has exactly
scalar1 and coupled3-201. Each case keeps directory/artifact/receipt/initial/
phases/checkpoints/sources and adds `program_irs` with the actual IR path/hash.
The receipt also records its actual compiled component and Program hash.
`other_junit_cases` is ROOT's explicit exact list of non-Frozen test names in the
FULL authentic per-rank XML, not a filtered or synthesized XML. Each rank must
contain exactly the two Frozen witnesses plus that complete list. Any global
failure/error/skip, count disagreement, duplicated/unlisted name or incomplete
rank-specific Frozen properties refuses the batch. Additional clean cases do
not become Frozen scientific qualification. Serial and MPI2 owner inventories
may have different explicit lists.

ROOT separately supplies this exact approval and both file hashes out of band:

```json
{"schema":"sol61.captured-d-root-approval@3","approved_by":"ROOT",
 "pins_sha256":"EXTERNAL_64_LOWERCASE_HEX",
 "qualification":"saved-states-original-residual@3"}
```

Use the existing assemble command with `--fixture-version 3` and one repeated
`--other-junit-case 'EXACT_NAME'` argument per additional case. Receive takes the
same four external pins/approval path/hash arguments. No seal is minted here.

### Independent checks and exact limits

131 source/protocol/math tests passed (80 preserved @2 tests plus 51 @3 tests),
Ruff passed, and the standalone checker imported no PoPS. New synthetic attacks
cover actual IR/body/input/point/version/CPP/receipt hash linkage, semantic
provenance, block order, rank-owned diagnostic offsets/rank/body/name/bits,
fully resealed lag/ring/clock/controller/event/counter metadata, closed11-file
inventory, and full mixed-JUnit failures/duplicates/omissions. All temporary
inputs are labelled synthetic; no positive native archive or owner approval is
manufactured.

Read-only **unsealed format/math admission**, not native qualification, was
performed against ROOT's two actual Serial archives under:

```
/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/
 installed-sdk2e4-diffusion-archive-and-uniform-diagnostics-serial-dim2/
 pytest-tmp/test_public_captured_diffusion{0,1}/captured-D-MMS
```

Both closed eleven-file inventories, actual IR10/CPP/checkpoint hashes,
Uniform8/POPSDIA1 five diagnostics, physical histories/logical cursors and exact
continuous/replay payloads passed. Maximum original relative L2 residuals were
2.9075324825715963e-15 (scalar1) and 1.883688360197054e-15 (coupled3-201).
The authentic full Serial XML admitted seven clean names: two Frozen, two
Candidate, two Uniform diagnostic tests and one geometry source admission. This
was a parser admission using the supplied files, not reception via `receive`;
ROOT's external execution-owner pins and approval were not yet consumed.

The @3 result explicitly retains gaps: original artifact aggregate payload,
block compiler CPP, independently reloaded checkpoint, separately saved
in-memory comparison images, private capture lease/point image, CPP-to-DSO
linking, and AMR/GPU/convergence/arbitrary-D/empty-rank qualification. No native
execution, install, JIT, environment mutation or production change occurred in
this review. Native Serial/MPI results belong to ROOT's authenticated campaigns.

### Version 3 shared-helper source contract follow-up

ROOT's first Serial assembly failed before any qualification because the old
source AST contract required the @2 inline expression
`DivCoeffGrad(q, float(Dij)*(1+material[0]))`. The actual shared helper at
`457e07000c35a466be3c80a7d7864f44a5f546c6` (SHA256
`0226d9889de555b20c4a4a97d3ce29ff055fb719c1d6829cc4935240e8ca6f34`)
assigns that exact frozen coefficient to a local variable, conditionally
multiplies it by the Candidate recipe, then calls DivCoeffGrad. The helper and
public runner default `candidate_diffusion=False`; the Frozen public entry
calls that runner without overriding the selector. The method's PerCandidate
policy is selected only in its true branch. The saved Frozen IR10 and original
F are unchanged. The earlier unsealed archive/math admission did not invoke
`origins`/assembly and therefore did not qualify this source-contract seam.
Both ROOT assembly failures remain historical harness evidence.

The follow-up gives @3 a strict, separate shared-helper AST contract: exact
entire coefficient branch (including condition/order/sign), exact false default,
no selector reassignment, exact conditional method policy, unchanged original
reaction/equations/captures `.n`/stage c0/exterior consumer, and the exact Frozen
entry/default/forwarding route. @2 still requires its original inline recipe;
neither source spelling is automatically accepted under the other scope.
Additional tests reject future-state captures, changed stage/duration/material,
coefficient sign/conditional/body, true or integer defaults, reassignment,
unconditional PerCandidate and forged public selection. No physical guard was
weakened. The source suite now has 151 passing tests, including the same 80
historical @2 tests; Ruff passes.

Read-only actual ROOT Serial execution-owner leaves and the two exact457e source
AST contracts now pass `origins(..., fixture_version=3)`. This is metadata/source
admission without consuming ROOT's external approval. MPI2 likewise has two
collectively published Frozen case archives under rank0-tmp, each storing both
ranks' diagnostics, plus two raw rank XML files; there are not four independent
Frozen physical cases. The two actual MPI2 archives' format/math/IR/history/
clock/replay admission passes, as do both full clean eight-case XML files. The
same residual maxima as Serial were observed. External MPI2 owner seals remain
required; no native execution or multiplied-by-rank scientific claim occurs.
