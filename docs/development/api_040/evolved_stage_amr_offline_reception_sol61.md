# Independent offline reception protocol for composite AMR Stage

Prepared on 1 October 2026. This is a **source-only reader preparation**, not a
native reception. No actual AMR Stage dataset or ROOT seals have been received.
No PoPS import, runtime, JIT, build, installed environment or ROOT worktree was
used or modified. The three new files are the autonomous reader, its labelled
SOURCE_ONLY tests and this note.

## Original equations and scope

The four future cases are base grids 8×8 and 16×16, each with one or two evolved
fields on two genuine AMR levels, refinement ratio (2,2), a periodic unit square
and partial fine coverage. The two-field case has three actual unknowns because
it also solves the auxiliary constraint. The original equations read

\[
 Q_0(T)=T_0+T_0^2\quad\text{(one field)},
\]
\[
 Q_0(T)=T_0+T_0^2+0.1T_1^2,\qquad
 Q_1(T)=T_1+T_1^2+0.2T_0T_1,\qquad
 z-0.25T_0-0.5T_1=0.
\]

The declared spatial rates contain the signed, nonsymmetric matrix
`[[.012,.002],[-.001,.014]]`. The actual closed witness is homogeneous in
temperature, so its physical gradients vanish. The reader receives
`Q(T_new)-Q_previous-.01*f=0`, the original nonlinear projection and the
auxiliary constraint, **not a nonconstant diffusion discretization**. Transposing
D cannot be detected from zero gradients; an explicit test records that
identifiability limit rather than inventing a passing matrix test.

Initial temperatures are .15 and .25; the first T0 target is .16. In the
two-field case conservation of total Q fixes the positive T1 target. The reader
spells this scalar quadratic calculation and Q directly, without importing
`closed_data`, `accumulation` or `check_saved`. The prescribed load comes from
that declared target and the fixed .01 duration. Its two components are equal
and opposite, so individual composite Q amounts change while their sum remains
constant. Subsequent states are checked against their actual accepted previous
Q, not against a regenerated saved solution. Positive temperatures select the
declared branch; no universal norm or EOS domain is asserted.

Finest masks are reconstructed from the checkpoint's authenticated global
patch boxes. Coarse cells covered by aligned fine boxes are removed. Each valid
global box is counted once; duplicate/overlapping boxes, missing base coverage,
invalid owner maps and owners outside the actual rank count refuse. Volumes are
exact Fractions `1/(N*2**level)**2`; their finest sum must equal one. Q sums use
Fractions of the observed binary64 values. NPZ active masks must equal the
topology-derived masks. Global gathered arrays are not multiplied by rank count.

This does not qualify nonlinear restriction, nonconstant AMR fluxes, AMR
convergence, GPU execution or complete M06/M13 models. In particular it cannot
prove that an opaque native owner performed an operation merely because a
global NPZ contains a plausible value.

## Historical slot defect and corrected archive

The independent review found the historical `4c7412a8` fixture labelled raw
`history_global(name,level,0)` as latest. Actual `AmrSystem::history_global`
reads `ring.at(slot)`. Store writes slot zero and the emitted publication rotates
the two slots at the end of the step: latest is physical slot one, previous is
slot zero. After the first publication they coincide, hiding the inversion;
after the second they differ. The old fixture's Q projection and previous-slot
comparison would therefore inspect the wrong temperature.

Author fixture-only correction
`d3f79a5935cdf6f8654d8daa5d758f5fc6e97f50` has been read independently. It keeps
`build`, `closed_data`, `check_saved`, F and all seven Newton controls unchanged;
it changes capture labels, archives real POPSHID1 bytes, and emits
`pops.evolved-stage-amr-native-fixture@2` with explicit latest=1/previous=0 policy.
The reader requires @2 and never upcasts @1. Each level/ring's NPZ uint8 sample
must equal its durable sample and metadata hex. POPSHID1 authenticates name,
level, depth, kind, exact binary64 start and duration, and ordinal. For two
accepted steps the raw slots have starts `(0,.01)`; both intervals are .01,
kind=Publication and ordinal=1. Slot ordering is thus checked by earned sample
identity, not just by a human label.

## Required ROOT inventory and two seals

Run `contract` to display the exact protocol skeleton. ROOT supplies one owner
pins JSON and one approval JSON, and their SHA256 values externally. The reader
does not produce approvals or native owner pins. Approval must name ROOT, the
qualification `homogeneous-original-composite-Q-durable@1` and the exact pins
digest. Owner pins use `sol61.evolved-stage-amr.owner-pins@1`, specify Serial or
MPI2, and include:

- Exact runtime source and native-build source Git commits, ABI, native DSO,
  SDK archive and authenticated installed-source manifest file pins.
- Exact source file pins for the AMR helper, declared Q constants, Newton
  controls and native fixture. Source admission uses AST without importing any
  helper. The complete reviewed `build` AST is pinned independently of file
  residence/line numbers; its digest is
  `48ceca9d578ac14ca43571a8e8568ea3218d8685ff227ed3d405fe73a445f411`.
  Any change requires explicit source review, not reminting from altered input.
- Exactly four collective archives: cells8/16 × width1/2, not four per MPI rank.
  Each has a receipt pin, externally authenticated artifact/bind/semantic tokens
  and a closed path→SHA inventory covering every receipt reference.
- Ten level NPZs per archive: initial, accepted, continuous, reloaded and replay,
  with both levels. Three **distinct** checkpoints: accepted, continuous, replay.
  Checkpoint paths must not overwrite one another. Each reference's immediate
  fixture hash must equal the final externally pinned bytes.
- Every actual block/Program DSO and compilation sidecar referenced by the
  fixture. The same compiled Program component's dumped CPP and dumped IR are
  mandatory, with its actual `program_hash` and the source-selected IR16 or IR17.
  The reader checks IR→hash→CPP exported hash, full-product dimension, seven
  numerical controls, issued duration, FullResidualBasisLU resource choice and
  qualified history port. It does not regenerate IR or an artifact aggregate.
- One complete, raw JUnit XML per actual rank, with a ROOT-pinned exact full
  batch name list. No XML filtering is accepted. Failures/errors/skips anywhere
  in that batch refuse; all four parameter cases must appear once on every rank,
  with exact artifact, dimension, rank, size and collective receipt path.

The reader hashes all referenced native/package/source evidence. **ROOT is
responsible for the Git/installed-manifest and source→native build provenance
attestation**: hashing a package manifest does not independently reconstruct
each installed source or a CPP→DSO compilation graph. The result explicitly
sets `cpp_to_dso_crypto_link=false`. It never attributes a local synthetic XML,
fake DSO or synthetic owner seal to ROOT.

## Durable checks and remaining gaps

AMR payload v11 is checked against the run-origin checkpoint envelope v1 and
the exact typed-array hashes/CBOR restart identity. The spatial layout carries
the unit-square bounds, periodicity, base shape and refinement ratio. State Q
and forcing arrays match their observations byte-for-byte. Two-level histories
match physical raw slots, scalar width, initialization, fill, duration and
publication sample bytes. FixedDt and accepted temporal cursors retain the
exact accepted clock. The reader checks the live creator's checkpoint tokens,
run request digest, continuation start, one-step controls, parent run and
restart lineage. The schema7 native report additionally carries both level
macro steps at rational phase 0/1, their decimal display times and the primary
logical tick; those reports and zero accepted transaction depths are checked.
The display times do not replace the temporal envelope's exact binary64 bits.

POPSDIA1 is decoded independently as rank-owned images with exact offsets,
Real width, sorted opaque byte keys and untouched float bits. No artificial
cross-rank equality is imposed. Each rank's original-F residual must be finite
and meet the unchanged 1e-10 bound; rank-zero observed diagnostics match its
durable bits. Empty POPSEX01 and POPSEX02 ledgers remain distinct. This Stage
witness declares no integral or moving interval: a nonempty or other-version
ledger refuses rather than being silently ignored.

Continuous and restarted replay must have identical NPZ images and every
checkpoint array except their run/restart identity envelopes. Reopened accepted
observations must equal accepted observations, and the second previous history
must equal the first accepted temperature. AMR Program accepted state, its
source authority, per-level auxiliary images, native ledger/clock reports and
flux snapshots are externally authenticated and compared in this exact replay.
Their full opaque native codec bodies and private leases/attempt authorities
are **not decoded or inferred** by this reader. A genuine ROOT dataset is still
required to receive them as actual captured bytes. Those gaps remain explicit
even if the physical and durable checks pass.

## Source-only validation

**70 SOURCE_ONLY tests pass** (0.40 s), with no PoPS import. They cover original scalar
and coupled balances at both declared resolutions, equilibrated wrong linear
Q, stale temperatures/captures, initial shifts, constraint and replay faults,
ownership/coverage defects, physical history slot samples, diagnostic opaque
bits/rank faults, malformed offsets, raw versus resealed checkpoint mutations,
and rehashed IR control/resource changes, numeric-type aliases and stale
rational/logical clocks or provisional ledger reports. Synthetic checkpoint builders test
only the typed-array protocol; their deliberately arbitrary opaque native bytes
are not native codec reception. The full `receive()` path has intentionally
not been given fabricated ROOT seals or synthetic native evidence.

AST admission was also executed against the exact `d3f79a59` helper retrieved
read-only from its Git object: the reviewed build, original constants and seven
controls pass, with no helper import. Ruff and whitespace checks pass. The
author's source tests and the previous 44 history-authority checks remain
distinct from these 70 offline protocol/math tests.

```
rtk proxy env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest tests/review/test_sol61_evolved_stage_amr_saved_reception.py -q -p no:cacheprovider
rtk proxy env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_evolved_stage_amr_saved_reception.py contract
```

Future actual reception uses `receive --pins ... --pins-sha256 ... --approval ...
--approval-sha256 ...`. No scientific/native positive result has been produced
in this preparation.
