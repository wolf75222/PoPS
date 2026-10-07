# Prepared AMR checkpoint capture @1

The real original IMEX run on Source `9d4cdcca19abd7d8acfc2151497d8f2a967cddcc`
and Native `64c8922e58cf484a1b1188b899e25e84569707ae2090ecbd26f515daf2e1e9c6`
compiled and bound, then failed during checkpoint-consumer staging. The external step
envelope begins, advances, stages consumers, commits, publishes consumers, then finalizes.
The ordinary state-carrier checkpoint correctly refused the still-active transaction.
The original negative is closed at
`/Users/romaindespoulain/dev/tmp/sol61-imex-final-native-20261007`, pins
`05e9e2fc1e97a2b29985e182d313c3cb96bb1e66bef899d3e55816e6c604a893`.
It contains no accepted State/Field/restart qualification.

The consumer now prepares an opaque native handle with contract
`pops.amr.prepared-checkpoint-capture@1`. Preparation requires one live outer transaction,
an uncommitted external rollback image, a completed advance, and no restart, bootstrap,
active native advance or pending resource refresh. The rollback image identifies the
transaction; the numerical bytes come from the **current completed candidate**, never
from that older image.

Completion is a private marker produced only when `step()` returns successfully or an
AMR Program region returns an empty final port after completion. Clock setters cannot
produce it. Rollback metadata restores an earlier genuine completion point for the same
session, while mutation revisions keep old image handles stale. A fresh capture of the
restored point is permitted, including local diagnostic or history updates at that point.
Entering a new Program region clears completion authority. A nonempty Map port cannot mint
a capture while `cadence_dispatch_active_` remains true; the existing continuation guard
applies to preparation and publication validation. An empty completed port establishes the
new private marker. Nested rollback restores only a previously genuine marker for its session.

A fresh shared lease control block identifies each outer session. The handle holds a weak
lease, owner identity, generation, local mutation revision, physical time, macro-step and
topology epoch. This rejects foreign owners, expired sessions, pointer-address reuse and
old retry handles. Nested transitions and the runtime mutation entry points invalidate old
captures before modifying data. Revision validation is local and its failure is voted
collectively. Generation, phase, clock and topology agree exactly across ranks; local
diagnostic tables may legitimately produce different revision counters.

Preparation freezes the native all-rank POPSCAR1 image, including all blocks, levels,
patches and ghost bits, plus each rank's native diagnostic image. The existing checkpoint
codec captures the scientific projections, Field values, histories, Program state, clocks,
exchange ledger and ownership metadata at the same live point. It validates the native
authority before and after that synchronous capture and seals the complete immutable NPZ
proof using the existing descriptor-owned staging transaction. The codec's capture path
uses getters and serialization; it invokes no physical model callback or additional step.
Any supported mutation between these reads invalidates the handle and rejects the capture.

Publication validates the same committed outer lease and atomically publishes the already
sealed proof. It does not gather State, Field or history again. Rollback retains the existing
cleanup and publication-quarantine behavior. Finalization, rollback and owner destruction
expire the lease. Commit alone does **not** authorize the ordinary public checkpoint:
`checkpoint_state_carriers()` and `checkpoint_program_diagnostics()` retain their strict
depth-zero guards. Initial/end consumers outside a transaction use that ordinary route.
The AMR candidate seam has a distinct Python name, preserving Uniform's existing
`_prepare_checkpoint_capture(path)` interface.

ABI12 names the new public opaque C++ capture interface. The repository policy in
`include/pops/runtime/module_capabilities.hpp` requires a native ABI bump when public native
layout or capability meaning changes. The release source is regenerated with
`scripts/generate_release_contract.py`; the textual header signature also changes and
requires rebuilt artifacts. POPSCAR1, POPSDIA1 and CP12 archive formats are unchanged.
The IMEX tableau, State/Field equations, consumer policy, numerical thresholds and accepted
rollback rules are unchanged.

Validation required on the exact integrated build: the original manual IMEX State/Field
restart case, manual/preset parity and rejected-attempt rollback; the original C++ implicit
transaction case with prepared/accepted/lifetime assertions; MPI2 with unequal local
diagnostic mutation counts before fresh capture and a one-rank stale handle afterward;
and Uniform checkpoint-consumer non-regression. Source parsing/review does not establish
these Native results. The negative9d4, W11 and running CUDA artifacts remain immutable and
cannot qualify this new interface.

## Native reception at Source23810

The official incremental build/install passes on Source
`23810e143d24b901d1ba1410a0e8b1c64a6f7770`, without repeating setup. All seven binding
translation units compile. Native SHA is
`77d5e29cd8146cdb17cba08c88d03fc63e9b14ea32673ff99f3baff8db800a4c`,
header signature `c04a7fa8d0852fb3dd8542ed7907485264350e1764227d077a993fe5e8712c77`,
ABI12 and release contract `d71b9667557abba7bb87e8e0a5f6a4d5e32dc91bd4bb9b9424b6dc37ec9ce823`.
The first original manual IMEX run captures and seals the candidate checkpoint successfully,
then fails during its publication: `AMR prepared checkpoint capture owner/session/point is not live`.
One XML test has one error, no failure/skip; no accepted scientific step or restart is received.

The preceding HDF5 effect carries an Integral diagnostic. Its publication calls
`_publish_diagnostics()`, which writes the Native inspection registry and correctly invalidates
the earlier checkpoint token. This identifies a missing preparation rule for the joint effect
transaction: the candidate Native diagnostic projection must be prepared before immutable
checkpoint capture; its Python inspection registry and external effects still publish after
commit. The stale-token guard must remain strict. Correction and a new original execution
are required; this negative is not a successful checkpoint reception.

ROOT authenticates all2 845 declared entries (2 602 files,242 directories,one link) in
`/Users/romaindespoulain/dev/tmp/root-imex-publication-negative-reception-23810-20261007.json`,
SHA `ee91711a02f96c7876682a410ca67111d1ec6b2a200d4b57fb899e743f4fc741`.
The closed producer pins are
`92e30bdf1fbc1562c4d93795b56e104c5d7fe5d1cabaf0b9e1512d1923347ba5`.
The complete2563-member installed package is unchanged through the attempted run, and
the preceding ABI11/64c package is preserved in full.

The initialized real Uniform routing regression and six bounded checkpoint unit protocols
pass on the same installed package. Four older capture fixtures fail because their local
Native doubles omit `checkpoint_state_carriers()`. Both the fixture and Uniform capture
source are byte-identical to parent9d4; the four failures are preserved separately.
ROOT receipt `root-uniform-native12-bounded-nonreg-reception-23810-20261007.json` has SHA
`17eda869a7792ccb92de17bcfbfe51ae01450afefa7ddffcdedd199d289fc495`.
World1/OpenMP2 are measured by separate probes under the same controls; pytest-process
effective concurrency is not independently recorded. These unit protocols do not qualify
physical IMEX, MPI2, CUDA or3D.


The original direct C++ transaction case now passes in a separate standalone runtime:
MPI1 one test,85ms; MPI2 one test per rank,112ms, zero failure/error/skip. It executes the
clock-setter refusal, current-candidate image, mutation invalidation, unequal local
revision counters, one-rank collective refusal, nested rollback, commit validation and
finalization expiry. All706 C++ source entries remain exact to Source23810. Runtime is
O3/C++20; the heavy test TU is O0 per repository policy. The first MPI2 shared-XML capture
is preserved, followed by the correctly ranked capture. ROOT seal
`root-cpp-candidate-checkpoint-control-reception-23810-20261007.json` has SHA
`7c54ea8648543a5b7e1f69f7ba3d6654d6dcb8917c1d607550b150f741cc995c`.
The standalone binary SHA is
`33b2e258bdd53f3096b1a047564efeaf004832e7542278949d34d3d8a6e2a5d5`;
it is distinct from the preserved installed extension77d. This confirms direct C++ commit
and lifetime behavior; Python IMEX/publication/restart, spatial partitioning, GPU and3D
remain separate qualification requirements.

## Candidate diagnostic projection @1

The corrective Python rule is `pops.consumer.candidate-diagnostics@1`: within an active
outer rollback transaction that prepares a final checkpoint, prepare each Native diagnostic
projection before checkpoint capture. Its prepared receipt distinguishes these writes from
publication of Python registries, baselines, console samples and cursors after commit.
The ordinary initial/end route and SkipSampleReported without a checkpoint retain their
existing behavior. A rank-local staging failure is voted before peers enter writer or
checkpoint collectives; the outer Native snapshot restores the diagnostic map on failure.

A transaction-local immutable projection cache preserves the same values and baseline
updates across a ScientificOutput publication Retry. Re-preparation restores pending
Python data without reducing the State or writing Native diagnostics again after commit.
The checkpoint remains sealed at the prepared revision. The cursor override is cleared
even if construction of this preparation mode fails. Equations, C++ ABI12, headerc04a,
archive formats and stale-token guards are unchanged by this Python correction.

The author and independent Source cohorts each pass22 tests. The independent report is
`/Users/romaindespoulain/dev/tmp/sol61-diagnostic-candidate-source-independent-20261007/report.json`,
pins `706d481666b9eed953c5d14745cec5be1153e1f6306c93551b53a0f5db532fd9`.
Imports explicitly use Main/python and load no PoPS Native module. The first independent
command selected the older IR17 site package and failed ten cases; that targeting error
and its XML remain preserved. The corrected Source replay changes no product code.
These Source protocol tests do not receive a real IMEX run or distributed MPI execution.

Retry of the checkpoint itself remains an identified extension requirement: rollback
destroys the current provider's staging proof, and re-preparation after commit cannot
call the uncommitted candidate factory. It requires a versioned retain/replay interface
for the same sealed checkpoint image and new tests. It is not qualified by the diagnostic
projection cache. The next required reception is the rebuilt original manual IMEX,
then its parity, rollback and independent publication-failure variants.

## Diagnostic inventory correction

The rebuilt a67 original reaches prepared diagnostic capture, then its declared78-byte budget
refuses the larger image. The compiler-retained Program contains only `pops.frontier.duration`:
40 envelope bytes plus16 record bytes plus22 UTF-8 name bytes. The failing Native image and
extra key were not saved by that process; their exact values remain unknown. ROOT receives
all2 849 declared entries in `root-imex-diagnostic-capacity-negative-reception-a67-20261007.json`,
SHA `a3d5019d84337277139de3f15fb0cc841a9da05aac028b49a69ebefab00d9dd6`.

The corrected default inventories the union of verified Program records and all potential
ConsumerGraph diagnostic records. The canonical naming helper is shared with the actual sink,
including conservation and accepted-balance reductions. Records aggregate across selected AMR
levels; there is no record per cell or arbitrary level multiplier. Deduplication and UTF-8 byte
lengths follow the existing POPSDIA1 formula. Dynamic Program names remain unknown, requiring
explicit configuration; user-chosen capacities and the collective over-budget guard remain strict.

Author and independent Source cohorts each pass32 tests. The original declared bound becomes
383 bytes per rank from two names; this is a Source calculation, not an observed new Native image.
The independent pins are `8b455036d8d9fdf8df592426d0132b6f6ff115d8f6f6d88a30b91f00b581400d`.
No wire, API, ABI12, header signature, equation or numerical threshold changes. A new official
package and original Native run are required to receive accepted State/Field/restart behavior.
