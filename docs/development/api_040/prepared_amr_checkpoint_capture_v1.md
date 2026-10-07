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
