# Independent C38 review - 2026-09-29

## Final source review: gaps resolved in 8ca34ec

Reviewed the complete `06dfa98..8ca34ec` production/test diff independently on the
clean `PoPS-numerical-bodies` checkout. Commit `7790bbe` registers the MPI cases;
`8ca34ec302ac0713c8300e5cb942de8ae99cdfca` closes post-capture mutation admission.
The earlier findings below are preserved as the review history and are resolved
for the explicitly identified store/rotate/restore routes. No remaining blocker
was found in this bounded source review.

- `store_history_` and `rotate_histories_` call the new collective freeze check
  before obtaining/mutating their history manager or preparing numeric copies.
  `prepared_execution_lane()` itself requires a live facade, so the subsequent
  query cannot dereference a missing facade in the spatial-only context.
- The facade restore guard votes first on committed state and then on the frozen
  image; both decisions occur before history publication. A frozen image on any
  participating rank rejects authoring everywhere. Outside these phases the guard
  returns normally. The new native test checks this normal route again after
  rollback with a distinct numeric sample, 19 instead of 7.
- The guard is intentionally attached to authored store/rotate and explicit
  facade restoration. Internal remap services still use their original shared
  mutation preparation; they do not call this new authoring guard. Thus the second
  parent can continue consuming the original frozen image and remapping descendants.
- Rollback/finalize cleanup from 06dfa98 remains intact. Finalization is still
  `noexcept` and does not introduce a collective or a new failure after commit.
- CMake now includes the three-level DSO case in its existing MPI2 filter and
  registers `test_amr_history_ring` at two ranks with both C38 tests. The new
  `FrozenRestartImageRefusesLaterStoreRotationAndRestore` witness checks refusal
  before numeric/metadata change and normal authoring after rollback.

Validation performed by this reviewer: source/diff/caller inspection and
`git diff --check 06dfa98..8ca34ec` (passed). No native compilation or execution
was run. The new witness freezes on every rank; its execution does not itself
provide a rank-asymmetric failure injection. Central serial/MPI2 reception on the
rebuilt source remains required. This review closes the bounded source tranche,
not the complete C38 scientific/runtime qualification.

Reviewed source: `PoPS-numerical-bodies`, commit
`06dfa98a2730895fe78de432c8523f37c3ea4124`. Production review is read-only.
No native binary, MPI run, GPU run or heavy build was executed for this review.

## Findings requiring follow-up

1. **A frozen sequence does not prevent subsequent history mutation.**
   `begin_restart_regrid_history_sequence()` stores a detached image. Public
   `AmrProgramContext::store_history` and `rotate_histories` remain callable after
   this capture: their shared preparation checks the lane and collective operation
   contract, but does not check the sequence. `regrid_parent` at source line 9804
   reuses the image without rechecking the live `store_pending` flags. The
   `RetainedChild` branch copies the saved child ring and its saved metadata; it
   need not inspect the updated child ledger. Thus capture -> store -> regrid can
   consume an obsolete accepted image despite the newly added capture-time guard.
   Capture -> store -> rotate -> regrid is a second counterexample candidate:
   rotation clears pending, so a pending-only consumption check is insufficient.
   This is a source-established admission gap, not a native-executed numerical
   failure. The author acknowledged the public sequence and is implementing the
   bounded correction/test. Freeze externally requested history mutation for the
   sequence, or authenticate its mutation generation at consumption; do not reject
   the legitimate internal remaps between the two parent replacements.

2. **The new three-level test is omitted from the explicit MPI2 filter.**
   `tests/CMakeLists.txt:1232` does not include
   `RestartRegridRetainsDescendantHistoriesAcrossTwoParentReplacements`.
   `test_amr_history_ring` also has no explicit MPI2 registration in this snapshot.
   Serial discovery does not provide the intended collective reception. The author
   is adding the test inventory before integration.

## Independently checked source properties

- `prepare_history_hierarchy_images` catches local pending-store failure, then
  reduces errors collectively before exact-contract agreement and sparse gathers.
  Preparatory ring copies allocate detached storage; the live hierarchy is not
  published by this routine. The normal uncached regrid calls it before tagging.
- Sequence begin collectively requires one uncommitted restart and no overlapping
  external step or existing sequence. Rollback clears the sequence before snapshot
  restore, including when restore later fails. Finalize clears it only when actual
  committed finalization is allowed; premature finalize retains rollback ownership.
- The existing `restart_regrid_` callback already brackets the complete parent loop
  with begin/end and ends the sequence on exceptions. Missing multi-parent freezing
  is **not** a defect fixed by this patch.
- The new DSO test drives the actual three-level restart/regrid route, requires both
  child layouts to change, verifies retained deepest-level coverage and every saved
  physical slot there, and checks fill/sample identity on all levels. The public
  `history_global` uses `all_reduce_sum_inplace`; its arrays are available on every
  rank, so the test's all-rank comparisons are valid.
- The test does not compare all numeric history values on levels 0/1, newly covered
  deepest cells, or post-regrid slot dt explicitly. These are limits of that witness,
  not evidence that those values are wrong. Its synthetic scalar history does not
  qualify arbitrary vector history physics.

## Reception to run centrally

- `test_amr_history_ring.RestartRegridImageRejectsPendingStoreAndEndsWithItsTransaction`
- `test_amr_synthetic_program_loader_transaction.RestartRegridRetainsDescendantHistoriesAcrossTwoParentReplacements`
- New mutation-after-capture regression, with both pending and store+rotate paths.

Run serial and MPI2 with a finite timeout after rebuilding the exact corrected
source. Include a single-rank pending injection before the collective preflight if
supported by the native test seam. Syntax checks alone cannot establish absence of
collective hangs or successful descendant restoration.
