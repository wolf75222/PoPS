# Independent diagnosis of the two MPI2 runtime fixture failures

Exact source: `f36176fa2a822c67b5ccc114bd21de1bef4be23e`.
Exclusive checkout `work/PoPS-sol61-mpi2-fixture-review`, branch
`codex/api040-sol61-mpi2-fixture-review`. The historical ALE review commit
`43f0686cc7273c0df62f0ccb619010ea52d939a1` and its checkout are preserved.
No MAIN, native SDK, header, shared environment, installation or JIT was changed.

Authentic external evidence:
`outputs/native-integrated-moving-consumption-scope-green-candidate-dim1-20260930/ctest.log`.
The serial cases pass, while the MPI2 wrapper reports the moving projection
and poisoned vector pairing failures. This review does not rerun or qualify
the central native binary.

## Moving projection: global stride applied to a local image

`System::set_state` calls `write_global`: the expected input is the complete
global component-major image. The writer maps every component and global
coordinate into each rank's local Fab after exact collective payload preflight.
`System::get_state` calls `gather_local_compact`: the result contains
`components * local_cells`, and its component stride is `local_cells`.

The fixture initializes a correct global image, but reads the compact local
return with `component*n+cell`, loops through `n` global cells, and compares
the local rollback return to the global initial vector. At `n=8`, MPI2 owns
four cells per rank. A three-component return has twelve values; the old loop
reads indices 0 through 23. The first component's indices 4 through 7 actually
hold the second component; indices 8 through 11 hold the third; the remaining
reads are out of bounds. This directly explains the authentic zeros, energy
values and garbage reported against the wrong expected components. The old
rollback comparison also compares a twelve-value local image to twenty-four
global values. It does not demonstrate that production initialized only
component zero.

The fixture now authenticates local cell count from the actual native boxes,
checks the compact image size with an all-rank vote, and verifies every initial
component against its declared constant. It checks the all-rank owned cell
count against `n`. Accepted values use the same authenticated local stride;
rollback compares the exact initial native local image. The GCL, issued
durations, generations, receipt, ledger and existing `4e-13` numerical guard
remain unchanged.

## Vector pairing: poisoning a participant without a stored cell

`unit_domain_config` leaves `boxes` empty. `materialized_boxes()` therefore
creates one domain box. SystemDomain applies its declared round-robin prepared
load-balance provider: patch ordinal zero belongs to process zero, and the
distribution is partitioned. Rank one owns no Fab in MPI2.

The fixture selects `lane.size()-1` to poison. On this layout, that rank has
an empty MultiFab. `MultiFab::set_val` only iterates local Fabs, so no NaN is
written. The local contraction also iterates local Fabs; an empty rank
correctly contributes zero and participates in the collective votes. The
stored owner remains finite, so the global contraction correctly succeeds.
The failure at log line `program_dot_all_contract.inc:53` matches this exact
source. The target source already uses `set_val`, not the earlier described
self-aliasing `axpy` injection.

The test now elects a real owner from the actual local cell counts, verifies
that its poison support is nonempty through an all-rank sum, and retains the
strict expected exception. A distinct case elects an empty rank, confirms
its no-op scalar fill does not invent a NaN contribution, and checks the exact
finite all-component contraction. Wrong widths, foreign halos, product
overflow, local-sum overflow and all original vector/gamma values remain
checked. No noncontributing *stored replica* coverage is claimed from the
empty-rank case. Production's replica branch validates the local contraction
before suppressing its SUM contribution, but authentic replicated execution
requires a separate native fixture.

## Scope and validation

Only two integration fixture includes and this report change. No production
defect is demonstrated by these two failures. The diagnoses use the exact
marshaling, load-balance, storage and finite-contraction source paths, not a
replacement physics implementation.

`rtk git diff --check` passes. No C++ compilation, CTest, installed Python,
native execution, SDK update or JIT was run here, as requested. Central native
reception must rebuild `test_program_runtime`, receive both corrected serial
cases, then receive `test_program_runtime_np2` on the exact integrated SHA.
The complete MPI wrapper must pass; a source diagnosis cannot establish that
there is no further failure after fixing its out-of-bounds fixture reads.
