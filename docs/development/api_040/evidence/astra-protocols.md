# T4/T5 native prepared-resource publication - 2026-09-28

Checkout: `work/PoPS`, shared branch `codex/api-040-native-20260928`, supplied base
`3a93ba7`. Existing edits by other workers were preserved. No commits, environment
setup, global build, version change, or reference-runtime installation were performed.

## Implemented defect and correction

`PreparedResourceCache::acquire` used `optional.emplace` on its currently cached
resource. A failed replacement destroyed the prior numerical storage. Constructor
failure on rank 0 was not converged, so rank 1 could return successfully with a
replacement while rank 0 unwound. This is the real cache used by uniform
`ProgramContext` and AMR program contexts, not a separate protocol oracle.

The cache now builds a separate holder, gates its allocation collectively before
provider construction, catches constructor failure and converges that result on
the execution lane. Only unanimous success publishes the holder. Publication
does not allocate or move the resource; noncopyable/nonmovable resources retain
their addresses. Failure preserves the original buffer and makes the acquisition
retryable. Cache hits keep their existing collective count; misses/replacements
add two reductions and temporarily retain both old and candidate storage.

Changed files:

- `include/pops/runtime/program/prepared_resource_cache.hpp`
- `tests/cpp/unit/runtime/test_prepared_resource_cache.cpp`

## Executed red/green evidence

Two added tests own actual `Kokkos::View<double*>` storage and inject a constructor
exception only on rank 0. They cover failed replacement, original holder/buffer
identity, original contents, successful retry, and failed initial construction.
The resource explicitly deletes its copy and move constructors.

Before the header correction:

- Serial: exit 1, 3/4 passed. Failed replacement caused an extra construction and
  changed the retained values from 17 to 42.
- MPI2: exit 1. Rank 0 passed 3/4 and rank 1 passed 2/4. The peer rank did not throw
  during either initial or replacement construction failure.
- `outputs/astra-protocols-red-serial.xml` preserves the serial failure.
  `outputs/astra-protocols-red-mpi.xml` preserves one failing rank's report;
  that initial command did not enable rank-qualified XML and the two reports
  shared a path. The two-rank counts above were observed in tool stdout.

After correction: serial 4/4 passed; MPI2 4/4 passed on **each** rank, exit 0.
Rank-qualified green XML was explicitly rerun with `POPS_TEST_EXPECT_RANKS=2`
to avoid shared output paths. `git diff --check` on the two changed sources passed.

Commands (from repository root, prefix with `rtk proxy`):

```sh
env -u PYTHONPATH PATH=/Users/romaindespoulain/miniforge3/envs/pops-api040/bin:/usr/bin:/bin:/opt/homebrew/bin cmake --build build-mpi --target test_prepared_resource_cache -j2
env OMP_NUM_THREADS=2 OMP_PROC_BIND=false build-mpi/bin/test_prepared_resource_cache --gtest_output=xml:outputs/astra-protocols-green-serial.xml
env OMP_NUM_THREADS=2 OMP_PROC_BIND=false POPS_TEST_EXPECT_RANKS=2 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/mpiexec -n 2 build-mpi/bin/test_prepared_resource_cache --gtest_output=xml:outputs/astra-protocols-green-mpi.xml
```

The target build automatically reran CMake for dependency/header signature changes.
It used existing `build-mpi`, Kokkos 5.2.0 OpenMP/Serial, MPI and native Dim=2.
Only the target and its GoogleTest dependencies were built. Existing HDF5 detection,
GoogleTest character conversion, and duplicate-rpath warnings were observed; build
and final executions exited successfully. These results are local, not GitHub CI.

SHA-256 at validation:

```text
d6ca06c78aca69eeb2802167ef798c169a879a3ef447fae130277bbda4c10546  prepared_resource_cache.hpp
f98adfdc9e14eaaaef115011d0ab856d7c98443aa40218a5560102d244290d32  test_prepared_resource_cache.cpp
825721d2dd9cb8d71c730a4417130a202a25e9563a020e86b9b60a7daa513e78  build-mpi/bin/test_prepared_resource_cache
```

## Existing relevant tests inspected, not executed by this worker

- `test_amr_synchronized_continuation.collective_failure_abort_publication_rollback_and_retry`:
  existing MPI2 registration; rank-0 callback failure, cancellation during callback
  followed by candidate access, rank-0 final-validation failure, later-level
  publication failure, exact rollback and retry. The production
  `PreparedMultiBlockAmrSubcyclingEngine::abort_synchronized` defers destruction while busy.
- `tests/cpp/integration/amr/test_amr_system_contract.cpp` around lines 589–612:
  restart auxiliary data missing/truncated/nonfinite/wrong-owner; no partial
  publication and exact restart rollback. No dedicated MPI registration found.
- `test_program_reflux_ledger.InvalidCheckpointAndDuplicateFacesRejectBeforeMutation`:
  existing malformed accepted-state/reflux checkpoint coverage.
- `tests/python/integration/io/test_amr_history_regrid_replay.py::test_d_corrupted_fingerprint_refused`:
  restart refuses a corrupted regrid fingerprint.
- `test_amr_regrid_mpi_parity`: existing MPI ranks 1/2/4 registration.

## Limits and remaining work

This change covers collective preparation/publication and storage lifetime on
failed acquisition. It does not implement a new asynchronous task scheduler,
late-completion token protocol, independent concurrent-root conflict protocol,
checkpoint checksum/incarnation schema, or quiescence mechanism. It does not
qualify all T4/T5 contracts, checkpoints, regrid, GPU, Dim=3, or process-failure
tolerance. Those claims would require their own native integration evidence.

Providers that contain collectives must still synchronize their own internal
allocation/failure paths before unwinding, as required by the existing cache
contract. The outer gate cannot repair a provider that strands a rank inside an
internal collective. External constructor side effects are not rolled back.
Callers must continue to obey existing sequential/lane and lifetime discipline;
references returned by acquire remain invalidated by a successful replacement
or cache clear. Allocation failure of the candidate holder was guarded in code
but no allocator fault injection was executed.
