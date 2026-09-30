# Independent review of the native moving-interval carrier

Initial source pin: `88c755d31fd7a4f1ec3451d5917ed3d400a455cf`.
Collective correction: `b7dcaa6a33166f425ee0bfae39f3b9e3cbc353ef`.
Physical publication correction received:
`ed8b4e09358234053d7d47bbf21088d525002cec`.
All native inputs were exported with `git archive`; the author's working files
and shared SDK/environment were not changed. The review did not link System.

## Defect and corrected source reception

The original update checked only finite output before swapping the physical
state. For an Euler state with density `2.3`, unchanged cell volume `V`, zero
sweeps/physical flux, and finite integrated density source `-(2.3+1)*V`, the
actual `SweptInterval` algebra produces density `-1`. Coordinates and GCL remain
valid. The ordinary `ProgramContext::commit_many` additionally invokes the
physical block's prepared variable-recovery authority; the ALE path bypassed it.
The System step envelope has no extra terminal recovery check after its body.

`ed8b4e0` now calls `validate_program_state_publication_candidate_` before ledger
staging and both swaps, and refuses static embedded-boundary masks. This keeps
the existing physical admissibility authority in charge. The independent
structural receiver fails on `88c755d` (missing guard) and passes on `ed8b4e0`.
It also checks the exact box/owner/rank-space/domain/periodicity/lane contract
and the collective numeric-preparation exception vote added by `b7dcaa6`.
These are source checks, not an executed MPI refusal.

## Executed bounded checks

The host C++ probe uses the actual archived `ProgramRuntimeState`, `Fab`,
`FaceField`, `MultiFab`, `CacheManager` and `SweptInterval` headers plus Kokkos.
It verifies independently allocated coordinates, swept volumes and measures in
the aggregate copy used by `AcceptedSnapshot`. It then prepares an accepted
restore, mutates the source image, publishes the prepared restore, and verifies
geometry, generation/interval, last duration, diagnostics and cache. This passes
against both `88c755d` and `ed8b4e0`; Kokkos views are not shallow aliases here.

The final probe against `ed8b4e0` also uses the actual Reynolds update on 8/24
cells with durations `.1/.2/.3` and starting times `0/.4`. An independent oracle
`U=2.3+.4*t`, `F=.7*x`, `S=.4+.7` uses exact midpoint space-time flux/source
integrals for each linear face path. Local densities and global amount change
match that oracle. Stale-duration swept measures and NaN sweeps are refused by
the primitive. Output:

```text
deep_copy_prepared_restore_pass
finite_unrecoverable_candidate -1
manufactured_reynolds_balance_stale_nan_pass
```

The complete archived `test_program_runtime.cpp` plus the independent fixture
`tests/cpp/integration/runtime/ale_carrier_independent_review.inc` passes Clang
C++20 syntax checking with explicit `POPS_NATIVE_DIM=1`, Kokkos/OpenMP/MPICH/HDF5,
and the central gtest headers. There is one upstream gtest `char8_t` warning.
The fixture calls the real `System`/`ProgramContext::advance_moving_intervals`;
only rank zero injects the finite unrecoverable candidate. It requires refusal
and unchanged fields, coordinates, swept volumes, measures, generation,
interval, time, step, cache and ledger. Its runtime execution remains pending.

## Reproduction and integration

From the review checkout, structural reception is:

```sh
python3 docs/development/api_040/evidence/ale_carrier_review.py --revision ed8b4e0
python3 docs/development/api_040/evidence/ale_carrier_review.py --revision 88c755d
```

The second invocation is expected to fail. Export the selected headers before
compiling the host probe; `ALE_REVIEW_DIR` is a separate temporary directory:

```sh
ALE_REVIEW_DIR=$(mktemp -d /tmp/pops-ale-review.XXXXXX)
ALE_REVIEW_ENV=/Users/romaindespoulain/miniforge3/envs/pops-api040
git archive --format=tar --output="$ALE_REVIEW_DIR/headers.tar" ed8b4e0 include
tar -xf "$ALE_REVIEW_DIR/headers.tar" -C "$ALE_REVIEW_DIR"
/usr/bin/clang++ -std=c++20 -DPOPS_NATIVE_DIM=1 -DPOPS_HAS_KOKKOS -DKOKKOS_DEPENDENCE \
  -I"$ALE_REVIEW_DIR/include" -I"$ALE_REVIEW_ENV/include" \
  -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include \
  docs/development/api_040/evidence/ale_carrier_review_88c755d.cpp \
  -L"$ALE_REVIEW_ENV/lib" -lkokkoscore -lkokkoscontainers -lomp \
  -Wl,-rpath,"$ALE_REVIEW_ENV/lib" -o "$ALE_REVIEW_DIR/probe"
env OMP_NUM_THREADS=1 OMP_PROC_BIND=false "$ALE_REVIEW_DIR/probe"
```

After integrating the carrier and the ABI-coherent native rebuild, include the
independent `.inc` at the end of `test_program_runtime.cpp`, and run
`ProgramRuntime.MovingIntervalsRefuseFiniteUnrecoverablePhysicalStateBeforePublication`
in the serial Dim1 target and its MPI2 aggregate. The Dim1 syntax command used
`-fsyntax-only -DPOPS_RUNTIME_SHARED_EXCEPTION_ABI -DPOPS_HAS_MPI
-DPOPS_HAS_PARALLEL_HDF5` in addition to the flags above; the central gtest
include root was `work/PoPS/build-mpi/_deps/googletest-src/googletest/include`.

No rebuilt System execution, shared-face MPI execution, empty-owner rank
execution, GPU run, ALE checkpoint/restart or public `StateGeometry` lowering
is qualified by this review. Those remain central reception obligations. The
carrier's persisted checkpoint codec is absent in this tranche, as documented
by its author; the public moving-geometry request must remain fail-closed until
its lowering and saved moving geometry are authenticated.
