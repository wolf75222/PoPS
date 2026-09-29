# API 0.4.0 specification migration

This directory tracks the integration into the production PoPS package. The
provided specification is version 0.4.0; it is not the production package's
version. The production additions use package 1.1.0, public API revision 2 and
semantic IR revision 2. Release native ABI 3, generated System package protocol 6,
and component interface 1 are separate authorities. Joint primitive coordinates
use artifact manifest schema 10. The [native boundary decision](c26_c27_native_boundary_decision_v1.md)
records the original protocol 5 decision; the [prepared boundary extension](m07_prepared_boundary_route.md)
records its protocol 6 successor.
Exact artifact identities invalidate incompatible compiled plans.

The migration remains in progress. Start with the [current checkpoint](CHECKPOINT.md)
for actual source/native identities, completed receptions, failures and next work.
The [independent mission audit](mission_gap_audit_a458113.md) distinguishes the
remaining implementation, numerical and backend obligations. Historical receipts
below remain scoped to their own source and binary, even after fixes are integrated.

## Historical registry audit at 83b2b12

`contracts.csv` and `corpus.json` now distinguish the inspected source at
`83b2b1239973f3e744f9e3975734a40a3abf65d6` from each historical receipt's exact
source commit, dirty-diff hash, native hash and backend. `current_head` is **not**
a claim that every row was rerun. Fourteen contract mappings retain older evidence
without new qualification. All unresolved obligations remain active; missing
expression/implementation is never converted to SCOPE.

The [receipt manifest](evidence/83b2b12/manifest.json) preserves copied identities,
results and XML with SHA-256 digests. Original workspace locations are recorded;
large native state archives remain there. The bounded audit did not rerun any
numerical campaign. It records these completed receipts separately:

* Native used by that audit `4574ed6096650aa708ad040fd6ee056414a3cfa1735a0440bd74220170265beb`,
  1,079 verified source files, source digest
  `c1f5d8141ac858aa80efc7e24ca1c2960b2733b83084b97b7715fb08ffeacaf5`:
  initial critical run **50 passed / 21 setup errors** (Kokkos root absent).
  Corrected PDE run **12 passed / 9 failed**: six Primitive and six LocalResidual
  passes; two C17 test-API failures and seven principal/User generated-package
  compilation failures. Neither run is globally green.
  Subsequent source fixes `6ce6782` (C17 diagnostic facade) and `3d06cab`
  (principal versus constituent-storage policy ownership) are recorded as
  pending reception; they do not replace or turn the failed receipts green.
* Historical unit directory `installed-38faddb-unit` actually authenticates
  source `044fade`: **209 passed / 1 failed**, the absent-versus-empty C34 cursor
  defect subsequently fixed in `db486be`. The current critical run receives 41
  C34/transaction tests successfully. Historical CTest: **123 run / 9 not run /
  1 failed**; its obsolete Path FixedDt expectation was adapted in `db486be`,
  with a fresh C++ receipt still required.
* M02 shock and rarefaction each passed the full prescribed grids/time on native
  `220b48d…`, Dim=2 CPU, one rank/thread. M04 Forward Euler still fails its
  original order criterion; the distinct SSPRK2 experiment passes. M17's old
  FixedDt rejection is retained pending a full new scientific receipt.

Check registry identities, source paths, receipt hashes and XML counts without
running numerical tests:

```sh
python docs/development/api_040/check_registry.py
```

`build_registry.py` is the historical bootstrap inventory, not the updater for
these reviewed records; rerunning it would discard their evidence and statuses.

## Baseline and provenance

The isolated worktree starts at `3a93ba7f1b3fc46ee85a06d9d3482c9b4ece7feb`
on `codex/api-040-native-20260928`. It continues the existing PR 680 work;
it is not based on remote master (`bb9aea0cc56fabe6b003bfbee5daee28d38ac7e7`
when inspected). PR 680 was draft, with successful checks on its own exact head.
Those checks do not qualify this migration. Existing worktrees and user changes
were preserved. The new worktree was initially clean.

The supplied ZIP passed CRC and its own manifest verification: 325 declared
files, 300 inherited digests, two identity checks, no missing/extra entries.
The normative task is the ZIP's `PROMPT_CODEX.md`, not the different standalone
prompt attachment. The original mathematics is retained unchanged in the input.

| Input | SHA-256 |
| --- | --- |
| Handoff ZIP | `a73bc1bbe85a8f4e0f4850ff3a324866a57017b582eb95ba9898dd8479fe4033` |
| Design principles | `b52ecf2e854f7e5a10e13147fe8f59c366f35bef89f9bf8179c1f9678510b1be` |
| Separate prompt r2 | `070c18307a2e470c4f0c5c38bec2d4fa4b344dc0ec5d009085a478899883564a` |
| API PDF (45 pages) | `8e89f49eb44fe5b8b5a05e30ee891b42a69ef8078059ee4a4adfc8240b3de2e5` |
| Original mathematics | `8e0f4b2abe43f7fcd380113931fe61eaea46d276a75d28644ee0a7d70190a4f7` |

`scripts/setup_env.sh` was run once in the worktree. The shared `pops`
installation had a newer, unrelated native-manifest schema; the first build
refused to overwrite it. A clone named `pops-api040` preserves that installation
and provides a reproducible Python 3.12 / AppleClang / Kokkos / MPICH / parallel
HDF5 lane on the local Apple M1 Pro. This is a local MPICH lane; it does not
qualify the release matrix's OpenMPI lane.

The baseline imported the installed package from that environment and its
Dim=2 extension, not any handoff `pops` module. Native extension SHA-256:
`ac9129b1ca51cec68c977c658d170be8a996fe39461a00a2499c9e281d38dda0`.
The baseline had 34 passing source/contract tests. Five runtime tests initially
skipped without the native compiler environment; the explicit rerun passed all
five in 118.17 seconds. The skips are not counted as native passes.

The isolated prototype replay passed 57 current reference groups and 51 legacy
records, separately. These are evidence about the supplied CPU demonstration;
they are not production PoPS integration or MPI/AMR qualifications.

## Integrated mechanisms under validation

* T1/T3: reusable scalar/vector expressions on Program values, compact shared
  expression DAG, exact component ownership, immutable Newton captures distinct
  from the initial guess, and guards on every native floating intermediate.
* T2: authored state paths and quadratures applied to retained physical matrices,
  emitted through the ordinary expression compiler into the existing native
  path finite-volume execution. Fan–Li authentication belongs to its library
  path; the compiler does not recognize a scientific model by name.
* T4/T5: failure-atomic native prepared-resource replacement with collective
  agreement, plus a real buffer/history/cache/exchange nested rollback test.
* Numerical robustness: signed subnormal/extreme Minmod/Van Leer/MC/Superbee;
  finite WENO evaluation at extreme scales and explicit invalid-input behavior.
* C30: frozen descriptor deletion is prohibited as well as assignment.
* C11/C12: shared principal operators, per-row parameter captures, explicit joint
  primitive inverse/domain, and AMR halo/reflux routes are integrated. Current
  principal/User generated compilation failures remain implementation defects.
* C17/C34/C38: original-residual stagnation refusal, exact cursor compensation,
  and frozen restart-history mutation guards are integrated. Their distinct
  source and native reception states are recorded in the registry.

The contract and corpus registries are scoped source/evidence mappings, not a claim that
C01–C40 or M01–M28/W01–W12 have all been implemented or executed. Results and
unresolved obligations must remain separate. See [decisions.md](decisions.md)
and [agents.md](agents.md). This migration is still in progress until the native
integration receipts and the remaining gaps are reviewed.

## Reproduction

Use the repository setup/build scripts. In an environment named `pops-api040`,
with `POPS_PREFIX` set to its absolute conda prefix:

```sh
env -u PYTHONPATH POPS_ENV_NAME=pops-api040 \
  Kokkos_ROOT="$POPS_PREFIX" POPS_KOKKOS_ROOT="$POPS_PREFIX" \
  CMAKE_PREFIX_PATH="$POPS_PREFIX" POPS_HEAVY_MODULE_TU_POOL=2 \
  CMAKE_BUILD_PARALLEL_LEVEL=4 bash scripts/build_python.sh --dim 2 --mpi

env -u PYTHONPATH PYTHONNOUSERSITE=1 POPS_NATIVE_DIM=2 \
  POPS_REQUIRE_NATIVE_TESTS=1 CONDA_PREFIX="$POPS_PREFIX" \
  Kokkos_ROOT="$POPS_PREFIX" POPS_KOKKOS_ROOT="$POPS_PREFIX" \
  POPS_INCLUDE="$POPS_PREFIX/lib/python3.12/site-packages/pops/include" \
  OMP_NUM_THREADS=1 "$POPS_PREFIX/bin/python" \
  docs/development/api_040/run_installed_checks.py --output /tmp/pops-api040-evidence
```

The runner refuses source/prototype imports, verifies the selected extension,
compares all tracked package/SDK files with the checkout, requires a healthy doctor,
and writes imports, hashes, commands, pytest XML and exit status. It does not
promote skipped checks to successful runtime coverage.

Build `--dim 1 --mpi` as well to receive true Dim1 examples; the repository
script preserves the sibling extension while refreshing common Python files.
Receive an explicit native group on two MPI ranks with per-rank XML, dimension
selection and package/source authentication before and after execution:

```sh
env -u PYTHONPATH PYTHONNOUSERSITE=1 CONDA_PREFIX="$POPS_PREFIX" \
  Kokkos_ROOT="$POPS_PREFIX" POPS_KOKKOS_ROOT="$POPS_PREFIX" \
  CMAKE_PREFIX_PATH="$POPS_PREFIX" \
  POPS_INCLUDE="$POPS_PREFIX/lib/python3.12/site-packages/pops/include" \
  OMP_PROC_BIND=false "$POPS_PREFIX/bin/python" \
  docs/development/api_040/run_installed_mpi_checks.py \
  --dimension 2 --ranks 2 --threads 1 --timeout 900 \
  --output /tmp/pops-api040-local-products-mpi2 \
  --test tests/python/integration/runtime/test_local_residual_product_runtime.py \
  --test tests/python/integration/runtime/test_local_product_operators_runtime.py \
  --test tests/python/integration/runtime/test_local_product_readonly_runtime.py \
  --test tests/python/integration/runtime/test_m11_w10_constrained_runtime.py \
  --test tests/python/integration/runtime/test_m18_discrete_entropy_runtime.py
```

For the owner-only periodic rejection, select `--dimension 1` and
`test_m15_mpi_collective_rejection.py`. A PASS on each rank is one shared test,
not two independent experiments. Freeze production and test sources until the
runner has completed its final authentication.

Full scientific examples use the same identity check before and after the run:

```sh
env -u PYTHONPATH PYTHONNOUSERSITE=1 CONDA_PREFIX="$POPS_PREFIX" \
  Kokkos_ROOT="$POPS_PREFIX" POPS_KOKKOS_ROOT="$POPS_PREFIX" \
  POPS_INCLUDE="$POPS_PREFIX/lib/python3.12/site-packages/pops/include" \
  "$POPS_PREFIX/bin/python" docs/development/api_040/run_scientific_checks.py \
  --case m01 --ranks 2 --threads 1 --output /tmp/pops-api040-m01-mpi2
```

The output directory must be empty. The runner also includes `m04-ssprk2`,
`m06`, `m08`, `m13` and `m17`; use its `--help` for the exact current choices.
Availability in the runner does not imply acceptance. M02's full serial receipts
are historical native evidence; M04's directional restriction is fixed, while
the original Forward Euler order criterion still fails. M03 stiffened passes
receipt schema 2, which independently reconstructs the discrete incoming
boundary integral from saved native states. Its previous far-field-budget
failure is retained in the checkpoint. Grids, final times and criteria are
fixed in each linear example.
M04 receipt schema 3 separates its spatial and temporal methods and checks an
independent discrete Fourier oracle. It uses the exact tensor `diag(.01, 0)` for the requested
x-only law. `m04-isotropic` retains the previous `diag(.01, .01)` variant to
reproduce and repair its directional-frequency failure. Both retain the same
error, conservation and bound criteria, with the combined step computed from
their actual constitutive tensor. Neither is a native Dim=1 qualification.
Saved native states, exact/independent references, measurements, hashes and logs
remain in the output directory. `--threads 4 --ranks 1` selects the local OpenMP
configuration; the runtime log reports the native execution concurrency.

The C++ lane uses `cmake --preset mpi -DPOPS_TESTS_FAST_O0=ON`, followed by
targeted builds and CTest labels. Keep its source frozen during compilation;
a header edit during one earlier rebuild was detected by `doctor.headers_sync`
and that installation was excluded from native acceptance.

GPU execution, untested spatial specializations, process-loss tolerance and
full scientific corpus qualification are not implied by these CPU/MPI receipts.
