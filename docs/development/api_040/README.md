# API 0.4.0 specification migration

This directory tracks the integration into the production PoPS package. The
provided specification is version 0.4.0; it is not the production package's
version. The production additions use package 1.1.0, public API revision 2 and
semantic IR revision 2. Native component ABI 3 and checkpoint payload schemas
are unchanged. Exact artifact identities invalidate incompatible compiled plans.

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

The contract and corpus registries are initial source mappings, not a claim that
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
compares modified package/SDK files with the checkout, requires a healthy doctor,
and writes imports, hashes, commands, pytest XML and exit status. It does not
promote skipped checks to successful runtime coverage.

The C++ lane uses `cmake --preset mpi -DPOPS_TESTS_FAST_O0=ON`, followed by
targeted builds and CTest labels. Keep its source frozen during compilation;
a header edit during one earlier rebuild was detected by `doctor.headers_sync`
and that installation was excluded from native acceptance.

GPU execution, other spatial specializations, distributed failure tolerance and
full scientific corpus qualification are not implied by a Dim=2 CPU/MPI build.
