# Archived M09 finite, AMR local storage and W12 reception

This bundle preserves completed **local CPU Kokkos/OpenMP, Dim2, serial and
MPI2** receptions. It is attached to the pre-M23 installation, not to later
source or SDK rebuilds. It does not qualify GPU, remote HPC, a meshed/global
M09 solver, M14, or the whole API 0.4.0 mission.

| Archived receipt | Observed XML result |
|---|---|
| `installed-finite-amr-repaired-unit` | 21/21 passed |
| `installed-finite-amr-repaired-serial` | 13 tests: 11 passed, 2 failed |
| M09 subset of that serial XML | 10/10 passed |
| `installed-finite-m09-mpi2` | 10/10 passed on each of 2 ranks |
| `native-w12-reviewed-ctest` | 3 CTest rows: 2 passed, MPI aggregate failed |
| `native-w12-snapshot-repaired-ctest` | 3/3 CTest rows passed |
| `installed-affine-amr-interface-serial` | 3/3 passed |
| `installed-affine-amr-interface-mpi2` | 3/3 passed on each of 2 ranks |

Counts are reconstructed from testcase elements and checked against XML
suite counters, not copied from a terminal summary. The W12 MPI row is a
ten-case GoogleTest aggregate. Its untruncated `build/w12-LastTest.log`
contains both ten-case success summaries. The CTest XML itself truncates
this output, so it is not used alone to infer the inner case count.

The two retained serial failures are the AMR fixture's checks that refinement
contains a real coarse/fine interface: the old tag choice refined the whole
fine level. `fb017cc4` selects an interior refinement band and the later
AMR receipts cover that exact fixture. The older W12 aggregate failed on
the rank-owned global snapshot comparison; test-only correction `210868a1`
was rebuilt and received. Both previous red XML/log pairs remain in this
bundle. The earlier compilation and first AMR failures are preserved in
the separate `evidence/finite-amr-preliminary` bundle (`4c92f36d`); that
bundle is neither replaced nor copied here.

## Identities and source authority

Installed native DSO SHA-256:
`aed8c1582c3344bd5afb19ab784adec260bcdbd4bf944b26a29242b53889c862`.
SDK headers signature:
`396719235acdb96435dba0dd27c5e5579aefbda4936f657262b9b2553d13bb4c`.
The raw identity receipts preserve compiler, C++ ABI, MPI ABI, package paths,
doctor, import paths, command and the full **1089-member** source hash maps.
The version string in those receipts is retained as observed, `1.1.0`.

Finite unit/serial receipts name `86646b6c`; AMR final serial names
`fb017cc4`. MPI receipts preserve per-rank DSO identity, exact test hashes,
before/after authentication results, unchanged-installation/test flags,
and identical rank testcase inventories. The runner's before/after log files
are empty; the result JSON carries their exit status, not a invented log.
Exact tests/helpers/examples/runners are extracted from Git into `source/`
with commit, Git blob and SHA-256 recorded in `source/snapshots.json`.

All 1089 installed-source hashes were independently compared to exact Git
blob bytes at `86646b6c`, `fb017cc4` and documentation-only `03e25b14`:
all match, recorded in `source/git-production-proofs.json`. The later live
disk capture crossed the M23 integration and found eleven committed source
differences at `4b5ad60d`; this is retained in
`source/current-production-proof.json`. The working-tree production diff
was empty. This explicitly does **not** authenticate the new M23 tree as
the old installation.

The actual existing `build-mpi/bin/test_program_runtime` was hashed before
the next build, at 2026-09-29 07:56:18 UTC, while HEAD was `03e25b14`:
`87c96448ae2ca84693bfbbdd86f55198e767325512ba2b9a0f5f51d7c5d916e7`.
Its size is 383,415,176 bytes. Only its identity/stat metadata is archived;
the binary is deliberately omitted. Build logs, selected CMake cache keys,
the actual compile command, exact matching CTest command lines and the
full original files' hashes preserve the relevant build provenance. This
is a post-run capture of the unchanged existing binary, not a new run.

## Saved-state evidence and independent recomputation

There are **16 serial + 16 MPI rank-0 M09 NPZ archives**, preserving their
original `pytest-tmp`/`rank0-tmp` paths. No generated `.so`, object file,
JIT cache, or executable is included. The archived independent root script
`raw/recheck_finite_m09_states.py` imports NumPy only. It reconstructs G
from RNG seed 20260928, the four original block matrices and old operands,
checks all original equations and solves the monolithic reference directly.
It does not call the compiled provider or the example's oracle function.

Recomputed maxima on these 32 saved states:

- Original residual: **3.552713678800501e-15**.
- Error against monolithic NumPy solve: **7.771561172376096e-16**.
- Crank–Nicolson energy error on the original W06 saved states:
  **1.0658141036401503e-14**.

All use the original **1e-11** threshold. The receipt also retains results
for two seeds, distinct old captures, same-artifact new binds, singular
elimination versus nonsingular monolithic system, and repeated owner-only
nonfinite rejection with an empty MPI rank. Assertions and exact fixtures
provide those runtime proofs; the NPZ recomputation covers accepted states.
These are finite **8+4 DOFs** carried in a batch cell, not 12 spatial cells
and not a distributed meshed elliptic solve.

The AMR fixtures saved no NPZ files. Their exact native assertions and
serial/MPI XML/logs cover local affine body/solve, one and two levels with
an actual interface, and refusal/rollback. This archive does not claim an
independently reopened AMR state dataset or dynamic-regrid qualification.

## Reproduction of archive checks

```sh
python docs/development/api_040/check_finite_amr_converged.py
python docs/development/api_040/check_finite_amr_converged.py --recompute
```

The first command uses only the standard library and verifies exact file
inventory, SHA/size, XML status, rank parity, identities, source/test
snapshots, binary fingerprint and NPZ hashes/counts. The second additionally
requires NumPy and recalculates every accepted M09 metric from the archived
states, without importing PoPS or modifying the archive/environment.

Validation during assembly: both commands passed, the NumPy record arrays
match the independently supplied root receipt exactly, and **5/5** checker
tests passed. Negative tests remove a rank XML, corrupt an NPZ, hide an old
failure while recomputing its file hash, and introduce a cache binary.
Ruff passed. No native solver/compiler/rebuild was invoked during archival.
