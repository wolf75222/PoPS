# M18 moderate discrete entropy: SDK375f native reception

These receipts receive the original five-node discrete entropy equations for
twenty interior targets in Dim2 on Serial and two MPICH ranks. Both campaigns
import the installed package in the dedicated `pops-api040` environment, with
`PYTHONPATH` removed. The native build source is `32235b93296fe6d4337194e17d0d45569dc63a08`;
the observed fixture/source commit is `33dee906850d56eec542dd0a7d0b21954aa244d3`.
Their 1,114 production Python/header files are byte-identical. The subsequent
oracle correction changes only tests/docs. Exact digests and paths are recorded
in [the companion receipt](m18_sdk375f_native_reception.json).

Each backend runs one native test without skips, stores ten phases and preserves
the readonly target exactly. The original moment residual is
`4.548139642679416e-12 < 2e-11`; minimum population is `0.07362741560528462`,
minimum entropy gap `1.1062088624547162e-4`, entropy tangent defect
`1.362028979184915e-18`. The independent reader reconstructs populations,
quadrature moments, entropy and finite-cone classification from the saved arrays.
Actual outside-cone attempts refuse twice and preserve state, reports and clocks.
A fresh bind to the original interior problem succeeds. Reducing a time step on
the impossible outside-cone problem is not claimed to repair its equations.

Independent reception uses two external ROOT seals. Capture records actual
Python/SDK/native origins, both System libraries and retained generated Program
sources. System-generated C++ is not exposed by this provider and remains an
explicit provenance gap. No missing source is reconstructed or substituted.
The empty-exchange reader contract is `sol61.m18-empty-exchange-reader@2`;
reception is `sol61.m18-offline-reception@2`. The original reader incorrectly
expected 32 bytes for empty `POPSEX01`; the actual wire image is 16 bytes. Empty
`POPSEX02` is 32 bytes. Historical red receipts are preserved.

Each backend then clones its authentic saved files into twelve separate
countermodels, recomputes all file/checkpoint seals, and receives the expected
scientific refusal. These attack multipliers, readonly target, axes, population
overflow, cone target, rollback, clock type/value, ownership, target availability,
original tolerance and quadrature. They execute no native solver; the original
files are independently reverified unchanged after all attacks.

Reproduce the native campaigns from the repository with a coherent installed
Dim2 SDK, the repository build script and a fresh output directory. The original
profile used Apple LLVM21, Kokkos5.2.0/OpenMP1, MPICH4.1.2 and binary64:

```sh
env -u PYTHONPATH POPS_ENV_NAME=pops-api040 bash scripts/build_python.sh --dim 2 --mpi
conda run -n pops-api040 env -u PYTHONPATH POPS_NATIVE_DIM=2 \
  POPS_KEEP_GENERATED=1 OMP_NUM_THREADS=1 POPS_THREADS=1 \
  python docs/development/api_040/run_installed_checks.py \
  --output /absolute/fresh/m18-serial \
  --test tests/python/integration/runtime/test_m18_discrete_entropy_runtime.py
conda run -n pops-api040 env -u PYTHONPATH POPS_NATIVE_DIM=2 \
  POPS_KEEP_GENERATED=1 OMP_NUM_THREADS=1 POPS_THREADS=1 \
  python docs/development/api_040/run_installed_mpi_checks.py \
  --output /absolute/fresh/m18-mpi2 --dimension 2 --ranks 2 --threads 1 \
  --test tests/python/integration/runtime/test_m18_discrete_entropy_runtime.py
```

Assemble the exact captured execution-owner metadata with
`tests/review/sol61_m18_owner_assemble.py`, then authenticate the actual inventories
before providing the external ROOT approval. The exact seal protocol and CLI are
in [the assembly contract](m18_owner_assembly_sol61.md). Supply the external
owner-pins SHA256 to the independent mathematical reader and mutation harness:

```sh
conda run -n pops-api040 env -u PYTHONPATH \
  python tests/review/sol61_m18_entropy_offline_oracle.py \
  --pins /absolute/received/owner-pins.json --owner-sha256 EXTERNAL_OWNER_SHA256 \
  --output /absolute/fresh/scientific-reception.json
conda run -n pops-api040 env -u PYTHONPATH \
  python tests/review/sol61_m18_entropy_countermodels.py \
  --pins /absolute/received/owner-pins.json --owner-sha256 EXTERNAL_OWNER_SHA256 \
  --countermodels-dir /absolute/fresh/m18-countermodels \
  --output /absolute/fresh/countermodel-reception.json
```

Use the actual CLI arguments reported by the assembly/harness help; do not reuse
old owner seals for a new execution. The archived output paths in the companion
receipt are relative to this task workspace, outside the Git checkout.
Near-boundary entropy behavior, other quadratures, complete Vlasov–Poisson/BGK,
AMR entropy and GPU execution remain unqualified by these receipts.
