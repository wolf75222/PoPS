# Genuine Dim3 CoupledGradient reception — SDK20d

This is a completed execution of the installed production package. Source is
`48871851f10cd0acb4c0dfa45bd0f3195ab2ef0a`; the native extension is
`pops/_native/dim3/_pops.cpython-312-darwin.so`, SHA256
`0ed823f5f1639172e9913a5fe5e1f15755a70a153e128e7a0f3a1274183c1bdb`.
The SDK is `20d956005fe76c85c846c9ef36ddf769a80fb0d5d78d9f1e98ad220238f99e58`,
native ABI5, public API3, package1.1.0. Apple LLVM21, Kokkos5.2 OpenMP CPU,
MPICH4.1.2 and parallel HDF5 are the actual local backend. The build doctor
passes. Both native loaders require Dim3; no prototype or Dim2 substitute is
used. Root authenticated all 1,109 installed Python/header files against their
tracked source hashes and the exact executed Git revision.

The existing physical fixture remains unchanged: two components, both
permutations, a periodic 6×4×3 grid of lengths1×2×3, SPD D plus skew R, two
physical occurrences of weights.5 and1.5, and one SSPRK2 step1e-5. Both physical
components vary along every axis. Its original guards remain unchanged.

| Execution | Actual pytest cases | Skips/failures | Duration |
| --- | --- | --- | --- |
| Serial | 2 pass | 0/0 | 30.0787s |
| MPI2 | 2 pass on each rank | 0/0 | 76.4446s |

MPI authentication before and after returns0. Native SHA, installed sources,
test sources and per-rank case inventory stay unchanged. Serial identity SHA is
`f166230cf2ba0bf64b2bf0673ca8e0eaf7d1a5ef89d2fc4eba584fb824261c94`.
The serial log SHA is
`bbddc53c923171d84ef4ed5e10bd5fedbbec671c9cd3349bab1caecbb49ecfa4`.

The independent saved-state verifier imports NumPy/stdlib only. It constructs
the72-DOF neighbor-incidence Laplacian and a Kronecker evolution matrix, checks
the independent Fourier amplification polynomial, and decodes the real raw
ledger bytes. It receives3,456 unique face incidences per permutation, including
all three axes, both sides, both stages, both components and both occurrences.
Both backends and permutations have the same bounded errors:

| Quantity | Maximum error |
| --- | --- |
| Native state vs original discrete equation | 4.440892098500626e-16 |
| Physical raw face flux | 6.661338147750939e-16 |
| Integrated face amount | 2.541098841762901e-21 |

Energy decreases from6.267161828458659 to6.267152942896044. The external owner
identities link each actual passing JUnit case, native ABI/SHA, original fixture,
artifact identity, operation and exact stage contexts. They also pin the launch
identities, source manifests and build log. Saved payloads are copied without
changing their bytes; owner identities stay outside those payload directories.

| Backend | External identity SHA | Manifest SHA |
| --- | --- | --- |
| Serial | `5ff0b84abb22189d56d66dac26a00d13e0ba2fd10162652347ca2c636b86405d` | `652357acd9fef343ab0a7a9723ee34a9d520f69d66cbecf7c2f956d116882cbf` |
| MPI2 | `439694a9377920b26d63af81d36c7e1ce007b3236fd47de2b9df0538063a431f` | `5c2d35ef17db2536269b214f97eec6dc238a0b3143733fa2917a705b7f65916c` |

Evidence is in the task workspace
`/Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/outputs`:

- `installed-first-coupled-gradient-fourier-dim3-sdk20d-20260930`
  and `installed-first-coupled-gradient-fourier-mpi2-dim3-sdk20d-20260930`;
- `coupled-gradient-dim3-sdk20d-saved-{serial,mpi2}-20260930`;
- `coupled-gradient-dim3-sdk20d-{serial,mpi2}-owner-identity-20260930.json`;
- `coupled-gradient-dim3-sdk20d-{serial,mpi2}-offline-reception-20260930.json`;
- `seal-first-coupled-gradient-dim3-real-receptions-20260930.py`, the owner
  authentication/copy/sealing command, and the accepted build log
  `build-python-first-coupled-gradient-dim3-sdk20d-20260930-accepted-command.log`.

For reproduction from this source revision, use the repository setup once per
new worktree and its incremental build script. Use a fresh output directory for
each campaign and authenticate the new actual source/native SHA instead of
copying these historical owner identities:

```sh
POPS_PROOF_ENV=/Users/romaindespoulain/miniforge3/envs/pops-api040
POPS_PROOF_PY="$POPS_PROOF_ENV/bin/python"
POPS_PROOF_OUT=/absolute/new/evidence-directory
rtk proxy env -u PYTHONPATH POPS_ENV_NAME=pops-api040 Kokkos_ROOT="$POPS_PROOF_ENV" POPS_KOKKOS_ROOT="$POPS_PROOF_ENV" CMAKE_PREFIX_PATH="$POPS_PROOF_ENV" POPS_HEAVY_MODULE_TU_POOL=1 CMAKE_BUILD_PARALLEL_LEVEL=4 FI_PROVIDER=tcp bash scripts/build_python.sh --dim 3 --mpi
rtk proxy env -u PYTHONPATH POPS_NATIVE_DIM=3 POPS_REQUIRE_NATIVE_TESTS=1 POPS_HEAVY_MODULE_TU_POOL=1 OMP_NUM_THREADS=1 POPS_THREADS=1 OMP_PROC_BIND=false FI_PROVIDER=tcp "$POPS_PROOF_PY" docs/development/api_040/run_installed_checks.py --output "$POPS_PROOF_OUT/serial" --test tests/python/integration/runtime/test_coupled_gradient_dim3_fourier_runtime.py
rtk proxy env -u PYTHONPATH POPS_REQUIRE_NATIVE_TESTS=1 POPS_HEAVY_MODULE_TU_POOL=1 FI_PROVIDER=tcp "$POPS_PROOF_PY" docs/development/api_040/run_installed_mpi_checks.py --dimension 3 --ranks 2 --threads 1 --timeout 1200 --output "$POPS_PROOF_OUT/mpi2" --test tests/python/integration/runtime/test_coupled_gradient_dim3_fourier_runtime.py
```

Offline replay of the frozen serial evidence uses two externally supplied pins:

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM "$POPS_PROOF_PY" -I docs/development/api_040/check_coupled_gradient_dim3_reception.py "/Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/outputs/coupled-gradient-dim3-sdk20d-saved-serial-20260930" --identity "/Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/outputs/coupled-gradient-dim3-sdk20d-serial-owner-identity-20260930.json" --identity-sha256 5ff0b84abb22189d56d66dac26a00d13e0ba2fd10162652347ca2c636b86405d --manifest-sha256 652357acd9fef343ab0a7a9723ee34a9d520f69d66cbecf7c2f956d116882cbf
```

This qualifies one discrete Fourier witness on Uniform Cartesian geometry.
It does not establish spatial convergence, arbitrary modes/material coefficients,
AMR/EB/moving geometry, GPU, ROMEO, the official Kokkos/OpenMPI matrix or
exact-head GitHub CI. Further independent fully resealed countermodels are
pending separate review. Physical-global detached authority has a separate
known primitive-recipe defect under correction; this fixture has no such ports.

