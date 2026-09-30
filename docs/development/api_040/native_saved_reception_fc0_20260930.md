# Retained native reception, 30 September 2026

`evidence/native-fc0-20260930/manifest.json` freezes 161 payloads from ten actual
receptions. The native scope is **ABI4, SDK fc0bfd9…**, Kokkos 5.2.0 CPU and
MPICH 4.1.2 over TCP on the local Apple LLVM 21 build. Dimension 1 and 2 module
digests are recorded separately. The archive preserves failures alongside passing
states. It qualifies its recorded source and installed module images, not a
subsequent ABI5 build, the official OpenMPI matrix, GPU or ROMEO.

The receipt manifest and offline checker use **schema 2**. Independent review of
the preliminary schema-1 checker found seven false positives: omitted state
inventory, an unanchored ND2 initial condition, nonfinite face data and inconsistent
native, state and JUnit metadata. The frozen historical checker and adversarial
report are retained in `evidence/native_saved_reception_checker_224bbd6.py` and
`native_saved_reception_independent_review.md`. Version 2 requires every declared
scientific NPZ exactly once; finite typed ledger values; prescribed initial
conditions; native/SDK/source identity links; saved-state hashes; JUnit counts;
rank result links; source snapshot hashes; and MPI before/after installation
consistency. These are consistency and equation checks of retained receipts;
the archive does not embed the native shared libraries or a signature.

The independent recomputation imports only NumPy and the standard library:

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python \
  docs/development/api_040/check_native_saved_reception.py \
  docs/development/api_040/evidence/native-fc0-20260930 --recompute
```

It receives **18 actual saved states**: six M26 measured finite interactions,
ten M27 original linear mixed pairs and two ND2 SSPRK2 tensor-face cases. Two
M27 states are N16/N32 from the first run that subsequently failed at N64 with
GMRES restart 30. Their equations can be checked independently; the outer
reception remains failed. The later N16/N32/N64/permutation reception uses restart
128 without changing physics, tolerances or guards. M26 is twelve measured DOFs
at an actual cell, not a spatial aggregation PDE. M27 is the original linear
mixed system, not the nonlinear double-well case. ND2 retains 1536 face incidence
records per case, including both constitutive occurrences and both SSPRK2 stages.

The checker recomputes M27 cell means, both mixed equations, time, mass and
energy; M26 quadrature, potential, pairings, adjoint defect and energy increment;
ND2 prescribed cell means, full vector update, each face flux/area/time weight,
integrated amount, retained increments and global cancellation. Its benchmark
constants are explicit scientific fixtures. No production runtime is replaced
or invoked by these calculations. Eighteen passing recomputations are separate
from the ten outer reception statuses in the manifest.

Reproduction of current native fixtures requires a real installed PoPS build:

```sh
env -u PYTHONPATH POPS_ENV_NAME=pops-api040 \
  Kokkos_ROOT=/Users/romaindespoulain/miniforge3/envs/pops-api040 \
  POPS_KOKKOS_ROOT=/Users/romaindespoulain/miniforge3/envs/pops-api040 \
  POPS_HEAVY_MODULE_TU_POOL=1 CMAKE_BUILD_PARALLEL_LEVEL=4 \
  bash scripts/build_python.sh --dim 1 --mpi
env -u PYTHONPATH POPS_NATIVE_DIM=1 POPS_REQUIRE_NATIVE_TESTS=1 \
  FI_PROVIDER=tcp OMP_NUM_THREADS=1 POPS_THREADS=1 OMP_PROC_BIND=false \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python \
  docs/development/api_040/run_installed_checks.py --output /tmp/pops-m26-m27-new \
  --test tests/python/integration/runtime/test_api040_m26_finite_interaction_runtime.py \
  --test tests/python/integration/runtime/test_api040_m27_mixed_linear_runtime.py
env -u PYTHONPATH FI_PROVIDER=tcp OMP_NUM_THREADS=1 POPS_THREADS=1 OMP_PROC_BIND=false \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python \
  docs/development/api_040/run_installed_mpi_checks.py \
  --dimension 1 --ranks 2 --threads 1 --timeout 1200 --output /tmp/pops-m26-m27-mpi-new \
  --test tests/python/integration/runtime/test_api040_m26_finite_interaction_runtime.py \
  --test tests/python/integration/runtime/test_api040_m27_mixed_linear_runtime.py
```

Use a fresh output directory and supply the installed Kokkos/include roots as
in the repository environment. For ND2 build `--dim 2 --mpi` and select
`test_coupled_gradient_dim2_fourier_runtime.py`. These commands receive the
current checkout's installed package; recreating the archived ABI4 image requires
its recorded source commit, diff and build inputs. The runner refuses source/SDK
drift before executing tests. A later ND2 MPI launch in this archive was refused
for precisely that reason and has no passing rank results. Serial captures
authenticate installation before execution; only the dedicated MPI runner
claims authentication both before and after. The original OFI finalization
failure and the subsequent explicit TCP results remain available.
