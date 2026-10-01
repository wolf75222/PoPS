# SDK2e4 actual build and reception window

The migration remains open. [The snapshot](sdk2e4_native_reception_snapshot.json)
pins the three actual builds, wheel members, source receipts, C++ inventory and
closed Python attempts. Raw evidence is retained in
`/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001`.
It is local evidence, separate from GitHub CI and the original corpus acceptance.

All dimensions share the same C++/header SDK. Dim1 was built at 0298d696;
Dim2/Dim3 and the current common Python were built at e01c12a7. Actual packaging
removes an rpath and may strip symbols: an uninstalled build DSO's hash can differ
from its installed hash. The retained wheel member exactly matches the installed
binary in each dimension. Build receipts retain that distinction.

The actual C++ selection passes 235 of 240 cases with five MPI-only Serial skips.
The N16/N32 original AMR case explicitly selects FullResidualBasisLU@1. Its
N32 active quotient has 144 DOFs and uses 165,888 dense-matrix bytes per rank;
four factorizations assemble 576 columns and perform 1,174 original-F evaluations.
This is a replicated host dense realization; its declared numerical workspace
does not account for allocator, topology metadata or process RSS.

The actual installed-package API/source/math/host/unit selection passes 451 tests.
The separate Serial native campaign passes four diffusion cases and two AMR
diagnostic cases. Two Uniform diagnostic helpers still call an AMR-only accessor.
All eight Stage attempts fail at actual generated C++ compilation, before a
scientific runtime result. Those red receipts remain immutable and are not
overwritten by correction runs. Independent saved-state owner reception and
fresh MPI2 closure are still required.

Independent inspection also found that the diffusion fixture writes observation
NPZs to the same names as its native checkpoints after the actual in-process
restart/replay. Those real test passes establish the in-process comparisons,
but their retained NPZs do not preserve the checkpoint codec/history image for
offline reception. The same fixture retains compiler-owned C++ but does not
dump the compiled component's carried IR. A corrected fixture will use distinct
checkpoint paths and the existing component `dump_ir()` API. No checkpoint or
IR proof is fabricated for these earlier archives.

Reproduce from the native checkout after the repository's one-time environment
setup. The original task setup attempt and its pre-transaction Conda failure are
already retained; do not repeat it during this window. These commands use the
real installed package and clear PYTHONPATH.

```sh
cd /Users/romaindespoulain/dev/tmp/pops-api040-native-reception-20261001
export POPS_TASK_ENV=/Users/romaindespoulain/miniforge3/envs/pops-api040
export PATH="/Users/romaindespoulain/miniforge3/bin:$POPS_TASK_ENV/bin:$PATH"
export CONDA_PREFIX="$POPS_TASK_ENV"
export Kokkos_ROOT="$POPS_TASK_ENV" POPS_KOKKOS_ROOT="$POPS_TASK_ENV"
export CMAKE_PREFIX_PATH="$POPS_TASK_ENV"
export POPS_INCLUDE="$POPS_TASK_ENV/lib/python3.12/site-packages/pops/include"
export PYTHONNOUSERSITE=1 POPS_THREADS=1 OMP_NUM_THREADS=1 OMP_PROC_BIND=false
export FI_PROVIDER=tcp POPS_REQUIRE_NATIVE_TESTS=1 POPS_KEEP_GENERATED=1
export POPS_ENV_NAME=pops-api040 POPS_HEAVY_MODULE_TU_POOL=1 CMAKE_BUILD_PARALLEL_LEVEL=1
env -u PYTHONPATH bash scripts/build_python.sh --dim 2 --mpi \
  --wheel-dir /tmp/pops-api040-reproduction-wheel -- \
  -C cmake.define.CMAKE_EXPORT_COMPILE_COMMANDS=ON
env -u PYTHONPATH POPS_NATIVE_DIM=2 "$POPS_TASK_ENV/bin/python" \
  docs/development/api_040/run_installed_checks.py \
  --output /tmp/pops-api040-diffusion-reproduction-serial \
  --test tests/python/integration/runtime/test_public_captured_diffusion.py::test_public_captured_diffusion_nonconstant_saved_and_exact_replay \
  --test tests/python/integration/runtime/test_public_captured_diffusion.py::test_public_candidate_diffusion_nonconstant_saved_and_exact_replay
env -u PYTHONPATH POPS_NATIVE_DIM=2 "$POPS_TASK_ENV/bin/python" \
  docs/development/api_040/run_installed_mpi_checks.py \
  --output /tmp/pops-api040-diffusion-reproduction-mpi2 \
  --dimension 2 --ranks 2 --threads 1 --timeout 1800 \
  --test tests/python/integration/runtime/test_public_captured_diffusion.py::test_public_captured_diffusion_nonconstant_saved_and_exact_replay \
  --test tests/python/integration/runtime/test_public_captured_diffusion.py::test_public_candidate_diffusion_nonconstant_saved_and_exact_replay
```

Use a fresh output and wheel directory for each run. Select dimension 1 or 3 in
the build command to reproduce the sibling builds. No GPU, ROMEO, official
OpenMPI/Kokkos4.4, full scientific corpus or exact-head CI result follows from
this local CPU/MPICH window.
