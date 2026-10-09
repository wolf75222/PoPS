# Original isolated self-BGK: installed native reception

Both original cases pass on the installed PoPS package: `32-32-False` and `64-64-True`, two SSPRK2 steps each, with zero failures, errors or skips (147.595 seconds including compilation). Test revision `190cd3c60de7ca8be947beaf745d59df4a5a9eee` retains production `7edddce1255012700fac7d098c94a3bf18d23141`. The native extension is `d725dcc30a411f7782ee290de989143f83a14bf2bca6c8b17579e545a0e6994c`, ABI11/header c190, Apple LLVM, MPI-enabled Dim2, singleton world. All 2,563 physical installed package/distribution members are unchanged before and after the runs. Thread parameters are recorded; effective concurrency was not measured for these tests.

## Equations and realization

The public [collision body](../../../tests/python/support/m19_bgk_case.py) authors the continuous self-Maxwellian from density, first moment and second moment. It uses the original velocity domain `[-8, 8]`, `nu = 3/2` and `dt = 1/128`. No discrete equilibrium fit or moment renormalization is introduced. This isolated relaxation has no transport term.

At each SSPRK2 stage, three explicit product-support reductions compute the velocity moments, then three lifts supply them to the collision body. Phase storage and physical moment storage remain distinct layouts. The reversed case changes block order, rather than selecting a different model recipe in the compiler. Each case retains eight actual Model and two Program source/binary records before bind, plus all six source-compiled transfer components.

The first actual `32-32-False` run compiled these components and then failed before bind because compiled metadata contained digest bytes that plain JSON could not serialize. Commit `190cd3c` uses the existing lossless `json-with-bytes-hex.v1` evidence codec and records its tag. Production code, binary bytes, equations, numerical data, mesh sizes, thresholds and all six source/binary checks are unchanged. This is a test metadata correction, not a new public Python/C++ contract. The original failure remains retained.

## Independent saved-state reception

An independent NumPy calculation starts from each actual initial distribution, first checks its analytic initialization, then recomputes the two SSPRK2 steps without importing PoPS, its provider, compiled Model or existing BGK oracle. It receives:

| Check | Result |
|---|---|
| Distribution error | At most `1.1102230246251565e-16`, against unchanged `2e-11` |
| Reduced/lifted moment error | At most `8.881784197001252e-16`, against unchanged `2e-11` |
| Moment conservation error | At most `9.325873406851315e-15`, against unchanged `1e-12` |
| Positivity and thermal variance | Positive in every saved phase |
| Saved checkpoints | All eight NPY arrays bitwise equal to the two actual CP9 payloads in each phase |
| Time and mappings | Exact clocks; each of six mappings has counters `0 → 2 → 4` |
| Unrelated state | Both sentinel components remain bitwise unchanged |

Original runs, commands and failure (`/Users/romaindespoulain/dev/tmp/root-m19-bgk-current-native-20261006/report.json`), pins `cfb168851453a7fec86a1b7a2062cb102e14db62579e93923244e564fa0814e5`, preserve 559 regular payloads and two literal links. The independent report (`/Users/romaindespoulain/dev/tmp/sol61-m19-bgk-local-independent-20261006/report.json`), pins `5f2f90b0a03d45d1547c2d4170d4e6c24eb97142b9ea8115e42f184089065134`, and ROOT reception (`/Users/romaindespoulain/dev/tmp/root-m19-bgk-local-reception-20261006.json`), SHA `3d16db81a8f76d99cab527bb47e6cf27dad70be75841f0294788d7dd678a263e`, receive this scope.

## Reproduction

Use a checkout of test revision `190cd3c` to reproduce this exact test body. Setup has already run once in the current worktree. After the official incremental build in the `pops` environment, run the original file through the installed-package driver:

```bash
conda activate pops
export POPS_ENV_NAME=pops
export POPS_KOKKOS_ROOT="$CONDA_PREFIX" Kokkos_ROOT="$CONDA_PREFIX"
export POPS_NATIVE_DIM=2 POPS_REQUIRE_NATIVE_TESTS=1
export OMP_NUM_THREADS=2 POPS_THREADS=2 PYTHONDONTWRITEBYTECODE=1
TASK_EVIDENCE=$(mktemp -d /tmp/pops-bgk-reproduction-XXXXXX)
env -u PYTHONPATH -u PYTHONOPTIMIZE bash scripts/build_python.sh \
  --dim 2 --mpi --wheel-dir "$TASK_EVIDENCE/wheels"
env -u PYTHONPATH -u PYTHONOPTIMIZE "$CONDA_PREFIX/bin/python" \
  docs/development/api_040/run_installed_checks.py \
  --output "$TASK_EVIDENCE/native" \
  --test tests/python/integration/runtime/test_m19_bgk_runtime.py
```

The actual command (`/Users/romaindespoulain/dev/tmp/root-m19-bgk-current-native-20261006/corrected/command.json`) uses `conda run -n pops` with explicit Kokkos roots and no `PYTHONPATH`. The driver authenticates installed Python/headers and the exact native extension in the pytest process. Later revisions require their own identity and results.

## Exact remaining scope

No checkpoint restore/replay was exercised. Histories, consumer cursors, Field slots and cache nodes are empty, so this receipt establishes no populated History/Field/cache behavior. Predictors were not directly dumped; stage moments and accepted distributions agree with the independent calculation. C25 records are received, while the complete compiler-to-binary graph proof remains false. Transport, Landau damping, the full BGK/Vlasov–Poisson family, convergence, comparable cost, MPI2, CUDA, ROMEO, 3D and final CI remain separate obligations. No completion of all 94 mission IDs follows.
