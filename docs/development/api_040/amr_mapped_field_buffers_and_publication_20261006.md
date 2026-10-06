# AMR consumed Field: solve-owned buffers and publication barriers

Production fix: `0d15499906093ee01217667dfdfa09de86a262c4`.

The original installed M19 test reached its initial checkpoint on node-local XFS, then failed at the first temporal step on ROMEO Source92/Native035: the mapped source buffer was allocated through a State-block scratch service even though its prototype belonged to a consumed Field solve. The C++ owner check correctly rejected that buffer. Emission now obtains both mapping candidate and status storage from the exact solve's hierarchy scratch service. Destination State imports retain their existing State-owned allocation.

After that repair, the real local first-step arrays exposed a second defect: a destination layout with a mapped Field publication and no local solve staged components without collectively publishing them before its provider-backed RHS. Qualified top-level Field publications now participate in the existing hierarchy continuation and publication barrier. Equations, model names, constants, thresholds, the C++ owner guard and all five original numerical oracles are unchanged. This fixes lowering of existing operations; it adds no public Python/C++ contract or primitive.

The current installed local wheel passed 74 affected Source/codegen tests and both original installed M19 orderings (2 tests, zero failures/errors/skips,176.925s). Independent recomputation from the actual five CP12/NPY phases confirms two accepted steps, topology changes0→2 and1→3, and full saved-State/cursor rollback after a nonfinite vote. The retained installed inventory (1164 files) is identical before/after. Local Native SHA `a15fe6c2d6180d0ce905d37a4dbc4ffa400978138b355caa9065094e0fcc3b7d`, ABI11/headerc190, AppleClang Dim2/MPI. Rank count1 is recorded by the original test. `OMP_NUM_THREADS=8` and `POPS_THREADS=8` are recorded configuration; effective local concurrency was not retrospectively measured.

[Closed author evidence and reproduction](/Users/romaindespoulain/dev/tmp/sol61-amr-field-buffer-production-fix-evidence-20261006/README.md), pins `0bee31c80afffe9a41f5419d8c8c22c3c682417c782026ab38e87cd72aef28a0`, binds the patch, commands, wheel, installed inventories and2518 physical evidence files. [Independent review](/Users/romaindespoulain/dev/tmp/sol61-amr-mapped-field-publication-independent-r2-20261006/report.json), pins `9aa485b5902032a1106f52fd0371245df7fd3c145ca4be574e0a5f5f7146ead1`, rechecks the actual saved arrays and exact source/package joins. The previous local package and its older Native variants were preserved during environment repair.

Reproduce from this checkout with the project's `pops` environment:

```bash
export POPS_ENV_NAME=pops
env -u PYTHONPATH -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 bash scripts/build_python.sh --dim 2 --mpi --wheel-dir /tmp/pops-amr-field-fix-wheels
env -u PYTHONPATH -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 POPS_NATIVE_DIM=2 "$CONDA_PREFIX/bin/python" -m pytest -q tests/review/test_sol61_mapped_field_route.py tests/python/unit/runtime/test_mapped_field_release_admission.py tests/python/unit/fields/test_program_field_problem.py
env -u PYTHONPATH -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 POPS_NATIVE_DIM=2 POPS_REQUIRE_NATIVE_TESTS=1 OMP_NUM_THREADS=8 POPS_THREADS=8 "$CONDA_PREFIX/bin/python" -m pytest -q tests/python/integration/runtime/test_m19_amr_consumed_runtime.py --basetemp=/tmp/pops-amr-field-fix-native-data
```

Activate `pops` before these commands; use a fresh output directory and a filesystem supporting the existing atomic checkpoint operations. `scripts/setup_env.sh --cpu` ran once in this worktree; the build uses the repository's incremental flow. The closed actual command record contains the interpreter path, environment and output paths used for the received runs. These tests import the installed package, whose production emitter files and Native match the built wheel.

Limits remain explicit: ROMEO735043 ran the previous Source92/Native035 and failed before its first accepted step; its result is not qualification of this fix. ROMEO735054 separately received MPI world1/OpenMP CPU concurrency2 on Native035, without physics. Rebuild/install and the original AMR campaign on the new production revision,27 native C++ cases, MPI2, GPU,3D, final-revision CI, independent analytic initialization, full PDE convergence and comparable costs remain pending. No broader94-obligation completion follows from these local results.
