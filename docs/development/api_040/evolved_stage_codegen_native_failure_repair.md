# Stage generated-C++ repair after installed-native refusal

Source baseline a5b05ff37c601c34f2c84fa6a34655a50c8311ca integrates
production e01c12a7. ROOT's immutable installed SDK2e4 serial Dim2 campaign
rejected all eight Stage cases at GENERATED_CPP compilation, before any
scientific measurement. The original evidence is preserved under
`/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/installed-sdk2e4-evolved-stage-serial-dim2/`.
No donor file or environment was modified.

## Actual causes and minimal repair

1. The duration producer and residual guards called `ctx.step_dt()`, which
   exists in neither actual Uniform ProgramContext nor AmrProgramContext.
   `boundary_evaluation_point(operation_id).dt` is their public issued-window
   duration: Uniform obtains it from current_dt_ set by begin_step and logical
   interval frames; AMR obtains it from its current interval. Stage fraction
   changes neither duration. The emitter now calls this actual API with the
   consuming SSA operation identity. Positivity, finiteness, MPI equality,
   frame/point/attempt/lease checks and authenticated execution lane stay in
   place. A lexical macro dt cannot replace this duration for child windows.
2. Each accumulation publication inserts its capture views after the output
   view declaration. The old unbounded `lines.index` selected the *first*
   publication's declaration while emitting the second Q. It redeclared three
   captures in the first kernel, left them absent in the second, and shifted
   the profiling insertion offset so `_pt9` was not in scope at its record.
   The search now begins at this publication's recorded output start. Each
   capture stays in its own patch/kernel scope, and the existing per-node
   profiling wrapper brackets the correct node. No profiler API change occurs.

The production diff is five Python files, 11 insertions/10 deletions in
commit e15445be7df87a740681092a3822551cb9fdbc17. No C++ header, ABI, wire,
IR version, equation, physical declaration, coefficient order, seven Newton
controls, stopping criterion or scientific acceptance threshold changes.

## Source and compiler reception

* 37 author/source tests pass (34 existing source/math plus three actual
  emission regressions), in 58.24 seconds. The new checks cover both scalar
  and partitioned cases, actual operation-scoped point.dt, exact one capture
  declaration per publication, and visible unique profiling timers.
* Ruff and diff whitespace checks pass.
* Actual newly emitted coupled CPP (51588 bytes), not a hand replacement or
  stub runtime, passes clang++ `-fsyntax-only` Dim2 against the frozen source
  headers and installed dependency includes. It creates no DSO and executes
  no runtime. The same command on ROOT's immutable failed coupled CPP still
  rejects missing step_dt, capture redefinitions and the undeclared timer.
* Three historical Stage profiles have byte-identical complete IR, Module
  hashes/manifests and SolveRequests before/after. Their CPP must change:
  retaining calls to a nonexistent API cannot preserve compilation. The
  historical Stage CPP/failed artifacts remain preserved as red evidence.
* Six non-Stage historical profiles are compared fresh at the same authoring
  call site: Uniform original, captured Uniform, AMR Jacobi, captured AMR
  Jacobi, candidate-D Uniform and candidate-D AMR. Complete IR, CPP, Module
  hashes/manifests and SolveRequests are byte-identical (aggregate SHA256
  dc254843124ffc15378c267cde4cadcc1a50b2b86c157d52d2b42d19889687ec). No emission
  change is taken on branches without TemporalTau/accumulation publication.

The previous independent source/math suite was also rerun without editing its
reviewer-owned assertions: 78 pass, seven fail solely because they require the
literal string `ctx.step_dt()`. Its author will independently receive the
native API correction; those failures are not hidden or described as math
regressions. They show why source substring reception alone was insufficient.

Exact syntax command (no linking, JIT, SDK write or native execution):

    /usr/bin/clang++ -std=c++20 -fsyntax-only \
      -DPOPS_NATIVE_DIM=2 -DPOPS_RUNTIME_SHARED_EXCEPTION_ABI \
      -DPOPS_HAS_KOKKOS -DKOKKOS_DEPENDENCE -DPOPS_HAS_MPI \
      -DPOPS_HAS_PARALLEL_HDF5 \
      -I/Users/romaindespoulain/dev/tmp/PoPS-sol61-stage-codegen-repair/include \
      -I/Users/romaindespoulain/miniforge3/envs/pops-api040/include \
      -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include \
      /tmp/sol61-stage-fixed-coupled.cpp

ROOT must install/authenticate the corrected Python generator and rerun the
same eight native witnesses. This receipt does not assert native solves,
original residuals, Q projection, checkpoint/replay bytes, MPI convergence or
scientific qualification. No MAIN, reception worktree, SDK or ENV mutation was
performed by the author. The independent reviewer receives e154 separately.
