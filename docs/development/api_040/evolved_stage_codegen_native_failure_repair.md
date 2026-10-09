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

The separate native-fixture follow-up uses receipt schema
`pops.evolved-stage-native-fixture@2`. Schema @1 has no positive native
reception: all eight installed witnesses failed generated-CPP compilation.
Its checkpoint paths `accepted`, `continuous` and `replay` could also resolve
to the same `.npz` files subsequently written as observations. Historical
failed evidence remains unchanged.

The fixture now requests `accepted-checkpoint`, `continuous-checkpoint` and
`replay-checkpoint`, seals each returned native file immediately with the
bounded reader before taking observations, and checks those hashes again
after all archive writes. Resolved checkpoint and observation paths must be
disjoint, including the initial image. That follow-up retained the original
raw continuous/replay checkpoint comparison; the subsequent continuation
correction below replaces it with an exact authenticated payload comparison.
Each actual compiled Program
component also exports its carried IR through `dump_ir`, with file SHA256
and the same component's `program_hash`; no new builder or emitter supplies
that archive. The eight cases, original Q/source/diffusion equations, seven
Newton controls and predeclared acceptance of 3e-8 are unchanged. This change
prepares authentic future reception and asserts no new native result.

Source-only validation collected exactly eight parametrized native nodes
(`pytest --collect-only`, source Python path, no compilation). An AST probe
checked all three native-call/seal/capture sequences and executed the actual
final guard statements on a disjoint archive image, an overwritten hash and
a colliding path: the valid structure passes and both corruptions refuse.
Ruff and `git diff --check` pass. The source guard probe is an archive-policy
check, not a substitute for ROOT's forthcoming native checkpoint reception.

## Continuation identity and exact checkpoint content

The following installed Stage@2 campaign compiled and ran all eight cases
through two steps, restart and replay, with `same_images` passing. All eight
then refused the fixture's raw checkpoint-file equality check. Evidence is
unchanged under ROOT's
`installed-sdk2e4-evolved-stage-corrected-serial-dim2` directory. This is not
a positive final fixture receipt and does not replace scientific reception.

The difference follows the real lifecycle contract, rather than a state
error. `_lifecycle._restore_checkpoint_run_identity` restores the accepted
source run as `_restart_lineage_identity`. `_run_manifest.begin_run` places
that identity in the replay run's `continuation_identity`; the uninterrupted
run has no restart lineage. `_checkpoint_manifest` seals the entire run
identity into the restart digest. Consequently the final run digest and the
derived restart digest differ legitimately. On each of the eight real pairs,
all payload entries other than the canonical manifest and restart token are
bit-identical (47 for scalar and 74 for coupled cases). Independent integrity
inspection of each envelope passes.

The replacement fixture contract is
`pops.evolved-stage-checkpoint-equivalence@1`. Immediately after each native
checkpoint and hash, it calls `authenticate_checkpoint_payload` against its
actual creator RuntimeInstance and retains that runtime's canonical
`last_run_manifest`, run/restart identities, accepted clock, and exact
semantic/artifact/bind identities. Later comparison authenticates every
payload digest and the full derived restart identity, decodes each retained
RunManifest with its strict native contract, and matches those identities
and clock against the retained creator evidence. Replay must name the
accepted run as continuation and retain the accepted checkpoint as its last
restart. Both final runs must have identical bind, start clock, and complete
controls. Every physical entry must have identical dtype, shape and bytes;
the complete final manifests must be equal except for the authenticated
`run_identity` and derived `restart_identity` leaves. No additional key,
array hash, clock, ABI, artifact or semantic difference is ignored. The
checkpoint-file hashes and disjoint path guards remain unchanged.

Read-only host validation used ROOT's eight real checkpoint triples and
reconstructed the exact RunManifest identity from the original FixedDt
controls, accepted clock and continuation. Thirteen tests pass: all eight
pairs plus twelve corruption classes, including resealed changed physical
value/dtype/shape and foreign clock, identities, lineage, last restart,
controls, run digest and restart token. Corruption copies are written only
to pytest's fresh temporary directory. This host reconstruction does not
claim live-runtime provenance; the forthcoming native fixture obtains that
provenance directly from each creator. The equations, eight cases, seven
solver controls and 3e-8 acceptance remain unchanged.

    env PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH=/Users/romaindespoulain/dev/tmp/PoPS-sol61-stage-codegen-repair/python:/Users/romaindespoulain/dev/tmp/PoPS-sol61-stage-codegen-repair \
      POPS_STAGE_CHECKPOINT_EVIDENCE=/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/installed-sdk2e4-evolved-stage-corrected-serial-dim2 \
      /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q \
      tests/review/test_sol61_stage_checkpoint_continuation.py
