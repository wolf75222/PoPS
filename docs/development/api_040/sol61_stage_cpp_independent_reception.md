# Independent Stage C++ reception after genuine native compilation failures

Producer fix: e15445be7df87a740681092a3822551cb9fdbc17, parent
a5b05ff37c601c34f2c84fa6a34655a50c8311ca. Review worktree:
`/Users/romaindespoulain/dev/tmp/PoPS-sol61-stage-compile-review`.
No production file, MAIN, native reception checkout, environment, SDK, header,
library or donor evidence was changed by this review. Compilation here is
`clang++ -fsyntax-only`, without linking, native loading, MPI launch or JIT.

## Preserved red evidence and earlier source limits

The actual ROOT campaign
`/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/installed-sdk2e4-evolved-stage-serial-dim2`
contains eight native fixture failures, before numerical solving or an accepted
saved-state proof. Its scalar failed.cpp has three invalid ctx.step_dt calls.
The coupled failed.cpp additionally redeclares capture0/1/2 and uses _pt9
without a declaration. Running the real compiler against these immutable files
and the genuine private headers independently reproduces exactly 3 and 7
errors, respectively; logs remain `/tmp/sol61-stage-scalar-baseline.stderr`
and `/tmp/sol61-stage-partition-baseline.stderr`. This does not relabel the
original eight runtime failures or manufacture positive state evidence.

Review 1c22df20 received 85 source/math cases, explicitly marked native pending.
It did NOT compile the generated Program. Its public compile sentinel stopped
before the first compiler call; textual ctx.step_dt assertions matched an API
which neither real Context provides. The old report's claim of that *native
method* was wrong even though its source/math results and pending execution
scope were explicit. The source checks also did not detect the cross-publication
lines.index bug or resulting profiling offset. Math/IR tests cannot attest C++
well-formedness. The earlier Stage-independent review had the same missing
compilation coverage. Those historical source receipts remain historical;
this gate supplies the missing actual header/type check.

The old additive test assertions are now changed to check the real native point
duration read, retaining all original equation/tau/capture/refusal tests. The
new CLI is opt-in and never substitutes a fake Context class or scalar runtime.

## Duration authority and scope fix

ProgramContext.begin_step stores current_dt_. LogicalIntervalFrame replaces it
with the issued child duration and restores it on scope exit. Its public
BoundaryEvaluationPoint.dt reports that actual current window. AmrProgramContext
and its logical/subcycling guards provide the same current-window field.
There is no public Context.step_dt or Context.dt method. Reusing the outer macro
dt variable would lose subinterval authority.

The five-file producer delta reads
`ctx.boundary_evaluation_point(operation_id).dt` inside the existing collective
error guard, with finite/positive and exact lane consensus. Uniform callback
revalidation retains lane, live attempt and complete point comparison and now
compares that same operation's dt. AMR body and coefficient callbacks compare
their real point.dt; the prepared original operator retains its independent
owner/attempt/level/capture authority. No equations, tau factor, tolerance,
controls, schema, ABI or header were changed.

The publication insertion is bounded by the current publication_start. Both
capture views and profiling declarations therefore remain in their respective
Q publication scopes. The profiler implementation itself is unchanged.

## Independent gate

`tests/review/sol61_stage_cpp_reception.py` emits genuine public programs from
independent witnesses in Uniform and synchronous AMR, one or two actual State
carriers. It calls the real compiler with the real Context/Kokkos/MPI headers.
Baseline a5: four of four syntax compilations fail (Uniform 3/6 errors, AMR
7/9), for the same duration/capture/profiling defects. Closed baseline receipt:
`/tmp/sol61-stage-source-baseline-four-v3/receipt.json`.

Fix e154: four of four syntax compilations pass. Receipt:
`/tmp/sol61-stage-source-fixed-four/receipt.json`.
The final stronger two-carrier witnesses include captured coefficients in Q,
so both publications actually read their capture views; all four pass again:
`/tmp/sol61-stage-source-fixed-captured-four/receipt.json`.
These are header compatibility receipts, not generated CPP-to-DSO link proofs
or native solver/restart/conservation results.

Six independent legacy fixtures are emitted at the same tool/callsite against a
fresh a5 Git archive and fixed e154. Complete IR and C++ digest tuples match:
ImplicitStage, nonlinear-map ImplicitStage, mixed linear, its permutation,
State copy and explicit vector pairing. The JSON files are byte-identical:
`/tmp/sol61-stage-six-parent/legacy-six.json` and
`/tmp/sol61-stage-six-fixed/legacy-six.json`, SHA256
`7705d8609042758d690964931bad5c2086a9ecc56bd6e6c5d98a6065be81b5c2`.
Old EvolvedStage C++ was invalid and necessarily changes; no byte-parity claim
for that generated source is made. The producer separately received its Stage
IR/metadata and broader Module/manifest/request parity; those are not silently
upcast to this review's six IR/C++ comparisons.

## Commands and limits

```sh
env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B tests/review/sol61_stage_cpp_reception.py --source-root /Users/romaindespoulain/dev/tmp/PoPS-sol61-stage-compile-review --out /tmp/sol61-stage-next-syntax --dependency-prefix /Users/romaindespoulain/miniforge3/envs/pops-api040 --openmp-include /opt/homebrew/opt/libomp/include
env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B tests/review/sol61_stage_cpp_reception.py --legacy-only --source-root SOURCE_ROOT --out NEW_OUTPUT --dependency-prefix /Users/romaindespoulain/miniforge3/envs/pops-api040
env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q tests/review/test_sol61_stage_additive_independent.py tests/review/test_sol61_stage_additive_mms_math.py
```

The updated independent source/math selection receives 51 PASS in 39.71 s;
Ruff and staged diff checks pass.

The dependency prefix is read-only; compiler flags and CPP digests are recorded
per case. The CLI directory must be new, and any nonzero syntax exit refuses
its gate. `--emit-only` is explicitly uncompiled and has no syntax qualification.

ROOT owns rebuilt Python/native artifact authentication and the fresh eight
solver, MPI, history, checkpoint, restart/replay and physical conservation cases.
The Stage fixture checkpoint/IR-export @2 patch is a separate review and not
received in this codefix report. No M06/M13 complete-family claim is made.
