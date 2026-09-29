# C22 bounded decision: exact ExternalTimeGrid frontier, version 1

This decision applies to the existing `ExternalTimeGrid` controller only. It does
not establish expression-valued endpoints, relaxed RK output-interval semantics,
independently progressing clocks or all of T5. The C22 general gap remains open.

## Reproduced failure

On the pre-fix source, a public `ExternalTimeGrid("grid")` with controls
`(0., 1.e-18, 2.e-18)` and requested `t_end=5.e-19` accepts a native interval to
`1.e-18`. The controller's `4*ulp(max(1,abs(left),abs(right)))` comparison also
accepts an undeclared nearby entry time and allows a one-ULP overshoot at ordinary
and large scales. These are distinct authored instants, not an error estimate.

The independent initial controller/publication tests produced 9 failures and
5 passes: eight behavioral failures and one diagnostic mismatch (the existing
temporal envelope already rejected a double macro-step). The subsequent finite
loss-of-endpoint test additionally covers `-1 -> 1.e-18`, where finite subtraction
and addition land at zero instead of the requested point.

## Contract and implementation

The controls already detach a finite strictly increasing sequence of binary64
times. Numeric equality now identifies the entry point; distinct adjacent floats
are never merged. The next point must not exceed the requested run frontier.
The interval must be positive, finite and satisfy `now + (next - now) == next`.
Failure refuses execution; the controller does not relabel a rounded native time
or replace an authored grid point. Signed zeros denote the same physical instant
and cannot form two strictly increasing grid points.

After the native step, both native time and macro-step must equal the prepared
next point and entry macro-step plus one. This check runs inside `_native_attempt`
before `TemporalRestartState.accept`, using its existing collective error vote.
The enclosing `RuntimeInstance` transaction owns state/time/history rollback on
all ranks before any accepted publication. The direct controller seam supplies no
extra native rollback and is not advertised as a replacement transaction owner.
An invalid future frontier may be detected after earlier valid grid points have
already been committed; this change does not make an entire run atomic.

No output-schedule tolerance, FixedDt rounding rule, native kernel, numerical
method or scientific threshold changes. Existing every_dt handling of `.1,.2,.3`
continues to pass.

## Executed evidence and pending native reception

With source from this checkout and the installed Dim2 extension used only for the
existing Python test fixtures:

- 15 new unit cases pass: tiny/subnormal/ordinary/large/adjacent grid points,
  undeclared entry, overshoot, nonrepresentable interval, and wrong native
  time/macro-step through the actual Python publication envelope with a test
  executor. The latter restores state, time, temporal envelope and publication.
- 78 total targeted cases pass: the new file, all `test_step_strategy.py`, and
  the four existing ExternalTimeGrid/every_dt cases.
- The native integration test is collected, not executed. Ruff passes.

The first broad `-k grid` invocation also selected an unrelated existing restart
fixture, `test_regrid_restart_derives_distinct_run_identity_from_global_receipt`.
It failed because its SimpleNamespace lacks `_consumer_cursor_authority`; the
39 other selected cases passed. No production or test edit hides that failure.

Central native reception target:
`tests/python/integration/runtime/test_external_grid_frontier_runtime.py`.
Run serial and MPI2 with the runner's timeout. It compiles a public generic
source-balance Program, refuses a tiny undeclared frontier without mutation,
checks `.1,.2,.3` against a forward-Euler cell oracle, then injects a wrong clock
observation only on rank 1 (rank 0 in serial) **after a genuine native step**.
All ranks must reject, restore native state/time and the temporal envelope, and
successfully retry with the original target. The injected observation is a fault
seam, not a second solver or a claim that the native backend normally mislands.
No native compile, JIT, MPI run or GPU test was executed by this worker.
