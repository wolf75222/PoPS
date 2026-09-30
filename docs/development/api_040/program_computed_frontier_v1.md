# C22 Program Scalar computed frontier, version 1

The public authoring contract is `ComputedDt(requested_dt, endpoint_ulps=0,
shrink=.5, max_rejections=0)`, `Program.requested_dt()` and
`Program.reached_duration(duration_scalar)`. A duration must be an owned top-level
Scalar, declared exactly once and consumed by the installed ComputedDt strategy.
The canonical strategy and node manifests version both the computation and the
interval rule. The Program computes its effective duration in native C++, using
ordinary scalar arithmetic and collective reductions. A Program Scalar can now
be broadcast into the existing pointwise State expression kernel; no Python
per-cell or endpoint callback computes the candidate.

For the contract's relaxed Heun rotation, the ordinary source is `(-y,x)` and the
unrelaxed increment is `d = .5*h*(f(U)+f(U+h*f(U)))`. The Program computes
`gamma = -2*dot(U,d)/dot(d,d)`, returns `U+gamma*d`, and declares `gamma*h`.
The request `(U,h)=((1,0),1)` therefore returns the independently verifiable state
`(.6,.8)` and effective duration `.8`. Neither gamma nor that endpoint is
hard-coded in the runtime.

The existing native CadenceContinuation owns this result. A finite positive
Scalar duration produces `reached = accepted_entry + duration`; the result must
strictly advance the native coordinate. The prepared final coordinate changes,
while earlier source/stage evaluations retain the requested-step coordinates.
One public macro-step is then published by the existing transaction owner.
The continuation compares entry, requested duration, computed duration and
reached coordinate exactly across MPI ranks before accepted publication. The
controller independently checks the native time/step against the returned
reserved result and the run frontier. An explicitly authored endpoint_ulps bound
allows only those binary64 neighbours of the target; zero is exact. The clock is
never reset or relabelled. A shortened duration does not implicitly finish a run
unless it reaches the declared frontier under that public bound.

The version 1 interval rule is `no_spatial_exchanges`. Uniform cadence 1/1
Programs can return a Scalar frontier; spatial flux/diffusive/principal/coupled
interval outputs, persistent transfers, AMR, nested clocks, held scheduler caches,
post-synchronization and cross-layout exchanges are refused during preparation.
This is an explicit capability refusal, not invented exchange rescaling. The
reached-duration effect also breaks same-invocation field reuse. Existing earlier
history stores retain their state and entry identity, but their pending outgoing
duration and sample interval bits become the effective duration before rotation.
A later history store after the declaration is refused. The accepted last-dt and
context/history/diagnostic images use the ordinary runtime rollback snapshot.

A declared RejectAttempt can request another genuine native Program evaluation
with `requested_dt *= shrink`, bounded by max_rejections. Each retry runs inside
the existing RuntimeInstance transaction. The controller's detached retry cursor
alone survives the failed attempt. A terminal failure or a computed overshoot
refuses acceptance and leaves no computed-frontier receipt for that attempt.

The accepted strict restart receipt contains the requested/effective durations,
entry/reached coordinates, run limit and rejection count. Validation binds it to
the canonical ComputedDt policy, exact repeated shrink proposals, accepted native
clock, last accepted displacement and endpoint bound. An accepted computed step
cannot omit that receipt. ExternalTimeGrid version 2 validation also binds its
receipt index/start to the preceding declared grid point and checks last-dt;
forged index, previous interval, last-dt and missing receipt are refused live and
on restart before another native step.

## Evidence and reception obligations

Source/host selection: 118 tests passed, covering the existing exact grid and
strategy cases, adversarial computed-grid restart/live receipts, Scalar source
emission, DCE/CSE retention, generic broadcast expressions, ComputedDt restart and
synthetic MPI preflight comparisons. Ruff and git diff --check pass. These tests
select the installed Dim2 extension, but import Python sources from this checkout;
they do not authenticate a rebuilt extension or establish a PDE trajectory.

Integration target: `tests/python/integration/runtime/test_computed_program_frontier_runtime.py`.
It uses public compile/resolve/bind/run on the native relaxed rotation, a NumPy
algebra oracle, checkpoint/restart continuation, a genuine compiled guard with
one shrinking retry, zero/negative Scalar refusal, overshoot rollback, and MPI2
rank-local run-limit disagreement. The fixture explicitly authors four endpoint
neighbours for repeated reduction rounding; its state oracle thresholds remain
unchanged. Run it after rebuilding/reinstalling both native dimensions and
re-authenticating the headers/extension; run serial and MPI2 with the existing
timeout harness. The worker has collected/emitted these fixtures, not executed
native compilation, JIT, MPI or GPU reception. General C22 multi-clock frontiers,
computed-endpoint AMR exchanges and the full T4/T5 mission remain open.
