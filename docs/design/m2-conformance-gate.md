# M2 temporal-execution conformance gate

`python scripts/run_m2_gate.py` is the reviewed readiness matrix for the M2
temporal core. It validates the manifest before executing an exact list of
mandatory pytest nodeids and CTest cases. Renaming, skipping or deleting a
supporting proof therefore fails source-only CI instead of silently reducing
coverage. At execution time, a mandatory pytest plugin also turns every
runtime skip, xfail or xpass into a gate failure.

The executable supporting battery covers the landed contracts:

- typed phase pipeline and immutable canonical `ProgramGraph`;
- manual/factory SSPRK2 and IMEX normalization to the same exact graph;
- typed schedules, exact residual operators and real index-1 DAE
  consistent-initialization/refusal behavior;
- explicit `SolveOutcome`, including `RejectAttempt` lowering;
- uniform and AMR step transactions, including topology/state/history/cache/diagnostic/clock rollback;
- emitted refined-hierarchy gather → solve once → publish → synchronize ordering;
- accepted AMR transaction commit across topology, state, history and clock;
- strict shared `TemporalRestartState` round-trip and rejected-attempt checkpoint refusal;
- history restart round-trip and mismatched-program refusal.

The two native example checks are stronger than launch smoke tests. The scalar
check compares the manual SSPRK2 continuation against the
`pops.lib.time.SSPRK2` continuation bit for bit. The IMEX check requires the
manual/preset graph identity and a bit-identical accepted runtime snapshot.
They run in separate process groups so one native abort cannot hide the other
result or corrupt the lightweight Python conformance process. Each has a
30-minute default watchdog, configurable with `--example-timeout`.

Those checks are not sufficient to close ADC-668. The manifest records seven
explicit acceptance gaps: stable native manual/factory SSPRK2 plus IMEX
execution; native `SolveOutcome` fault injection; the complete transaction
fault matrix including outputs and checkpoints; strict next-attempt restart; a
native multi-block implicit phase; a refined hierarchy native oracle; and
deletion of legacy temporal/fallback routes. These are blockers, not waivers.
The default command returns a non-zero status while any `[[deferred]]` row
remains.

Use `--check-only` for the cheap source-only CI integrity proof. Use
`--available-only` to execute the landed battery without claiming completion,
`--python-only` to skip only the CTest stage, and `--build-dir` to select the
CTest tree. Pytest proofs can still compile or execute native code. The
unqualified command is the final fail-closed gate.
