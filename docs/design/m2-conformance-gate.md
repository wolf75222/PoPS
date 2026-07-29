# M2 temporal-execution conformance gate

`python scripts/run_m2_gate.py` is the reviewed acceptance matrix for the M2
temporal core. It validates the manifest before executing an exact list of
mandatory pytest nodeids and CTest cases. Renaming, skipping or deleting a proof
therefore fails source-only CI instead of silently reducing coverage. Executable
pytest proofs always run with `POPS_REQUIRE_NATIVE_TESTS=1`; their JUnit report is
also rejected if any skip or xfail survives prerequisite enforcement.

The final executable scope covers the landed contracts without a waiver:

- typed phase pipeline and immutable canonical `ProgramGraph`;
- typed schedules and exact residual operators;
- explicit `SolveOutcome`, including `RejectAttempt` lowering;
- manual and preset SSPRK2/IMEX execution through one normalized `ProgramGraph`;
- one native multi-block implicit phase with exact name-qualified routes;
- uniform and AMR step transactions, including topology/state/history/cache/diagnostic/clock rollback;
- accepted AMR transaction commit across topology, state, history and clock;
- strict shared `TemporalRestartState` round-trip and rejected-attempt checkpoint refusal;
- real history restart round-trip and mismatched-program refusal;
- Program-only Uniform/AMR temporal facades with no compatibility stepper or
  explicit-Euler fallback for unsupported semantics.

`deferred = []` is normative. The validator rejects any deferred row and requires
positive plus refusal/rollback coverage for every declared issue and requirement
using exact pytest nodeids or built CTest selectors.

Use `--check-only` for the source-only CI integrity proof, `--python-only` when
the mandatory pytest half is desired, and `--build-dir` to select the CTest tree.
`--python-only` does not make native pytest prerequisites optional. The default
command is the full local gate.
