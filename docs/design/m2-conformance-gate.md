# M2 temporal-execution conformance gate

`python scripts/run_m2_gate.py` is the reviewed acceptance matrix for the M2
temporal core. It validates the manifest before executing an exact list of
mandatory pytest nodeids and CTest cases. Renaming, skipping or deleting a proof
therefore fails source-only CI instead of silently reducing coverage.

The final executable scope covers the landed contracts without a waiver:

- typed phase pipeline and immutable canonical `ProgramGraph`;
- typed schedules and exact residual operators;
- explicit `SolveOutcome`, including `RejectAttempt` lowering;
- uniform and AMR step transactions, including topology/state/history/cache/diagnostic/clock rollback;
- accepted AMR transaction commit across topology, state, history and clock;
- strict shared `TemporalRestartState` round-trip and rejected-attempt checkpoint refusal;
- history restart round-trip and mismatched-program refusal.

`deferred = []` is normative. The validator rejects any deferred row and requires
positive plus refusal/rollback coverage for ADC-648 and ADC-667 using exact pytest
nodeids or built CTest selectors.

Use `--check-only` for the source-only CI integrity proof, `--python-only` when
no native build is available, and `--build-dir` to select the CTest tree. The
default command is the full local gate.
