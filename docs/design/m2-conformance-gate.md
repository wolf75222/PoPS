# M2 temporal-execution conformance gate

`python scripts/run_m2_gate.py` is the reviewed acceptance matrix for the M2
temporal core. It validates the manifest before executing an exact list of
mandatory pytest nodeids and CTest cases. Renaming, skipping or deleting a proof
therefore fails source-only CI instead of silently reducing coverage.

The current executable scope is intentionally limited to the landed contracts:

- typed phase pipeline and immutable canonical `ProgramGraph`;
- typed schedules and exact residual operators;
- explicit `SolveOutcome`, including `RejectAttempt` lowering;
- uniform step rollback with no state, history, cache, diagnostic or clock publication;
- history restart round-trip and mismatched-program refusal.

ADC-648 (atomic AMR/regrid transaction) and ADC-667 (multirate history and
restart) remain explicit `[[deferred]]` rows with close conditions. They are not
counted as executable coverage, and the manifest validator rejects any attempt
to list them as completed checks.

Use `--check-only` for the source-only CI integrity proof, `--python-only` when
no native build is available, and `--build-dir` to select the CTest tree. The
default command is the full local gate.
