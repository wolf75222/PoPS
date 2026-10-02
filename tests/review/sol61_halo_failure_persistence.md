# Accepted Halo stage failure: persistence extension

Base: eef92c681e4df85497cebda9ec4fe154242fd316, exact Native fixture including independent preflight adversaries. Only fixture and SOURCE tests change; SDK, production, equations and failure request are unchanged.

`failure-proof.json` now contains `divergent_request_applicable` and `divergent_request_refusals`: Serial false/[], MPI true/all-rank refusal tuples. Existing assertions remain mandatory.

After successful unarmed retry and continuous control, all ranks call the genuine runtime checkpoint API, saving `retry-checkpoint.npz` and `continuous-control-checkpoint.npz` (runtime determines final filenames). Rank zero compares every checkpoint member's key, dtype, shape and bytes except precisely the existing MANIFEST_KEY and IDENTITY_KEY lifecycle seals. CP12 and accepted schema9 are required. `retry-control-checkpoint-proof.json` records resolved actual file paths, SHA256 of actual saved bytes, excluded seals and which seals differ. Archives remain the authoritative complete payload, including any available Field/Aux/history/diagnostic members.

SOURCE validation (no Native simulation):

```
rtk proxy env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest --noconftest -p no:cacheprovider -o pythonpath=python tests/review/test_sol61_halo_failure_persistence.py tests/review/test_sol61_halo_stage_failure_source.py tests/review/test_sol61_halo_failure_independent.py -q
```

Result: 15 passed in 3.04s. Synthetic arrays exercise persistence and negative physical-member comparisons only; AST checks authenticate collective runtime/control checkpoint call sites. No checkpoint manifest validity, MPI execution, Native failure, rollback or science reception is inferred from these SOURCE checks. ROOT owns the first ENV10 executions; Banach independently reviews the saved receipts. No simulated receipt is saved into runtime evidence directories.
