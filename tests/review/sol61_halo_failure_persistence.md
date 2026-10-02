# Accepted Halo stage failure: persistence extension

Base: eef92c681e4df85497cebda9ec4fe154242fd316, exact Native fixture including independent preflight adversaries. Only fixture and SOURCE tests change; SDK, production, equations and failure request are unchanged.

`failure-proof.json` now contains `divergent_request_applicable` and `divergent_request_refusals`: Serial false/[], MPI true/all-rank refusal tuples. Existing assertions remain mandatory.

After successful unarmed retry and continuous control, all ranks call the genuine runtime checkpoint API, saving `retry-checkpoint.npz` and `continuous-control-checkpoint.npz` (runtime determines final filenames). Rank zero compares every checkpoint member's key, dtype, shape and bytes except precisely the existing MANIFEST_KEY and IDENTITY_KEY lifecycle seals. CP12 and accepted schema9 are required. `retry-control-checkpoint-proof.json` records resolved actual file paths, SHA256 of actual saved bytes, excluded seals and which seals differ. Archives remain the authoritative complete payload, including any available Field/Aux/history/diagnostic members.

SOURCE validation (no Native simulation):

```
rtk proxy env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest --noconftest -p no:cacheprovider -o pythonpath=python tests/review/test_sol61_halo_failure_persistence.py tests/review/test_sol61_halo_stage_failure_source.py tests/review/test_sol61_halo_failure_independent.py -q
```

Result: 15 passed in 3.04s. Synthetic arrays exercise persistence and negative physical-member comparisons only; AST checks authenticate collective runtime/control checkpoint call sites. No checkpoint manifest validity, MPI execution, Native failure, rollback or science reception is inferred from these SOURCE checks. ROOT owns the first ENV10 executions; Banach independently reviews the saved receipts. No simulated receipt is saved into runtime evidence directories.

## Attempt-aligned comparison (@2)

The first true Serial reception of @1 remains FAILED (47.0869s): retry/control have equal 68-member inventories but differ in program_accepted_state, program_accepted_state_source_authority and the two lifecycle seals. The accepted wire differs at 113 bytes, each encoding attempt3 versus attempt2. This is causal provenance, not a physical discrepancy. `amr_program_context_subcycling_runtime.inc` retains allocated attempt authority on rejection (1–4,116); snapshot rollback restores accepted authority while retiring the engine (history_checkpoint_definitions104–117). Reconstruction restores both counters (subcycling_runtime316). Accepted state writes accepted_attempt (checkpoint.hpp1018) and each face fragment attempt (644); source authority v2 hashes the complete serialized accepted state (amr_system5122–5130).

The corrected experiment has three real owners. Continuous control remains uninterrupted and has before/final full checkpoints and raw images. Its physical image must equal retry exactly. Their exhaustive checkpoint divergence must be exactly the four members above; the actual current POPSAND9 wire prefix is decoded, requiring retry accepted_attempt3 and continuous2. No member is normalized or ignored to force equality.

A third attempt-aligned control starts identically, receives the same genuine rank-local prepublication failure, saves before/after images and full checkpoints, requires the same receipts and exact same-owner rollback, then retries without rearming. Its final image equals retry; final full checkpoints must compare byte-exact outside only MANIFEST_KEY and IDENTITY_KEY. The new `retry-aligned-control-checkpoint-proof.json` uses distinct `pops.accepted-halo-test-failure.retry-aligned-control@2`; it does not relabel @1. `aligned-failure-proof.json` preserves raw failures/receipts; `retry-continuous-checkpoint-differences.json` records exhaustive differences, actual attempts and archive paths/hashes. @1 failure evidence remains untouched externally.

Updated coherent SOURCE/host command above: 16 passed in 1.98s. No new Native execution or proof; Source prefix parser/orchestration tests are not simulated Native receipts. Production/SDK unchanged.
