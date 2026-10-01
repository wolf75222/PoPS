# Independent reader @3 counter-review — SOURCE_ONLY

Author freeze 66e18852, detached private review checkout. No production edit, native import/build/JIT or environment mutation. Existing focused v3 suites: 40 PASS in 0.75 s. This is not a native receipt.

## P1 — incomplete interpolation provenance authentication

`accepted_contract` validates aggregate maximum halo=(2,2), maximum lookahead=1 and some per-row bounds. It does not require complete operation coverage for each subject, bind key operation row[1] to serialized route operation row[9], or authenticate the route's space/centering/representation/storage (columns5–8). It allows order1 for a prolongation whose authored witness uses order2 as long as another operation supplies lookahead1.

Pure SOURCE_ONLY probes based on the author's source_contract, each independently accepted by the frozen reader:

- Remove the restriction row entirely.
- Change prolongation order from2 to1.
- Change prolongation representation to `foreign`.

Actual `AmrSystem::checkpoint_transfer_routes` serializes all these columns (CPP22525). Maximum support is a geometric union requirement, not an exact interpolation/provenance oracle. Resealing modified records cannot establish the missing scientific contract. Before receiving @3, derive the complete per-subject operation inventory and exact per-operation properties from authenticated authored registration/IR; do not create field-name recipes. Add these three refusals and operation-key mismatch as genuine negative checks. Keep historical schemas isolated.

## Reviewed without blocker

FunctionType checkpoint reuse has a distinct globals dictionary with only accepted_contract replaced and the original code object; historical v2 callback/module remains intact. Schema8, exact tag-selection profile, integer ABI6 module/capability/release assertions and external @3 approval scope are explicit. Candidate assembly remains pending/nonapproval. Legacy owner pins/approval are refused instead of silently upcast. CP12 full-carrier and prior physical/body checks remain in the reused checkpoint code.

## Commands and principles

`env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest -q --noconftest -p no:cacheprovider tests/review/test_sol61_evolved_stage_amr_saved_reception_v3.py tests/review/test_sol61_amr_owner_candidate_v3.py`

Principles1.1/1.3/1.8 → authored interpolation semantics are authority → reader accepted_contract/CPP checkpoint_transfer_routes → three independently accepted corruptions above → pure source probe → BLOCKED for scientific reception until fixed. Principles1.6 → real native files plus external ROOT seals still required; SOURCE_ONLY admission is not runtime proof. Principle1.7 → no cost claim from this review.
