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

## Corrected freeze ccc77c9 — P1 CLOSED (SOURCE_ONLY)

Independent rereview: expected subjects come from retained ProgramIR commits, read under external inventory hashes and subsequently checked by program_image against Program hash/generated CPP/body expectations. They are not learned from transfer rows. Expected subjects×four operations is complete, keys and per-operation descriptors/order/halo/rank/ratio are exact, provider qualified identity is independently reconstructed from canonical CBOR SHA256 and Handle URI semantics. Source authoring inspected at transfer.py474–524 and model/handles.py111. No physical field names enter this transfer registry.

Replayed the exact reported five source suites: 176 PASS1.46s. Added separate pure independent CBOR encoder/hash checks for all four providers and the original three attacks plus missing whole subject, injected subject and operation-key mismatch:10 PASS0.13s. These tests produce no native/scientific data.

Historical @1/@2 reader bytes compared against66e1885 and unchanged. Private checkpoint globals continue to replace only accepted_contract while retaining the historical code object; old approvals/schemas do not upcast. No new correctness blocker found in the bounded corrected gate. Full resolved-transfer graph hash is syntax/domain authenticated plus immutable ROOT inventory/seals, not independently reconstructed; this boundary is explicit and no stronger provenance claim is made. ROOT still owns real native reception/ABI attestation/external seals.

## Actual integer-control codec freeze ba267dd — CLOSED (SOURCE_ONLY)

ROOT's true bb416 retained IR exposed the historical helper's raw-int assumption. Independently inspected canonical_data.py strict_data and ScalarLiteral.to_data: control integers are exactly {scalar:{kind:integer,value:decimal-string}}, while floating controls are direct binary64 hex images in the retained Program IR. The new @3 program_image compares this exact shape, without permissive integer parsing or modifying historical helpers. IR Program hashing and CPP identity checks occur on the original full semantic IR before inspecting controls; only the established provenance projection is excluded as before.

Independent combined replay:198 author tests plus10 independent review checks =208 PASS2.08s, zero skips. The actual retained-IR positive ran, as did21 rehashed negative mutations (three controls × rawint/bool/delta/leadingzero/wrong-kind/extra/float-value). Rehashing IR and CPP makes these real semantic refusals rather than stale-hash refusals.

Additional read-only actual-receipt loop verifies four unique cases (N8/N16 × width1/2) using receipt-pinned Program IR, generated CPP, recorded Program hash and width. All pass @3. Paths were resolved/deduplicated to exclude the pytest-current directory alias. Native was neither imported nor rerun. Historical @1/@2 bytes again equal66e1885. The interpolation P1 closure above remains intact. No further bounded correctness blocker; this is a decoder/source receipt, not ROOT scientific reception or external approval.
