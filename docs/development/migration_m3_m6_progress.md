# M3-M6 active evidence ledger

This is the active qualification ledger for the 15 M3-M6 execution leaves. The current integration checkout is 08ff8a6c689fea8b5ba30480343b3ee35524f019 (08ff8a6), with the recent serial-prepare, Newton, execution-context and AMR fixture/report repairs in its lineage. The authoritative no-overlay interaction/provider receipts were executed at source 34a8acc with fixture/test follow-up 3826eb4; the exact periodic SG and combined-face receipts are bounded executions at source 6b2a1c3. Those pins identify the receipts and are not a current-source closure claim. This ledger remains active and does not claim final closure.

The machine-readable per-leaf ledger is migration_evidence/m3_m6_foundation/index.json. Its source contract is migration_m3_m6_contract.md. Acceptance numbering comes from the pinned manifest at /Users/romaindespoulain/Documents/Codex/2026-09-07/infer-the-intended-scope-from-the/notes/migration-manifest.json (SHA-256 b64efb3c33266c769aa58a7ef44ad7dc873867e7dac18792972a5df00f4c676d). Retained receipts are under /Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence.

Each leaf records the lifecycle validate -> resolve -> compile -> bind -> run and all seven evidence levels: representable, validated, resolved, emitted, executed, numerically_checked and performance_characterized. Statuses use pending, pass, fail, unsupported and unavailable. Source assertions and structured refusals remain distinct from runtime evidence; the JSON index carries the exact command, artifact path, provenance, hash and limits for each receipt.

## Current dispositions

| Leaf | Bounded receipt | Open boundary |
| --- | --- | --- |
| M3.1 | Field representation, physical-boundary/storage contracts and immutable review-native field selectors pass | Broader repository/native gate |
| M3.2 | Dim2 variable-scalar and joint-field review-native matrix passes | Broader repository/native gate and wider undeclared matrix |
| M3.3 | First-order FV field transport passes its predeclared order >=0.8 criterion with observed orders 1.3176677704849828 and 1.1117397898166301; current no-overlay provider matrix passes 12 Euler/SSPRK2 rows | Current source/native lineage, CTest and 485-file source gates; same-source distinct-output publication remains an explicit refusal |
| M3.4 | Joint constants and smooth-gradient field consumers pass in the immutable review-native replay | Broader repository/native gate |
| M4.1 | Current no-overlay closure interaction package: publication 13 rows per MPI2 rank, guard/solve 15 rows per rank and rank-one 19 rows, all without failures or skips | Final 1065-entry CTest closure |
| M4.2 | Current no-overlay closure covers six native face rows, rank-one interaction rows and MPI2 derivative/guard/solve rows; the former N16 face issue was a fixture-oracle axes mismatch and is historical | Full multi-component diffusion integration is outside this receipt; CTest closure remains open |
| M4.3 | Current no-overlay MPI2 exact/approximate/FD solve, guard and rollback rows pass 15 cases per rank | Final 1065-entry CTest closure |
| M4.4 | Current no-overlay MPI2 inventory/timing rows pass with the recorded shared/unshared protocol; concurrent-host timing is bounded | Final 1065-entry CTest closure; GPU/device and memory are bounded extensions |
| M5.1 | Immutable Dim2 explicit artifact passes 9 tests with 24 accepted and 3 structured rejected native scenarios | Current source/native proof lineage; combined Euler/SSPRK2 reconciliation is recorded in M5.3 |
| M5.2 | Immutable Dim2 variable-scalar and diagonal rows pass; off-diagonal tensor, AMR and GPU remain bounded extensions | Current source/native proof lineage |
| M5.3 | Exact 6b2a1c3 periodic native receipt passes the combined Euler/SSPRK2 accepted-face matrix over N16/32/64 and both EB refusal cases; maximum face defect 5.69e-15, cell-change defect 3.05e-18 | Current source/native proof lineage; AMR local-h/subcycling and distributed exchange remain bounded extensions |
| M5.4 | Immutable review-native implicit artifact passes the six public cases with no skips or failures | Broader repository/native gate and current source/native proof |
| M5.5 | Exact installed Dim1 SG proof/report passes 6 cases with no overlay in 226.013 seconds, including solved-potential consumption and combined bounds | Current source 08ff8a6 Dim1 artifact/proof and final source/native runtime lineage |
| M6.1 | Solve source/residual/lifecycle checks and finite-budget retry proof pass | Final source/native artifact and wider solve matrix |
| M6.2 | Source/native transaction checks and finite-budget retry publication proof pass | M6.3/M6.4 consumer use and R1 entry review, outside the current 15-leaf gap list |

The authoritative current M4 closure is closure-interactions-final.json (SHA-256 a23b73d1a808c6e0a0c4f5a93a731112d1b9f575566d700c46d92e8fef5ed7d1): source 34a8acc, no Python overlay, and unchanged 966-member package/12-fixture integrity. Its rank-one native batch has 19 passed in 462.23 seconds; its guard/solve batches have 15 passed per MPI2 rank in 235.05 seconds; its publication matrix has 13 passed per rank in 523.99 seconds. The authoritative current provider closure is combined-accepted-exchanges/closure-provider-result.json (SHA-256 649dc3c9d3a504f8210e32d4e29308cabc410246788720f2937b1b0860a97d9d): 12 rows pass without an overlay, with maximum flux defect 2.858824288409778e-15 and maximum per-cell balance defect 1.370499308657458e-18.

The exact periodic receipts remain bounded and are retained with their hashes. periodic-dim1-sg-final-proof.json (SHA-256 c170e251b4c3ca4b7ff4fe60f2d74221f8c79ebdfe9c7f828b7a72e4b7d61708) records 6 passed Dim1 cases in 226.013 seconds without a Python overlay. combined-accepted-exchanges/periodic-native.xml (SHA-256 9985fa32b257baca1e5b247e9486191887b258628e01d657764afbdafc8463cc) records the 4 passed Dim2 combined-face/refusal cases in 120.089 seconds. These receipts close their bounded rows while the current-source lineage gate remains open.

## Active acceptance gaps

- **G1 — final native/CTest inventory.** The 225-target native build is complete with returncode 0 in 1241.053 seconds (final-native-v3-build.json, SHA-256 a35966579db1b11a16a6ef6945162ea074ce473a8de5d3f147b465270a947e22). The exact CTest inventory has 1065 entries: 963 normal, 20 MPI-np1, 49 MPI-np2, 3 MPI-np3, 25 MPI-np4 and 5 standalone. The completed batches report 9 failures and 17 skips in normal, 1 failure in MPI-np1, 3 failures in MPI-np2, 0 failures in MPI-np3, 1 failure in MPI-np4 and 0 failures in standalone. The inventory is complete but not all-green, so repository closure remains open.

- **G2 — final non-MPI Python source gate.** The first exact 485-file current-source command completed with returncode 1 after 562.376 seconds: 20 passed, 8 failed and 247 deselected (closure-source-summary.json, SHA-256 4945fdadeb439768dee2605c3e912c5f8192221d8309dbcf5f8b23465ba2790c). Its log hash is 18ca03b0f3116e319e3eee5c767b838797aa7c919ae5073cb8c321385f2548d7. A separate 472-file diagnostic remainder is running; it is retained as diagnostic evidence and does not close the source gate.

- **G3 — final immutable artifacts and current source/native proof lineage.** The exact 6b Dim1 proof/report and current closure Dim2 proof are authenticated bounded artifacts. The current repository source is 08ff8a6, while the Dim2 closure proof is pinned to 34a8acc and the Dim1 proof to 6b2a1c3; the current Dim1 artifact/proof and final current-source native/runtime lineage are still required. This gate is separate from G1 and G2.

## Downstream status

M6.3 and M6.4 remain pending downstream expansion, and R1 remains pending across the eight representative paths. They are recorded in the JSON downstream section and are not counted among the current 15-leaf acceptance gaps. GPU, AMR, off-diagonal tensor and wider rank/layout/backend cells remain bounded extension limits unless a specific contract item requires them.

Exact command retention, source/native pins, wheel/build hashes, receipt hashes, per-leaf seven-stage records, active gaps and downstream status are maintained in migration_evidence/m3_m6_foundation/index.json.
