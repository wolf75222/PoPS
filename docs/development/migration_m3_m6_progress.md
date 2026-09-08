# M3-M6 active evidence ledger

This document is an active qualification ledger for the 15 M3-M6 execution leaves. The current integration source is 6b2a1c3402535209e0169b543e81e5537fec3551 (6b2a1c3), after native combined-face, embedded-geometry admission, fingerprint/projection, and periodic prepared-flux changes. Several immutable receipts retain historical native pins at 24b58985e89b194ebd505672ca0b1e04d75ee4f4 (24b5898) or 14a0f74; those pins identify receipt artifacts and are not the current integration revision. This ledger does not claim final closure.

The machine-readable per-leaf ledger is migration_evidence/m3_m6_foundation/index.json. Its source contract is migration_m3_m6_contract.md. Acceptance numbering comes from the pinned manifest at /Users/romaindespoulain/Documents/Codex/2026-09-07/infer-the-intended-scope-from-the/notes/migration-manifest.json (SHA-256 b64efb3c33266c769aa58a7ef44ad7dc873867e7dac18792972a5df00f4c676d). Retained receipts are under /Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence.

The lifecycle is validate -> resolve -> compile -> bind -> run. The seven evidence levels are representable, validated, resolved, emitted, executed, numerically_checked and performance_characterized. Statuses use pending, pass, fail, unsupported and unavailable. Source assertions and structured refusals remain distinct from runtime evidence. Every leaf has all seven levels in the JSON index with receipt references, command retention state, provenance and limits.

## Current dispositions

| Leaf | Bounded receipt | Open boundary |
| --- | --- | --- |
| M3.1 | Field representation, boundary/storage contracts, and immutable review-native field selectors | Broader repository/native gate |
| M3.2 | D2 variable scalar and joint field matrix; immutable review-native replay passes | Broader repository/native gate and wider undeclared matrix |
| M3.3 | First-order FV field consumer matrix passes its predeclared order >=0.8 criterion; Euler-Poisson orders 2.002789/2.000696; transport orders 1.3176678/1.1117398 | Broader repository gate only |
| M3.4 | Joint constants/smooth gradient field consumers pass in the immutable review-native replay | Broader repository/native gate |
| M4.1 | Publication MPI2 matrix passes 13 rows on each rank in 1572.82 seconds; N16/32/64 and 100-step Heun covered | Final source/native proof and 225-target repository gate |
| M4.2 | Goodall consolidated proof passes six actual face rows, final rank-one native rows, and MPI2 guard/solve rows | Full multi-component diffusion integration is outside this receipt; final source/native proof remains |
| M4.3 | MPI2 exact/approximate/FD, guard and failure rows pass per rank | Final source/native proof and 225-target repository gate |
| M4.4 | Publication MPI2 paired timing rows pass on each rank with 2 warmups and 7 samples | Final source/native proof; GPU/memory are bounded extensions |
| M5.1 | Final immutable Dim2 explicit matrix passes 9 tests, 24 accepted and 3 rejected native scenarios | Combined-face reconciliation and final source/native proof |
| M5.2 | Final immutable Dim2 variable and diagonal rows pass; off-diagonal/AMR/device remain bounded extensions | Final source/native proof |
| M5.3 | D1/D2 restrictions and weighted diffusion exchanges pass; current integration retains native transport faces | Combined N16/32/64 Euler/SSPRK2 per-cell reconciliation |
| M5.4 | Six implicit cases pass in the immutable review-native artifact without a Python overlay | Broader repository/native gate only |
| M5.5 | D1 public SG six-case matrix passes in 255.22 seconds; final proof/report covers solved-potential consumption and combined bounds | Immutable Dim1 artifact and final source/native proof |
| M6.1 | Solve source/residual/lifecycle checks pass | Final source/native artifact and wider solve matrix |
| M6.2 | 79 source, 36 native transaction tests and the finite-budget retry proof pass | M6.3/M6.4 consumer use; MPI2-only rank-divergent selector cells are outside this MPI1 retry receipt |

## Acceptance gaps

- The final native gate covers 225 selected targets against a 1065-entry CTest inventory; its latest attempt was interrupted at `test_program_context_schur_free`, and the broad 475-file Python run must resume after the fingerprint-scope fix.
- M3.3 first-order FV transport meets its predeclared order >=0.8 criterion with observed orders 1.3176678/1.1117398; the review artifact closes the field-specific boundary, while broader repository proof remains.
- The Goodall consolidated M4 receipt supersedes the historical face-oracle failure logs: six actual face rows, final rank-one native rows, MPI2 guard/solve rows and the full MPI2 matrix pass. Its explicit limits retain no claim for full multi-component diffusion integration.
- M5.3 now has native transport-face retention in the current integration, but the combined N16/32/64 Euler/SSPRK2 per-cell reconciliation receipt remains required; four matrix cells are pending, and historical combined probes do not close that receipt.
- Final immutable Dim1 and Dim2 artifacts and final source/native proofs remain required; off-diagonal, AMR, GPU and wider cells are recorded as bounded extensions unless an acceptance item explicitly requires them.
- M6.3 and M6.4 have no integrated receipts, so R1 remains pending across the eight representative paths.

Receipt hashes, exact wheel/build pins, command-retention status, per-leaf seven-stage records and unqualified cells are in migration_evidence/m3_m6_foundation/index.json.
