# M3-M6 active evidence ledger

This document is an active qualification ledger for the 15 M3-M6 execution leaves. It records bounded receipts at integration source 24b58985e89b194ebd505672ca0b1e04d75ee4f4 (24b5898) and does not claim final closure.

The machine-readable per-leaf ledger is migration_evidence/m3_m6_foundation/index.json. Its source contract is migration_m3_m6_contract.md. Acceptance numbering comes from the pinned manifest at /Users/romaindespoulain/Documents/Codex/2026-09-07/infer-the-intended-scope-from-the/notes/migration-manifest.json (SHA-256 b64efb3c33266c769aa58a7ef44ad7dc873867e7dac18792972a5df00f4c676d). Retained receipts are under /Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence.

The lifecycle is validate -> resolve -> compile -> bind -> run. The seven evidence levels are representable, validated, resolved, emitted, executed, numerically_checked and performance_characterized. Statuses use pending, pass, fail, unsupported and unavailable. Source assertions and structured refusals remain distinct from runtime evidence. Every leaf has all seven levels in the JSON index with receipt references, command retention state, provenance and limits.

## Current dispositions

| Leaf | Bounded receipt | Open boundary |
| --- | --- | --- |
| M3.1 | Field representation, boundary/storage contracts, native foundation/preflight | Final field-specific publication artifact |
| M3.2 | D2 variable scalar and joint field matrix; 2 overlay tests pass | Immutable replay and wider matrix |
| M3.3 | Three-test/nine-row field consumer matrix passes; Euler-Poisson orders 2.002789/2.000696; transport orders 1.3176678/1.1117398 with finest L2 9.98748e-05 | Final integrated replay and strict transport-rate qualification |
| M3.4 | Joint constants/smooth gradient overlay N16/32/64 | Final integrated joint native replay |
| M4.1 | Publication MPI2 matrix passes 13 rows on each rank in 1572.82 seconds; N16/32/64 and 100-step Heun covered | Final integrated native artifact and broader advertised matrix |
| M4.2 | Authenticated wheels; imported derivative routes execute | N16 face oracle fails by 1.98245226e-05 at atol 1e-11 |
| M4.3 | MPI2 exact/approximate/FD and failure rows pass per rank | Final package and broad matrix |
| M4.4 | Publication MPI2 paired timing rows pass on each rank with 2 warmups and 7 samples | GPU, memory, final artifact and broader timing matrix |
| M5.1 | Explicit scalar rows, restrictions and exchange ledgers | Immutable Dim2 9-test gate |
| M5.2 | Variable and diagonal rows; diagnostic 4-pass native run | Final package and off-diagonal/AMR/device cells |
| M5.3 | D1/D2 restriction and weighted diffusion-exchange receipts | Combined transport-plus-diffusion RHS ledger omits transport faces; AMR local-h/subcycling and distributed exchange |
| M5.4 | Six implicit cases pass on a08 native plus four Python patches | Unchanged wheel and broader implicit matrix |
| M5.5 | D1 public SG six-case matrix passes in 255.22 seconds; final proof/report covers solved-potential consumption and combined bounds | Final integrated SG artifact and broader matrix |
| M6.1 | Solve source/residual/lifecycle checks pass | Final artifact and wider solve matrix |
| M6.2 | 79 source, 36 native transaction tests and retry proof pass | Skipped rank-divergent cells and M6.3/M6.4 |

## Acceptance gaps

- 178 C++ targets are selected, but retained native build candidates stop in compilation; the broad 475-file source run has no completion receipt.
- M3.3 consumer execution now passes the retained three-test/nine-row matrix, including the field-transport route. Observed transport orders are 1.3176678/1.1117398, so strict second-order qualification and a final integrated replay remain open. The authoritative Case registry refusal is a separate diagnostic.
- M4.2 retains a completed N16 face runtime, but its oracle fails. Runtime completion does not turn that result into a numerical pass.
- M5.3 exchange receipts cover staged diffusion only. Exact transport-face capture and Euler/SSPRK2 per-cell reconciliation are still required for the combined RHS before AMR or distributed qualification.
- The final immutable Dim2 explicit diffusion receipt is still running; diagnostic variable/tensor receipts retain their overlay limits.
- M6.3 and M6.4 have no integrated receipts, so R1 remains pending across the eight representative paths.

Receipt hashes, exact wheel/build pins, command-retention status, per-leaf seven-stage records and unqualified cells are in migration_evidence/m3_m6_foundation/index.json.
