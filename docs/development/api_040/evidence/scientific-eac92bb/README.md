# Scientific reception on native 6b5f452e

These are copied execution receipts on the actual installed PoPS package,
Dim2 OpenMP, one thread and one MPI rank. They include the failed attempts;
no native binary, environment or scientific NPZ array is committed. The raw
workspace directories and saved-state hashes remain in the receipts.

| Case | Executed result and exact limit |
|---|---|
| M06 | Eight scenarios pass after correcting two final history reads from working slot 0 to accepted slot 1. The previous failed result is retained. Equations and thresholds are unchanged. |
| M13 | Eight reaction-chain runs pass; maximum state error 5.757e-7, inventory defect 2.887e-15. |
| M08 | Final-state and current-field-versus-stale-field checks pass on 32²/64². A later audit found the saved stage-0 diagnostic reads an older history slot. This archive does not qualify that diagnostic; its correction and discriminating rerun are required. |
| M17 | Canonical and reversed component orders pass: DOP853/Gauss24 error 4.461e-10, exact permutation equality, active nonconservative term 7.564e-3. The earlier postprocessing-index failure is retained. This is an x-dependent Dim2 extrusion, not Dim1/AMR reception. |
| Cattaneo | Twelve runs pass after binding the exact authored external grid name. First-order temporal convergence is measured (orders 1.0037–1.0170), with zero permutation difference. No spatial convergence claim. The earlier grid-name rejection is retained. |

Each `result.json` records the invoked command, before/after import authentication,
unchanged shipped sources, selected native image and backend. Scientific receipts
record the criteria and actual state hashes. Use `run_scientific_checks.py` with
`env -u PYTHONPATH`, the authenticated environment and a new empty output directory.
The native hash is `6b5f452e432976a94b69635b96aad1a04c42c8d5e933cad5cf785301089488fb`;
SDK signature is `dfcd85eba73c963aaceeaf85738296e5a46272e3f76aff1890a2f7850c4b94d8`.

`manifest.json` hashes the copied receipts. `SHA256SUMS` covers the complete bundle
except itself. Verify it from this directory with `shasum -a 256 -c SHA256SUMS`.
These results do not qualify the subsequently integrated coordinated-face, local
product or AND9 changes; those require their rebuilt package and new receipts.
