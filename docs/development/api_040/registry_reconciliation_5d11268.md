# Registry reconciliation through 5d11268

`contracts.csv` and `corpus.json` describe source integrated through
`5d1126811ff72fbe54d8686682ff2eb1ca6254d1`. Their historical mapping
baseline is `83b2b12`, which retains the failed critical-83 and earlier M17
receipts. Registry schema 3 adds two separately hashed reception manifests:
`evidence/eac92bb/manifest.json` and
`evidence/scientific-eac92bb/manifest.json`. `check_registry.py` verifies their
file hashes, statuses, selected AMR pytest and MPI-rank counts, the 73-row CTest
inventory, scientific receipt identities, and the shared native digest. It does
not rerun any simulation.

The completed installed package was eac92bb, native
`6b5f452e432976a94b69635b96aad1a04c42c8d5e933cad5cf785301089488fb`:

- Seven selected User/principal AMR tests and one MPI2 rank-one invalid-body
  rollback test passed. Two earlier MPI2 runs failed on a diagnostic expectation
  and remain in the manifest. The old critical-83 generated compilation failures
  remain archived, but are no longer the status of these selected routes.
- M06 passed eight homogeneous enthalpy scenarios after the accepted-history
  observation fix; its import and stale-history failures remain archived. M13
  passed eight homogeneous reaction-chain runs. M17/W08 passed canonical and
  reversed eight-step Uniform Dim2 x-dependent Fan–Li15 runs. The scoped
  Maxwell–Cattaneo example passed twelve runs after an external-grid name fix;
  its initial failed attempt remains archived. These receipts do not qualify
  Stefan fronts, streamer radiation, Dim1/AMR Fan–Li, or general Cattaneo
  spatial convergence.
- M08 passed N32/N64 final-state and current-field checks. A later saved-state
  audit proved that its original stage0 diagnostic read the previous history
  slot. M08/W04 remain partial pending a run of the corrected observation and
  independent SSPRK2 stage oracle. The original runner's `passed` status is
  preserved as a fact about that narrower check, not promoted to a complete
  stage/cache claim.

The eac92bb native CTest inventory has 69 run, two failed, and two not run
rows. Both failures concern the three-level C38 second-parent history
transition. Source through 5d11268 contains a later AND9 repair and new
coordinated-face/product work, but the installed eac92bb receipts predate those
changes. Their native reception remains pending. The selected MPI2 rollback
test is distinct from the failed C38 restart transition. No global contract,
GPU, OpenMPI, or later-source acceptance follows from this reconciliation.

Read-only validation:

```sh
python3 docs/development/api_040/check_registry.py
```

Expected: `40 contracts, 28 models, 12 witnesses; 11 historical and 13 scoped
receipt snapshots consistent`. The checker reports inventory and provenance,
not numerical correctness by itself.
