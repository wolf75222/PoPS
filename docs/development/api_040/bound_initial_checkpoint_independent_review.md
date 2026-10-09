# Independent bound-initial checkpoint reception

GPT-6.1 Sol counter-review, 2026-09-30. Production author: separate agent.
Reviewed commit: `c310f10a69b237a0143e53e723ba7043cb046381`, parent
`5e69b3425184446ea2cd318fe2d31d038cca3035`, checkout `PoPS-sol61-initial-checkpoint`.
No production edits, native builds, installation, shared-environment mutation or
primary-checkout mutation were performed by this reviewer.

## Findings and qualified scope

No remaining production defect was demonstrated on the frozen correction.
Independent source lifecycle/envelope probes pass in 0.35 s. A second reception
of the automatically combined production projection with central `ff726dc`
support metadata 2 also passes (0.62 s). This qualifies protocol and source
ordering; it does **not** qualify native initial capture, real public restore,
first PDE run, MPI agreement, AMR execution or central ABI5.

The integrator's earlier installed native log recorded 6/6 IntegralState tests
failing at checkpoint capture with `NoneType.to_data`, before physics. Its
installed initial-control log also recorded the source fixture's missed capture
seam. These are baseline failure evidence, not results of the correction:

- `outputs/installed-integral-native-initial-checkpoint-fc0-20260930/pytest.log`
- `outputs/installed-initial-checkpoint-controls-fc0-20260930/pytest.log`

Both production capture-plan writers now call `checkpoint_lifecycle_evidence`
and include that evidence in the capture identity: Uniform at `_system_io.py`,
AMR at `_amr_checkpoint_v3.py`. Their old direct `last_run_identity.to_data`
expressions are replaced. This source wiring addresses the observed failure
path; the integrator must still receive it against the refreshed installed code.

## Independent probes and identities

`freeze_frontier_source.py` archives tracked Python and three header inputs from
the named Git object. The runner verifies all file hashes before import and
asserts the selected `pops.__file__`. Frozen projections:

| Source | Projection SHA-256 |
| --- | --- |
| Baseline 5e69 | `5993364d4d0f2bba7872037aea7caea4ba85627f0174afe6b2676bae84ca2865` |
| Correction c310 | `5d0fb6a30c199d4c9a855683fe5000d9f2e656d18256ce8cc011f35d6e3f2b9c` |
| Central production merge tree | `1d2ad2eb7a771552ec13bd55750d21ebeac943760a212a02067b5496862e381a` |

The independent fixture uses the actual `_LifecycleMixin`, exact domain
identities, `TemporalRestartState`, real manifest producer/inspect/authenticate
functions, `begin_run`, `_CompositeTemporalRestartState`, resource-budget class,
NPZ decoder and child-origin verifier. Its bind snapshot and ABI equality seam
are deliberately synthetic; no backend is selected or loaded. This is a protocol
fixture, not a simulated native checkpoint reception.

- Uniform and AMR initial envelopes emit v2 with exact `bound_initial` origin and
  null run identity. Repeated capture leaves temporal state, controller and run
  identity unchanged.
- Fourteen raw alterations are rejected. Twelve variants recompute array evidence
  and the restart content identity, proving semantic rejection beyond a checksum
  mismatch: wrong origin/run/schema/clock/controller/transaction count.
- Ten initial clock variants are rejected: signed-zero, noncanonical hex,
  text counter, binary64 counter and bool counter on both runtime kinds. Binary64
  metadata is refused during canonical identity construction; other variants
  reach the initial-envelope validation. Four invalid owner conditions are
  rejected: noninitial native clock, manifest without run identity, unbound owner
  and nonzero attempted-transaction statistics.
- Restoring initial provenance through the actual lifecycle setter clears an
  existing run identity, manifest and continuation lineage. Calling the real
  `begin_run` request function then creates the first run request identity with
  no invented continuation. The following envelope emits legacy v1. This is
  source lifecycle reception, not execution of public `pops.run`.
- Two-layout Uniform and AMR initial children pass the real bounded NPZ decoder,
  collective manifest validator and child-origin verifier. A decoded child with
  an existing run origin is rejected before composite publication.
- Sentinels prove archive-size refusal occurs before NPY header decoding and
  manifest-size refusal before array decoding. The initial manifest fits the
  derived schema budget. Fixture budgets are explicit host capacities; no claim
  is made about native-installed capacities.

Composite child inflation correctly occurs after each child's resource
admission. The outer generic sealer/inspector checks byte-array shape only and
does not inflate child archives. `_MultiLayoutUniformExecutor` prepares all
children, derives the authenticated composite temporal state, invokes
`require_bound_initial_children`, then returns the prepared restart. The probe
targets this admitted-child boundary, avoiding an invalid expectation that the
outer sealer should decode nested children without resource authority.

Legacy v1 bytes and token remain exactly identical across baseline, correction
and central support-metadata-2 projection:

- Manifest SHA-256: `eef0d0c581509257b2bab5678e2c01c4a645ca4254491ab9473fb5eca7edb9c4`
- Restart: `pops.restart.v1:sha256:7d846e571b77eecc88beeb19ff4f031c475e44a80080166117244669e7ccb7f2`

## Integration and reproduction

Direct `merge-tree` of c310 and ff726 reports an add/add conflict only in
`tests/python/unit/runtime/test_initial_checkpoint_strategy.py`. Production
Python automatically merges. Tree `58ffd9028afd51c7a179225d3c91db6697ecd0bc`
contains that unresolved test conflict; the reviewed projection deliberately
excludes tests. This is not a claim that the complete merge is ready. The
integrator must reconcile both copies' final assertions when cherry-picking.

Run from the exclusive review checkout; snapshot outputs must be new:

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/freeze_frontier_source.py --checkout ../PoPS-sol61-initial-checkpoint --revision c310f10a69b237a0143e53e723ba7043cb046381 --output outputs/initial-checkpoint-sol61-c310f10-source
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_initial_checkpoint.py --source outputs/initial-checkpoint-sol61-c310f10-source --receipt tests/review/evidence/sol61_initial_checkpoint_c310f10.json
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_initial_checkpoint.py --source outputs/initial-checkpoint-sol61-before-source --receipt tests/review/evidence/sol61_initial_checkpoint_legacy_5e69.json --legacy-only
rtk git merge-tree --write-tree c310f10a69b237a0143e53e723ba7043cb046381 ff726dc
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/freeze_frontier_source.py --checkout . --revision 58ffd9028afd51c7a179225d3c91db6697ecd0bc --tree --output outputs/initial-checkpoint-sol61-central-tree-source
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_initial_checkpoint.py --source outputs/initial-checkpoint-sol61-central-tree-source --receipt tests/review/evidence/sol61_initial_checkpoint_central_tree.json
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m ruff check tests/review/sol61_initial_checkpoint.py
rtk git diff --check
```

The three checked-in receipts contain exact source and probe hashes and state
the limited host scope. No production repair was needed in this counter-review.
