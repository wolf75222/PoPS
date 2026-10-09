# Program frontier and diffusive trace coherence

Independent integration-contract review, 2026-09-30, GPT-6.1 Sol.
This review combines diffusion `b03b47b231776324832a358b8dcb3d4261b53ba5`
and Program Scalar/ComputedDt `ccfa25af07f592638c6a4b09925f116a08543804`
without changing a checkout, branch, shared environment or production header.

`git merge-tree --write-tree` returns tree
`7e3361ffcbb2674b9f46d8df3aa1ff5d9fc00eb8`, exit 0, no textual conflict.
This is a Git **tree object**, not a compiled source commit. Its authenticated
Python/three-header projection is
`7489ade1ea244d5b40f57d56c1c1d1f754f95dcf4fa9a9fd900f2f5c31c859ec`.
The same projection for diffusion-only b03 is
`b122fd064e59e923f7631a23dfaaded144a7d8501b5dd0fdeb903e7bcbf822e1`.

Both public Uniform and AMR diffusion fixtures retain byte-identical generated
C++, Program IR hashes, semantic hashes and exact TraceSelection lines after the
combination. The lines retain operation identity, occurrence 0, accepted
evaluation 1, axis/side/component and weight 1. Program schema remains v6 for
these admitted diffusive selectors. Fixture source SHA-256 is
`955c53e60a09068d7076b14a8d24522428c89887f0c33fb48935ee124599a567`.
The fixture predates this integration review and is identified as a production
extension fixture, rather than a fresh independent diffusion runtime oracle.

| Fixture | IR hash | Generated C++ hash |
| --- | --- | --- |
| Uniform | `6f3bb3b447bb550fc4d822eab3ede7bcc9adfce92a4a3f3ed75cd039122d361b` | `6b29d1b9c86882f94dd519b8d573c4b0d42da8341c8b25a18f27ab3bd9147136` |
| AMR | `6ded69369c3fb1f8bcb95d622d935d02007bc20e0409cea39130f8e938102cf5` | `e1bf40056f1b1df3235e1acd12c52f5829625ee36f31eefb535be878d71a8361` |

Adding a real ComputedDt frontier to the public diffusion fixture retains the
v6 selector serialization but refuses emission explicitly: v1 has no remapping
of spatial interval exchanges. Thus a requested duration cannot silently become
a different accepted duration while consuming the original diffusive amount.
Pure source RK with ComputedDt remains v5 and still passes the complete five
independent frontier math/source/host checks on this combined tree (3.90 s).
The old FixedDt fixture keeps exactly the same v5 IR and semantic identities
reported in `program_frontier_independent_review.md`.

## Region counterexample and the actual refusal boundary

The independent scope probe derives a branch from the real public diffusion
fixture: each callback evaluates `rate(initial_value)` in its own region, both
return an RHS, the branch result is committed, and its actual internal RHS
handles are captured for the two physical trace declarations.

Passing the branch result itself as `rate=` is refused during authoring because
it is not an exact `rhs` or `diffusive_rhs` operation. Passing a captured internal
diffusive RHS is currently accepted during authoring and resolution. Its
provisional serialization is v5, because the conditional v6 detector searches
top-level `_values`. However, emission refuses with
`integral transfer requires an accepted, exact conservative face occurrence`:
the downstream accepted-face proof cannot authorize that nested evaluation.
No native Program is returned, and no persistent checkpoint is produced through
this lowering path. This is a late fail-closed restriction, unlike the earlier
ComputedDt gate defect that emitted an inadmissible Program successfully.

There is no demonstrated silent admission of a nested trace. Earlier owner/
region validation and an explicit authoring diagnostic would improve this
boundary, while preserving all admitted legacy identities. A future version
that admits branch-local trace consumers must make selector version detection,
accepted-face proof, branch-dependent consumption and graph conversion recursive
together; changing the version scan alone is insufficient.

## Reproduction and receipts

From the review checkout:

```sh
rtk git merge-tree --write-tree b03b47b231776324832a358b8dcb3d4261b53ba5 ccfa25af07f592638c6a4b09925f116a08543804
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/freeze_frontier_source.py --checkout . --revision 7e3361ffcbb2674b9f46d8df3aa1ff5d9fc00eb8 --tree --output outputs/frontier-sol61-b03-ccfa-tree-source
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/freeze_frontier_source.py --checkout . --revision b03b47b231776324832a358b8dcb3d4261b53ba5 --output outputs/frontier-sol61-b03-source
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_frontier_diffusion_coherence.py outputs/frontier-sol61-b03-source tests/review/evidence/sol61_frontier_diffusion_b03.json
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_frontier_diffusion_coherence.py outputs/frontier-sol61-b03-ccfa-tree-source tests/review/evidence/sol61_frontier_diffusion_merge_tree.json
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_frontier_program.py --source outputs/frontier-sol61-b03-ccfa-tree-source --receipt tests/review/evidence/sol61_frontier_merge_tree.json
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_frontier_legacy_identity.py outputs/frontier-sol61-b03-ccfa-tree-source
```

Coherence probes exit 0 (b03: 3.53 s, combined: 4.11 s). The checked-in receipts
contain the complete comparison fields and refusal diagnostics. This qualifies
source and lowering coherence only. Real native diffusion/IntegralState,
ComputedDt rollback, MPI, AMR execution and central ABI5 remain integrator-owned.
