# Independent Source review of accepted-halo preparation

Reviewed author freeze `fb530d5f804990f12502c51ec79638951e78704e` (baseline `a2408d6f`), in a separate worktree. Source approval is bounded to its documented opt-in bootstrap and aligned synchronous/subcycled publication route. No Native, JIT, MPI runtime, device, ENV, Main or external seal was changed. Ordinary standalone regrid/rebalance is explicitly refused by this freeze and is not received here.

No blocking Source defect found. The engine validates all block/level candidates and stages the complete matrix before its first publication. The generic lambda preserves legacy per-level staging and invokes the whole-matrix callback with const clock/history matrices. The callback uses macro dt times the actual rational history phase width, authenticates the ending clock, and creates each level point; it does not manufacture a positive bootstrap interval. Full-grown extent is an explicit global opt-in constraint, not an inferred getter readiness claim.

The system preflights all candidate contracts and typed boundary field points, dirty auxiliary authorities, exact storage extents, and request identity before mutation. It allocates full accepted field backups under the collective failure vote. It temporarily stages the complete candidate tower for ancestry reads, then prepares every level/block, restores accepted storage, and restores candidate pointer bindings through RAII before publication. Inner preparation also restores its own live level and ancestor staging before its collective failure vote. The outer catch restores the accepted full fields before propagating preparation failure. Publication exceptions retain the existing engine accepted snapshot restore. These are source causal findings; MPI allocation failures, callback failures and restoration failures still require genuine runtime fault injection. Histories are separate and are not halo-prepared by this request.

Typed external GhostBoundary ABI1 requires positive dt and is therefore refused for bootstrap dt0 before bootstrap snapshot/staging. Later field dependency plans must match candidate time, dt, step, stage, substep, phase and clock exactly. Bootstrap is constrained by existing accepted time0/macro step0 guards. Default accepted8 remains distinct from requested accepted9; ABI7 advertises the request while CP12/carrier codec1 remain unchanged. No scientific qualification follows from these checks.

## Independent validation

`rtk proxy env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops/bin/python -m pytest --noconftest -o pythonpath=python tests/python/unit/amr/test_accepted_halo_preparation.py tests/python/unit/runtime/test_amr_bind_lowering.py tests/python/unit/runtime/test_amr_checkpoint_contract.py tests/python/architecture/test_release_contract.py -q --tb=short`

Result: **119 passed in 13.38s**, zero skip.

`rtk proxy env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops/bin/python -m pytest --noconftest -o pythonpath=python tests/review/test_sol61_accepted_halo_callback_independent.py -q --tb=short`

Result: **1 passed in 1.05s**, zero skip. The independent host compiler probe extracts the frozen production dispatch lambda and compiles it in a minimal explicit Source-only type harness. It verifies const clock/history invocability, five legacy level calls, and one complete 3-block by 5-level matrix call. It does not compile AmrSystem, generated kernels, Kokkos, or MPI and is not a native backend test.
