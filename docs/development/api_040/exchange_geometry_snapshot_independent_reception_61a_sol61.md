# Independent reception of accepted-exchange geometry snapshots

Target: `61a3cb3fd175de8ee5a7bcfe6f5b1a007c2351bd` plus
`9471bf8ebceda549e75355ed0b9f2d77133d8a06`, merged in the exclusive review checkout
as `eb61cf0b`. Reviewer: GPT-6.1 Sol. Historical receipts and other worktrees are
preserved. No MAIN, installed environment, production header, native build or JIT
was changed by this review.

## Result

No new production defect was demonstrated. **31 source/host tests pass**:
eleven snapshot/producer/coverage checks and twenty legacy reduction byte
comparisons. Ruff and whitespace checks pass. The new snapshot test compiles
the actual frozen Uniform and AMR method bodies with explicit host geometry and
topology types; **90 assertions** receive dimensions 1/2/3, negative domain
origins, all face sides, periodic qualification, invalid axis/side refusal,
mutation of the original facade after preparation, and closure use after its
context/facade is destroyed. Both bodies acquire geometry/topology once and their
closures make no further facade calls. This is source/host evidence, not native
execution in three dimensions.

The pre-existing independent transport/diffusion producer probes keep all their
physical amounts, incidence signs, two components, finest-owner/EB intersection,
seven replicated ranks, distributed nonzero owner, empty-rank, Uniform,
fully-covered level and staged-prefix assertions. Their Context adapters now
provide the explicit geometry type and snapshot API. The producer phase is
guarded: direct geometry, snapshot preparation, or the old facade trace callback
throws if reached inside the batch producer. Every successful call prepares the
predicate, including suppressed replicas and empty ranks. The two error votes
and batch completion remain required; malformed masks and injected preparation
failure refuse before publication. The vote spies are **local control-flow
evidence**, not simulated proof of MPI synchronization.

Importantly, the same positive owner baseline and phase guard also execute the
actual old producer bodies from `61a3cb3^`. Transport is rejected precisely for
its geometry callback and diffusion for its trace callback inside the producer.
This prevents a false green from never visiting any face. The new bodies pass.
Byte normalization against that parent confirms that transport's numerical
emission and the entire PreparedDiffusion header change only through the declared
snapshot additions/substitutions: equations, orientation, component identities,
flux density, face measure, dt and coverage/EB filtering are unchanged.

## Collective phase review

`AmrSystem::prepared_amr_level_geometry` enters `ensure_engine`, verified in the
actual source. The authentic SDK506 PMPI diagnostic cited by the integrator in
`accepted_exchange_collective_geometry_snapshot.md` places that getter opposite
an exchange vote on another communicator at entry 3483. It is evidence of the
old failure, not reception of this correction.

Transport now acquires its measure geometry and its domain/topology predicate
in the existing all-rank preparation try block, before owner selection. Diffusion
already owns its prepared measure geometry and obtains the predicate in the
same preparation phase. Error and layout votes precede `stage_exchange_batch`;
the producer's contributor return stays *inside* that batch, so noncontributors
still reach preparation/error/staging collectives. The producer uses only owned
snapshot values. Finest-owner coverage remains the local accessor following the
active-mask refresh; no new collective was added inside that lookup.

This closes the demonstrated rank-dependent geometry calls. It does not certify
every facade getter or every possible hierarchy preparation failure. Native
prepared-context phase consensus, communicator behavior and rebuilt AMR/MPI
execution must still be received by the integrator.

## Fixture compatibility correction

Source inventory found two additional C++ unit Context adapters in
`tests/cpp/unit/runtime/test_prepared_diffusion.cpp` which still implemented only
`is_external_trace_face`. Their stage_accepted_exchanges instantiations now need
`prepare_external_trace_face_predicate`. This review adds owned snapshots to those
two **test adapters only**; it changes no existing numerical assertion. The shared
Python coverage adapter receives the same additive API/type adaptation. The C++
unit target has **not** been compiled here.

Required central native reception includes rebuilding `test_prepared_diffusion`
and `test_mpi_exchange_batches` (the latter configured with three ranks), then
rerunning installed serial and isolated AMR two-rank transport/diffusion/restart.
The previously failing SDK506 trace cannot qualify the new SDK.

## Exact source/host reproduction

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I -c 'import sys; from pathlib import Path; root=Path.cwd(); sys.path[:0]=[str(root/"python"),str(root)]; import pops,pytest; assert Path(pops.__file__).resolve()==root/"python/pops/__init__.py"; raise SystemExit(pytest.main(["-q","--tb=short","tests/review/test_sol61_exchange_geometry_snapshot_independent.py","tests/review/test_replicated_exchange_1f777538_independent.py","tests/python/unit/codegen/test_accepted_exchange_coverage_independent.py","tests/review/test_dot_all_compensation_legacy.py"]))'
```

The assertion authenticates source-package imports. Small clang++ host probes
compile only the selected real bodies with explicit adapters; they do not import
the installed native module, launch MPI, build Kokkos or create a Program JIT.
