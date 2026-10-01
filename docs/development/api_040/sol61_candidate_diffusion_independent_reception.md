# Independent candidate diffusion reception — 1 October 2026

Reviewed production: `48292403c0f42192e17c61a3be9f97b3dd2d3c19`, allocation drain follow-up `2e9a07329d008cf910f442998e54f5a47ed6e6d4`, and mandatory true-correction follow-up `5394fda54aaf9623f784a74703da69635f4d2dae`. Parent: `7d91f5ab06eca3db1d32f2c0164359a5426c2c9c`. This commit contains independent tests/documentation only. The author's later fixture freeze `be6665bd19fb5a32a276a0e4ae9e03734bdfb2c4` is an additional native test inventory, not an execution receipt.

## What was received

Actual public authoring, request validation, registered physical recompilation, resolve and C++ emission receive scalar and product widths 3/5, different component orders, Uniform and AMR. The independent input diffusion matrix is signed, nonsymmetric and singular; its coefficient depends on the actual unknown and a separate exact material State capture. No positivity/SPD assumption is added. Seed, captures and unknowns remain distinct inputs.

The selected method carries Arithmetic@1, PerCandidate@1 and `pops.field.linear.true-correction-residual@1`. Source/token/request identities agree; the Program promotes conditionally to IR11/request2. Missing or changed correction criteria, an older residual contract, seed-frozen evaluation, harmonic faces, the unsupported SpatialBasisJacobi combination, reordered capture inputs, a foreign coefficient point, changed coefficient expression or changed storage role are refused by the actual validator. Restoring the original objects restores the original IR identity. A consistent rehash of a forged zero-D descriptor and request passes internal request consistency, then fails the independent registered physical recompilation with `equation_identity_drift`; self-consistency is not physical authority.

Inspection plus source regression confirms AMR's actual order: copy/synchronize owned candidate q (including hierarchy restriction/halos), evaluate D from that q and owned capture snapshots, prepare coefficients (fine-to-coarse restriction before coefficient halos), apply the resulting operator, then add the original local residual. Both central directional samples call the whole evaluate path. The evaluation resource is private and its generation changes independently of the borrowed original operator. Authority validation uses the prepared operator lane for local failure votes and exact agreement; point, attempt, ownership, topology, materialization, capture identity/components and coefficient generation remain authenticated. The allocation catch paths added by 2e9 drain launches before their votes, only for the opted-in paths.

Uniform emission writes coefficients inside its original F callback and prepares/applies the actual field operator with local collective guards. Owned capture copies are made before Newton; candidate coefficient values are not copied from a seed-frozen field. The emitted PreparedSpatialResidual enables local guards and true-correction verification as two separate explicit arguments. The actual source still checks original terminal F before outcome/publication. This inspection does not establish a distributed runtime execution.

## Concrete correction criterion counterexample

`test_sol61_uniform_true_correction_independent.py` extracts the actual `solve_linear_`, `update_correction_` and Hessenberg accessor from the frozen header and compiles a small host C++20 program. It substitutes storage with vectors, serial reduction and a fence; it does not substitute the GMRES method bodies. Its callback is normalized central FD of `F_i(q)=q_i+q_i^3` at zero, with a positive explicit step, rhs (3,4), restart/budget 4 and stop 1e-10.

Legacy projected acceptance reports convergence with actual full-correction residual 0.280434 after 2 reported JVP evaluations. On 539, the explicit true policy performs the complete correction JVP and refuses within the same column budget: actual residual 0.135287, 6 reported JVP evaluations. Extra complete-correction evaluations are counted by the existing report. The test requires the new policy and refuses either convergence above the true stop or no additional verification. This discriminates the projected-residual gap in 482/2e9; it neither asserts false publication nor changes legacy behavior, nonlinear tolerance, FD step, restart or budget. Terminal nonlinear F alone could not establish the linear contract.

## Exact legacy comparison

The explicit `legacy_images(CHECKOUT)` entry point in the independent source test uses the same public fixture file and callsite in fresh interpreters for both checkouts. It neither fetches historical objects nor runs during ordinary unit tests. Four profiles compare strictly equal: original AMR2/permutation10 (IR8), original Uniform3/permutation201 (IR8), SpatialBasisJacobi AMR5/permutation42031 (IR9), captured Arithmetic Uniform3/permutation201 (IR10). Compared fields include authored/resolved IR hashes, full canonical SolveRequest, physical/solver identities, C++ digest, all three Module hashes, and all three full manifests including actual provenance. No normalization or golden replacement is used. The parent/candidate JSON receipts have the same SHA256 `3d5601c636438194c23daedec7f8e1da1a9d2b6da27b7c1ffd843c8ff62d5809`.

Example source commands (Python is the existing pops-api040 interpreter; package path is selected explicitly, without installation):

```sh
rtk proxy env PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q tests/review/test_sol61_candidate_diffusion_independent.py tests/review/test_sol61_uniform_true_correction_independent.py -p no:cacheprovider --tb=short -s
rtk proxy env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/test_sol61_candidate_diffusion_independent.py CHECKOUT
```

Validation: 20 cases passed in the complete source/host selection (96.27 s); the additional allocation-drain test passed separately after correcting its extraction window (1 passed, 19 deselected, 0.03 s). The earlier AMR-emission assertion incorrectly assumed a generated identifier prefix; it was replaced with the actual coefficient callback signature and passed on replay. These were independent test defects, not production failures. Ruff and diff checks passed.

The previously frozen independent Fraction/math obligations in `85325074cc7ea76eb0f42878b12c8bd1889aa65a` were replayed: 9 passed. They discriminate missing delta-D terms, seed freezing, signed/singular matrix/component permutations and nonlinear restriction ordering on shifted/anisotropic interface witnesses. Those mathematical arrays are not native saved states and do not qualify the entire AMR discretization.

## Limits

No installed package, SDK, MAIN checkout or production source was edited, and no Native/JIT/full build was executed. The host GMRES extraction is not a Kokkos/MPI/GPU test. The emitted callback and source order are received, but distributed fault convergence, actual hierarchy fluxes and solve/rollback/publication must still be executed by the central native campaign on rebuilt artifacts. No convergence guarantee follows for arbitrary candidate-dependent coefficients; SpatialBasisJacobi remains explicitly unsupported for this branch. No M22 pressure closure, M27 physical benchmark, physical equations/boundaries not specified by the corpus, or future LU realization is received here.
