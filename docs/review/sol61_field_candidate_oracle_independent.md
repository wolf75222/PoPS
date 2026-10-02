# Independent review of consumed Field oracle 76a8c103

Reviewed exact author freeze 76a8c103f4c808adbdf2c27a78b891408d3a628e in a separate clean worktree. No Native/ENV/Main writes or execution. Verdict: Source preparation blocked by two evidence-validation findings.

## Verified geometry scope

AmrSystem builds each Field EllipticBuildRequest from the actual engine layout.patches/distribution and state.local_rank (src/runtime/amr/amr_system.cpp:5708–5724). The builtin CompositeFAC Level allocates phi using request.boxes/distribution/local_rank, scalar width one and its own unit ghosts (include/pops/numerics/elliptic/mg/composite_fac_poisson.hpp:540–545). Therefore matching State/Field valid patch keys and owners is justified for this fixed scalar builtin fixture; matching grown extents is not a generic promise and the reader correctly keeps them independent. Reconstruction uses valid slices of authentic per-rank archives and independent composite restriction/quadratic interpolation/reflux/Neumann OriginalF, with the unchanged 1e-10 guard. Cache getter images are excluded from endpoint math and only checked for restart bit equality.

## Findings and exact independent adversaries

1. Accepted consumer point is admitted by ranges only. tick999, stage17 and fraction0/1 each pass independently while owner stamp/time and all numeric images remain exact. The actual prepare_accepted_publication_halos_ builder (include/pops/runtime/program/amr_program_context_subcycling_runtime.inc:440–476) fixes stage/substep zero and takes tick and phase from the candidate ClockStamp. For this one-step synchronous FE profile the endpoint is tick0/phase1, while owner completion increments to owner_macro_step1. Authenticate this scoped temporal contract and the actual primary-clock identity, without imposing consumer tick==owner macro in the generic observer.

2. The reader ignores the actual Field grown strip supplied to physical xmin Ghost. Replacing only candidate Field cells x=-1, y in the valid tangential range by 12345 preserves valid phi, OriginalF, State and State Ghost. It nevertheless passes check_observed. The actual callback receives these pixels through pack_dependency_region, and the declared public expression is phi+1+t. Cross-check that consumed Field strip against the recorded State Ghost in this profile. This is not a blanket Field halo-readiness certificate; it is the exact support consumed by this expression. The current author note claims this comparison but check_observed does not perform it.

## Reproduction

    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest --noconftest -p no:cacheprovider -o pythonpath=python tests/review/test_sol61_field_candidate_oracle_independent.py tests/review/test_sol61_field_candidate_math_migration.py tests/review/test_sol61_initial_field_ghost_native_preparation.py -q --tb=short

Actual result: 4 FAIL / 22 PASS, 3.84s. All four failures are DID NOT RAISE in the independent negative tests. Inputs in these adversaries are explicitly synthetic offline codec fixtures; none is a Native capture, transport error, installation authority or SCI receipt. The genuine public Source resolve test passes. Author correction and replay are required before positive verdict.
