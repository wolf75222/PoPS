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

## Corrective replay: Source preparation accepted

Reviewed corrective author freeze de6becf1d9b81f0e9a6dba1420592db0cc236fc4 on the same independent tests (local cherry-pick eac4f36cd9c15fd653b374a064570c9b7355bf5c). The command above now closes **27 PASS in 4.36s, zero skip**. The four original red tests are unchanged. The prior red result remains historical evidence, not a successful run.

The real fixture and saved reader now pass the primary clock from authenticated checkpoint temporal_restart_state.program_schedule.primary_clock. Consumer tick0/stage0/substep0/fraction steps/1 and empty builder graph/rate/application identities are the exact scoped synchronous FE contract; owner cursor remains separately authenticated. Pure synthetic math calls may omit the external clock argument, and cannot mint SCI authority. No constraint consumer.tick==owner_macro_step was added to the generic observer.

The full candidate grown strip at physical xmin, tangential-valid coordinates, is now checked against the recorded State Ghost for the declared phi+1+t expression. The five-operation IEEE bound is analytical (machine epsilon, conservative relative to unit roundoff), not fitted to captured discrepancies. OriginalF and valid-phi guards remain 1e-10. Candidate valid geometry/owners match State only under this builtin FAC fixture; candidate grown storage uses its own axes. This does not certify corners, arbitrary external Field layouts, or general Field halo readiness.

The generic public emitter uses qualified dependency identity and declared component widths; reads and writes use the same coordinate-to-stride mapping. Its reversed linear traversal is a Cartesian permutation of the Native pack traversal, not a component or cell substitution. In this 2D xmin strip the fixed x coordinate and varying tangential coordinate are the exact support checked above. No central physical-name or first-species special case is introduced by this test-only corrective.

Verdict: no remaining blocker in this bounded Source/offline preparation. No Native execution, MPI transport, SDK qualification, scientific reception, or Root seal follows from these tests. Initial and accepted witnesses must come from real solves before Ghost; reloaded observation is absent and warm-start cache is compared only for restart bits. ROOT must execute the corrected fixture on the authentic fresh SDK before any positive SCI reception.
