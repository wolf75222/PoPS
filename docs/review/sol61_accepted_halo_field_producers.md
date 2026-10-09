# Accepted-halo Field producer preparation: third Source freeze

Author Hooke, GPT-6.1 Sol. Follows fb530d5 + 863fa6b + 72384a6. Source implementation and a tiny host guard probe only; no Native/SCI acceptance.

The previous `FieldPlan.boundary_point` is setter configuration, not evidence that its value was computed. The stronger route now requires a temporary witness from a real successful Field candidate solve. Exact comparison includes primary clock, level, time, dt, tick, stage, substep and rational fraction. No setter creates this witness.

The actual GhostBoundary declarations retain their exact Field slots. Preparation closes the required producer DAG, uses real existing field solver/RHS/typed BC machinery at the authenticated local candidate point, and forces its exact non-Field InputAux dependencies from the complete staged state tower. Upstream Field outputs come from the real preceding producer solve. The solver must report an available solved candidate; failed outcomes follow their authored failure action.

Successful candidates are copied into temporary Field/provider images under collective copy/fence votes, then consumed through the existing `DiscardCandidate` disposition and their retained boundary bindings are restored. This is a genuinely solved temporary numerical dependency, not a synthetic successful Native result or failed-outcome downgrade. Canonical Field acceptance's old noexcept copy hook is not used by this preparation route. BC reads these real temporary outputs at their exact point. Complete original Field potential, provider grown storage, accepted registry provenance and dirty identities are restored before state publication, preserving borrowed allocation addresses. No FieldReady-after-step promise is added to the state HaloReady effect.

Local request validation and exact rank agreement complete before collective Field materialization. Complete grown state, Field and provider backup allocations are then voted before staging. Snapshot size arithmetic covers all stored numerical images. Callback metadata/string allocations are voted before dependent collectives; candidate publication validation, copies, fences and temporary provenance errors fail collectively. The enclosing accepted transaction remains the broader rollback authority. The optional request no longer imposes a global clean-auxiliary precondition: only its exact required inputs are refreshed temporarily, and prior provenance is restored.

Bootstrap does not invent a positive dt. Typed GhostBoundary ABI1 with absent initial interval still refuses before staging; periodic initial dt0 remains admissible. Default executions and legacy contract8 are unchanged. Root owns final ABI8 integration and actual module compilation, protocol faults, Field cases, MPI/GPU/SCI reception.

## Actual Source host probe

```sh
rtk proxy python3 tests/review/sol61_accepted_halo_field_witness_host.py
```

The probe extracts the actual retained C++ guard and compiles it with the actual BoundaryEvaluationPoint/amr_clock headers using clang C++20 Wall/Wextra/Werror. A minimal dependency-table harness is used; this is not the Native FieldPlan/solver. Actual result: ten refusals (including exact old setter without producer and one-ULP time/dt contradictions), exact producer witness accepted, periodic initial dt0 accepted. Guard SHA256 `929bdb63ac0e2cd0bd0783377be1fea343a9d475fcd6b3569c5f8f52f7bdfee6`; point header SHA256 `5618c7ed416bea593056aebf6816b9f317b7eb84c8b0a4c6c6590c47c888d19c`.

The earlier Python Source checks remain 119 PASS for first gel and 21 PASS for the primary-clock emitter follow-up. They do not verify these C++ producer routes. `git diff --check` passes. No PoPS import in the host probe, no Native/JIT/heavy TU/ENV mutation.

## Limits and required real reception

The extra cost is material: complete Field/provider snapshots plus potentially a composite Field solve for each block/level's required producer closure. No negligible-performance or GPU claim. Causal cycles still require an existing typed coupled solver; there is no invented iteration recipe. Independent Source review must examine outcome ownership, temporary provenance restoration, exact clocks/Field input causality and allocations before collectives. Root must exercise real typed BC/time/Field cases, aligned subcycling/regrid/rebalance, asymmetric failure/rollback, strict grown bit-one guards and nonregressions on the rebuilt SDK before any backend or scientific claim.
