# Temporal execution and restart contract

ADC-667 makes logical time an explicit execution authority. A `Program` has one primary `Clock` and
may declare fixed-ratio child clocks with `Program.subcycle(...)`. A value crosses clock domains only
through `Program.synchronize(..., relation=...)`; `SampleAndHold()` is the first native relation.
There is no implicit clock cast, inferred parent relation, or fallback to the macro-step counter.

```python
fast_state = T.synchronize(
    state.n, at=TimePoint(fast), relation=SampleAndHold())
fast_next = T.subcycle(
    fast_state,
    clock=fast,
    within=T.clock,
    count=3,
    body_fn=lambda P, q: advance_fast(P, q),
)
state_next = T.synchronize(
    fast_next, at=state.next.point, relation=SampleAndHold())
T.commit(state.next, state_next)
```

`subcycle` is structured IR. The child duration is exactly the enclosing duration divided by
`count`; nested subcycles compose their ratios. Generated native code opens an exception-safe clock
scope, checks every child iteration in order, and closes it only after the exact authored count.
Uniform programs additionally lower `InterpolateHistory(...LinearInterpolation())` against native
retained slots owned by either the primary clock or a child clock. Every generated child-clock
store publishes the active child interval in the exact accepted `slot_dt` ledger. Dense-output
capabilities and AMR interpolation remain rejected until they have an equally exact native
provider; no callback or sample-and-hold fallback is substituted.

## Qualified histories and schedules

`Program.temporal_manifest()` is the canonical data-only contract for execution and restart. It
contains every qualified clock and its derived ticks per macro step, parent/child relations,
synchronization points, typed schedules, cache requirements, and each history's owner, state,
space, clock, maximum lag, ring slots, interpolation provider, validity domain, and checkpoint
policy. A non-primary clock without one unambiguous route to the primary clock is invalid.

Native history registration carries the same owner/state/space/clock/interpolation tuple. Reusing a
history name with another identity is an error. Uniform execution rotates only histories owned by
the active logical clock. AMR refuses child-clock histories until a composed AMR-level/logical-clock
dense-output provider exists; it does not run them at a false macro cadence.

## Accepted boundary and schema v2 restart

The temporal restart payload schema v2 persists the exact program schedule and accepted cursors for
clocks, subcycles, synchronization nodes, schedules, histories, held caches, the event queue,
controller proposal state, and transaction statistics. Field/history/cache values remain in their
native checkpoint sections, authenticated by this envelope.

For `ErrorControlledDt`, the queue head is the exact next proposal: it determines the native `dt`,
survives a rejected attempt until the controller explicitly replaces it with the typed reduced
proposal, and is consumed exactly once by acceptance. The accepted boundary immediately publishes
the following proposal, so restart replays the same next decision rather than recomputing it from a
display-only field.

A checkpoint is legal only at an accepted fully synchronized boundary. Rejection and failure leave
all accepted cursors unchanged and make checkpointing ineligible. Restart compares the checkpointed
program schedule with the installed program before native state mutation and requires the exact
checkpointed step strategy for the next attempt. Schema v1 and other historical payloads require an
offline migration; runtime restart contains no compatibility branch.

## Embedded-boundary active-cell contract

Under an active staircase/cut-cell boundary, a qualified pointwise `local_transform` or local
nonlinear solve evaluates only active cells and preserves the accepted values of inactive cells.
The default model source follows the same rule: an inactive cell is short-circuited before any
model or provider read. Generated reductions are raw owner/block/layout/lane-authenticated
reductions over active cells; they do not apply kappa or volume weights. Physically weighted
integrals remain explicit System services rather than an implicit reduction policy.

Terminal recoverability and admissibility validation ignores inactive cells, while transactional
publication preserves their accepted bits after every candidate in the batch has been validated.
The same active-mask semantics apply at every AMR level; the implementation must not narrow the
mask to `nlev == 1`. Unmasked pointwise Cartesian operations (`where`/`cell_compare`) and
Cartesian stencil operators (`laplacian`, `gradient`, `divergence`, condensed-RHS stencils, and
matrix-free stencils) fail closed under an active embedded boundary. Their exact
owner-authenticated preflight rejects before the unsupported operation evaluates; for matrix-free
stencils it rejects before preparation or iteration enters Krylov. Persistent scratch or other
resources may have been prepared earlier. These statements document the contract; they are not a
claim that the corresponding serial, MPI, GPU, or performance runtime gates are green.

## Cell-local temporal-partition restart foundation

The AMR Program accepted image now has an explicit temporal-partition section. Its cell-local form
stores a prepared-provider identity, hierarchy topology epoch, integer synchronization tick and tick
denominator, plus canonically ordered `(level, cell, rung, accepted_tick)` records. Floating-point
cell clocks and rank-local addresses are not checkpoint authority. Every persisted cell must be at
the same rung-aligned synchronization tick; a provisional attempt cannot be serialized.

`BatchedCellTemporalPartition` supplies the execution-provider-independent transaction semantics:
one attempt target, ordered same-rung batches, synchronization barriers, commit, rollback, strict
restore and an accepted-state manifest. The AMR restart path decodes and validates this state before
replacing accepted bytes. A malformed provider identity or topology/level mismatch leaves the
previous image untouched.
The public Program report consumes the same native image and exposes provider identity, accepted
tick, denominator, cell count and per-rung counts.

For this bounded slice, a cell-local checkpoint restarts with the recorded MPI cardinality.
`RestoreRecordedHierarchy` restores the exact partition directly; `RegridOnRestart` rebuilds its
topology-derived records at the same exact synchronization tick through the retained route table.
Rank-change remains refused because redistributing those records to a different rank cardinality is
not implemented.

`PreparedBatchedCellTemporalExecutor` is the first executable ADC-756 rung slice. Preparation
authenticates the exact provider identity recorded by the accepted image, groups canonical records
into compact device-accessible arrays and reserves the host clock transaction. During an attempt it
orders events by exact integer end tick and rung, then launches one Kokkos batch for every active
rung event rather than one task or kernel per cell. The provider sees the exact rational begin/end
time of each cell. The prepared hot loop does not allocate PoPS storage.

The executor accepts only a typed provider exposing one combined device operation:
`evaluate_local_stage_and_record_space_time_flux`. There are no independent Boolean declarations
for a local stage or ledger. A provider that needs a coherent neighbouring-cell image additionally
owns the optional `prepare_rung_batch_local`/`materialize_rung_batch_snapshot`/
`finalize_rung_batch_candidate`/`complete_rung_batch` lifecycle. Local preparation reaches exact-lane
consensus before halo materialization; finalization prepares attempt-local metadata after the kernel
outcome consensus, and completion only rotates the prepared candidate. None publishes accepted
state. An accepted result therefore means that the provider evaluated the stage and wrote its
attempt-local integrated-flux record before that cell clock advanced. All provider records and cell
clocks commit together only at the synchronization barrier. A malformed outcome, rejection,
provider-preparation refusal or kernel failure rolls back the complete attempt and leaves the
accepted checkpoint unchanged.

`PreparedSameLevelTransportEulerPackStageFluxProvider` is the first scientific consumer of this
executor. It reuses every independent AMR block's real compiled transport closure to materialize
`-div(F)` and the exact ranked face-flux fields, advances the complete block pack with forward Euler,
and aggregates one bounded time-integrated flux basis per route and hierarchy window. State and
diagnostic view stay in fixed attempt-local storage; the existing AMR transition ledgers remain the
sole conservation authority and the barrier commit is the sole accepted publication. The exact
provider contract includes route/block identities, model-owned transport identities and parameters,
limiter/Riemann routes, spatial options, hierarchy/materialization identity, clock, tick scale,
layout, distribution and exact execution lane. A type-erased spatial closure without that
builder-owned contract is refused rather than authenticated from a caller label.

This scientific route is deliberately bounded to exact-rank host execution over independent
multi-block, multi-level and MPI-owned multi-box AMR with transport-only forward Euler. The authored
rung is the finest-level base; integral power-of-two temporal relations derive one homogeneous rung
per level-group and exactly one FE batch per hierarchy window. Global/interface block coupling,
non-dyadic clocks, heterogeneous per-cell rungs, source/field stages, physical or non-periodic
boundaries, GPU default execution or memory spaces, performance qualification, rank-change
rematerialization and diagnostic-ledger checkpoint persistence remain unavailable. Boundary and
device exclusions fail collectively on the exact lane before a prepared state or boundary stage.

`Program.cell_local_time(tick_denominator=..., rung=...)` now selects this bounded route explicitly.
Generated AMR code accepts one exact Forward-Euler transport route per Program block, prepares the
complete provider at an accepted boundary and installs
`ProgramExecutionServices::advance_same_level_cell_temporal`.
The context routes checkpoint, attempt, commit and rollback through that sole executor; the ordinary
hierarchy-global driver still refuses a cell-local image and never substitutes a global `dt`.
Same-rank restart/regrid restores the numerical image and exact partition clocks. Because the
accepted-state schema does not persist the last interval's diagnostic face view, restart invalidates
that publication until the next accepted interval instead of exposing stale fluxes.

The remaining production extensions are explicit dependencies, not capabilities inferred from this
slice: device-resident provider storage and publication for GPU; temporal neighbour interpolation
and subface synchronization for heterogeneous rungs; global and interface block coupling; prepared
source, field and physical-boundary stage contracts; rank-change rematerialization; an accepted-state
schema extension if the last diagnostic ledger must survive restart; and backend allocation and
performance qualification. ADC-707/ADC-708 continue to own the prepared patch/task graph. This
bounded route makes no end-to-end heterogeneous-rung or coupled-block local-time claim.

Offline envelope inspection authenticates only the integrity of a canonical checkpoint; it is not
a migration. The frozen release-v2 Uniform checkpoint predates the envelope and omits lifecycle
identities, temporal state, consumer cursors, and field-provider state. The explicit
`pops.codegen.checkpoint_migration` route can migrate exactly that frozen, store-all Uniform v2
schema-4 mapping only when the caller supplies both a complete, separately authenticated current-v9
authority checkpoint and a reviewed mapping. The v2 source contains no auxiliary authority. The
authority must carry an empty POPSAUX2 image natively attested by the installed specialization; the
mapping pins the exact source and authority bytes, source ABI and Program hash, authority restart
and target lifecycle/ABI/Program identities, the SHA-256 of the exact auxiliary image bytes and of
the binary registry contract bytes, every block/component/history correspondence, and the closed
set of current metadata inherited from the authority. For the native dimension, that image is
provider-/payload-empty only when persisted groups, components and providers are all zero while
the opaque sealed registry contract is nonempty and `accepted_generation` is in
`[0, UINT64_MAX)`. It need not equal a freshly sealed bare-registry contract: real code generation
may install zero-valued consumer plans (the real authority contract is 1312 bytes). The attestor
alone does not establish target-registry compatibility; the full-image and raw-registry-contract
SHA-256 pins, byte-identical copy and strict live target restart provide that exactness. Migration
copies that POPSAUX2 image byte-identically from the authority and never fabricates it from v2. The
generation is preserved provenance, not rewritten: this AB2 fixture records `0`, structurally empty
publication may produce a value greater than zero, and `UINT64_MAX` is refused as wrap poison. The
emitted `checkpoint_migration` member is reserved by the live Uniform resource budget in a fixed 16 Ki
character envelope. The supported route requires the same grid and accepted clock, Dense fully
stored histories with a matching outgoing-`dt` ledger, and no qualified field providers, scheduled
caches, or ConsumerGraph state. It validates and reopens the complete current payload before an
atomic no-clobber publication; the source and authority are never modified. Other v2 variants
remain unsupported. Runtime restart contains no migration import or compatibility branch and
continues to reject every historical payload.

`RuntimeInstance` obtains each consumer moment from the accepted cursor of the consumer's qualified
clock. A missing clock, provisional phase, or desynchronized cursor is an error. Consequently a
child-clock consumer sees its child tick, while `wall_tick` and `accepted_step` retain the accepted
macro-step coordinate.
