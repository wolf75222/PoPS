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

For this bounded slice, a cell-local checkpoint restarts only with the recorded MPI cardinality and
`RestoreRecordedHierarchy`. Rank-change and `RegridOnRestart` are rejected during Python preflight,
before the native restart transaction, because rematerializing canonical cell ids onto a new owner
or topology is not implemented yet.

`PreparedBatchedCellTemporalExecutor` is the first executable ADC-756 rung slice. Preparation
authenticates the exact provider identity recorded by the accepted image, groups canonical records
into compact device-accessible arrays and reserves the host clock transaction. During an attempt it
orders events by exact integer end tick and rung, then launches one Kokkos batch for every active
rung event rather than one task or kernel per cell. The provider sees the exact rational begin/end
time of each cell. The prepared hot loop does not allocate PoPS storage.

The executor accepts only a typed provider exposing one combined device operation:
`evaluate_local_stage_and_record_space_time_flux`. There are no independent Boolean declarations
for a local stage or ledger. A provider that needs a coherent neighbouring-cell image additionally
owns the optional `begin_rung_batch`/`complete_rung_batch` lifecycle; these hooks can materialize and
rotate attempt-local storage but cannot publish it. An accepted result therefore means that the
provider evaluated the stage and wrote its attempt-local integrated-flux record before that cell
clock advanced. All provider records and cell clocks commit together only at the synchronization
barrier. A malformed outcome, rejection, provider-preparation refusal or kernel failure rolls back
the complete attempt and leaves the accepted checkpoint unchanged.

`PreparedSameLevelTransportEulerStageFluxProvider` is the first scientific consumer of this
executor. It reuses the selected AMR block's real compiled transport closure to materialize
`-div(F)` and the exact x/y face-flux fields, advances the conservative candidate with forward
Euler, and accumulates four time-integrated face records per valid cell. Both state and ledger stay
in fixed attempt-local storage; the barrier commit is their sole accepted publication. The exact
provider contract includes the block state identity, model-owned transport identity and parameters,
limiter/Riemann route, spatial options, hierarchy/materialization identity, clock, tick scale,
layout and distribution. A type-erased spatial closure without that builder-owned contract is
refused rather than authenticated from a caller label.

This first scientific route is deliberately bounded to a host/serial 2D hierarchy with exactly one
block, one level, one rank-owned box, one common cell rung, frozen attempt auxiliary fields,
built-in periodic/Foextrap transport boundaries and transport-only forward Euler. A prepared
physical-boundary plan is refused until its exact executable contract can join the provider
identity. The route also has no MPI, GPU, heterogeneous-rung interpolation, coarse/fine ledger,
source-stage integration, regrid/rank-change rematerialization, diagnostic-ledger checkpoint
persistence or performance proof.

`Program.cell_local_time(tick_denominator=..., rung=...)` now selects this bounded route explicitly.
Generated AMR code accepts only the exact single-state Forward-Euler transport graph, prepares the
provider at an accepted boundary and installs `AmrProgramContext::advance_same_level_cell_temporal`.
The context routes checkpoint, attempt, commit and rollback through that sole executor; the ordinary
hierarchy-global driver still refuses a cell-local image and never substitutes a global `dt`.
Same-topology restart restores the numerical image and integer clocks. Because the accepted-state
schema does not persist the last interval's diagnostic face ledger, restart invalidates that
publication until the next accepted interval instead of exposing stale fluxes.

The remaining production extensions are explicit dependencies, not capabilities inferred from this
slice: canonical rank/box ownership and halo-stage snapshots for MPI; distributed face-ledger
reconciliation and collective failure draining; device-resident provider storage and publication for
GPU; temporal neighbour interpolation and subface synchronization for heterogeneous rungs;
coarse/fine space-time ledgers, reflux and local refinement ratios for multilevel AMR; exact provider
rematerialization after regrid or rank migration; prepared source, field and physical-boundary stage
contracts; an accepted-state schema extension if the last diagnostic ledger must survive restart;
and backend/allocation/performance qualification. ADC-707/ADC-708 continue to own the prepared
patch/task graph. No end-to-end heterogeneous or multilevel locally subcycled AMR conservation claim
is made by this bounded route.

Offline envelope inspection authenticates only the integrity of a canonical checkpoint; it is not
a migration. The frozen release-v2 Uniform checkpoint predates the envelope and omits lifecycle
identities, temporal state, consumer cursors, and field-provider state. The explicit
`pops.codegen.checkpoint_migration` route can migrate exactly that frozen, store-all Uniform v2
schema only when the caller supplies both a complete authenticated current-v6 authority checkpoint
and a reviewed mapping. The mapping pins the source bytes, source ABI and Program hash, the
authority restart and target lifecycle/ABI/Program identities, every block/component/history
correspondence, and the closed set of current metadata inherited from the authority. The supported
route requires the same grid and accepted clock, Dense fully stored histories with a matching
outgoing-`dt` ledger, and no qualified field providers, scheduled caches, or ConsumerGraph state.
It validates and reopens the complete current payload before an atomic no-clobber publication; the
source and authority are never modified. Other v2 variants remain unsupported. Runtime restart
contains no migration import or compatibility branch and continues to reject every historical
payload.

`RuntimeInstance` obtains each consumer moment from the accepted cursor of the consumer's qualified
clock. A missing clock, provisional phase, or desynchronized cursor is an error. Consequently a
child-clock consumer sees its child tick, while `wall_tick` and `accepted_step` retain the accepted
macro-step coordinate.
