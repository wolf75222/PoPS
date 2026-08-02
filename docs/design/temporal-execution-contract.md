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

This foundation deliberately does not pretend that the existing global AMR driver is cell-local
stepping. If a cell-local image reaches that driver, execution fails before the Program body instead
of silently falling back to a global `dt`. ADC-707/ADC-708 still own the prepared patch/task graph;
ADC-756 still requires Kokkos rung batches, actual local-stage boundary evaluation, time-integrated
same-level/MPI/coarse-fine flux ledgers, regrid/rank-change rematerialization, device determinism and
performance evidence. None of those execution or conservation claims is made by this restart slice.

Offline envelope inspection authenticates only the integrity of a canonical checkpoint; it is not
a migration. The frozen release-v2 Uniform checkpoint predates the envelope and omits lifecycle
identities, temporal state, consumer cursors, and field-provider state. The explicit
`pops.codegen.checkpoint_migration` route can migrate exactly that frozen, store-all Uniform v2
schema only when the caller supplies both a complete authenticated current-v5 authority checkpoint
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
