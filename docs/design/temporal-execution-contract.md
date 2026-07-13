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
An unsupported synchronization provider is rejected during lowering, before a binary is published.

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

A checkpoint is legal only at an accepted fully synchronized boundary. Rejection and failure leave
all accepted cursors unchanged and make checkpointing ineligible. Restart compares the checkpointed
program schedule with the installed program before native state mutation and requires the exact
checkpointed step strategy for the next attempt. Schema v1 and other historical payloads require an
offline migration; runtime restart contains no compatibility branch.

`RuntimeInstance` obtains each consumer moment from the accepted cursor of the consumer's qualified
clock. A missing clock, provisional phase, or desynchronized cursor is an error. Consequently a
child-clock consumer sees its child tick, while `wall_tick` and `accepted_step` retain the accepted
macro-step coordinate.
