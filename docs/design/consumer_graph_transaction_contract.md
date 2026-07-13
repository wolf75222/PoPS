# ConsumerGraph transaction contract

ADC-685 adds consumer planning after `RuntimePlanBundle`, without placing diagnostics or output
work in the scientific `OperatorGraph`. Diagnostics, scientific outputs, checkpoints, and monitors
are distinct `ConsumerManifest` nodes. Each node owns a canonical `Handle(kind="consumer")`, exact
qualified dependencies, selected quantities, layouts and levels, field contexts and typed read
policies, a typed schedule, a publication target and format, a parallel mode, and one failure action.

`ConsumerGraph` canonicalizes declarations by qualified identity and computes one deterministic
topological order. Ready-node ties use the qualified consumer id. Manifest and graph identities
therefore change with selections, scheduling, target/format, parallel requirements, recomputation,
dependencies, or failure policy, but never with declaration insertion order.

## Pure effect planning

`plan_accepted_side_effects(runtime_plan, graph, moment, cursors)` is pure. It authenticates every
selected resource against an exact call/layout in the `RuntimePlanBundle`, requires a planned
collective for every collective quantity, and records every quantity in a
`LoweringCoverageReport`. A field read is resolved through its exact `FieldContext` and current
`LayoutBinding`. A stale, provisional, off-schedule, or regridded field fails unless its manifest
contains an explicit typed hold/recompute policy. An explicit recompute is recorded in the payload;
the consumer planner never invokes a solve.

Only due, previously uncommitted occurrences become `AcceptedSideEffect` values. An effect contains
the authenticated manifest, a `PublicationTarget`, a deduplicated `ConsumerPayload`, its failure
action, and the cursor transition that may be applied after publication. Planning does not prepare
files, publish artifacts, or mutate cursors.

## Writer boundary and acceptance

Format implementations subclass two nominal interfaces from `pops.runtime.consumer`:

```python
class ConsumerPublisher(ABC):
    def prepare(self, effect: AcceptedSideEffect) -> PreparedPublication: ...

class PreparedPublication(ABC):
    @property
    def effect_identity(self) -> Identity: ...
    @property
    def payload_identity(self) -> Identity: ...
    def publish(self) -> PublicationReceipt: ...
    def discard(self) -> None: ...
```

`prepare()` creates only an incomplete temporary. `publish()` must make that one artifact visible
atomically (normally commit/rename) and returns `PublicationReceipt` only after success. `discard()`
is idempotent and removes all preparation residue. HDF5, NPZ, ParaView, and checkpoint writers are
outside ADC-685 and live behind this boundary.

`ConsumerTransaction` prepares every effect while the step attempt is provisional. `reject()`
discards all temporaries, publishes nothing, and returns the original cursor set. `accept()` is the
only publication path. A cursor advances only after a matching receipt authenticates both the exact
effect and payload.

Failure actions are exact:

- `FailRun()` aborts the consumer transaction and reports unchanged cursors for the failed sample;
- `Retry(max_attempts)` performs a bounded fresh preparation/publication attempt and fails when the
  bound is exhausted;
- `SkipSampleReported()` returns a structured skipped-sample report, no receipt, and no cursor
  advancement.

Already published artifacts from an earlier consumer are not rolled back if a later independent
publication fails. No artifact is considered complete without its receipt, and no failed or skipped
sample advances its scheduling cursor.
