# ConsumerGraph transaction contract

The sole public authoring import is `from pops.output import ConsumerGraph`. Runtime planning and
publication keep their private records under `pops.runtime`; they do not form a second public graph
namespace.

Consumer planning follows `RuntimePlanBundle` without placing diagnostics or output work in the
scientific `OperatorGraph`. Diagnostics, scientific outputs, checkpoints, and monitors
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

## Typed diagnostic execution

Every diagnostic embedded in `ScientificOutput` lowers exactly once to a `DiagnosticQuantity`.
That value owns a qualified diagnostic handle, exactly one resolved conservative state, its layout
and level selection, and a closed native instruction containing the reduction, scalar transform,
metric-weighting rule, optional physical role, and optional conservation tolerance. An unqualified
diagnostic is accepted only when the selected output resolves to exactly one conservative state;
multi-state ambiguity is an error, never a first-field convention. A diagnostic cadence, when
provided, must equal its parent `ScientificOutput` schedule.

Cell traversal, AMR coverage masking and MPI reduction execute in the C++ runtime. Python applies
only the declared scalar post-transform (for example the square root after a native sum of squares)
and stages the resulting immutable `DiagnosticPayload` in the same accepted-side-effect transaction
as the writer. Uniform metric weighting is derived from the authenticated normalized geometry; AMR
composite reductions already include level metrics and covered-cell exclusion and are not weighted a
second time.

`ConservationCheck` is an invariant check for a genuinely closed conservative quantity, not a
substitute for an open-domain balance. Its first accepted sample establishes a non-zero baseline;
later samples compare against that same baseline. Baseline creation, diagnostic publication and
cursor advancement commit or roll back together, and the canonical baseline map is part of the
checkpoint/restart payload. An open system must instead report its storage, outward-boundary, source,
reflux and projection terms explicitly.

## Writer boundary and acceptance

The author-facing extension seam stays under `pops.output`: a custom scientific format subclasses
`pops.output.FormatInterface`, provides deterministic `consumer_data()`, and returns a writer from
`writer()`. The writer implements the public structural `pops.output.ScientificWriter` protocol:
`preflight(execution_context)` returns canonical capability evidence and effect-free
`prepare_session(snapshot, request, target, communicator=...)` returns a
`pops.output.WriterSession` on every rank. No inheritance or runtime-package import is required.

The session exposes deterministic `authority` plus its authenticated `identity`, then `stage()`,
idempotent `abort_prepare()`, `publish()`, `rollback()`, and idempotent `finalize()`.
`prepare_session()` cannot create a temporary or enter backend I/O. The runtime first compares
session authority across communicator rank order and only then calls `stage()` on every session.
`ROOT` therefore has an active rank-zero session and a no-op participant session on every non-root
rank; `COLLECTIVE` has an active session on every rank; `PER_RANK` has one active local session and
target on every rank. If any stage fails, every rank calls `abort_prepare()` before the failure
escapes, including ranks whose own stage succeeded. After `publish()`, `rollback()` remains available
until the enclosing transaction seals. Sealing first makes rollback permanently unavailable, then
calls the idempotent `finalize()` release operation through the runtime bridge on every pending
session. A successful release is removed from the pending set; a failed or non-`None` release is
reported as a post-commit diagnostic and remains pending for an idempotent retry. Built-in writers
release the retained staging descriptor/inode authority in that phase. The runtime never tests a
concrete session class or selects behavior from a provider id.

`RuntimeInstance` retains each failed finalization owner and retries it before and after subsequent
consumer fires, accepted steps and checkpoints. `retry_consumer_finalizers()` is the narrow explicit
retry boundary. The accepted report holds only the current failure for that owner: another failed
attempt replaces it, and a successful retry removes it instead of accumulating stale diagnostics.

After resolution, private execution code adapts accepted effects to two runtime-owned nominal
interfaces:

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
    def rollback(self) -> None: ...
    def finalize(self) -> None: ...
```

These runtime interfaces are implementation details, not extension base classes. Their private
`prepare()` adapts the already staged public writer session to an accepted effect. `publish()` must
make that one artifact visible atomically (normally no-clobber link/commit) and returns
`PublicationReceipt` only after success. Runtime `discard()` delegates to the session's idempotent
`abort_prepare()`; compensation delegates to `rollback()`. `finalize()` is a release phase, not a
second commit: it runs only after the outer transaction has committed, must return `None`, is
idempotent at the public writer-session boundary, and releases rollback-only resources such as the
retained staging file descriptor. HDF5, NPZ, ParaView, external native writers and checkpoint
providers all live behind this boundary.

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

Receipted artifacts remain compensatable until the enclosing accepted-step transaction seals. If a
later publication in that transaction fails, the runtime rolls earlier artifacts back in reverse
dependency order and restores the original cursor set. `ConsumerTransaction.seal()` first makes the
transaction non-compensatable, then finalizes every accepted publication and drops its rollback
ownership. The native step's successful finalization is the irreversible `native_finalized`
boundary: release failures and contract violations are attached to the accepted report as
post-commit diagnostics, while the engine state, cursors, receipts and artifacts remain accepted.
They can never trigger consumer abort, native rollback, envelope restoration or artifact removal.
After sealing, those receipts are final and are never removed by a later independent transaction.
No artifact is considered complete without its receipt, and no failed or skipped sample advances
its scheduling cursor.

A quarantine race is not reduced to diagnostic text. The transaction transfers its typed recovery
authority into the owning `RuntimeInstance`; `consumer_recoveries` exposes immutable inspection
records, while `restore_consumer_recovery(id)` and `cleanup_consumer_recovery(id)` provide the only
restore/cleanup lifecycle. Cleanup is refused until exact restoration has authenticated the public
inode.
