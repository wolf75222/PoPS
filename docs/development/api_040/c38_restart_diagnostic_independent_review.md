# Independent source review of a6d2ca0

Reviewed exact commit `a6d2ca0fff88f7ffb478f21cfef479e5007625b4`, its parent
restart sequence, the existing `collectively_rethrow_exception` implementation,
its communicator overload and chunked byte broadcast, and the strict injected
second-refresh regression around `f3d3205`. No CPU test, compilation, native JIT
or performance operation was run during the performance freeze.

**Verdict: favorable, bounded source review; native MPI reception still required.**

The old wrapper voted on `regrid_error`, preserved the exception in serial, then
replaced every distributed diagnostic with a generic string. The replacement
helper begins with the same `all_reduce_max(error)` on the same `published_lane`.
On collective success it returns immediately, without any extra collective.
On failure every rank takes the same branch: lowest failing rank selection,
diagnostic preparation vote, length validation vote, length reduction,
allocation-success vote, then the same byte broadcast sequence. It forwards the
lane's exact communicator; there is no MPI_COMM_WORLD fallback or communicator
replacement. The regrid has finished mutating its prepared hierarchy before this
borrowed lane is selected, and the helper does not mutate that hierarchy.

Serial execution rethrows the original `exception_ptr` immediately after the
failure vote, retaining its type and text. MPI intentionally returns a common
runtime_error containing the selected rank and original reason, including a
nested resource-refresh diagnostic. Selection does not promise to aggregate all
errors: the lowest failing rank is the existing helper's explicit authority.

Diagnostic string construction on the authoritative rank and diagnostic resize
on each rank are inside separate catch/vote phases. Failure leads all ranks to
throw `bad_alloc` before any following broadcast. Excessive length is likewise
voted before conversion/broadcast. Broadcast chunks are deterministic from the
shared length and use the same communicator. An allocation failure constructing
the final runtime_error can change its local exception type, but it occurs after
all helper collectives and still throws; it cannot fall through into publication.
This review does not claim recoverability from arbitrary MPI transport failure or
from a failure before ranks reach the existing outer error rendezvous.

Publication ordering is unchanged. `restart_history_authority` and restored-slot
swaps occur only after the helper returns successfully. The helper itself neither
commits nor finalizes the restart transaction. Importantly, this does **not** mean
there was no internal mutation: the tested first child topology publication can
precede failure of the second resource refresh. That provisional hierarchy remains
under the existing restart snapshot and is restored by the caller's rollback.
The exact test still checks boxes, epoch, accepted bytes/revision, exchanges, flux
shards, clock/cadence, all level states and history slots/provenance, then retries.
No assertion or scientific threshold was relaxed by a6d2ca0.

The reported 79 passes / one MPI aggregate failure / two skips and previous
rollback/retry behavior are root's prior execution evidence, not results executed
by this reviewer. A new MPI2 receipt on rebuilt a6d2ca0 remains necessary to mark
the diagnostic assertion and aggregate as passed.
