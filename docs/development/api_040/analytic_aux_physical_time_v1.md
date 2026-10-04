# Analytic auxiliary physical stage time

`AnalyticAux(target, expression, frame=...)` accepts `time(program.clock)` in a
parameter-free scalar analytic expression. Geometry, target ownership and declared
Clock identities remain exact. The temporal descriptor carries
`temporal_contract="analytic-aux-time@1"` and its authenticated `time_clock`.
Expressions without time retain their former descriptors and emitted C++ bytes.
An expression cannot treat distinct logical Clocks as one physical value: this
consumer boundary supplies one declared Clock, not an implicit clock conversion.

The generated launcher reads its physical time from the consuming
`AuxiliaryEvaluationPoint` before issuing the kernel. The time is a scalar capture,
using a postfix input slot separate from CellBounds inputs. Existing analytic
consumers keep the default time input slot zero. No Python callback runs per cell.
The temporal prepared launcher uses identity `pops.analytic-aux-time@1.*`, version 2;
the static launcher keeps its former identity and version 1.

The Uniform and existing single-level AMR Program preparation paths qualify an
auxiliary point from their actual `BoundaryEvaluationPoint` inside the existing
collective preflight. The boundary already contains the Program clock, accepted
tick, level/substep/stage, exact rational stage fraction, trial dt and physical time.
Preparation verifies the logical identities and finite physical values, and the
existing rank agreement includes the full boundary before publication. A missing
temporal payload or another Clock is refused before a temporal kernel. The existing
stage-state layout validation and evaluation sequence bind the consumer invocation;
analytic expressions do not read mutable state leaves.

Point contract version 3 adds an optional physical lease. Cache equality includes
its rational fraction and exact binary64 dt/time bytes, along with all former
integer epochs and event fields. Neither approximate time comparison nor inference
of a stage from a floating value is used. Unqualified static points retain version 2
serialization bytes. Candidate rejection and full runtime Registry/carrier snapshots
retain their existing publication/rollback semantics; snapshots copy the whole live
point. A retry with a changed trial dt or physical stage cannot reuse the rejected
lease. The actual auxiliary provider remains evaluation-fresh.

Physical leases are transient. Durable auxiliary checkpoint capture copies accepted
points and clears only that optional lease. The checkpoint wire shape and discrete
accepted provenance remain unchanged; after restoration, a temporal live point
differs from the discrete-only restored point and forces a recomputation. When merged
with accepted-publication invalidation, keep the checkpoint invalidation tombstones
and clear the lease at the same copied-provider insertion. Do not erase invalidations
or synthesize physical time from a durable accepted tick.

The internal C++ layout changes: on the measured Host toolchain the point grows
from 64 to 104 bytes. LaunchContext remains 56 bytes and refers to the changed point.
This requires rebuilt modules and generated packages. Both modified headers are
authenticated `sdk-support` rows in `pops_headers.manifest`; the official header
signature changes. The true `_check_headers_match_module` compilation guard and
`check_compiled_matches_module` cached-package wiring guard reject differing known
signatures. The C++ loader compares the complete TU-local `POPS_ABI_KEY_LITERAL`
before the package protocol, manifest and registrar callbacks. Builds reporting an
unknown header signature retain the existing degraded Python guard; use a real
authenticated SDK and C25 inspection for qualification. This patch does not declare
binary compatibility or independently change the shared published release ABI;
ROOT owns that decision and official regeneration after the coherent integration.

Source proofs cover four complete public compositions, including SSPRK2 fractions
0 and 1, quarter/three-quarter stages, read-only catalyst and permuted inputs. Real
Host launcher execution checks two non-square layouts, every value against the
original analytic formula, exact accepted storage/points/generation after rejection
and rollback, retry dt/2, cache invalidation, and durable checkpoint restoration.
The complete Program/Model TUs are compiled for syntax; they are not a native
Program execution. A valid externally constructed C++ boundary with a wrong finite
time cannot be distinguished from an issuing scheduler merely by reading its DTO;
the production authority is the Program boundary/Clock service, not an auxiliary
reconstruction of time. No Native, MPI, GPU, end-to-end coupled evolution, or
scientific checkpoint qualification is claimed by these Source/Host checks. Fresh
installed-header/DSO/C25 and runtime stage/rollback tests remain required after ROOT's
new build.
