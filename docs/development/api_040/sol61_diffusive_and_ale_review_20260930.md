# Independent diffusive/ALE review — 30 September 2026

This review reads exact commits `b03b47b231776324832a358b8dcb3d4261b53ba5`
and `88c755d31fd7a4f1ec3451d5917ed3d400a455cf`. It does not install them,
change the shared environment or claim native/MPI qualification.

## Diffusive traces: source/host approval for the bounded extension

An exact `git archive b03b47b` snapshot under
`outputs/diffusive-sol61-b03b47b-source` supplied Python source, public fixture,
tests and production headers. Its source identity was printed before reception.
The author's four source tests passed; the actual extracted production diffusion
producer with `AcceptedExchangeLedger` passed its host consumption/restore test
(2.36 s). These checks were replayed, not inferred from the author's report.

The independent `tests/review/sol61_diffusive_b03b47b_selectors.py` adds five
metadata probes. It checks selector identity against the actual producer emitter:
a source occurrence cannot contaminate the sole constitutive selector; fitted
drift/diffusion selects exactly their joint ordinals; two constitutive currents,
mixed transport/constitutive currents and source-only balances refuse an
ambiguous selector. All five passed. Numerical coefficients/quadrature remain
in the producer, while source evaluation identity remains in the consumer.
Conditional Program version 6 scopes the persisted selector extension; old
transport-only identity remains version 5.

No source defect was found in this bounded extension. Its native Uniform/AMR
operator, collective consumption, rollback and checkpoint restoration still
require reception against the matching rebuilt SDK. The author's native fixture
uses independently prescribed `u=1+x`, `D=.1`, finite dt and opposite boundary
amounts; those prepared tests were inspected but not executed here.

## ALE: two required guards before collective approval of 88c755d

Ownership is real: `FaceField` owns `Fab`; `Fab` copy allocates storage and performs
`Kokkos::deep_copy`. Copying `MultiFab` and face vectors therefore detaches trial
geometry/state. `AcceptedSnapshot` copies the geometry map and prepared restoration
swaps it back. The final field/carrier swaps are statically nonthrowing. Integrated
source and physical flux enter amount algebra once, without another dt multiplier.
Endpoint, previous measure, positivity, finitude and GCL checks precede ledger
append and accepted publication. Empty local ranks still enter the numeric vote.

**Confirmed defect:** the local numeric reduction loop at lines 206–248 is outside
a catch/vote guard. A rank-local execution/launch failure skips `all_reduce_max`
at line 249 while peers can enter it. The independent host script
`tests/review/sol61_ale_88c755d_fault.py` extracts that production phase unchanged
and injects an exception from its reduction launcher. A compiled C++20 executable
confirms one local launch and zero numeric/error collective votes. This is a
control-flow fault injection, not a real MPI hang claim. Catch local numeric
preparation/execution exceptions and collectively propagate them before the
invalid-value reduction.

**Missing local authority:** publication authenticates layout count, but the
following broadcast order/roots depend on exact boxes/distribution. Either cite
an enforced existing all-rank invariant or include domain, boxes, owners,
rank-space and periodicity in the pre-broadcast exact contract. The author has
accepted both findings and is preparing a separate correction. Re-review that
commit before granting source collective approval.

Supplied flux/source/face resources contain no runtime interval provenance.
Endpoint/sweep equations detect inconsistent geometry, but cannot authenticate
the physical/source quadrature duration of otherwise consistent arrays. That is
an explicit caller responsibility in this tranche. Public lowering, persistent
moving-carrier checkpoint/restart, AMR moving transfer and complete temporal
resource authority remain open; no PDE or public ALE closure follows.

## Reproduction

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python \
  tests/review/sol61_diffusive_b03b47b_selectors.py
rtk proxy /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python \
  tests/review/sol61_ale_88c755d_fault.py
```

The ALE probe accepts an optional replacement commit argument and checks whether
the same injected local failure reaches a collective error vote. It compiles one
small host translation unit; it does not load a native extension. Extracted
snapshots, XML results and executable remain untracked evidence, not product code.

## Corrective reception and temporal publication boundary

`b7dcaa6a33166f425ee0bfae39f3b9e3cbc353ef` resolves both collective findings:
the exact publication v2 contains boxes/owners/rank-space/domain/periodicity/lane;
local view/launch/reduction/fence exceptions converge before the invalid-value
vote. Replaying the same extracted-phase injection confirms **one collective
error vote and zero numeric votes** after the injected launch failure. The same
numeric seam passes on `ed8b4e09358234053d7d47bbf21088d525002cec`.
These are source/host control-flow receptions; actual MPI execution stays open.

The latter commit calls the existing generic prepared physical-recovery validator
on the candidate before ledger append/swaps and refuses static EB masks. Its
placement addresses the bypass in 88c755d; physical rejection execution is
independently received by the frontier reviewer/parent, not by this numeric probe.

The next temporal resource carrier must authenticate interval authority rather
than relabel raw arrays with the current point. `boundary_evaluation_point(0)`
includes stage/child phase, but `current_dt_` is the outer step duration.
Swept-interval amount algebra consumes an endpoint-to-endpoint interval. Bind
explicit start/end phases and actual interval duration/quadrature, accepted
geometry generation, physical state/layout/frame and attempt lifetime to an
owning proposal. Reject stale resources before numeric/ledger/state publication.
Until partial-interval rules exist, reject nonzero stage/partial child phase
instead of applying a full-interval swept update at an intermediate point. A
point stamp created at consumption cannot establish the preparation time of
physical flux, face reconstruction or source quadrature.
