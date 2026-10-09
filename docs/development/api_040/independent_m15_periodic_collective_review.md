# Independent review: M15 periodic MPI rejection

Reviewed SolReference's `30a0cb6`, test follow-up `33e687a`, and exact-route
consensus patch `698b91c` in `PoPS-degenerate-diffusion`. This review is source
inspection only: no JIT, native build or MPI execution was performed here.

## Failure and repaired route

The reported installed Dim1 failure is consistent with the source. For a
periodic generated block, `prepare_bound_physical_group_` prepared the transport
evaluator but published its core only when physical boundaries existed.
`block_rhs_group` consequently selected the legacy unprepared group route. A
face-domain exception on the sole box owner could escape while a rank with no
local faces continued into later collectives.

The correction publishes the prepared evaluator when both periodic callbacks
exist and invokes the callback for the requested full/flux mode even without
face retention. Generated prepared face materialization has its own collective
failure phase before divergence/source assembly. The containing transaction
uses detached candidate state/output and restores its journals on refusal.
This repairs the identified face-domain phase without changing the physical
closure, admissibility test, periodic topology or numerical method.

Mixed groups retain a legacy fallback for blocks without the selected prepared
callback. Its new collective phase prevents a local return/throw from sending
one rank into the next block while another leaves the group. This does not prove
arbitrary callbacks safe if they themselves enter collectives after an unvoted
rank-local failure.

## Findings closed during review

1. Optional callback presence was initially absent from the collective contract.
   A two-block group with one prepared block on every rank, but a missing
   `periodic_full_at_point_prepared` on only rank zero of the second block, could
   retain group route two everywhere and then split into fallback/prepared paths.
   The final patch compares physical/boundary/full/flux presence and block count
   before selecting periodic preparation or its absence, and encodes per-block callback presence
   again in the RHS group contract before evaluation. Exact mismatch refuses
   before either branch is entered.
2. The MPI test initially asserted local rollback time before `state_global`.
   A broken rollback could therefore strand a peer in that collective. The test
   now gathers clocks before asserting. It also gathers actual local box bounds
   and requires exactly one global box, establishing that non-owning ranks have
   no local face at which to raise the injected negative-density error.

The test keeps the exact numerical refusal message and verifies unchanged
global state and clocks. An external timeout remains necessary when running the
MPI regression against the old implementation, whose expected failure is a hang.

## Scope and remaining evidence

Source verdict: favorable for the repaired Uniform periodic route and the two
closed collective-ordering findings. Existing physical-boundary dispatch and
exact point/lane checks remain in place. `mark_bound` already obtains the prepared
execution lane before this helper, so the new route vote does not invent a
fallback communicator for old bindings.

No `AmrSystem` path calls these `System` group helpers; this patch does not
qualify AMR failure handling. A local Kokkos failure while materializing detached
state before a callback's internal collectives is also outside the demonstrated
face-domain failure window. Direct legacy callbacks and group-wide atomicity of
arbitrary scratch outputs are not newly established by this test.

Central reception must run the exact new MPI2 test under timeout, plus existing
periodic/physical interface session tests and successful M15 evolution. The
review does not substitute source reasoning for that native evidence.
