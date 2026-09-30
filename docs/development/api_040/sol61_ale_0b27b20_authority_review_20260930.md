# Independent ALE interval-authority review — 30 September 2026

Exact candidate: `0b27b20240a3a23061a2197e262b603783a6d258`.
The source/host review finds one required guard before approval: publication
authenticates the old physical state exactly, but does not authenticate the old
geometry exactly. No production file, header, installation or shared environment
was changed in this review. Native C++/MPI reception belongs to the integrating
worker and is not claimed here.

## Confirmed defect: geometry changed after preparation can be overwritten

`ProgramContext::runtime_state()` is public (`program_context.hpp:1604`) and
`ProgramRuntimeState::moving_interval_geometry_` is public. At
`program_context_moving_interval.inc:429–442`, publication checks the attempt,
point, generation and physical block. It then compares every physical-state
component against the evaluation's independent initial state. It does not
compare the accepted geometry's frame, previous interval, coordinates or
measures against an independent previous image.

Changing the accepted frame or measures **without increasing generation**, after
preparation but before publication, therefore passes the current commit guard.
The final swap replaces the changed geometry with the older prepared candidate
and appends its ledger. This is the same stale-input window already guarded for
physical state. It cannot be dismissed by assuming the publicly accessible map
immutable.

The independent compiled probe executes the complete, unmodified production
`commit_moving_interval` with actual carrier private access, actual attempt
leases and actual point equality. Two injections, a changed frame and a changed
measure, confirm publication and overwrite at constant generation. Scalar
storage and collective/physical/ledger execution are explicit host seams. This
is a demonstrated control-flow defect; actual Kokkos storage execution is not
claimed. The author accepted the finding and is preparing a separate correction
to authenticate previous frame/interval/layout and coordinate/measure values
against the owned receipt before publication. Re-review that correction before
granting source approval for stale-geometry protection.

## Received guards and ownership

The same harness receives **42 checks** covering successful publication and
refusals for foreign owners at preparation and publication, superseded/revoked
real leases, stale dt/tick/time/graph/stage point, stage and partial interval,
generation change, changed frame before preparation, changed physical input,
physical recovery failure, static EB mask, contract mismatch, local launch
failure, ledger admission failure, reused proposal, producer failure and
producer changes to point/generation/attempt. Missing transaction, wrong frame
and empty quadrature refuse at evaluation. The two confirmed-gap probes above
are reported separately and are not counted as passing refusal checks.

The harness compiles the real `RuntimeIntervalEvaluation` and
`PreparedMovingIntervalUpdate` headers. Compile-time probes confirm their
constructors are inaccessible to ordinary callers and the proposal is not
copyable. The real `PreparedResourceCache` supersedes/rejects its old attempt;
the real `BoundaryEvaluationPoint` compares its complete tuple. The real exact
contract serializer is used. The evaluation method captures the physical state
before the callback, passes its issued point to the callback, and checks point
and generation again afterwards. Preparation checks the old lease even if a
callback superseded it without changing the point.

Production storage ownership was inspected separately: `Fab` copy allocates
new storage and performs `Kokkos::deep_copy` (`fab.hpp:71–84`), `FaceField` owns
its `Fab`, and `MultiFab` owns its local fields. Carrier/receipt copies therefore
detach their buffers. The harness's scalar storage does not prove this property
numerically. Runtime snapshots copy the moving map together with the accepted
exchange ledger (`program_runtime_state.hpp:668`); prepared restoration revokes
resource work and publishes both by swaps (`:716`, `:743–744`). A rejected outer
transaction, including a parent rejecting after a child accepts, remains the
sole rollback authority. Arbitrary producer side effects are not undone by
`evaluate_moving_interval` itself; they require that enclosing transaction.

## Collective order and publication boundary

Preparation votes local preflight/allocation errors before the exact publication
contract. Version 3 authenticates frame, quadrature, interval, generation,
tolerance, component count, domain, rank space, periodicity, lane, every box and
every owner before ordered endpoint broadcasts. Global topology is checked as
a full fixed-domain partition. Each local numeric launch and fence is caught;
its error vote precedes the invalid-value vote. Empty local ranks still enter
these votes. Prepared physical recovery and EB refusal precede exchange-record
preparation. Record preparation and candidate copies each have a collective
failure boundary, and preparation does not append the live ledger.

The numeric preparation phase is byte-identical to the previously independently
fault-probed `ed8b4e0` phase: SHA-256
`bdf4c47b5f92b19b3f59b99ba623727cb11db7a0d781c25bb9361a243cd37d37`.
This equality was checked during this review; that earlier host execution is
not relabeled a fresh native receipt.

Publication votes target lookup/authority/layout errors, agrees its exact
contract, catches local physical-input comparison launches, votes changed input,
calls the generic prepared recovery validator, rejects EB, copies candidate
storage under a collective error guard, then appends the ledger and swaps state
and geometry. Its last swaps have static nonthrowing assertions. The real
`System::stage_program_exchanges` delegates to
`stage_exchange_batch_collectively`: a staging failure restores the prior
ledger prefix before rethrow. The generic validator performs collective block,
layout, recovery-authority, local recovery-error and failed-cell checks in
`system_fields.cpp`. Those real collective/storage implementations were read,
not executed by the scalar harness. No rank-order hang or MPI rollback is
claimed from serial seams.

## Provenance and remaining scope

The issued carrier binds the producing callback to one complete root interval,
attempt, point, geometry generation, frame and quadrature. Stage/partial
intervals are refused, including a cached `.1` resource under a `.15` point.
An authored callback can still compute an incorrect physical law, choose an
incorrect dt multiplier or use stale external data; the runtime metadata does
not prove those mathematical statements. Integrated source/flux amounts enter
the inspected Reynolds update once, without a second dt multiplier. Independent
manufactured and saved-receipt verification remain required at native reception.

Export, validation and restore explicitly refuse checkpointing a nonempty
moving carrier until a coupled geometry/receipt codec exists
(`system.cpp:277–302`). There is no completed checkpoint codec, public Python
SSA lowering, PDE qualification, AMR moving transfer or GPU qualification in
this review. Higher-dimensional swept geometry and moving physical-domain
boundaries remain refused by this provider.

## Reproduction and archive identity

```sh
rtk proxy /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python \
  tests/review/sol61_ale_0b27b20_authority.py
```

The script authenticates `git archive` of the exact commit's `include` tree and
`src/runtime/system/system.cpp`, SHA-256
`def4e9b962e340ae0aa6e400365bcd7eb657f738ddbddcb3992c39dbae549df8`,
then emits fragment hashes, generated C++ and the host executable under
`outputs/ale-sol61-0b27b20-source`. One serial C++20 compiler process was used;
the final execution completed in 1.49 s. Ruff passed. No native JIT/build or
package install was performed. Output explicitly distinguishes the 42 received
checks from the two confirmed missing refusals.
