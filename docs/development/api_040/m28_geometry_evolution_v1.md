# M28 geometry-evolution request and swept-interval primitive

Contract extension: `pops://geometry-evolution/moving-control-volumes@1`.
It does not change the input API 0.4.0 document, package version or native ABI.
The request describes a fixed physical domain and fixed topology. Physical
boundary motion and topology changes require separate boundary/continuation
contracts. Arbitrary AMR is still an implementation obligation, not excluded
from the production objective by this first tranche.

The mathematical authority is the handoff original, lines 3085–3130: Reynolds
balance uses physical flux minus density times mesh velocity; the same swept
measure must enter the volume and amount balance. A static regrid is different.

`GeometryEvolution` accepts coordinate maps made with the existing generic
`ScalarExpr` algebra. `MovingControlVolumes(layout, evolution=...)` composes a
normal layout with this requirement. Frame/rank and all coordinate, clock,
parameter and input identities remain explicit in its canonical data. The
coordinate law affects layout identity even when the reference grid is equal.

Resolution refuses the request before compilation, with gate
`native_geometry_evolution_unavailable`, classification `IMPL`, and the missing
native obligations. `Geometry<Dim>` is an immutable uniform bounds/spacing
mapping. The existing Program/AMR flux and observation paths still use that
static spacing; StateStorage has no accepted/trial per-cell moving measure.
Changing a frame parameter, adding a geometric source to an Eulerian density,
or calling regrid would not establish this contract.

`SweptInterval` supplies one native device-clean geometric primitive. In 1D,
each oriented swept face measure is its endpoint displacement. Host preparation
checks actual old/new coordinates independently against supplied sweeps and
GCL; collapsed/inverted/nonfinite cells and stale-duration sweeps are refused.
The device update accepts integrated physical fluxes and source amounts; it
does not multiply them by a duration a second time. Face reconstruction and
source projection remain responsibilities of the caller. It is not a complete
ALE execution provider or a second numerical runtime.

The counterexample is already enough to disprove a static carrier: old faces
`(0,.5,1)`, new faces `(0,.52,1)` give measures `(.52,.48)`. A stationary
physical density 2.3 requires amounts `(1.196,1.104)`. Keeping old amounts
`(1.15,1.15)` conserves global mass but produces densities approximately
`(2.21153846,2.39583333)`; changing only coordinates fails constant preservation.
The `.1` duration sweep `.01` cannot be reused for `.2` endpoints `.02`, even
though global length remains one. These are exact local mathematical examples,
not runtime simulation results.

Source-only tests authenticate the preserved body, identity and rejection path.
Native primitive tests added to `test_geometry` cover constant preservation,
oriented exchange, nonzero physical flux/source and stale-duration refusal.
They require central native rebuild/reception; no installed PoPS artifact,
MPI/AMR execution, scientific saved-state receipt or GitHub CI is qualified by
this document.

## Native transaction connection (source implemented, execution pending)

`MovingIntervalGeometry<Dim>` now lives in `ProgramRuntimeState`, alongside the
existing ledger/history/cache. It owns Kokkos face coordinates, swept measures
and cell measures, the exact bound physical state, generation and runtime
interval identity. The System's existing AcceptedSnapshot deep-copies it; the
prepared restore explicitly copies and swaps the geometry map. Child acceptance
therefore remains provisional inside a parent transaction.

`ProgramContext::initialize_moving_interval_geometry` initializes the actual
represented geometry at an accepted boundary. `advance_moving_intervals` takes
shared-face coordinates, swept measures, physical integrated fluxes, reconstructed
densities and cell-integrated source. It verifies patch/partition identities,
exact patch endpoint agreement across ranks, fixed physical boundaries,
periodic trace compatibility, cell validity and endpoint/GCL equations before
publishing. Only patch endpoints cross ranks; the state is updated in its native
Kokkos partition. The complete proposal/point/component/tolerance/generation
declaration must agree across the prepared execution lane.

The native first provider implements actual 1D intervals. A higher-dimensional
call is refused because swept surfaces/orientations have not been prepared;
this is a missing provider, not a production dimensional restriction. It is not
served by the public descriptor yet. AMR moving-volume transfer/reflux is also
unimplemented. Existing static flux, CFL, source and scientific output paths
cannot be advertised as consumers of this new carrier.

The candidate state, geometry and rank-local exchange records are prepared
before publication. Source projection is an explicit integrated cell amount.
Reynolds relative flux records and swept-volume records retain the same cell/side
occurrence and temporal quadrature; source records retain cell/component
occurrence. The facade remains the sole commit/rollback authority. Duplicate
interval publication is refused. Reuse across `.1/.2/.3` is qualified by the real
runtime point and independently checked against proposed endpoints.

Three Dim1 native tests are added to `test_program_runtime` and its MPI2
aggregate: constant-preserving motion at 8/24 cells with component permutation,
post-publication rejection/child acceptance/parent rollback/retry; stale-duration
sweeps; and a manufactured open-boundary balance with nonzero physical flux
`F=.7*x`, growth `.4`, projected source `.4+.7` and exact linear space-time
quadrature. They inspect native state, coordinates, measures and ledger, and
independently reconstruct inventory. Their execution remains the central
integrator's responsibility; no runtime pass is reported in this isolated
worktree. Source syntax is checked against the actual Kokkos/OpenMP/MPICH headers.

The new carrier changes `ProgramRuntimeState` storage layout. Central integration
must rebuild host/generated code together and advance native ABI coherently;
there is no compatible mixed installation. No checkpoint codec for the carrier
is added in this tranche, and no ALE checkpoint/restart or persisted scientific
receipt is qualified. Public requests remain refused until common Program
lowering, saved moving geometry and restart can authenticate this carrier.
The SDK manifest includes both the primitive and carrier/include fragment.
# Collective publication review

The publication contract authenticates every global box, its broadcast owner,
the rank-space, prepared lane, physical domain and periodic topology before
the endpoint broadcast schedule begins. Local numeric preparation (including
device launches and the fence) catches failures and reaches a collective error
vote before the following invalid-value reduction. An exception on one rank
therefore cannot skip that reduction while its peers enter it.

This fixes the source defects found in independent review of `88c755d`.
Source syntax validation is separate from native MPI execution evidence.

Publication also invokes the physical block's existing prepared recovery guard.
A finite but negative gas density is rejected before ledger staging and swaps.
The native retry probe now injects that state through a genuine integrated
source and checks unchanged state, geometry and ledger before its next retry.
The interval provider rejects active embedded-boundary masks: static masks do
not supply the missing moving physical-cell measures.

## Issued intervals and coupled candidates

`RuntimeIntervalEvaluation` has a private constructor. The existing native
`ProgramContext` issues it while executing a generic producer receiving the
actual `BoundaryEvaluationPoint`. It owns the coordinates, swept volumes,
integrated physical flux, face density and integrated source. Its authority
includes the issuing context, accepted geometry generation, physical frame,
quadrature, exact point/duration and the existing revocable attempt lease.
The bare-vector publication overload has been removed. A rejected attempt's
allocations may remain owned, but its interval cannot be consumed by a retry.
The producer remains responsible for its authored physical formulas; arbitrary
C++ callback contents are not mathematically proven by a provenance stamp.

This first provider requires a complete root interval (phase `[0,1]`, start
stage zero, physical endpoints `point.physical_time` and that time plus
`point.dt`). Evaluation, preparation and commit all reject partial intervals or
stage offsets until their geometry quadrature is explicitly implemented.

`prepare_moving_interval_update` returns an owned `PreparedMovingIntervalUpdate`
without publishing the state, geometry or exchange ledger. Commit authenticates
the same point and live attempt and compares the original state against the
actual native state before its recovery guard and coupled publication. A
candidate is consumable once. This is the native seam for a coupled SSA value;
it does not yet open the Python public ALE backend gate.

Accepted geometry retains an independent receipt: previous coordinates,
previous measured volumes and physical state, integrated source, physical face
flux and density traces, runtime point, frame and quadrature. The native probe
recomputes Reynolds and GCL directly from these stored values without calling
the update primitive. Checkpoint codecs and public observer exposure remain
required before Python ALE acceptance is enabled.

Until that coupled wire codec exists, native exchange checkpoint export and
validation/restore explicitly refuse a nonempty moving-geometry carrier. This
also covers externally authored C++ Programs. The mandatory Uniform checkpoint
continuation capture reaches this refusal before publishing a checkpoint file;
an image that silently drops the accepted moving geometry is forbidden.
