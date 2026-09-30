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

Next connection: place a geometry carrier beside the actual ProgramRuntimeState
storage so the existing System AcceptedSnapshot copies/restores it with state,
cache/history and exchange ledger. The carrier must retain accepted coordinates,
cell measures and interval-qualified swept-face occurrences; its publication
must be transactional. Transport, CFL, source quadrature and saved outputs need
to consume that carrier explicitly. Public requests remain refused until those
connections are executable and independently received.
