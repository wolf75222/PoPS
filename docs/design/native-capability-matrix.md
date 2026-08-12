# Native Capability Matrix

ADC-549 makes route support explicit. A Python public feature must either map to a real native route
or report an unsupported route before compile, bind, or runtime execution.

## Matrix Schema

ADC-591 adds a versioned native report above the route rows:

- C++: `pops::native_capability_report(target)` returns a `NativeCapabilityReport`.
- Python native binding: `_pops.capability_report(target)` returns the same report as a stable dict.
- Public Python: `pops.inspect(obj)` is the sole inspection dispatcher. Layout, compiled-artifact,
  and bound-runtime reports embed the relevant native rows without exposing a second root facade.
- Runtime: `pops.inspect(sim)` includes the native report plus the authenticated bind/install
  identity, layout plan, execution context, consumer graph/cursors, profile, diagnostics,
  history/cache metadata, and runtime environment facts. The internal adaptive runtime view returns a
  `pops.runtime.amr.RuntimeInspection` composing `hierarchy` (config envelope + live patches),
  `patches` (the live patch census), `regrid` (cadence + union-tag criteria), and `limitations`
  (the non-available rows of the same native report, filtered to `status != "available"`).
- Compiled artifacts: `compiled.inspect().to_dict()["capabilities"]` carries the same route IDs and
  statuses, projected from the artifact manifest without loading or recompiling the `.so`. On the
  AMR route, `pops.inspect(compiled.layout)` reports the exact layout `pops.compile` attached,
  including its refine/regrid policies. There is no artifact-specific layout inspector or override.

Pretty strings are views of these objects only. Tests should assert on `to_dict()` fields such as
`schema_version`, `abi_version`, `runtime`, `capabilities`, `routes[*].route_id`, `status`, and
`reason`; they should not parse the printed table.

Every route row is a plain metadata record with these fields:

| Field | Meaning |
| --- | --- |
| `feature` | Stable feature token, for example `layout:AMR`, `elliptic:fft_amr`, or `checkpoint:parallel_hdf5`. |
| `route_id` | Stable native route identifier. Today it equals `feature`; it is explicit so route IDs can diverge later without breaking tests. |
| `layout` | Layout envelope the row applies to: `uniform`, `amr`, `uniform|amr`, or `context`. |
| `backend` | Execution authority required: `production`, `runtime`, `module`, `native`, `external_cpp`, or `none`. |
| `platform` | Platform axis: `host`, `mpi`, `gpu`, or `context`. |
| `mpi` | Whether this row is backed by an MPI-capable route for the current build/artifact. |
| `gpu` | Whether this row is backed by a GPU-capable route for the current build/artifact. |
| `status` | `available`, `unavailable`, `partial`, or `unknown`. Known unsupported routes use `unavailable`. |
| `limitation` | Short human-readable limitation or constraint. |
| `reason` | Same limitation in the native C++ report; `limitation` is the compatibility alias. |
| `error_message` | For unavailable rows, the message shape used by validators: requested route, available route, alternative. |

The same shape is exposed by:

- `pops.Case(...).explain_routes()`
- descriptor `capability_matrix()` methods
- `pops.inspect(compiled)["capabilities"]`
- `pops.inspect(sim)["capabilities"]`
- the internal compiled-artifact manifest used by bind validation
- `_pops.capability_report()["routes"]`

## Native Inventory

The canonical inventory lives in `pops._capabilities.native_capability_matrix()`.

Supported native routes include:

- Uniform single-level layout. A `Uniform(...)` layout with an active AMR refinement criterion
  attached is refused by `Case.validate` by default (ADC-589/ADC-555); the explicit escape is
  `Uniform(mesh, refine=..., ignore_amr=pops.amr.IgnoreAMRCriteria())`.
  The current native `SystemConfig` has one `n`, one `L`, and a two-coordinate origin, so public
  `CartesianGrid` lowering requires equal axis lengths and equal axis cell counts. Its typed
  per-axis periodicity lowers every Cartesian `PeriodicAxes` partition. Rectangular and anisotropic
  grids fail before engine construction; translated and partially periodic grids retain their exact
  authored geometry and topology.
- Multiple distinct Uniform layouts in one `RuntimeInstance` when the Program is exactly separable
  per layout and every directional exchange is backed by an authenticated native `Transfer`
  component. Mixed Uniform/AMR, heterogeneous AMR, implicit reverse mappings and co-located
  cross-layout kernels are unavailable rather than collapsed onto a representative layout. The
  executable route requires exact `FixedDt`, no unresolved aux storage, no `FieldOperator` plan and
  no boundary plan without a per-layout installation authority. A global CFL step is unavailable
  until a qualified inter-layout reduction is present. `before-step@1` snapshots every transfer
  source before any target write, so chains and explicit cycles read one pre-transfer state.
  Concurrent overwrite transfers to one target at one synchronization point are rejected until an
  explicit merge operation/provider is selected. The supplied
  `CONSERVATIVE_CELL_AVERAGE_V1` operation accepts distinct integer-aligned resolutions only on the
  same physical Cartesian extent and boundary topology. Non-coincident geometry requires a separate
  coordinate-aware Transfer operation/provider and is refused by this operation.
- Conservative two-block interfaces through one authenticated native `NumericalFlux` evaluation
  and opposite residual scattering. Endpoints must be co-located on one layout and their explicit
  default-flux RHS evaluations must be simultaneous and contiguous in one Program point.
  `MPI_COMM_WORLD` layouts may distribute the two face decompositions independently: native C++
  collectives reconstruct both traces, require a finite bit-identical shared flux on every rank,
  then scatter only into locally owned residual cells. `MultiFab`/`DistributionMapping` ownership is
  still indexed in the process-world rank space, so interface installation performs one explicit
  admission comparison against that storage rank space. It then retains the world-congruent
  communicator carried by `ExecutionContext`; trace, failure, flux and registry collectives never
  reacquire the process world in the numerical hot path.
  Each endpoint trace plan retains its projection Handle, authenticated reconstruction-provider
  identity, operation and provider-derived stencil depth in the collective identity
  `pops.multiblock.interface-plan.v2`. The current
  type-erased scheduler executes the exact first-order cell-average operation. A MUSCL/WENO endpoint
  retains its higher-order reconstructed-face requirement but fails before native installation
  because no mapped-halo reconstruction provider is installed; it is never silently replaced by a
  cell-average trace.
  A public `AMRRegrid.frozen()` hierarchy may contain any positive materialized L0 prefix. A
  dynamic hierarchy is also executable when at least two levels are configured, its complete
  configured prefix is active at bind, and every accepted regrid preserves that active depth. The
  scheduler rematerializes the authenticated per-level routes on the replacement BoxArray and
  DistributionMapping before the next Program stage; missing full-face coverage or a depth change
  rejects the regrid and restores the accepted registry. At depth greater than two, the current
  coarse-to-fine regrid transaction can replace the finest transition while all ancestors remain
  unchanged. Replacing a non-finest transition temporarily removes its descendants, so that route
  fails closed until rematerialization can stage the complete candidate hierarchy atomically.
  Endpoint-qualified canonical fragments retain exact Program weights and authoritative local
  substep duration. An interior level publishes the same canonical evaluation to both adjacent,
  level-qualified coarse/fine audit pairs. Those fragments authenticate the paired RHS update; they
  are not injected again into reflux because that would duplicate the same face flux.
  Both endpoint hierarchies must expose matching full-tangential fine-face coverage. The level-zero
  route is installed before hierarchy bootstrap. Each successfully created fine route is installed
  before that level becomes the parent of the next transition; only those exact routes can authorize
  proper-nesting support across an omitted physical-boundary face. This route does not mirror one
  endpoint's AMR tags through the interface mapping.
  One narrow shared implicit JVP route is production-executable on host/serial. Resolve authenticates
  exactly two participant blocks connected by one interface on a frozen two-level hierarchy, two
  state-only `rhs_jacvec` nodes in one packed matrix-free apply, and both base residuals in the same
  top-level atomic RHS round. An unrelated third block may carry the two-component packed Krylov
  vector, but cannot participate in another interface. Code generation consumes that exact resolve
  evidence before emitting the paired call. The native
  `level_rhs_jacvec_pair` primitive perturbs both endpoint states before one shared-flux evaluation,
  so its finite difference includes both cross-interface derivatives. A generated Program now
  compiles, binds and runs GMRES through more than one paired interface evaluation per level while
  preserving the uncommitted packed carrier state. The public capability is therefore `partial`,
  not unavailable.
  Field-coupled boundaries, dynamic hierarchy mutation, additional participating interfaces and mixed
  apply operators fail closed. Bind requires the exact materialized prefix `(L0, L1)` and rejects
  MPI, non-host devices and non-host memory before native interface installation.
  Cross-layout interfaces without an explicit Mapping/Transfer provider, dynamic active-depth
  changes, non-finest dynamic replacements at depth greater than two, and historical
  shared-interface rates remain unavailable. Frozen and depth-preserving dynamic
  refined interfaces use the same exact `MPI_COMM_WORLD` trace and replacement-registry consensus
  as the flat route. Dynamic rematerialization stages a detached collective candidate; a
  rank-local failure restores the accepted layout, topology epoch, evaluator audit count and
  executable registry before retry. Every rank evaluates the canonical shared flux and scatters
  only to its locally owned endpoint cells. Rank-changing dynamic refined rematerialization remains
  unavailable.
- AMR through the native production route with hierarchy depth controlled by resolved resource
  policy. Transitions are exactly 2D, isotropic `ratio == (2, 2)`, share one isotropic buffer and
  one lookahead across the hierarchy, and currently select the exact native policy routes
  `shared_n_level`, `berger_rigoutsos`, `box_array`, and `round_robin`. Physical transfer providers
  expose exact dense cell/face-x/face-y/node contracts; restriction, coarse-fine fill and temporal
  interpolation are cell-centered on the supplied route. Derived fields use `elliptic_solve` and
  caches use `patch_topology`; unsupported provider contracts fail before artifact creation.
- Finite-volume spatial discretisation on the 2D core.
- One prepared, model-aware 2D transport-boundary plan shared by Uniform and AMR native/compiled
  routes. The capability matrix marks this route `partial` and names its exact built-ins:
  periodicity, extrapolation, constant or `RuntimeParam` fixed state, conservative device-side
  analytic fixed state over typed `(x,y,t,params)`, fixed-state primitive inflow converted once
  through the exact compiled block-model `to_conservative` provider, typed-role slip wall, and a
  typed `NoFlux` face. `NoFlux` uses the plan's prepared extrapolation for reconstruction ghosts,
  then zeroes the already evaluated face flux before divergence and AMR reflux; it is not a masked,
  polar, or embedded-boundary side channel.
  Analytic programs are immutable postfix tables evaluated in native device kernels at the exact
  `BoundaryEvaluationPoint`; no Python callback or hot-loop allocation is retained. The analytic
  finite-value contract is strictly non-mutating: one device preflight and one communicator
  reduction complete before any same-level, periodic, MPI or physical halo write. The commit kernel
  then evaluates the program again; this deliberate two-pass route avoids a per-cell scratch field
  but retains one blocking collective per analytic boundary fill.
  The analytic route remains `partial`: primitive per-point conversion and discrete state/field/input reads are
  rejected, as is an analytic ghost depth larger than the normal domain extent. Analytic faces with
  axis-permuted periodic coordinates also fail closed until a prepared coordinate map exists. The
  conversion route is explicitly `partial`: conservative-to-primitive recovery and arbitrary
  representation components remain unavailable, and conversion does not invent a boundary
  admissibility projection. Characteristic no-inflow is now an explicit narrow `partial` route:
  `Inflow(state=U, value=U_ref,
  characteristic=pops.boundary.model_characteristic_no_inflow(U))` requires a conservative
  constant/`RuntimeParam` fixed reference and the exact generated `m.roe_from_jacobian()` provider.
  Its Kokkos kernel evaluates
  the complete model flux Jacobian (1..16 components), orients it with the physical-face normal,
  applies the strictly incoming spectral projector, and leaves the scale-relative sonic subspace
  neutral. A collective real-spectrum preflight precedes publication; any failure restores the
  complete ghost transaction and never selects scalar, Rusanov, or Euler-specific logic. This
  qualification is currently 2D Cartesian host serial; primitive/analytic reference states,
  state/field-dependent auxiliary eigenstructure, sonic-error policy, MPI/GPU qualification, 3D,
  polar and embedded/cut-cell geometry remain unavailable. The native selector now authenticates
  these limits with one `dimension x geometry x operation` spatial-provider matrix: a 2D
  staircase/cut-cell residual cannot be mistaken for a metric-aware characteristic or boundary
  linearization provider, and a non-Cartesian coordinate provider is refused before native
  allocation. Post-Riemann
  transformation is instead an explicit `partial` route: a typed
  `BoundaryFlux` component receives the already evaluated outward-normal flux and executes between
  the Riemann solve and divergence/reflux through the same prepared Uniform/AMR plan. The runtime
  converts lower and upper faces to outward orientation before the call and converts the result
  back to canonical positive-axis face storage afterwards. This route is currently 2D Cartesian
  host-batch execution; it has no device-native or embedded/cut-cell metric ABI, and the ordinary
  Uniform route materializes face fields when selected.
  These requests fail during resolution or lowering; none silently degrades to component-wise
  ghost filling. A native rank-1/2/4 regrid fixture removes and recreates the fine hierarchy, then
  proves that uncovered internal fine ghosts retain the conservative coarse-fine transfer and are
  never treated as physical faces by the rematerialized prepared boundary session. The explicit
  public route is
  `Inflow(state=U, value=primitive_values, representation=Primitive(),
  converter=pops.boundary.model_primitive_to_conservative(U))`; the converter is derived from the
  authenticated block state and cannot name an unrelated callback or kernel.
  `primitive_values` follows the model's declared primitive-variable order.
- Native Riemann routes: Rusanov, HLL, HLLC, Roe, subject only to exact model capability
  requirements. HLLC and Roe keep one native route each; the compiled model separately authenticates
  the model-side provider (`fluid_roles_v1`, `direct_action_v1`, or `flux_jacobian_v1`) and the
  typed Roe entropy policy (`riemann.Harten(delta)`, `riemann.NoEntropyFix()`, or provider-owned).
  Missing, unknown, or flag/provider-mismatched evidence fails before native installation, and
  compiled inspection reports every distinct provider/options record instead of collapsing it to a
  Boolean. Cartesian Uniform and AMR dispatch use the same provider identity. The annular
  `PolarMesh` descriptor is geometry/output-only and is refused before compilation.
  `riemann:typed_failure_outcome` is deliberately `partial`: every built-in returns the common
  device-copyable `FluxEvaluation` with typed status, stability bound, reason code, requested/used/
  last solver identity and attempt metadata. A single-solver route remains explicit, while a
  typed public `riemann.Recovery(primary=Roe(), fallbacks=(HLL(), Rusanov()))` descriptor lowers to
  the sole statically instantiated C++ `PreparedRiemannRecoveryPolicy<RoeFlux, HLLFlux,
  RusanovFlux, RejectRiemannRecovery>` in the ordinary Cartesian Uniform/AMR face hot loop. Other
  orders, duplicate or configured candidates, external descriptors, and untyped values are refused
  before compile; annular polar geometry is explicitly unavailable. Only a typed candidate rejection
  advances; retry and fatal outcomes remain terminal. The route remains `partial`: block/team and MPI
  fallback counters, GPU qualification, restart publication metadata, backend matrices and
  performance budgets are not yet delivered.
- Prepared variable recovery is explicitly `partial`. One block-prepared closed-form method returns
  a device-copyable `RecoveryOutcome`/`RecoveryReport`. Type erasure retains both the selected and
  last-attempted method kinds, so a successful fallback or a refusal cannot be reported as an opaque
  chain index. System conservative-to-primitive and transactional analytic initial-state
  materialization plus Cartesian, masked, and embedded-boundary face reconstruction consume
  publication permission before copying a candidate or evaluating a flux. Primitive-to-conservative
  setup conversion similarly publishes only a finite candidate accepted by that prepared inverse
  authority. Accepted AMR regrid prolongation and restriction candidates also pass that
  block-prepared inverse authority collectively before replacing live hierarchy state. AMR
  bootstrap commits, rematerialized history slots, and physical boundary traces use the same
  publication gate and restore their complete transaction on refusal. Generated Program terminal
  commits also validate every Uniform or AMR live-state candidate before the first multi-block copy,
  including endpoints assembled from model-local and coupled sources. This route adds no implicit
  repair or fallback. The host Uniform `get_primitive_state` materializer now owns one per-block,
  per-local-cell warm-start slot qualified by exact conservative input plus topology and accepted
  state generations. It stages each slot through `RecoveryPublicationTransaction`, publishes the
  primitive array only after the complete batch succeeds, and explicitly invalidates every slot when
  a batch is refused. The separate `recovery:complete_consumer_cutover` capability remains
  `unavailable`: face-reconstruction kernels and AMR do not yet own persistent recovery warm starts,
  AMR regrid migration and checkpoint/restart do not persist such slots, and manual in-place Program
  writes, backend parity, and performance evidence do not yet share that authority.
- Native reconstruction routes: first-order, MUSCL, WENO5/WENO5-Z.
- Elliptic `CartesianCG` on a uniform 1D/2D/3D `System`, `GeometricMG`/FAC on AMR, and FFT on
  its separately qualified uniform periodic constant-coefficient route. `GeometricMG` is not an
  alias for the uniform CG backend.
- Matrix-free Krylov descriptors: CG, BiCGStab, GMRES, Richardson.
- ProgramContext install on System, and AMR program install when compiled for `target="amr_system"`.
- A native C++ `amr:cell_local_temporal_transport` route partially proves scientific consumption of
  the prepared cell-local executor. `Program.cell_local_time(...)` and generated
  `AmrProgramContext` code wire the exact bounded route. On an exact-rank host hierarchy it advances
  independent multi-block, multi-level and MPI-owned multi-box state packs with forward Euler and
  publishes one aggregated basis per route into the existing authoritative coarse/fine ledgers only
  at the synchronization barrier. The authored finest-level rung is lifted through integral
  power-of-two temporal relations to one homogeneous rung per level-group and one FE batch per
  hierarchy window. Its contract authenticates the block map, model-owned spatial parameters,
  selected limiter/Riemann routes, topology/materialization and exact execution lane. Same-rank
  restart/regrid rematerializes topology-derived records and clocks while invalidating the
  non-persisted last-interval diagnostic view until the next accepted step. It does not claim
  heterogeneous per-cell rungs, global/interface block coupling, non-dyadic clocks, source/field
  integration, physical or non-periodic boundaries, GPU default execution or memory spaces,
  performance qualification, rank-change rematerialization or diagnostic-view persistence. The
  physical-boundary and device envelopes are refused collectively on the exact lane during
  preparation, before any prepared state or boundary stage is entered.
- Generated local implicit-source Programs on synchronous two-level 2D AMR. `pops.lib.time.IMEX`
  lowers its local residual to the sole prepared `LocalNewton` service on every active level and
  consumes the returned `SolveOutcome`; it does not invoke a spatial-runtime time integrator. The
  executable route covers dynamic regridding, covered and uncovered coarse cells, active fine cells,
  finite no-root and non-finite failures, exact all-level/clock/topology rollback, and a rank-local
  failure reduced consistently over two MPI ranks. The capability remains `partial`: subcycled local
  solves, GPU qualification, field/global implicit coupling and performance evidence are not inferred
  from this pointwise synchronous route and require their own prepared execution proof.
- Prepared state-boundary residual/JVP pairs on Program matrix-free solves. The exact base
  `BoundaryEvaluationPoint` is transported into the apply closure, the core RHS is
  finite-differenced, and the authenticated state-only boundary JVP is added once with persistent
  conditional scratch. Field-coupled `rhs_jacvec` re-solves its exact prepared provider from the
  perturbed state on level zero and every refined level; if a transport boundary reads that solved
  field, its complete residual is finite-differenced before the perturbed provider is restored.
  Ordinary single-state field solves use that same owner-qualified provider ABI on Uniform and AMR:
  the generated call carries the exact `BoundaryEvaluationPoint`, provider slot, active level and
  stage state, with no AMR coarse-report reuse overload.
  Dynamic physical field boundaries may read level-qualified conservative states, already-solved
  fields and the exact stage/local time under both `LevelByLevelSolve` and
  `CompositeHierarchySolve`; the composite FAC provider requires one exact dependency carrier per
  materialized level before entering a solve. The generated resolve/source contract covers the
  field-dependent transport-boundary JVP route. A native L0/L1 level-local oracle now places that
  dependency on a physical face of a fully refined domain and checks the complete core-plus-boundary
  `rhs_jacvec(field_coupled=True)` against an independent centered finite difference; it also proves
  physical-face locality, provider sensitivity and restoration after every perturbation. The core
  field-coupled JVP has a two-rank level-local oracle over genuinely distributed L0/L1 state and
  provider storage. Its composite-policy MPI oracle exercises the ownership topology supported by
  the builtin FAC provider: one complete replicated L0 copy per rank and a genuinely distributed
  L1. Both check centered-difference parity, frozen-provider sensitivity and collective restoration
  of the complete provider hierarchy plus the active-level residual carrier. A second two-rank L0/L1
  oracle drives the level-local solved field through an x-low physical-face residual split across
  both ranks, proving that its JVP contribution is non-trivial, face-local, provider-sensitive and
  collectively restored.
  Partially refined FAC patches carrying a dynamic physical boundary must remain strictly interior;
  a patch touching a non-periodic domain face fails closed. A selected solve with a field dependency
  also fails closed until its complete dependency closure can share one transaction. Simultaneous
  multi-block stage solves use one exact hierarchy-qualified multi-state request carrying the same
  `BoundaryEvaluationPoint`, provider slot and active level; every provisional conservative state is
  restored before the provider candidate can be consumed.
- Runtime scientific output v1: typed `SERIAL`, `ROOT`, `COLLECTIVE` and `PER_RANK` publication on the
  exact modes advertised by NPZ, ParaView and HDF5, with native Uniform/AMR piece ownership.
- Runtime accepted-state checkpoint v6 for Uniform and v8 for AMR. The single-file MPI route captures
  collectively only after every rank agrees on the exact gather-plan identity, agrees again on the
  sealed payload identity, and publishes once on rank 0 with atomic no-clobber semantics. The provider
  authority is resolved into the compiled plan, including the builtin v5 manual route. It persists
  the release-versioned rank-generic spatial authority: one authenticated dimension and
  exact `Dim`-length shape, bounds, periodicity and per-transition refinement-ratio vectors. Restart
  compares that authority before native hierarchy allocation or state mutation; scalar `nx`/`ny`
  compatibility fields are not a runtime route. Restart reads and authenticates that file once on rank
  zero, broadcasts the exact bytes through the installed `ExecutionContext` communicator, preflights
  every rank before mutation, and keeps a rollback snapshot until apply/commit consensus. Multi-layout
  child payloads are decoded and replayed in memory without shared child files. AMR preserves
  multi-block/multi-level accepted state under active regridding, including topology ownership,
  clocks, the held cadence window, the last accepted Program interval, histories and transfer
  provenance. Native `SymbolicTagger` temporal hysteresis is part of that accepted image: non-zero
  `Hysteresis.min_cycles` is supported, restored transactionally, and rematerialized exactly across
  MPI rank-count changes. External Tagger components still refuse non-zero hysteresis before
  artifact creation until their adapter contract owns the same persistent-state route. MPI capture
  validates the rank-independent accepted-state image on every producer before sealing or
  publication; disagreement fails collectively and cannot leave a partial checkpoint.
- The prepared limiter registry exposes native `Minmod`, `VanLeer`, `MC`, and `Superbee` MUSCL
  policies. Each is a stateless `POPS_HD` compile-time provider with formal order 2 and exactly two
  ghost layers; Uniform, AMR, MPI and supported device targets consume the same route identity.

Explicit unsupported rows include:

- `elliptic:fft_amr`: FFT requires a single uniform periodic mesh; AMR uses GeometricMG.
- `checkpoint:parallel_hdf5`: parallel HDF5 is a scientific-output route, not a restartable checkpoint
  encoding; `RuntimeInstance.checkpoint()` and the typed `Checkpoint` consumer use uniform v5 or AMR
  v7 accepted-state payloads.
- `checkpoint:amr_dynamic_regrid` is available through the strict v7 accepted-state route. The single
  authenticated artifact carries one exact DistributionMapping and compiled-Program accepted image
  per native rank. `bit_identical=True` therefore requires the recorded rank count. With the default
  non-bit-identical guarantee, `RestoreRecordedHierarchy()` may rematerialize hierarchy ownership and
  the rank-owned accepted Program image onto a different MPI rank count only when every persisted
  history ring is Dense. The rematerialized image includes the exact runtime-owned persistent
  tagging payload and the rank-consensus accepted shared-interface flux audit. Every restored
  fragment is authenticated against the live topology, exact clock window, resolved Program weight,
  face measure and local duration. Selective history replay remains same-rank. Recorded patch boxes
  and refinement topology are not regridded or inferred from opaque local publications.
- `checkpoint:regrid_on_restart` has an explicit typed `RegridOnRestart()` identity and the weaker
  `accepted_state_after_regrid` guarantee. The builtin accepted-state-v5 provider supports one
  artifact-backed AMR layout at unchanged MPI cardinality: exact accepted replay precedes one real
  tagger/clustering regrid, history/flux topology is rebound, composite conservation is checked, and
  a global transform receipt derives a distinct run identity. Persistent tagging state is restored
  and rolled back with the accepted image, then advanced exactly once by a successful transform.
  Serial and exact-`MPI_COMM_WORLD` shared-interface groups are rematerialized at unchanged
  cardinality in the same transaction and execute conservatively after rollback or commit. The MPI
  acceptance proof covers one refined transition, a rank-local post-transform fault, exact
  all-rank rollback, retry with one receipt identity and post-restart interface execution. Uniform,
  multi-layout, elliptic-field, active-depth-change, unsupported non-finest replacements at depth
  greater than two, rank-changing dynamic shared-interface, and bootstrap-staggered/cache cases
  remain explicit refusals; `bit_identical=True` is incompatible with the policy. Exact phase-local dense-history consensus fingerprints are gathered only on this
  cold restart path; they prove all-rank agreement per hierarchy rather than bitwise continuity
  across interpolation. Accepted solution components retain the separate native composite
  conservation invariant. Fingerprint memory and collective-communication cost scales with all
  retained slots and active level-domain cells.
- `supports_partial_imex_mask`: no native C++ path backs partial IMEX masks.
- `supports_mpi` and `supports_gpu` when the loaded module/artifact was not built with the corresponding native backend.
- `runtime:explicit_gpu_context`: the final native `RuntimeInstance` providers are host/float64 and refuse a
  GPU Kokkos execution space before constructing `System`/`AmrSystem`; build-time availability is
  not launch authorization. The native providers do accept an explicit, authenticated
  `MPI_COMM_WORLD` context; custom communicators remain unavailable.
- `amr:composite_dynamic_boundary`: a fully refined hierarchy uses the exact finest-level uniform
  field solver and receives that level's logical time, state dependencies, distributions, and
  nonlinear/JVP context. A partially refined FAC hierarchy refuses the same request because its
  interface correction does not yet own the required homogeneous/JVP boundary operator per level;
  it never reuses the inhomogeneous primal closure as a correction boundary.
- `amr:external_field_solver_v2`: an authenticated `InstallPlan` now installs the exact
  `FieldTopology@2` + `FieldSolver@2` pair as one `AmrFieldSolverProvider`. One prepared request
  carries every hierarchy level, qualified by `metadata.level`, and binary material masks exclude
  fine-covered coarse cells. The serial integration oracle requires both a masked coarse cell and an
  active fine cell, then advances through repeated field solves and a layout-changing regrid. The
  provider performs one collective solve, validates every active candidate value before
  `SolveOutcome` publication, restricts solved fine values into covered coarse cells, materializes
  same-level/physical/coarse-fine potential halos before centered gradients, and
  destroys/rematerializes both component states when regridding invalidates the prepared solver.
  The current transfer proof is ratio-2. Host serial is available. The executable
  `test_external_amr_field_solver_mpi.py` oracle proves `mpiexec -n 2` with local patches on both
  ranks at L0 and L1, a layout-changing regrid/rematerialization, exact provider evidence, typed
  collective rollback/retry, and refusal of a rank-local non-finite candidate before publication.
  MPI is available only when both component manifests declare the host/MPI variant, the installed execution
  authority is exact `MPI_COMM_WORLD`/`MPI_DOUBLE`, and the coarse level is distributed (the v2 ABI
  has no replicated-coarse ownership marker); rank counts above two remain unqualified.
  GPU/device memory, embedded or cut-cell topology,
  dynamic/dependent boundaries, reaction coefficients and nonlinear/JVP solves remain fail-closed.
ADC-601 also records audited native subsystem limitations as `partial` rows. These rows are not
hard failures, but they make compatibility and performance constraints visible to reports and
future validators:

- `elliptic:mg_fac_defaults`: MG/FAC defaults and debug diagnostics still need a shared
  `SolverDefaults`/logger route.
- `mesh:nd_storage_arithmetic`: one `Index<Dim>`/`Box<Dim>`/`Fab<Dim>`/`MultiFab<Dim>` core is
  specialized at build time for the resolved artifact dimension (1, 2 or 3). The same retained
  specialization crosses native layout, storage, arithmetic and runtime binding; there is no
  parallel 2D storage authority or runtime dimension branch.
- `amr:refinement_ratio`: native AMR hierarchy, patch ranges, spatial transfers and reflux geometry
  are `ratio=2` only, and `validate_amr_refinement_ratio()` rejects other spatial ratios. Temporal
  parent/child ratios are explicit `ProgramGraph` data; `AmrRuntime` never infers or executes
  subcycling from this spatial descriptor.
- `amr:transition_envelope`: transitions are 2D/isotropic and buffer/lookahead are hierarchy-global.
- `amr:hierarchy_policy_routes`: only the reported shared hierarchy, clustering, patch-generation,
  and load-balance routes are installed.
- `amr:accepted_owner_migration`: a prepared `RebalanceDecision` can redistribute one active fine
  level at a clean accepted Program boundary. The consumer revalidates the exact decision against
  its prepared authority, source level, live topology epoch, materialization generation, boxes and
  current owners, requires all-rank byte consensus, migrates every block/aux/history carrier,
  rematerializes topology-bound providers, redistributes compact lagged-flux authority through the
  checkpoint rematerializer, invalidates audit reports qualified by the replaced topology epoch and
  republishes accepted Program state atomically. Stale, divergent, malformed and non-beneficial
  decisions do not mutate state; failures restore the complete accepted runtime/Program image.
  Level-zero migration, custom communicators, materialized staggered bootstrap fields and
  rank-change cell-local record rematerialization remain unavailable.
- `amr:transfer_contracts`: centering, representation, storage, operation, order and ghost depth
  must match an exact native transfer/materialization provider contract.
- `parallel:mpi_world_communicator`: the native `RuntimeInstance` providers consume the exact
  `MPI_COMM_WORLD` carried by its validated `ExecutionContext`; the C++ module owns initialization,
  collectives, ABI handles, rank and size. It calls `MPI_Init_thread(MPI_THREAD_MULTIPLE)` before
  worker threads exist, or attaches only to an externally initialized world whose queried level is
  already `MPI_THREAD_MULTIPLE`. PoPS finalizes only a world it initialized itself, after native work
  has ended; an embedding application retains its lifecycle. Python carries only the opaque native
  resource identity.
- `parallel:custom_communicator`: caller-provided custom MPI communicators remain representable but
  unavailable at the public bind surface because field storage does not yet carry a
  communicator-relative rank space. The native interface scheduler and layout-transfer consumers
  can execute on an authenticated `MPI_IDENT`/`MPI_CONGRUENT` lane, but admission must still compare
  that lane with the process-world-indexed field ownership. Subgroups and reordered communicators
  are refused before kernel launch.
- `precision:single_or_mixed`: `pops::Real` is `double`; single or mixed precision is unavailable.
- `runtime:kokkos_lifecycle`: `runtime_environment_report()` exposes whether PoPS will lazily
  initialize Kokkos, has initialized it, or is attached to an externally initialized runtime.
- `runtime:allocator_lifetime`: Kokkos builds use a process-lifetime managed arena whose blocks are
  returned by a Kokkos finalize hook.
- `program:hierarchy_scoped_solve`: a hierarchy-scoped `LinearProblem` requires an explicit
  matrix-free operator provider such as `CompositeTensorFAC()` and an executable Krylov solver. The
  full-tensor FAC authority is exact-rank in 1D, 2D and 3D; rank, hierarchy, coefficient and MPI
  ownership mismatches fail capability validation before execution.

## Error Policy

Unsupported routes must fail before they can compile or bind. Error messages must name:

- the requested route,
- the available route,
- an alternative when one exists.

Example:

```text
unsupported route: requested solver=FFT() with layout=AMR; available route: GeometricMG() on AMR; alternative: use pops.solvers.elliptic.GeometricMG()
```

Unknown values are not treated as false and are never repaired by a compatibility default. A public
artifact must carry the current authenticated manifest and required route facts; missing, unknown, or
incompatible evidence is refused before bind. Historical artifacts may only be converted by an
explicit offline migration tool that emits a complete current artifact. The one implemented
checkpoint route accepts only the exact frozen Uniform-v2 schema through
`pops.codegen.checkpoint_migration`: it requires a complete authenticated v5 authority and an
exhaustive reviewed mapping, preserves no runtime alias, and publishes atomically only after current
integrity and restart preflight succeed. Its supported envelope is same-grid/same-clock, Dense
store-all history, with no field-provider slots, scheduled caches, or ConsumerGraph state; every
other historical checkpoint remains a fail-closed refusal in the runtime.
