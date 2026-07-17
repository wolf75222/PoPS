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
  The current native `SystemConfig` has one `n`, one `L`, and no origin, so public
  `CartesianGrid` lowering requires `lower == (0, 0)`, equal axis lengths, and equal axis cell
  counts. Its global periodic switch lowers exactly an empty or all-axis typed `PeriodicAxes`
  partition. Rectangular, anisotropic, translated, and partially periodic grids fail before engine
  construction; no topology is widened or collapsed.
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
  then scatter only into locally owned residual cells.
  Cross-layout interfaces without an explicit Mapping/Transfer provider, shared implicit JVP, and
  refined or dynamically regridded AMR interfaces are unavailable; AMR accepts only one frozen
  level.
- AMR through the native production route with hierarchy depth controlled by resolved resource
  policy. Transitions are exactly 2D, isotropic `ratio == (2, 2)`, share one isotropic buffer and
  one lookahead across the hierarchy, and currently select the exact native policy routes
  `shared_n_level`, `berger_rigoutsos`, `box_array`, and `round_robin`. Physical transfer providers
  expose exact dense cell/face-x/face-y/node contracts; restriction, coarse-fine fill and temporal
  interpolation are cell-centered on the supplied route. Derived fields use `elliptic_solve` and
  caches use `patch_topology`; unsupported provider contracts fail before artifact creation.
- Finite-volume spatial discretisation on the 2D core.
- Native Riemann routes: Rusanov, HLL, HLLC, Roe, subject to model capability requirements.
- Native reconstruction routes: first-order, MUSCL, WENO5/WENO5-Z.
- Elliptic GeometricMG on Uniform/AMR and FFT on uniform periodic constant-coefficient grids.
- Matrix-free Krylov descriptors: CG, BiCGStab, GMRES, Richardson.
- ProgramContext install on System, and AMR program install when compiled for `target="amr_system"`.
- Prepared state-boundary residual/JVP pairs on Program matrix-free solves. The exact base
  `BoundaryEvaluationPoint` is transported into the apply closure, the core RHS is
  finite-differenced, and the authenticated state-only boundary JVP is added once with persistent
  conditional scratch. A field-dependent boundary closure under `field_coupled=True` is refused
  until a qualified tangent-field solve exists. Core field-coupled `rhs_jacvec` currently has an
  exact provider route only on AMR level 0.
- Runtime scientific output v1: typed `SERIAL`, `ROOT`, `COLLECTIVE` and `PER_RANK` publication on the
  exact modes advertised by NPZ, ParaView and HDF5, with native Uniform/AMR piece ownership.
- Runtime accepted-state checkpoint v3 for Uniform and AMR. The single-file MPI route captures
  collectively only after every rank agrees on the exact gather-plan identity, agrees again on the
  sealed payload identity, and publishes once on rank 0 with atomic no-clobber semantics. The provider
  authority is resolved into the compiled plan, including the builtin v3 manual route. Restart reads
  and authenticates that file once on rank
  zero, broadcasts the exact bytes through the installed `ExecutionContext` communicator, preflights
  every rank before mutation, and keeps a rollback snapshot until apply/commit consensus. Multi-layout
  child payloads are decoded and replayed in memory without shared child files. AMR preserves
  multi-block/multi-level accepted state under active regridding, including topology ownership,
  clocks, histories and transfer provenance.

Explicit unsupported rows include:

- `limiter:mc` and `limiter:superbee`: catalogued descriptors with no native C++ symbol.
- `elliptic:fft_amr`: FFT requires a single uniform periodic mesh; AMR uses GeometricMG.
- `checkpoint:parallel_hdf5`: parallel HDF5 is a scientific-output route, not a restartable checkpoint
  encoding; `RuntimeInstance.checkpoint()` and the typed `Checkpoint` consumer use accepted-state v3.
- `checkpoint:amr_dynamic_regrid` is available through the strict v3 accepted-state route. The single
  authenticated artifact carries one exact DistributionMapping and compiled-Program accepted image
  per native rank, so AMR restart currently requires the same rank count; rank redistribution is never
  inferred from opaque local publications.
- `supports_partial_imex_mask`: no native C++ path backs partial IMEX masks.
- `supports_mpi` and `supports_gpu` when the loaded module/artifact was not built with the corresponding native backend.
- `runtime:explicit_gpu_context`: the final native `RuntimeInstance` providers are host/float64 and refuse a
  GPU Kokkos execution space before constructing `System`/`AmrSystem`; build-time availability is
  not launch authorization. The native providers do accept an explicit, authenticated
  `MPI_COMM_WORLD` context; custom communicators remain unavailable.
- `amr:field_coupled_rhs_jacvec`: AMR level greater than zero is explicitly unavailable because the
  provider ABI does not transport a level-qualified tangent field. The reported error identifies
  the level-0 field-coupled route as the available route; a multi-level request must fail rather
  than silently reuse the coarse provider.

ADC-601 also records audited native subsystem limitations as `partial` rows. These rows are not
hard failures, but they make compatibility and performance constraints visible to reports and
future validators:

- `elliptic:fft_direct_dft_fallback`: non-power-of-two FFT grids use the correct direct `O(n^2)`
  DFT fallback and expose fallback calls through `poisson_fft_direct_dft_fallback_count()`.
- `elliptic:mg_fac_defaults`: MG/FAC defaults and debug diagnostics still need a shared
  `SolverDefaults`/logger route.
- `mesh:2d_storage_arithmetic`: the native mesh/storage/arithmetic core is `Box2D`/`Fab2D`
  2D-only, and `validate_dimension()` rejects `Dim != 2` requests.
- `amr:refinement_ratio`: native AMR hierarchy, patch ranges, reflux and subcycling are `ratio=2`
  only, and `validate_amr_refinement_ratio()` rejects other ratios.
- `amr:transition_envelope`: transitions are 2D/isotropic and buffer/lookahead are hierarchy-global.
- `amr:hierarchy_policy_routes`: only the reported shared hierarchy, clustering, patch-generation,
  and load-balance routes are installed.
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
  unavailable because the native engines expose no communicator-injection ABI.
- `precision:single_or_mixed`: `pops::Real` is `double`; single or mixed precision is unavailable.
- `runtime:kokkos_lifecycle`: `runtime_environment_report()` exposes whether PoPS will lazily
  initialize Kokkos, has initialized it, or is attached to an externally initialized runtime.
- `runtime:allocator_lifetime`: Kokkos builds use a process-lifetime managed arena whose blocks are
  returned by a Kokkos finalize hook.
- `program:hierarchy_scoped_solve`: a hierarchy-scoped `LinearProblem` requires an explicit
  matrix-free operator provider such as `CompositeTensorFAC()` and an executable Krylov solver. The
  currently audited native tensor route is 2D; unsupported dimensions or hierarchy shapes fail
  capability validation instead of selecting a named time preset.

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
explicit offline migration tool that emits a complete current artifact.
