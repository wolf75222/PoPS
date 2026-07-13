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
| `backend` | Backend or route required: `production`, `aot`, `prototype`, `runtime`, `module`, or `none`. |
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
  `Uniform(mesh, refine=..., ignore_amr=pops.mesh.amr.IgnoreAMRCriteria())`.
- AMR through the native production route with hierarchy depth controlled by resolved resource
  policy; transfer/reflux kernels currently require `ratio == 2`.
- Finite-volume spatial discretisation on the 2D core.
- Native Riemann routes: Rusanov, HLL, HLLC, Roe, subject to model capability requirements.
- Native reconstruction routes: first-order, MUSCL, WENO5/WENO5-Z.
- Elliptic GeometricMG on Uniform/AMR and FFT on uniform periodic constant-coefficient grids.
- Matrix-free Krylov descriptors: CG, BiCGStab, GMRES, Richardson.
- ProgramContext install on System, and AMR program install when compiled for `target="amr_system"`.
- Runtime output routes: npz, VTK, HDF5, plus AMR coarse/patch metadata output.
- Runtime checkpoints: Uniform v1 and strict AMR v3. AMR v3 preserves multi-block/multi-level accepted
  state under active regridding, including topology ownership, clocks, histories and transfer provenance.

Explicit unsupported rows include:

- `limiter:mc` and `limiter:superbee`: catalogued descriptors with no native C++ symbol.
- `elliptic:fft_amr`: FFT requires a single uniform periodic mesh; AMR uses GeometricMG.
- `output:plotfile_uniform`: Plotfile is an AMR per-level format, not a Uniform System writer.
- `checkpoint:parallel_hdf5`: parallel HDF5 is an output route, not a restartable checkpoint route.
- `checkpoint:amr_dynamic_regrid` is available through the strict v3 accepted-state route. A non-Dense
  history policy that reconstructs omitted slots by replay requires the restart to keep the rank count.
- `supports_partial_imex_mask`: no native C++ path backs partial IMEX masks.
- `supports_mpi` and `supports_gpu` when the loaded module/artifact was not built with the corresponding native backend.

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
- `parallel:mpi_world_communicator`: MPI collectives currently use `MPI_COMM_WORLD`.
- `parallel:custom_communicator`: caller-provided MPI communicators are unavailable.
- `precision:single_or_mixed`: `pops::Real` is `double`; single or mixed precision is unavailable.
- `runtime:kokkos_lifecycle`: `runtime_environment_report()` exposes whether PoPS will lazily
  initialize Kokkos, has initialized it, or is attached to an externally initialized runtime.
- `runtime:allocator_lifetime`: Kokkos builds use a process-lifetime managed arena whose blocks are
  returned by a Kokkos finalize hook.
- `program:condensed_implicit_preset`: `pops.lib.time.CondensedSchur` currently authors a 2D,
  two-component electrostatic-Lorentz reduction. The Program solve/provider protocol is independent
  of that preset and can host other operators, dimensions and hierarchy providers.

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
