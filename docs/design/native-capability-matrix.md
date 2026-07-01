# Native Capability Matrix

ADC-549 makes route support explicit. A Python public feature must either map to a real native route
or report an unsupported route before compile, bind, or runtime execution.

## Matrix Schema

Every route row is a plain metadata record with these fields:

| Field | Meaning |
| --- | --- |
| `feature` | Stable feature token, for example `layout:AMR`, `elliptic:fft_amr`, or `checkpoint:parallel_hdf5`. |
| `layout` | Layout envelope the row applies to: `uniform`, `amr`, `uniform|amr`, or `context`. |
| `backend` | Backend or route required: `production`, `aot`, `prototype`, `runtime`, `module`, or `none`. |
| `platform` | Platform axis: `host`, `mpi`, `gpu`, or `context`. |
| `mpi` | Whether this row is backed by an MPI-capable route for the current build/artifact. |
| `gpu` | Whether this row is backed by a GPU-capable route for the current build/artifact. |
| `status` | `available`, `unavailable`, `partial`, or `unknown`. Known unsupported routes use `unavailable`. |
| `limitation` | Short human-readable limitation or constraint. |
| `error_message` | For unavailable rows, the message shape used by validators: requested route, available route, alternative. |

The same shape is exposed by:

- `pops.Case(...).explain_routes()`
- descriptor `capability_matrix()` methods
- `CompiledProblem.capability_matrix()`
- `CompiledModel.capability_matrix()`
- `CompiledArtifactManifest.to_dict()["capability_matrix"]`

## Native Inventory

The canonical inventory lives in `pops._capabilities.native_capability_matrix()`.

Supported native routes include:

- Uniform single-level layout.
- AMR through the native production route, limited to `max_levels <= 2` and `ratio == 2`.
- Finite-volume spatial discretisation on the 2D core.
- Native Riemann routes: Rusanov, HLL, HLLC, Roe, subject to model capability requirements.
- Native reconstruction routes: first-order, MUSCL, WENO5/WENO5-Z.
- Elliptic GeometricMG on Uniform/AMR and FFT on uniform periodic constant-coefficient grids.
- Matrix-free Krylov descriptors: CG, BiCGStab, GMRES, Richardson.
- ProgramContext install on System, and AMR program install when compiled for `target="amr_system"`.
- Runtime output routes: npz, VTK, HDF5, plus AMR coarse/patch metadata output.
- Runtime checkpoint v1: npz rank-0 gather, with AMR bit-identical restart only for the frozen hierarchy route.

Explicit unsupported rows include:

- `limiter:mc` and `limiter:superbee`: catalogued descriptors with no native C++ symbol.
- `elliptic:fft_amr`: FFT requires a single uniform periodic mesh; AMR uses GeometricMG.
- `output:plotfile_uniform`: Plotfile is an AMR per-level format, not a Uniform System writer.
- `checkpoint:parallel_hdf5`: parallel HDF5 is an output route, not a restartable checkpoint route.
- `checkpoint:amr_dynamic_regrid`: bit-identical AMR checkpoint requires `regrid_every == 0`.
- `supports_partial_imex_mask`: no native C++ path backs partial IMEX masks.
- `supports_mpi` and `supports_gpu` when the loaded module/artifact was not built with the corresponding native backend.

## Error Policy

Unsupported routes must fail before they can compile or bind. Error messages must name:

- the requested route,
- the available route,
- an alternative when one exists.

Example:

```text
unsupported route: requested solver=FFT() with layout=AMR; available route: GeometricMG() on AMR; alternative: use pops.solvers.elliptic.GeometricMG()
```

Unknown values are not treated as false. Older artifacts that lack a manifest flag keep `None` in the
manifest and produce `unknown` rows. A validator may warn or report that limitation, but must not reject
a route solely because an old artifact did not emit a flag.
