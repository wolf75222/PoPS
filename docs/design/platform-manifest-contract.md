# Platform and field-view launch contract

`PlatformManifest` records the platform selected by a compiled artifact. `RuntimeBackendManifest`
records what the loaded runtime can prove. Both carry the backend, target, ABI, device, memory
spaces, communicator, capabilities, and four independent precision policies: storage, compute,
accumulation, and reduction. Every value is paired with evidence. A missing proof is unknown and
cannot authorize a launch.

The platform manifest participates in `CompiledSimulationArtifact.artifact_identity`.
`ExecutionContext` participates in the install/bind identity and therefore in the run identity. A
change of target, ABI, precision stage, device, memory space, communicator, or datatype invalidates
the corresponding identity instead of selecting a compatibility route.

## Execution resources

`ExecutionContext` is the sole runtime resource authority. It owns explicit identities and optional
opaque handles for the communicator, datatype, and device. A non-serial communicator or non-host
device requires a handle. The generic launch seam never reads `MPI_COMM_WORLD`, `MPI_DOUBLE`, a
current device, or a default execution space.

The native route currently proves two exact host configurations:

- dimension: 2;
- scalar: `float64` for storage, compute, accumulation, and reduction;
- centering: cell;
- device and memory space: host;
- communicator: `serial`, or `MPI_COMM_WORLD` when the authenticated module is MPI-enabled and the
  process is inside an active world launch.

The MPI route requires an explicit `ExecutionContext.mpi_world(artifact, MPI.COMM_WORLD)` at bind;
the artifact, native backend manifest, Python handle, native rank and native size must all agree.
Custom communicators, GPU devices, non-cell centerings, three-dimensional kernels, and single/mixed
precision are not advertised. No fallback or emulation is installed.

## Field views and pre-launch refusal

`FieldViewDescriptor` preserves dimension, extents, strides, centering, ghost widths, scalar,
memory space, patch identity, layout, and ownership. Three-dimensional views are representable as
metadata, but the current backend rejects them with its proved 2D-only capability. The launch gate
compares required and actual descriptors and refuses an unsupported dimension or a centering,
scalar, extent, memory-space, or communicator mismatch before calling the kernel.

The C++ contract is in `include/pops/runtime/config/platform_manifest.hpp`; the Python values live in
the private planning contract `pops._platform_contracts` and are installed by the private runtime
adapter. `runtime_backend_manifest()` is exposed by `_pops` from the same C++ source, so bind verifies
the native payload and its identity before installation.
