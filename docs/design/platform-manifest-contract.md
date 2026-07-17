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
opaque native handles for the communicator, datatype, and device. A non-serial communicator or
non-host device requires a handle. MPI handles are created and validated by the C++ runtime; Python
never imports an MPI implementation, executes a collective, or supplies `MPI_COMM_WORLD` or
`MPI_DOUBLE`.

The native route currently proves two exact host configurations:

- dimension: 2;
- scalar: `float64` for storage, compute, accumulation, and reduction;
- centering: cell;
- device and memory space: host;
- communicator: `serial`, or `MPI_COMM_WORLD` when the authenticated module is MPI-enabled and the
  process is inside an active world launch.

The MPI route requires an explicit `ExecutionContext.mpi_world(artifact)` at bind. That call asks the
native module to initialize MPI with `MPI_THREAD_MULTIPLE`, or attach to an external world only when
`MPI_Query_thread` proves the same level, and returns an opaque C++ world resource. Initialization
must precede worker-thread launch. PoPS finalizes only a world it initialized, after native work has
ended; an embedding application retains ownership. The artifact, native backend manifest, native
communicator identity, rank and size must all agree. Custom
communicators, GPU devices, non-cell centerings, three-dimensional kernels, and single/mixed
precision are not advertised. No Python MPI adapter, fallback, or emulation is installed.

Every C++ artifact compiled at runtime, including explicit `Program` loaders and authenticated
external components, inherits this same selected communicator. CMake serializes the complete
`MPI::MPI_CXX` contract into one private `_pops.__mpi_contract__` manifest: include directories,
compile options/definitions, link options/libraries, and SHA-256 values for every `mpi.h` and library.
Codegen re-hashes those files immediately before compilation, replays every flag, and folds the
manifest digest into both its cache key and `POPS_ABI_KEY_LITERAL`. An in-place MPI upgrade, missing
path, changed flag, Open MPI/MPICH mismatch, or incomplete manifest therefore fails before compile
or native installation; PoPS never substitutes a serial loader. On POSIX, `_pops` is promoted once
to `RTLD_GLOBAL` and its handle is retained for process lifetime before any component is compiled or
loaded, so plugins share the already-owned Kokkos/MPI runtimes. The external component manifest
records `MPI_COMM_WORLD` plus the MPI ABI proof and is checked against the explicit execution
context at installation.

`compile_native` has an explicit PE/COFF command and `_pops.lib` contract. By contrast,
`compile_problem` and `compile_component` are currently fail-closed on Windows because their final
authenticated PE/COFF symbol-inspection/publication pipeline does not yet exist. They never run a
POSIX `-shared -fPIC` command or publish a `.so` under Windows.

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
