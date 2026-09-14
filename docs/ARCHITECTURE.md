# Architecture of PoPS

PoPS separates physical authoring, spatial discretization, time programs and native
execution. The public Python lifecycle is implemented in
[`_api.py`](../python/pops/_api.py):

```text
Case -> validate -> resolve -> compile -> bind -> run
```

Python builds and inspects typed descriptions. Generated C++ executes model expressions
and program kernels; Kokkos executes cell/face operations. Runtime orchestration also
exists in Python, including control, consumers and restart. There are no Python per-cell
callbacks in the compiled numerical path.

## Lifecycle and ownership

| Phase | Responsibility |
| --- | --- |
| Author | `Model`, qualified state/operator handles, spatial descriptors, a whole-system `Program`, and `Case`. |
| Validate | Check contracts and freeze the authored graph into a detached snapshot. |
| Resolve | Resolve layouts, providers, operations, consumers and exact runtime requirements without native execution. |
| Compile | Select an exact native dimension/platform, lower the resolved model and Program, and authenticate compiled components. |
| Bind | Supply parameters, initial state, auxiliary values, resources and initial values; prepare a private runtime. |
| Run | Execute the installed program, handle attempts and accepted steps, and publish declared consumers. |

The implementations live in [`problem`](../python/pops/problem),
[`codegen`](../python/pops/codegen) and [`runtime`](../python/pops/runtime).
`RuntimeInstance` owns the bound state and public run lifecycle. `System<Dim>` and
`AmrSystem<Dim>` are private native execution engines, not competing Python front doors.

Handles carry their owner and instance identity. Display names are metadata; field
slots, operator dependencies and history samples are not resolved by a reserved physical
name. Provider consumers receive compact plans for exactly their declared components.

## Python layers

| Directory under `python/pops` | Role |
| --- | --- |
| `physics`, `model`, `math` | Physical expressions, model declarations and typed operator/state handles. |
| `mesh`, `numerics`, `fields`, `solvers` | Layout, boundary, reconstruction, flux and solve descriptors. |
| `time`, `lib/time` | Program graph and generic factories for explicit RK, IMEX, DIRK, multistep and splitting methods. |
| `problem` | Case assembly, transactional freezing, validation and resolution. |
| `codegen`, `external` | C++ lowering, compile/cache identity and external component packages. |
| `runtime` | Bind adapters, native installation, execution, transactions, consumers and restart. |

The [package map](design/python_package_layout.md) details import boundaries. Shared
immutable-container copying lives in `_frozen_data.py`; Case and Program builders use it
without conflating it with descriptor-specific freezing or canonical serialization.

Time factories build the same typed Program operations as manual authoring. For example,
`lib/time/rk.py` supplies one tableau-based explicit Runge-Kutta builder. The compiler
lowers normalized SSA operations; adding a scheme should compose those operations rather
than introduce a scheme-name branch in the runtime.

Uniform and AMR bind adapters retain different layout and topology preparation. Their
field-component authentication, ABI registration and nullspace installation share
[`_field_provider_install.py`](../python/pops/runtime/_field_provider_install.py).
Each adapter retains its engine-specific plan, topology publication and route checks.

## Native layers and build

| Directory | Responsibility |
| --- | --- |
| `include/pops/core`, `include/pops/physics` | Model concepts, state contracts and reusable physical bricks. |
| `include/pops/mesh`, `include/pops/parallel` | Ranked boxes, geometry, distributed fields, halos and communication. |
| `include/pops/numerics` | Finite-volume, diffusion, elliptic/Krylov, reconstruction and temporal primitives. |
| `include/pops/amr`, `include/pops/coupling` | Hierarchies, transfer, clustering, coupled operations and interface exchanges. |
| `include/pops/runtime` | Native runtime interfaces, installation contracts and program state. |
| `src/runtime` | Compiled uniform/AMR, field, layout-transfer, output and transaction implementations. |
| `python/bindings` | pybind11 translation units exposing the native execution boundary. |

The project is not entirely header-only. [`src/CMakeLists.txt`](../src/CMakeLists.txt)
owns compiled runtime object/static targets shared by the extension and C++ tests.
The transaction exception ABI is compiled once and made visible to generated components.
Consumers link the relevant CMake targets instead of recompiling runtime sources themselves.

Kokkos is required. CMake finds an installed package or builds the configured version.
MPI and parallel HDF5 are optional native capabilities. FFTW is an optional radix backend
for Cartesian Poisson FFT; CMake searches explicit dependency prefixes. pybind11 is needed
for the extension, and GoogleTest for the C++ tests. Dependency discovery is in the root
[`CMakeLists.txt`](../CMakeLists.txt).

A native build specializes one spatial dimension (1, 2 or 3). The package can retain
separate authenticated dimension leaves; one Python process selects one dimension.
Compiled artifact identity includes the native ABI, headers, platform and generated code.
A shared-object pathname alone does not establish compatibility.

## Spatial and temporal execution

`PreparedCartesianOperator` reconstructs traces, materializes a ranked `FaceField<Dim>`,
and assembles residuals from that same flux storage. AMR exchange accounting uses those
fluxes rather than a separate reconstruction. The production barrel
`numerics/spatial_operator.hpp` exports these ranked components.

`MultiFab<Dim>` stores distributed cell data. Box/layout metadata describe ownership;
ghost exchange and physical boundary filling establish data required by prepared stencils.
The AMR hierarchy adds coverage, coarse/fine transfer, regrid and accepted flux correction.
Field storage and state storage are resolved explicitly, including supported physical maps.

The Program owns stage calls, evaluations, solves, histories, exchanges and commits.
A runtime does not infer a hidden integrator from a model or time-method descriptor.
An attempt stages state and associated history/controller/consumer state. Rejection rolls
back the attempt; acceptance publishes the declared state and effects. Output failure after
acceptance is distinct from failure of the numerical step.

See [Algorithms](ALGORITHMS.md), the [temporal contract](design/temporal-execution-contract.md),
and the [consumer transaction contract](design/consumer_graph_transaction_contract.md).

## Capability boundaries

- Python native execution requires binary64 and exact installed platform compatibility.
- The current distributed runtime uses `MPI_COMM_WORLD`; arbitrary subcommunicator injection
  is not a general supported runtime route.
- One resolved runtime rejects mixed Uniform/AMR engine families. Multiple layouts require
  explicit supported mappings and transfers.
- Cartesian runtime support does not imply that every standalone polar or embedded-boundary
  numerical component is available through every public bind route.
- Solver/provider combinations are checked at their installation boundary. A descriptor or
  generated kernel is not evidence that every configuration executes successfully.

The [capability matrix](design/native-capability-matrix.md) and
[verification scope](development/migration_verification_scope.md) retain detailed contracts
and evidence limits. The large uniform/AMR runtime implementations remain maintenance
concentrations; splitting them requires preserving installation order, collective participation
and transaction ownership, not merely moving methods between files.

## Repository navigation

`docs/tutorials` contains linear teaching scripts. `examples` contains acceptance cases and
focused workflows. `benchmarks` contains declared measurement campaigns; its manifests and
baseline data are inputs, not disposable runtime output. `tests/test_manifest.toml` classifies
Python/C++ checks and supports change-aware CI selection. `scripts` provides build, packaging,
release and CI entry points; `.github` composes them into workflows.

Focused design contracts and migration evidence remain where live tests or release protocols
consume them. Historical progress diaries and duplicated style guides belong in Git history.
