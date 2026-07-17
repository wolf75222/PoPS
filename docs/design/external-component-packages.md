# External component packages

PoPS has three deliberately different values at its extension boundary:

1. `SourceComponentPackage` owns authenticated source, header or PoPS IR bytes. It may advertise
   generic C++ component types, but it owns no target symbols and no binary digest.
2. `CompiledComponentArtifact` owns one linked binary, its exported entry-point table, exact
   `PlatformManifest`, component manifest digest and binary identity. A source-compiled artifact and
   a precompiled fixed-signature artifact have disjoint authorities.
3. `InstalledComponent` is the only loadable value. It exists only after binary digest and exported
   symbols have been checked again on the atomically published file.

The public authoring path is object based:

```python
components = pops.external.load("my_flux.pops.json")
MyFlux = components.require(
    "my_flux",
    interface=pops.interfaces.NumericalFlux,
)
flux = MyFlux()
```

There is no `_native(...)` call, component-specific pybind class or concrete-component dispatch
table. `pops.external.compile_component(component)` compiles and links the authenticated payload for
one proved target, audits its exported table getter without loading the image, and only then returns
an artifact. Source and fixed-binary packages export the same generated C/POD protocol: one
`pops_component_interface_v1()` getter containing the exact versioned interface tables declared by
the manifest. PoPS does not synthesize an implementation-specific wrapper or guess methods from a
class. A runtime consumes `InstalledComponent.runtime_contract`; it never reopens source files or
interprets package JSON.

## Package identities and trust boundaries

The JSON package schema is strict and versioned. It contains the complete `ComponentManifest` data,
explicit exports, payload digests, protocol ABI and a package digest. Paths are canonical relative
POSIX paths; absolute paths, traversal and resolved escapes are rejected. Payload bytes are read and
retained at load time, so compilation does not trust a later mutable source path.

Source registration compares the complete component-manifest digest and source-package digest.
Compiled registration compares the component, exact platform identity, artifact identity and binary
identity. Both registries validate before locking/mutating, make identical registration idempotent,
reject non-identical collisions and refuse every mutation after `freeze()`.

A fixed binary must declare `signature.generic = false`, one exact component target, exact runtime
entry symbols and an exact `PlatformManifest`. Its target, manifest, binary digest and actual exported
symbol table are verified before it can enter `CompiledArtifactRegistry`. A `.so` is never treated as
evidence of C++ template genericity and is never `dlopen`-ed to discover its contract.

Installation uses a same-directory temporary file, flush/fsync, digest and symbol audit, then an
atomic no-clobber hard-link publication into a content-addressed filename. Existing addressed files
are reused only when their bytes authenticate to the same binary identity.

## Generated native interfaces

`schemas/component_catalog.v2.json` owns the interface names, independent versions, operations and
table layouts, plus the version of the common request/value ABI. The complete declaration feeds the
catalog digest. The generator emits `pops.interfaces`, the Python route data and
`generated_component_abi.hpp` together; `--check` makes any hand-edited drift fail CI. The current
protocol includes separate tables for numerical flux, ghost boundary, field-boundary closure,
tagging, clustering, transfer, reflux, field solve, writer and field topology. Adding an
implementation requires no central scientific switch.

The installed CPU route proves 2D, `float64`, host execution. It supports source/header payloads and
versioned C++ IR translation units. Interface-specific requests use dimensioned views and an explicit
execution context; an unsupported dimension, scalar, memory space, communicator, capacity or missing
operation is rejected before the component is called. Hot-path tables are resolved once during
installation and invoked in bulk; `dlsym`, Python and manifest lookup stay outside cell loops.

An external writer is selected by the output itself:

```python
from pops.output import ExternalWriter, ScientificOutput

output = ScientificOutput(
    format=ExternalWriter(component=writer_component, extension=".pops"),
    schedule=schedule,
    fields=(block[state],),
    target="fields/state",
)
```

The `component_id`, manifest identity and exact `Writer` interface cross compile, bind and install.
The runtime refuses a missing, differently signed or unloaded component and never chooses a
process-global « only writer ». Preparation receives the complete selected snapshot; publication
happens only after step acceptance, while discard/rollback remove temporary or already-published
artifacts on the declared failure path.

Other devices, scalar types and dimensions remain unavailable until a target variant and every
interface operation prove them. The wheel ships the exact signed PoPS header tree under
`pops/include`, so AOT compilation does not depend on a source checkout.
