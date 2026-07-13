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
table. `pops.codegen.compile_component(flux)` is the current advanced compiler seam used by the
simulation compiler: it instantiates the component against resolved protocol types, generates the C
ABI wrapper, compiles and links it, inspects symbols without loading the image, and only then returns
an artifact. A runtime consumes `InstalledComponent.runtime_contract`; it never reopens source files
or interprets package JSON.

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

## Current proved capability

The implemented generic `NumericalFlux` AOT route proves 2D, `float64`, host CPU and serial execution.
It supports source/header payloads and versioned C++ IR translation units. Other dimensions, scalar
types, devices, runtime component parameters and other interfaces are represented by the manifests
but rejected before compilation until a corresponding interface lowering proves them. External
provider requirements are also rejected until the compiler is given a resolver that can prove and
materialize every requested provider. The wheel ships the exact signed PoPS header tree under
`pops/include`, so AOT compilation does not depend on a source checkout.
