# Pybind and native component boundary

This note defines the final binding boundary. Python authoring never selects a native algorithm with
a string and never calls `System.add_block`. A typed descriptor contributes a versioned
`ComponentManifest`; resolution authenticates its small interfaces and produces an immutable route
identity. Pybind materializes the already-resolved plan and does not reinterpret scientific intent.

## Translation-unit ownership

Bindings live under `python/bindings/` and have only three responsibilities:

| Family | Responsibility |
| --- | --- |
| module initialization | register value types and the private native execution entry points |
| plan installation | decode authenticated generated records and construct a `RuntimeInstance` |
| generated template leaves | instantiate bounded builtin template products without owning route policy |

The `System` and `AmrSystem` C++ types are private execution engines. Their pybind classes may expose
installation seams used by `pops.bind`, but no public Python authoring object delegates to their old
registration methods.

## One component path

Builtin and external components cross the same contract:

1. A `ComponentManifest` declares URI/version, exact interfaces, requirements, capabilities,
   target variants, effects, restart data and entry points.
2. The registry validates the manifest and produces the same provenance/report shape regardless of
   origin.
3. Resolution binds each requirement to a qualified provider and selects a target proven by the
   platform manifest.
4. Lowering calls only the component's narrow interfaces. It never switches on a Python class name,
   performs `isinstance` scientific dispatch, or consults a handwritten allow-list.
5. Installation authenticates the compiled artifact and binds its declared entry points. Missing
   interfaces, symbols or target evidence fail before runtime mutation.

The native interface vocabulary is generated from
[`schemas/component_catalog.v2.json`](../../schemas/component_catalog.v2.json). Compile-time
conformers implement only the concepts they need (`Requirement`, `Lowering`, `Stencil`, `Stability`,
`Provider`, `Effects`, `Restart`, `Report`, `FallibleEvaluation`, `Format`). There is no universal
component base class and no `provides(any)` escape hatch.

## Generated builtin leaves and build memory

Some builtin C++ policies remain template-instantiated in `_pops` for zero-overhead device kernels.
That is an implementation strategy, not a second registration path. Their identities and supported
combinations come from the generated catalog; generated visitors map resolved numeric IDs to typed
leaves. A handwritten pybind `if/else` on transport, flux, limiter, layout or model is forbidden.

Large template products stay split across generated translation units so one compiler process does
not instantiate the full product. Ninja job pools bound concurrent heavy compilations. This build
partitioning must not leak into manifests, route identities or user-visible behavior.

## External packages

`pops.external.load(...).require(alias, interface=...)` is the only package entry. Source packages
are specialized AOT against the resolved target; fixed-binary packages use a distinct exact ABI
contract and never claim template genericity. Both yield authenticated component artifacts and use
the same installation registry. Raw `.so` paths, historical brick JSON and `compile_library` are not
accepted.

An external conformer can supply a flux, boundary provider, tagger, clustering policy, transfer,
reflux operation, solver or writer without editing a central dispatcher. Adding a builtin may add a
catalog row and generated template implementation; adding an external component changes no PoPS
source file.

## Enforcement

Architecture and conformance tests must prove:

- public imports and normative examples contain no old authoring method or string selector;
- generated catalog/schema products are current and Python/C++ identities match;
- component trust boundaries contain no scientific concrete-class branch;
- malformed manifests and missing interfaces fail before registry mutation;
- builtin and external provenance/report records have identical structure;
- at least one external component executes on both Uniform and AMR layouts;
- generated leaf coverage is complete without a handwritten per-combination binding file.
