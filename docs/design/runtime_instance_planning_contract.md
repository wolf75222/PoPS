# RuntimeInstance planning contract

This contract defines the immutable planning boundary consumed when constructing the single
internal `RuntimeInstance`. Planning performs no binary loading, backend initialization,
allocation, communication, or kernel call.

## Authorities

`build_runtime_plans(install_plan, component_manifests)` accepts an exact, self-verifying
`InstallPlan` and an exact `ComponentManifest` for every compiled block. The block set must match
the artifact exactly. The function consumes, rather than recreates, these authorities:

- `InstallPlan.bind_identity` and its explicit `ExecutionContext`;
- `CompiledSimulationArtifact.platform_manifest`;
- `CompiledPlanRecord.layout_plan`, including directional mapping providers;
- each component semantic digest, accesses, requirements, effects, clocks, precision, target and
  determinism declaration.

The selected platform must prove its dimension, scalar policies, device, memory spaces and
communicator. Unknown evidence is a refusal. A component target and precision are checked through
the existing `ComponentManifest` contract; this layer does not maintain a second component schema.

## Derived values

The result is one `RuntimePlanBundle` containing:

- ordered, block-qualified `RuntimeCall` values with exact component, layout, entry-point, access
  and semantic-manifest identities;
- a `CommunicationPlan` containing derived halo depths, authenticated directional layout
  transfers, collective order and strategy, cross-memory-space fences, and explicit clock joins;
- a `ResourcePlan` containing exact access lifetimes, buffer byte maxima, memory spaces, mapping
  providers, fence identities, and the complete declared requirement evidence;
- a `DeterminismGuarantee` containing the weakest declared component guarantee and the runtime
  assumptions that authenticate it.

All containers are deeply immutable. Every executable action, subordinate plan, and bundle has a
domain-separated PoPS identity. `RuntimePlanBundle.from_data()` accepts only the exact version-1
shape and re-authenticates every nested identity.

## Closed runtime requirement vocabulary

Component requirements remain canonical `ComponentManifest` rows. The runtime planner interprets
only the following execution-bearing rows:

```text
{capability: halo, depth: positive-int, resource?: declared-read}
{capability: collective, resource: declared-access, operation: text, strategy: text}
{capability: buffer, resource: text, bytes: positive-int, memory_space?: proved-space}
```

An omitted halo resource is valid only for a component with exactly one declared read. Multiple
halo requirements for the same call and resource derive one maximum depth; there is no independent
ghost-depth knob. A memory space may be omitted only when the platform proves exactly one space.
Unknown execution requirements are refused rather than ignored.

## Ordering, synchronization, and determinism

Calls follow compiled block order. Collectives receive a contiguous order from those calls. Layout
transfers come only from `LayoutPlan.mappings`; a provider or reverse direction is never inferred.
A write followed by a dependent access in another memory space derives a fence. A cross-space
transition hidden inside one opaque entry point is refused because no legal fence position can be
proved.

Distinct declared clocks must form a connected graph of explicit `access: join` rows naming a
target clock and policy. Bitwise plans always authenticate rank count, device, reduction order and
reduction strategy. `DeterminismGuarantee.require_assumptions()` rejects any later runtime facts
that differ; it never silently downgrades the guarantee.

## Runtime integration boundary

The bundle crosses into execution only through the private implementation of `pops.bind` and
`pops.run`. Author code neither constructs the planner records nor imports runtime engines.
`RuntimeInstance` is the opaque bound value returned by `pops.bind`; its explicit read, checkpoint,
restart and inspection methods are the only supported instance surface. Installation and execution
must authenticate the bundle's plan, bind, component and layout identities without rebuilding or
weakening them.
