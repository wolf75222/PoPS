# Completed original-field composite interaction, IR19

This opt-in port is based on complete baseline
`0abbe25395a44e570d8d5525693b8e2dcbf4d387`, with the earlier gap inventory
cherry-picked as `52cfeaa1`. It implements a post-consumed original-field source,
not a nonlocal term inside Newton or a qualification of full M26/M27.

The physical declaration and original equation precede the Program. An existing
`Case.field`/original nonlinear FieldProblem is solved with its declared method,
captures, seven numerical controls and original full-F acceptance. Its consumed
`field.observe(...)[unknown]` is still global: `space`, `block` and `state_ref`
remain `None`. The new explicit call is:

```python
I = program.spatial_interaction(
    observed[declared_unknown], kernel,
    output_space=declared_output_field_space,
    measure=CellVolumeMeasure(), quadrature=CellMidpoint(),
    realization=DirectSpatialInteraction(max_workspace_bytes),
    source_scope="completed_original", owner_block=declared_storage_block,
)
```

The owner has an already issued TimeState in this Program/Case/clock. It supplies
only an allocation route, and may have a different state width. It is not added
to the original solve's captures. Source geometry is authenticated through the
original FieldProblem's physical captures and actual prepared provider, not the
output FieldSpace or storage State. Known output units refuse while original
field units are unknown; no storage units are borrowed. The current physical
FieldProblem descriptor does not encode a field-unit authority for this port.

The interaction node is `pops.spatial-interaction@3`; its immutable source
contract is `pops.completed-original-field-source@1`. IR19 is conditional. The
original qualified storage declaration is held independently from node attrs,
with exact typed canonical comparison and authenticated transfer at
freeze/rebuild/detach. Removing the discriminator or resealing the owner/tuple
metadata cannot downgrade or replace the declaration.

## Actual native authority and scheduling

Only the original provider's actual Outcome Accept hook stamps the accepted
candidate pointer/generation. Staging, solved reports, rejection, discard and
failed solves do not issue that receipt. A subsequent original invocation
supersedes the previous receipt even if it reuses the same candidate address.
Generic accepted publication invalidates an original receipt. Old snapshots
must therefore be reacquired after a new provider invocation.

`seal_original_field_source` requires the real typed
`PreparedAmrFieldResidual<Dim>`, its current authority callback and the exact
prepared provider. It calls the gated `core.candidate(current, prepared_lane)`
accessor, then checks the provider's completed Accept stamp. It deep-copies every
**accepted provider solution level**, including the entire unknown tuple, after
validating layout/ownership/width against the original candidate. The active
masks are also deep-owned and the Geometry values are captured at that same
barrier, rather than read lazily from a subsequent phase. Fab copy uses
real Kokkos deep-copy, not shared borrowed Views. The immutable binding table
holds result SSA id, source SSA id, original tuple component, storage route and
exact emitted identity. It refuses a substituted component or another valid
storage route on an already sealed result.

The solve-once phase seals this source after actual Outcome consumption and
computes all output levels through `direct_spatial_interaction`. The observe and
publish phases retrieve only the already complete tower. Source integration uses
the original provider's active quotient and actual level geometry. The currently
supported original provider already refuses EB/shared-interface layouts; this
port does not invent kappa or bypass that existing refusal. Direct quadrature
and kernel semantics are unchanged. Storage layout/distribution/rank must match
each source level, but storage ncomp is not the observation width.

Local allocation, copy, kernel and result phases fence and vote on the context's
authenticated lane before the following collective. Only after the last authority
and receipt checks are successful does a detached map node publish the complete
output tower. Every getter/reduction rechecks actual core points, capture
identities, attempt leases, owner, lane, topology/materialization/coefficient
generation and Accept stamp. Rollback, retry, rebind, phase/point change or a new
provider invocation revokes visibility while strong-owned allocations can remain
alive. No accepted-State reconstruction or relabelled State.n occurs.

Raw `sum_component`, `abs_sum_component`, min/max scalar reductions consume the
complete active output tower, not the active-level scratch. Replicated levels
contribute only on lane rank zero; distributed levels use their actual ownership.
These are unweighted scalar reductions, matching their existing public meaning;
the physical source quadrature itself uses volumes. No MPI-size division occurs.

## Realization and resource scope

For S active source cells, O stored output cells, C selected components and L
levels, the direct realization costs O(S*O*C) arithmetic, with real owned/rank
broadcast quadrature. It retains the full original tuple plus all output levels;
there is no DOF/component-count cap. The explicit uint64 workspace budget accounts
numeric fields, field/topology metadata and serialized bindings. The original
solver's own Core/capture allocations and allocator bookkeeping remain separate
from this budget. Checked integer products/additions precede large allocation.
Rank-local retained-byte reservations use exact scalar-word integer transport;
the common maximum leaves the same direct scratch budget on every rank.

The realized public consumer here is a read-only complete native tower and the
scalar reductions above. Gradient, field/history consumers and arbitrary affine
aliases refuse until their own authenticated complete tower/ghost ports exist.
This does not say gradients or nonlocal implicit models are mathematically
impossible. Uniform original fields need their own Accept snapshot implementation
and refuse this named realization explicitly. IR17 State and IR18 history routes
remain available with their old semantics and bytes. Arbitrary gather-local State
candidates still refuse on multilevel AMR; only the actual original provider's
full, accepted candidate tuple supplies this new barrier.

Native ABI/wire ordinals and manifest schema are unchanged. These existing
headers change the installed SDK signature and context/provider storage layout;
ROOT must rebuild all native dimensions before native reception. Snapshot
registries are attempt resources, not a new durable checkpoint wire payload;
a restart must run/accept a new original solve before obtaining another source.

## Source evidence, not native qualification

The real public witness uses the original coupled FieldProblem with 1 or 3
unknowns, captures, source consumer and permutation. It emits the real complete
CPP. The final SOURCE_ONLY suite and negative inventory cover typed metadata,
owner/source-id reseal/downgrade, freeze/resolved detach, unknown units, scalar selection,
Uniform refusal, storage-only TimeState route and gradient refusal. Historical
IR17/18 defaults are compared at the same authoring callsite in fresh interpreters
with `sol61_closed_interaction_legacy_parity.py`.

Three genuine default original-field profiles (AMR2, Uniform3, AMR5) match the
baseline byte-for-byte in authored/resolved IR, full solve request, CPP and all
three module hashes/manifests. IR17 issued candidate/accepted State and IR18
selected two-lag history also match exact IR/hash/Uniform+AMR emitter bytes.

The complete initial combined source run received 47 properties (four host
compiler probes deliberately deselected). The final author run received 16
properties and exposed one fault-injection harness error: `_replace_value`
cannot replace immutable SSA inputs. Its corrected explicit post-issuance
injection then received the remaining source-id rejection property separately.
The storage-only, gradient and native source-envelope subset also received
three properties. These are source checks, not independent native reception.

One actual emitted Dim2 CPP TU passed fsyntax-only against this private source's
headers and the real Kokkos/MPI/OpenMP dependency headers. CPP SHA256:
`b3f85c2d350cc028e84beace83c982aaef2d4206a14cc9704c9546828f485baf`.
Production diff SHA256 at that syntax reception:
`a7bf970d57eac4ff4012f0dbdec3100fff5da5c586551109336b40e07af4f9ac`
at the final header syntax reception; subsequent Python-only source-id origin
checks add no native signature or C++ control-flow change.
The first syntax attempt exposed a wrong Core template argument; the same TU was
corrected and rerun. No DSO, Native execution, SDK installation, ENV or donor was
modified. Syntax does not receive MPI convergence, rollback or scientific output.
The recorded syntax command below used `pops-api040/include`; it is not
attributed to ROOT's newer `pops-api040-ir17` dependency environment. ROOT's
378f compile_commands authenticates that newer environment for the upcoming
native reception and reviewer syntax, which remain separate evidence.

```sh
rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:. \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q \
  tests/review/test_sol61_closed_original_interaction.py \
  tests/review/test_sol61_composite_snapshot_inventory.py
rtk proxy /usr/bin/clang++ -std=c++20 -fsyntax-only -DPOPS_NATIVE_DIM=2 \
  -DPOPS_RUNTIME_SHARED_EXCEPTION_ABI -DPOPS_HAS_KOKKOS -DKOKKOS_DEPENDENCE \
  -DPOPS_HAS_MPI -DPOPS_HAS_PARALLEL_HDF5 -Iinclude \
  -I/Users/romaindespoulain/miniforge3/envs/pops-api040/include \
  -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include \
  /tmp/sol61-ir19-full.cpp
```

ROOT's next reception should exercise genuine Accept vs discard/rejection,
partial/empty/replicated ranks, component/storage substitutions after seal,
mutable-provider-alias independence, stale parent/child leases, phase/capture/
epoch changes, point/lane consensus, and a fault on one rank at each local phase.
Those are required native checks, not successful results claimed by this source
receipt. This does not qualify convolution inside original F, arbitrary captured
candidate snapshots, arbitrary 2D AMR geometries, gradients or full M26/M27.
