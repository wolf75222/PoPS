# Accepted static provider reads, version 1

Contract identifier: `pops.accepted-static-provider-read@1`.

This compiler proof authenticates an `input_fields` availability token used by an
explicit diffusive rate. It extends the accepted-update coefficient-one SSP proof
for externally supplied field storage. It introduces no physical model, numerical
recipe, expression substitution, tolerance, or native execution route.

## Stage and storage meaning

`Program.input_fields(state, for_rate=rate)` carries a State SSA edge that identifies
the observation's exact owner, storage and evaluation stage. An authenticated
`runtime_input` provider owns the field values; the observed State does not compute
those values. Consequently an updated State may observe the same static external
storage at a later stage of an authored composition.

The proof checks the issued State and token objects, their block, clock, point,
region and State reference, the actual consuming rate handle, the source operator's
target/kind/signature, and the exact formal StateSpace/FieldSpace pair. Its
FieldContext must retain that field declaration, exactly that State SSA source,
and the complete declared component surface. The existing accepted-update proof
also checks each rate's mathematical stage coordinate against its affine row sum.

## Resolved producer authority

The required authority is an exact `ProgramModelGraph` built from resolved Case
blocks. It retains each block's instance owner, immutable resolved operation plan,
and originally issued plan identity. A detached Program authenticates operators
through the graph's source Module; it does not restore or borrow a live Program
operator registry.

For every static read, the compiler must:

1. Validate the current canonical plan payload with
   `ResolvedOperationPlan.from_data(plan.to_data())`, then compare its retained
   identity with the originally issued identity. Equality of cached identity
   fields alone is insufficient.
2. Authenticate the block instance and source Module. Recompute the plan's
   provider packs with `require_provider_packs`, including
   `program_field_publications` reprojection.
3. Compare this resolved auxiliary pack with the pack on `_model_impl(model)`
   actually used by the emitter. This comparison retains capacity, component
   owners, contracts, availability, slots and producers.
4. Select the exact declared FieldSpace through `runtime_input_pack`. Every
   claimed component must have an available, assigned `runtime_input` producer.
   The selection must cover the complete constitutive read union after primitive
   expansion.
5. Authenticate every executed field publication. A publication replacing a
   claimed component on the same block prevents its classification as static,
   including a publication outside the accepted value's direct data dependency.

An authored Module pack by itself is insufficient: a resolved field publication
can replace its `runtime_input` producer with a field-problem producer. A standalone
emit-model without resolved block provenance is therefore refused for this proof.

## Acceptance scope and refusals

The resulting accepted-update certificate proves the actual rational coefficient
arithmetic and its coefficient-one convex decomposition, conditional on every
contributing rate's existing forward-Euler spatial guard. It changes no authored
State expression, accepted exchange weight, global Program method/order, or
numerical guard. Independent effects retain the existing nonmutation obligations.

The proof refuses changed plan payloads borrowing cached identities, mismatched
emitter packs, computed or replacing producers, missing/unavailable components,
foreign owners or rates, stale stages or FieldContexts, detached SSA references,
unknown token attributes, incomplete read coverage, and executed State mutations.
Existing negative coefficient, inconsistent stage/step, Boolean coefficient and
terminal guard refusals remain mandatory. Frozen solved-field publications keep
their separate existing mathematical read-closure proof.

## Received Source coverage and limits

The actual public diffusion fixtures are resolved and detached at 16-by-16 cells
and `dt=1e-4`, under both authored Euler and SSPRK2 compositions:

| Fixture | Claimed static components and slots | Publication count |
| --- | --- | --- |
| `variable` | `diffusivity`: 0; `source_factor`: 1 | 0 |
| `diagonal_linear` | `a_x`: 0; `a_y`: 1; `forcing`: 2 | 0 |
| `diagonal_smooth` | `a_x`: 0; `a_y`: 1; `forcing`: 2 | 0 |

Their exact serialized Programs are unchanged by certification. The accepted
weights remain `(1)` for Euler and `(1/2, 1/2)` for SSPRK2. Durable tests are in
`tests/python/unit/codegen/test_static_provider_ssp_proof.py`; the existing frozen
input/publication, input-fields and diffusion-program cohorts remain required.
The corrected candidate based on Source `98804c68` received 89 Source passes,
zero failures/errors/skips, with Native imports blocked. The earlier candidate's
cached-identity defect is superseded by current-payload authentication.

This Source receipt establishes no new CPU numerical, MPI, GPU, AMR, convergence,
or installed-package qualification. A new installed build and actual public
runtime runs must bind subsequent numerical evidence to the integrated Source
and native artifact identities. Computed or analytic providers are not granted
static `runtime_input` status by this contract.
