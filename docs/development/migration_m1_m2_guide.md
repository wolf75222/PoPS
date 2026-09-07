# PoPS M0–M2 developer usage and migration guide

This note describes the operator-first Python boundary delivered by M0–M2. It is a developer
usage guide for typed authoring, validation, and resolution. The lifecycle remains explicit:

```text
Model / Module + Case + Program
        -> validate -> resolve -> compile -> bind -> run
```

`validate` and `resolve` produce typed, immutable records. They do not prove native emission,
binding, execution, numerical correctness, or performance. Application code should keep the
scientific declarations as its only authority and cross the native boundary only through the
resolved plan.

## Qualify state and field identity

Declare a `StateSpace` or `FieldSpace` once on its owning `Module`, then use the registry-issued
handle. A component name such as `rho` or `phi` is only a label; the owner, space metadata,
representation, support, sampling, shape, and domain are part of the typed identity.

```python
from pops.model import Module, StateHandle
from pops.problem import Case
from pops.time import Program

module = Module("two_states")
electrons = module.state_space("electrons", ("ne", "ue"))
ions = module.state_space("ions", ("ni", "ui", "energy"))
electron_handle = module.state_handle(electrons)
ion_handle = module.state_handle(ions)

fields = module.field_space("psi_fields", ("psi",))
field_handle = module.field_handle(fields)
phi, = module.field_symbols(fields)

case = Case("plasma")
block = case.block("fluid", module, states=(electron_handle, ion_handle))
program = Program("multi_state")
electron_time = program.state(block[electron_handle])
ion_time = program.state(block[ion_handle])

assert isinstance(electron_handle, StateHandle)
assert electron_time.space is electrons
assert ion_time.space is ions
assert phi.space is fields and phi.handle == field_handle
assert block[field_handle].declaration_ref == field_handle
```

`Program.state` accepts the block-qualified form `program.state(block[state_handle])`. Passing a
string, an unqualified model handle, a field handle, or a declaration from another `Case` is a
checked ownership error. A field equation is likewise case-owned: qualify its unknown and typed
providers through the relevant block before calling `case.field(...)`. A field output remains a
field record; it is not an implicit state or an implicit solve.

Space metadata is structural. Unknown dimensions (`units=None`) are distinct from an explicitly
dimensionless declaration, and different supports cannot be inferred from equal array shapes:

```python
from pops._ir.quantity import PhysicalDimension, PhysicalSupport
from pops.model import StateSpace

physical = PhysicalSupport((("x", "physical"), ("y", "physical")))
unknown = StateSpace("U", ("rho",), support=physical)
dimensionless = StateSpace(
    "U", ("rho",), support=physical, units=[PhysicalDimension()]
)
assert unknown != dimensionless
```

Use `module.state_symbols(space)` and `module.field_symbols(space)` for symbolic quantities. They
retain the declaration handle and its complete typed `Space` until an authenticated lowering
binding supplies native coordinates. A foreign support, representation, or shape is rejected by
`Module.apply` before lowering.

## Preserve signed balances and select one equation

`Model.rate` retains the complete signed physical equation. `rate.select(...)` creates a
partition/view of that balance and preserves source order, coefficient, target, and repeated
occurrence identity. It does not choose between alternative equations for the same state.

```python
from fractions import Fraction
from pops.math import ddt, div

# `model`, `state`, and `flux` are the typed handles from one pops.Model.
source = model.source("forcing", on=state, value=[state[0]])
rate = model.rate(
    "balance",
    equation=ddt(state) == (
        -div(flux) + Fraction(2, 3) * source - source + source
    ),
)

selected = rate.select(source)
assert selected.balance is rate.balance
assert selected.view.ordinals == (1, 2, 3)
assert [item.coefficient for item in selected.occurrences] == [
    Fraction(2, 3), -1, 1
]
```

When a state has alternative complete balances, select the root explicitly. A partition cannot
be passed to `select_balance`:

```python
first = model.rate("advection", equation=ddt(state) == -div(flux) + source)
first.select(flux)                         # a numerical view of `first`
second = model.rate("alternative", equation=ddt(state) == source)

try:
    model.validate_balance_selection()
except ValueError as error:
    assert "select_balance" in str(error)
assert model.select_balance(second) is second
```

The default accumulation is the identity. A nonlinear or nonidentity accumulation must carry its
discrete representation and quadrature; otherwise resolution records a refusal. Scaled,
multi-divergence, signed-source, and joint projection balances stay in the scientific IR and are
not rewritten into a legacy boolean flux/source switch.

## Capture once, apply once, project without duplication

A decorated operator is evaluated once during authoring. `Module.apply` accepts only its captured
immutable expression body and returns one joint application. Named output projections share that
application identity and therefore do not evaluate the body again.

```python
from pops._ir.application import ApplicationContext, ApplicationEvaluation
from pops.model import Module, Rate, RateBundle, Signature

module = Module("joint")
a = module.state_space("a", ("rho",))
b = module.state_space("b", ("rho", "energy"))
signature = Signature((a, b), RateBundle({"a": Rate(a), "b": Rate(b)}))

@module.operator("exchange", signature=signature, kind="coupled_rate")
def exchange(ua, ub):
    return {
        "a": [ua[0] - ub[0]],
        "b": [ub[0] - ua[0], ub[1]],
    }

ua, ub = module.state_symbols(a), module.state_symbols(b)
application = module.apply(
    exchange, ua, ub, context=ApplicationContext(stage="first")
)
left = application["a"][0]
right = application["b"][0]
values = ApplicationEvaluation({
    ua[0].qualified_id: 4,
    ub[0].qualified_id: 1,
    ub[1].qualified_id: 7,
})

assert left.application is right.application is application
assert left.eval(values) == 3 and right.eval(values) == -3
assert (left + left).eval(values) == 6
```

The captured body is frozen symbolic data; repeated lowering does not call the Python function.
An uncaptured callable is refused by `Module.apply`. Direct `to_cpp()` on a quantity, application,
or projection is deliberately rejected; only resolved lowering supplies native coordinates. Joint
field/product and interaction outputs may therefore be representable while their native route
remains explicitly unsupported.

## Declare EOS meaning explicitly

Parameter spelling has no physical meaning. A parameter named `gamma` remains an ordinary runtime
parameter until ideal-gas metadata is declared. `Module.constitutive` accepts a numeric value or a
registered compile-time `ConstParam`; it rejects a runtime parameter and values not greater than
one.

```python
from fractions import Fraction
from pops.model import Module
from pops.params import ConstParam, RuntimeParam

ordinary = Module("ordinary")
ordinary.param(RuntimeParam("gamma", default=3))
# The spelling alone does not select an equation of state.

gas = Module("gas")
ratio = gas.param(ConstParam("heat_capacity_ratio", Fraction(7, 5)))
gas.constitutive(gamma=ratio)
```

The explicit constitutive declaration is included in the module identity and manifest. No lowerer
or runtime path may infer EOS metadata from a parameter name.

## Use fresh temporal stages and qualified commits

`Program.stage(name, c=...)` allocates a fresh program-local stage identity every time, even when
the label and exact abscissa are equal. The label is diagnostic; the ordinal is semantic and
survives serialization, graph rebuilding, optimization, and detachment.

```python
from fractions import Fraction
from pops.time import Program, TimePoint

program = Program("fresh_stages")
first = program.stage("predictor", c=Fraction(1, 3))
second = program.stage("predictor", c=Fraction(1, 3))

assert first.time == second.time == TimePoint(program.clock, Fraction(1, 3))
assert first != second
assert (first.identity, second.identity) == (0, 1)
```

For an authored state, materialize and commit through its qualified endpoint:

```python
temporal = program.state(block[electron_handle])
candidate = program.value("electron_next", temporal.n, at=temporal.next.point)
program.commit(temporal.next, candidate)
```

Use the longer `StagePoint(name, {partition: TimePoint(...)})` spelling when a method has multiple
partition coordinates. Do not use a stage name, timestamp, or free string as a state identity.

## Inspect resolved operations and provenance

For a whole `Case`, resolution attaches one plan per block:

```python
pops.validate(case)
resolved = pops.resolve(case, layout=layout)
block_plan = resolved.resolved_operations[block.name]

report = resolved.explain(block.name)
assert report[block.name]["operations"]
assert all(
    not row["evidence"]["executed"]
    for row in report[block.name]["operations"]
)
```

`layout` above is the application’s typed `Uniform` or `AMR` layout authority. The resolved plan
records operation identity, signed occurrences, inputs, outputs, dependencies, halo/effect
boundaries, selected numerical method, and disposition. `resolved_operations` is inspectable
compiler evidence; it is not a native artifact.

For a module-level route, the exact plan API is also usable directly:

```python
from pops._ir.expr import Const
from pops.codegen.resolved_operations import build_resolved_operations
from pops.model import Module, Rate

source_module = Module("resolved_source")
source_state = source_module.state_space("U", ("rho", "momentum"))
rho, _ = source_module.state_symbols(source_state)
source_module.operator(
    "source", (source_state,) >> Rate(source_state), "local_source",
    expr=(rho, Const(0)),
)

plan = build_resolved_operations(source_module)
operation = plan.operations[0]
explanation = plan.explain(operation.identity)
assert explanation["operations"][0]["evidence"]["resolved"] is True
assert explanation["operations"][0]["evidence"]["executed"] is False

# Reauthenticate the current source and route before an emitter call.
native_route = plan.require_native(operation.identity, module=source_module)
```

`ResolvedOperationPlan.to_data()`/`from_data(...)` authenticate the canonical wire snapshot.
`require_provider_packs` and `require_native` recheck the current module, provider plan, scientific
occurrences, and current route. Source drift, missing routes, unsupported operator kinds, omitted
terms, or double coverage raise structured `LoweringRejection`; a historical operation type or a
string lookup is never substituted.

## Cross the native boundary through checked adapters

Application code uses `pops.compile(resolved)`. The existing emitter is retained only behind the
narrow `lower_and_validate(...)` adapter, which accepts the exact `ResolvedOperationPlan`, binds
the authenticated provider pack, and checks every selected program evaluation against the current
source module before emission. Repeated adapter entry must preserve the same resolved plan and
source-module hash.

Extension components use the public manifest-driven `ComponentAdapter` seam. Register a component
with its `ComponentManifest` and target platform, retrieve it by the manifest `component_id`, and
invoke a declared interface. A `fallible_evaluation` interface must return an explicit
`EvaluationOutcome` (`ok`, `retry`, `reject`, or `failed`); implicit values, missing bindings,
malformed signatures, and unsupported target variants are refused before mutation or execution.

The adapter is a checked interface table, not a scientific class switch. It does not provide a
fallback for an unbound native entry point and does not turn a manifest declaration into execution
evidence.

## Migration map and current boundary

| Older authoring habit | M0–M2 form |
| --- | --- |
| Free component names or model-local handles in a `Program` | Declare `StateSpace`/`FieldSpace`, obtain the registry handle, and use `block[handle]`. |
| State or field selected by a string | Use the exact owner-qualified handle and its typed `Space`; foreign support/representation is rejected. |
| Boolean flux/source switches | Author the signed `Model.rate` equation; use `rate.select(...)` for partitions and `Model.select_balance(...)` for alternatives. |
| Python callback re-evaluated at lower/run time | Capture the operator during authoring, call `Module.apply`, and keep projections on one application. |
| EOS inferred from `gamma` spelling | Register a `ConstParam` and call `Module.constitutive(gamma=...)`. |
| Reused stage name/time as a cache key | Allocate `Program.stage`; use its fresh identity and qualified value/commit endpoints. |
| Compiler choice hidden behind a string or historical route | Inspect `resolved_operations`/`explain`; call `require_native` against the current module. |
| Broad legacy/native adapter | Use the narrow checked adapter and manifest interface table; unsupported routes fail closed. |

The following are the next generalization slices in the migration. They extend existing routes;
the table does not withdraw current AMR, restart, rollback, runtime-I/O, or MPI behavior. It records
which new contracts still await their phase evidence.

| Future phase | New generalization still awaiting that phase |
| --- | --- |
| M3 — Unify field problems and solve results | Derive general field providers from one equation with independent field storage; add variable-coefficient scalar and joint multi-field/shared-normalization routes, direct multi-state loads, source/flux observations, and precise reuse invalidation. Existing scalar/Poisson paths remain current capabilities; this row scopes the new general routes. |
| M4 — Generalize interactions and typed native calls | Add heterogeneous multi-output interactions and constitutive laws plus authenticated typed native-call nodes, with one evaluation context and physical inventory maps. Existing specialized interaction, native, and runtime-I/O routes remain available where qualified; M4 broadens the vocabulary and its evidence. |
| M5 — Deliver transport–diffusion execution paths | Extend one physical diffusion declaration across explicit and spatial-implicit paths, variable coefficients and a bounded tensor scope, combined restrictions/exchange accounting, and one joint fitted drift–diffusion construction. |
| M6 — Consolidate temporal problems and accepted continuation | Generalize solve/result tuples, nested-attempt publication, accepted exchange transactions, checked explicit/implicit/IMEX/splitting expansions, and explicit histories/dense-output/restart/topology-transition validity. Existing SSA/region, rollback, restart, and continuation routes are retained while these extensions are qualified. |
| M7 — Extend coupled supports, layouts and AMR | Preserve the qualified scalar AMR path while adding synchronized composite-field coupling, a bounded different-support 1x1v-to-1x physical map, explicit physical maps, and the required collective/transfer/transition diagnostics. |

Private kernel reuse is eligible only for a transparent, field-free, parameter-free source
implementation of the same declaration. Separate per-block evaluations and owner-qualified
identities remain distinct; reuse does not merge their ownership or provenance. No performance gain
is claimed, and the native reuse test is still pending.

Typical structured refusals include `unsupported_balance_realization`,
`multi_state_field_provider_unsupported`, `operator_kind_not_lowerable`,
`native_realization_unavailable`, and `incomplete_term_coverage`. Keep the refusal and its phase in
the evidence record; do not simplify it into a dummy computation.

## Evidence status

The frozen Dim2 candidate `aec6b17` has two recorded tutorial profiles passing (`scalar_tutorial_openmp`
and `scalar_tutorial_mpi2`). The full scalar, multiphysics, and IMEX–AMR profiles remain under
repair. This guide makes no M0–M2 completion claim and reports no numerical-correctness or
performance-green result. The integration owner will fill the exact source/configuration/command/
artifact/oracle record in [`migration_m0_m2_results.md`](migration_m0_m2_results.md).

Focused source witnesses are `tests/python/unit/codegen/test_qualified_quantity_application.py`,
`tests/python/unit/problem/test_state_handle_space.py`,
`tests/python/unit/time/test_qualified_program_references.py`,
`tests/python/unit/time/test_fresh_program_stages.py`,
`tests/python/unit/physics/test_balance_occurrence_contract.py`,
`tests/python/unit/codegen/test_resolved_operations.py`,
`tests/python/unit/codegen/test_resolved_operation_pipeline.py`, and
`tests/python/unit/codegen/test_component_adapters.py`, together with the implementations in
`python/pops/model/module.py`, `python/pops/physics/_board_rate.py`,
`python/pops/time/_program/clocks.py`, `python/pops/codegen/resolved_operations.py`, and
`python/pops/codegen/module_lowering.py`.
