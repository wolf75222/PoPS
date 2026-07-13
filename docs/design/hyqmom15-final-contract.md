# HyQMOM15 final contract

ADC-694 represents the 15-moment Vlasov--Poisson--Lorentz system with the same small public
interfaces as any other PoPS physics. `HyQMOM15.vlasov_lorentz(...)` returns an ordinary
`pops.physics.Model`; its state, flux, explicit rate, electric source and implicit magnetic map are
retrieved from the model's immutable typed families. No preset-specific result wrapper, model-name
test or native `hyqmom15` dispatch exists.

The final executable target is
[`examples/final/EXEMPLE_SPEC_FINALE_15_MOMENTS_HYQMOM.py`](../../examples/final/EXEMPLE_SPEC_FINALE_15_MOMENTS_HYQMOM.py).
It adds an ordinary model-owned Poisson unknown and operator to the provided model. A fixed unit ion
background makes the periodic right-hand side neutral. `FieldOutput("phi", ...)` and
`GradientOutput("grad", ..., sign=-1)` materialize the canonical `phi`, `grad_x`, `grad_y` field
context consumed by the electric source. The numerical method, periodic boundary law, nullspace,
gauge and multigrid solver remain separate `FieldDiscretization` choices on the `Case`.

## Generic extension boundaries

- `LocalClosure(order, name, evaluator)` is the closure extension interface. The evaluator executes
  once on symbolic standardized moments during authoring and must return exactly the order `N + 1`
  keys. It is absent from native execution.
- `RealizabilityProjection` configures the smooth floors used by moment algebra. It does not pretend
  to be a time-step acceptance guard. A future realizability rejection policy must implement the
  ordinary typed `AcceptanceGuard` protocol and participate in the Program transaction explicitly.
- `Model.field_spaces()` derives solved storage from the generic field-output protocol. A scalar
  `FieldOutput` contributes one component; a Cartesian `GradientOutput` contributes two. This rule
  lets any provided or user model add a potential-plus-gradient solve without a model-specific
  compiler branch or a repeated manual component list.
- The field provider is the exact typed operator handle exposed by `FieldOperator.providers`; IMEX
  receives that handle explicitly. Other field solvers and other implicit operators compose through
  the same interfaces.
- Formal reconstruction order, required halo depth, the three field components and the local matrix
  dimension are derived from their selected providers and resolved manifests. The user does not
  repeat any of those values.

## Native and runtime proof

The final spatial plan uses conservative variables, MUSCL with Van Leer limiting and HLL with the
model's explicit signed wave pair. `pops.lib.time.IMEX(...)` constructs an ordinary inspectable
`Program`; the preset contains no alternate runtime route. Its local solve is specialized from the
resolved state manifest and therefore emits exact 15 by 15 storage and `mat_inverse<15>`, without an
eight-component fallback or family dispatch.

The example executes only:

```text
Model + Case + Program -> validate -> resolve -> compile -> bind -> run
```

One accepted step publishes authenticated HDF5, ParaView and scheduled checkpoint artifacts. The
script reopens both scientific formats, creates a manual checkpoint, restores it into a fresh bind,
compares the full 15-component state, solved field, clock, program identity and consumer cursors,
then advances the uninterrupted and restarted instances one more step and requires exact equality.
This is the final behavior, not a transition or compatibility example.
