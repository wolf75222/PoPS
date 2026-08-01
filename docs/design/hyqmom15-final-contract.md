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

- `LocalClosure(order, name, evaluator)` is the closure extension interface. The final script writes
  the six fifth-order HyQMOM relations under `@closure(4)` and passes that value to
  `HyQMOM15.vlasov_lorentz(closure=...)`. The evaluator executes once on symbolic standardized
  moments during authoring and must return exactly the order `N + 1` keys. Its arithmetic is folded
  into the ordinary flux graph, so there is no Python callback or mutable closure state in native
  execution; the installed Program hash authenticates the resulting graph across restart.
- `RealizabilityProjection` configures the smooth floors and the complete 15-moment projection.
  `guard_hyqmom15_candidate(...)` authors ordinary typed acceptance guards with
  `ProjectAndRecheck(on_failure=RejectAttempt())` inside the `Program` transaction. Rejection and
  rollback therefore use the shared runtime path rather than a HyQMOM-specific branch.
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
model's explicit signed wave pair. The example authors the ordinary inspectable IMEX `Program`
explicitly so its realizability guard is visibly inside the commit transaction. The
`pops.lib.time.IMEX(...)` preset remains an ordinary `Program` constructor with no alternate runtime
route, but it does not hide this model-specific scientific guard. The local solve is specialized from
the resolved state manifest and therefore prepares exact 15 by 15 stack storage for the shared
pivoted local provider, without an explicit inverse, eight-component fallback or family dispatch.
The executable also requires the installed transaction plan to own every typed provisional store:
states, fields, topology, flux ledgers, caches, solver warm starts, histories, clocks, schedules,
consumers, diagnostics and external effects. Its forced non-realizable attempt compares the
accepted state, solved fields, histories, Program identity and ConsumerGraph cursors before and
after rejection and refuses any published artifact. This Uniform case has no non-empty AMR reflux
ledger; non-empty multilevel ledger persistence remains the responsibility of the AMR final example.

The example executes only:

```text
Model + Case + Program -> validate -> resolve -> compile -> bind -> run
```

One accepted step publishes authenticated HDF5, ParaView and scheduled checkpoint artifacts. The
script reopens both scientific formats, creates a manual checkpoint, restores it into a fresh bind,
compares the full 15-component state, solved field, clock, program identity and consumer cursors,
then advances the uninterrupted and restarted instances one more step and requires exact equality.
Every retained state must remain realizable and conserve the integral of `M00` over the unit square
within a relative tolerance of `1e-10`; the machine-readable report exposes the measured particle
number and maximum relative error. This is the final behavior, not a transition or compatibility
example.
