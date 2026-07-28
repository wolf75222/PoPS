# Canonical `pops.lib.time` Program factories

`pops.lib.time` provides configured Programs, not scheme objects or alternate executors. Every
factory takes one Case-authenticated state instance plus its complete typed numerical choices:

```python
U = tracer[model_state]

explicit = pops.lib.time.SSPRK2(U, rate=transport_rate, fields=field_solve)
custom = pops.lib.time.RungeKutta(U, rate=transport_rate, tableau=heun)
imex = pops.lib.time.IMEX(
    U,
    explicit_operator=transport_rate,
    implicit_operator=stiff_local_map,
    fields_operator=field_solve,
    tableau=my_additive_runge_kutta_tableau,
)
nonlinear_imex = pops.lib.time.IMEX(
    U,
    explicit_operator=transport_rate,
    implicit_operator=stiff_local_source,
    tableau=pops.lib.time.IMEX_ARS222_TABLEAU,
    implicit_solver=pops.solvers.LocalNewton(),
)
split = pops.lib.time.Strang(U, first=transport_flow, second=collision_flow)
```

The public catalog is `ForwardEuler`, `SSPRK2`, `SSPRK3`, `RK4`, `RungeKutta`, `IMEX`,
`AdamsBashforth`, `BDF`, `PredictorCorrector`, `Lie`, and `Strang`. Every call returns an ordinary
`pops.Program`. Factories expand only through public Program operations:
`T.state(U)`, callable operator handles, `T.value(...)`, generic solves and `T.commit(...)`. The
equivalent manual operations normalize to the same `ProgramGraph` and semantic identity. Order and
SSP evidence are reconstructed from that graph; factory names never select a runtime route.

The state must be the live instance handle produced by `block[state]`. Rate and local operators must
be typed handles returned by model declarations. A field solve must instead use the sole Case-owned
handle returned by `Case.field(...)`; a model field-provider handle is only a physical RHS
contribution and is rejected as a solve selector. There is no `(block, state)` overload, free-string
operator, `bind_operators`, `linear_combine`, boolean flux selector, legacy IMEX alias or
preset-specific native stepper. A custom explicit or additive method is configured by an exact `RungeKuttaTableau` or
`AdditiveRungeKuttaTableau`; unsupported or incomplete selections fail while the Program is being
authored. `Lie` and `Strang` take two Program-IR builder callables. Each callable receives its exact
fraction and endpoint, and the intermediate `StagePoint` retains separate logical coordinates for
both subflows.

`IMEX` accepts either a field-independent `local_linear_operator`, solved by the existing DenseLU
stage, or a state-only nonlinear `local_source`. The nonlinear form requires an explicit typed
`implicit_solver`; it lowers every diagonal stage to
`Program.solve(LocalResidual(...), solver=LocalNewton(...))` and consumes its `SolveOutcome` through
the selected failure action. `IMEX_ARS222_TABLEAU` stores the standard ARS(2,2,2) coefficients as
fixed IEEE-754 binary64 authoring values, so semantic identity never depends on a platform `sqrt`
call. A nonlinear source that reads a solved field remains a manually authored residual Program:
the cell-local Newton kernel cannot run a hierarchy field solve while perturbing one cell.

A partial implicit update is expressed by the source operator itself, for example
`source_term("momentum_drag", [0 * rho, -k * mx, -k * my])`. Components with a zero source remain
equal to the stage predictor. There is deliberately no `implicit_vars=["mx", "my"]` or
`implicit_roles=...` selector: bare component strings are ambiguous across typed StateSpaces and
would let a block policy choose hidden temporal semantics outside the Program.

There is no physics-specific condensed-solve time preset. A coupled implicit method is authored from
the generic field, residual, local-linear, nonlinear-solve, and failure-action operations of
`Program`, so every field read carries an exact `FieldContext` and no shared auxiliary state is
selected by a preset name.
