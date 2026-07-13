# HyQMOM15 final contract

ADC-694 represents HyQMOM15 as an ordinary `pops.physics.Model`. Its fifteen raw moments,
transport flux, explicit electric source and implicit magnetic rotation are ordinary typed
declarations. A `Case` qualifies those declarations; an ordinary `Program` calls them. No
registry, compiler or runtime branch selects a model by the text `hyqmom15`.

The closure extension point is `LocalClosure(order, name, evaluator)`. The evaluator runs once
on symbolic standardized moments during authoring, must return exactly the order `N + 1` keys,
and is then absent from the runtime. `@closure(N)` creates the same object for user physics.
`moment_flux_expressions` consumes only the tiny `primitive(name, expression)` authoring
protocol, so both provided and user models share one binomial transform and one validation
boundary.

Realizability has two distinct roles. `RealizabilityProjection` selects the authored smooth
floors used by local algebra. `RealizableSet(4)` describes the acceptance guard. The Program's
transaction plan stages state, fields, flux ledgers, histories, schedules and consumers; a guard
rejection publishes none of them. Scientific diagnostics and checkpoints live in the existing
accepted-side-effect `ConsumerGraph`, outside the scientific operator graph.

## Native execution contract

Local dense solves are specialized from the resolved model manifest. HyQMOM15 therefore emits
`mat_inverse<15>` with exact 15 by 15 storage; there is no eight-component dispatch, truncation or
model-family branch. The final example executes `validate -> resolve -> compile -> bind -> run`,
checks the finite 15-component state, then authenticates a bit-identical checkpoint/restart through
a freshly rebound instance of the same artifact.
