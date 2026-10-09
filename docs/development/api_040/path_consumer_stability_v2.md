# Path residual and explicit consumer stability, version 2

The native SymbolicPath route retains the complete conservative flux, signed
left/right nonconservative contributions and speed bound at each face. The
frequency is computed from those actual faces, including corrected coarse/fine
faces on a synchronous AMR hierarchy. No model name selects this implementation.

The previous implementation required an active `step_cfl` proposal and charged
every evaluated residual the entire macro step. The full M17 reception on native
artifact `220b48d3…` compiled its generic 15-component model but failed before
accepting any step: `FixedDt` did not supply that proposal. An unconsumed
diagnostic residual and a consumer using `dt/100` exposed the same incorrect
coupling between discretization and time method.

Version 2 exports the finite, nonnegative incident-face frequency from residual
evaluation. For each explicit consumer, the generated code checks
`dt * sum(beta_j * frequency_j) <= alpha * courant`, with the retained exact
input state, stage and rational coefficients. Fixed and external time grids use
the unit incident-face budget; an adaptive CFL invocation uses its authored
Courant. Domain and finite-value checks still execute during residual evaluation.
Diagnostic residuals have no explicit time-step budget. Reusing a residual does
not exempt any of its actual explicit consumers from checking its own weights.

Acceptance guards preserve the guarded value's stability obligations. Their
Boolean diagnostics do not create additional explicit updates. Rational
substitution of retained affine SSA stage identities recognizes Butcher-form
SSP updates, including `U0+dt/2*(R0+R1)` with `U1=U0+dt*R0`. This only constructs
a sufficient convex certificate: it does not rearrange the generated floating
point calculation, assume that every RK tableau is SSP, or infer a model's
admissibility from its wave-speed estimate. Unsupported negative/non-affine
certificates still fail explicitly.

Uniform stage authentication uses `pops.uniform-path-rhs-stage.v2` and includes
the immediate/deferred mode. The old exported C++ `block_path_rhs_into_at`
signature remains available with its immediate full-step check; an added
overload publishes the frequency only after successful residual publication.
The internal closure now returns its actual frequency and requires a native
rebuild. The SDK/header signature authenticates this change independently of
the production native loader ABI number.

AMR stage authentication uses `pops.amr-path-rhs-stage.v2`. Each stage creates a
fresh producer-owned scalar initialized to NaN, returning an immutable shared
view retained by the continuation. Only the successful hierarchy publication
writes its value. Rejected barriers cannot expose a previous attempt's valid
frequency. The existing flux tuple, ledger, reflux and immutable trace
authorities remain the sources of the spatial result. This extension does not
qualify asynchronous parent windows or general mapped/embedded-boundary paths.

`test_api040_path_consumer_stability.py` is an independent public-authoring and
emitted-scalar check, including two real host C++ executions; it is not a native
mesh reception. `test_symbolic_path_uniform_protocol.py` exercises the installed
native fixed/adaptive route and rollback. The preserved initial source suite
also exposed the Butcher-form and acceptance-guard gaps; the original failure
logs remain under `outputs/path-consumer-source.*`. Native MPI/backend claims
must come from the later installed receipts, not these source checks.
