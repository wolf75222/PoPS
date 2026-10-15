# API 0.2: twelve method contracts and their evidence

This is a source audit of the completion work following SDK010. It is not a
native test receipt. The [reception record](api_v02_reception.json) identifies
the source, loaded artifact, parameters, execution lane and result of each
actual run. A historical success does not qualify subsequently edited sources.

The selected builders come from
[profile_contracts.py](../../examples/api_v02/profile_contracts.py), through the
examples-only dispatcher. `require_complete=True` checks the reviewed component
manifest and its exact source hashes. It does not execute C++, verify a method's
order conditions, or infer that a companion's receiver has passed.

| Method | Selected Python construction | Discriminating numerical receiver |
|---|---|---|
| T01 | `build_t01`: SSPRK2 with a Poisson solve at both stages and again on the final candidate. | [Stage fields](../../tests/python/integration/runtime/test_api_v02_methods_T01_T12.py): independent discrete Poisson residuals on retained inputs, final versus last-stage distinction, temporal refinement on a fixed spatial operator. |
| T02 | `build_t02_imex_second_order`: ARS(2,2,2) with distinct explicit and implicit rates and a global implicit spatial solve. | [Coupled methods](../../tests/python/integration/runtime/test_api_v02_coupled_methods_native.py): complete spatial state versus an independent discrete inverse and second-order ratios. |
| T03 | `build_t03_dirichlet`: nonautonomous extended-system Strang with explicit time-dependent boundary lifting, reaction and diffusion subflows. | [Boundary splitting](../../tests/python/integration/runtime/test_api_v02_splitting_boundary_native.py): physical clock, two different subflow clocks, actual boundary values, an incorrect unlifted equation, and temporal order. |
| T04 | `build_t04_nonsymmetric`: global coupled relation, authored nonsymmetric Krylov algorithm, Schur action and back substitution. | [Coupled methods](../../tests/python/integration/runtime/test_api_v02_coupled_methods_native.py): full reconstructed solution and original block residual; the old symmetric CG profile is not substituted. |
| T05 | `build_t05`: outer iteration containing a Newton entropy closure; tangent obtained from the selected relation's Hessian inverse. | [Methods](../../tests/python/integration/runtime/test_api_v02_methods_T01_T12.py): moment residual and an independent NumPy Hessian solve, in addition to the outer solution. |
| T06 | `build_t06_selected_global`: separate old and unknown arguments, spatial nonlinear relation, explicit admissible root selection. | [Coupled methods](../../tests/python/integration/runtime/test_api_v02_coupled_methods_native.py): frozen versus unknown argument and both root branches, compared as full states. |
| T07 | `build_t07`: BE startup followed by variable-step BDF2; retained state, full increment, previous interval and finite-solve defect enter the actual formula. | [Methods](../../tests/python/integration/runtime/test_api_v02_methods_T01_T12.py): nonuniform scalar recurrence, nonzero defect delivered once, strict restart and authenticated previous interval. [Spatial BDF2](../../tests/python/integration/runtime/test_api_v02_spatial_bdf2_native.py) separately tests a global implicit operator, wrong-lag counterexample, partial-AMR conservation and restart. |
| T08 | `build_t08`: partial AMR, recursive 2:1 windows, owned faces, reflux, average-down and rejection after child work. | [Methods](../../tests/python/integration/runtime/test_api_v02_methods_T01_T12.py): composite transport and late rollback. [History transfer](../../tests/python/integration/runtime/test_api_v02_history_transfer_native.py): authored nonuniform reconstruction, BDF3 consuming all retained lags, distinct flux-source identities, subsequent transport and restart. |
| T09 | `build_interval_transport`: characteristic reconstruction and Duhamel integral on the actual requested interval, with face and collision deliveries from one realization. | [Interval transport](../../tests/python/integration/runtime/test_api_v02_interval_transport_native.py) and [nonlinear companion](../../tests/python/integration/runtime/test_api_v02_nonlinear_interval_transport_native.py): full-state quadrature, balance, changed-request retry, and deliberate wrong interval/dependency witnesses. |
| T10 | `build_t10_composite`: one authored relaxation factor for the whole synchronized macro increment, reached time and exchange measure. | [Composite relaxation](../../tests/python/integration/runtime/test_api_v02_relaxed_composite_native.py), [history](../../tests/python/integration/runtime/test_api_v02_relaxed_history_native.py), [multiblock](../../tests/python/integration/runtime/test_api_v02_relaxed_multiblock_native.py), [cut cells](../../tests/python/integration/runtime/test_api_v02_relaxed_cutcell_native.py), and [authored stability](../../tests/python/integration/runtime/test_api_v02_authored_stability_native.py) are separate receipts. A scalar energy check alone does not qualify all of them. |
| T11 | `build_t11_network`: an explicit directed process graph and second-order modified Patankar stage equations, including shared transfers and empty inventories. | [Coupled methods](../../tests/python/integration/runtime/test_api_v02_coupled_methods_native.py): independent linear-network evolution, nonnegativity, inventory conservation, order, and equal-net-ODE graphs with different circulation. |
| T12 | `build_t12_amr`: high/low face candidates, a Python admissibility predicate, one replacement per shared face and recomputed incident deliveries. | [AMR faces](../../tests/python/integration/runtime/test_api_v02_stencil_amr_native.py), [selection at the interface](../../tests/python/integration/runtime/test_api_v02_stencil_selection_native.py), [dependent field](../../tests/python/integration/runtime/test_api_v02_corrected_field_native.py) and [Euler companion](../../tests/python/integration/runtime/test_api_v02_euler_correction_native.py). The interface receiver must distinguish selected/unselected updates on uncovered coarse cells, not merely check mass. |

Fixed quadrature in T05, a chosen nonlinear root in T06, frozen coefficients in
the T09 witness, and a particular admissibility predicate in T12 are declared
scientific choices. They do not imply that these choices are engine primitives.
The methods' equations, stages, residuals and choices are authored in Python.
Independent NumPy oracles run after native execution and do not provide its
trajectory. None of the selected method identifiers is dispatched inside the
Program code generator or native Program/AMR context.

The T10 cut-cell witness conserves its authored quadratic energy in the **fixed
discrete measure** exposed by the native geometry. It checks that fractions and
integration weights are bitwise unchanged during the attempt and that omitting
the fractions changes the relaxation factor. It does not certify exact continuum
volume or conservation when a regrid reconstructs a different discrete measure.
In particular, the existing Dim1/2 `CutCell` geometry uses products of
centre-to-neighbour ray lengths. For the plane `x = 1/2` on eight cells, the
adjacent fluid cell has fraction `(1 + 1/2)/2 = 3/4`, so the initial measure is
`1/2 - (1/8)/4 = 0.46875`. Refining the central half in both axes yields
`0.4765625` in Dim2. The storage receiver checks these independently derived
measures at unchanged tolerance; the compatible staircase case separately
requires unchanged mass and forbids invented transport flux. This is a property
of that selected geometric discretization, not a restriction of Python algebra
or an engine branch for a physical model.

## Compositions requiring their own receipt

The historical SDK007 flux-history receiver executed through time .32 after
reconstruction and restart, preserving distinct sources and exact rational
weights. Its successful receipt is evidence for those exact bytes; it is not a
PASS for the newer continuation-forest, field-port or target-field-allocation
implementations. SDK010 and the subsequent source changes require their own
native reception.

The current completion work includes temporal field reconstruction, retained
fields inside a complete continuation, a parent assessment over all children,
and explicit Field/Bundle representation ports during regrid. Their receivers
exercise geometry, auxiliary components and ghosts, field histories, and
publication/rollback. Allocating target storage does not interpolate data:
the transfer equations and target initialization remain explicit Python.

Two additional restrictions were found by the source audit and are tracked as
API composition work, not as scientific assumptions:

* A StencilFlux inside a Uniform branch or bounded iteration previously failed
  because accepted receipts referred to local C++ variables. The new lowering
  captures immutable native exchange expressions and publishes the expression
  attached to the executed candidate. Its [scoped receiver](../../tests/python/integration/runtime/test_api_v02_scoped_stencil_native.py)
  checks distinct evaluations of one loop node, branch laziness, full face
  integrals, restart, rollback and refusal of an unproved nonlinear rewrite.
  Source emission alone does not establish this native result.
* The original nonlinear AMR field route still needs to compose multiple
  solve/publication barriers and control regions with the invocation scheduler.
  The existing one-barrier [original-field receiver](../../tests/python/integration/runtime/test_public_amr_original_field.py)
  cannot establish this stronger contract. Removing the guard without retaining
  exact inputs, synchronization and lifetimes would be incorrect.

Final qualification must match the final reviewed source and native artifacts,
include the listed discriminating receivers on their requested MPI/backend
configurations, and retain the actual GitHub required-check results. Successful
source construction, a time-only frontier, or an earlier native revision is
insufficient for that claim.
