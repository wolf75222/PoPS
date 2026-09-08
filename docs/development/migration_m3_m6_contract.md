# PoPS M3-M6 execution contract and crosswalk

This document is the supporting contract for the M3-M6 leaf packages. It carries the full
acceptance boundary from the pinned migration manifest into an execution index; it does not report
an M3-M6 implementation or qualification result. Every leaf and every evidence level is `pending`
until an agent supplies an exact source, configuration, command, artifact, output, and oracle.

M0-M2 is already documented in
[`migration_m0_m2_contract.md`](migration_m0_m2_contract.md) and
[`migration_m0_m2_results.md`](migration_m0_m2_results.md). This document does not copy that
evidence. It uses the final M0-M2 production revision as the starting provenance for this work:

- Prior integrated baseline: `80fac141`.
- Final M0-M2 production source: `bd583faf196f3c1faeedec489e04d24e959ed00b`.
- M0 reference source: `51bcdc2e2399dbc922c58c3008ae30a11332f1d8`.
- Pinned manifest: `/Users/romaindespoulain/Documents/Codex/2026-09-07/infer-the-intended-scope-from-the/notes/migration-manifest.json`.
- Manifest SHA-256: `b64efb3c33266c769aa58a7ef44ad7dc873867e7dac18792972a5df00f4c676d`.
- Supplied specification SHA-256: `4b50b6dd75201c2175dcc8f987acb42d3c04898c6bca55fd12cee84324976e40`.
- Mathematical reference SHA-256: `dacaa9eaa8bdd55abd7366756601e91a4f8e7dfdc425d3b4495107ac3e613dfd`.
- Companion machine-readable crosswalk: `/Users/romaindespoulain/dev/tmp/PoPS-migration-20260907-evidence/contract-crosswalk.json`.

The existing Linear parent issues are [ADC-903](https://linear.app/romain7522/issue/ADC-903) for
M3, [ADC-904](https://linear.app/romain7522/issue/ADC-904) for M4,
[ADC-905](https://linear.app/romain7522/issue/ADC-905) for M5, and
[ADC-906](https://linear.app/romain7522/issue/ADC-906) for M6. The names M3.1-M3.4, M4.1-M4.4,
M5.1-M5.5, and M6.1-M6.2 are execution-index packages only. They are not claims that child
Linear issues exist.

## Contract boundary

The public authoring lifecycle remains equation-oriented:

```text
validate -> resolve -> compile -> bind -> run
```

M3-M6 preserve `Model`, `Case`, `DiscretizationPlan`, and `Program` as the public decision
boundary. A `FieldProblem` owns unknown/equation tuples, dependencies, physical boundaries,
compatibility and normalization. A solve request owns solver coordinates, seeds, derivative
strategy, requested outputs, residual interpretation, and failure disposition. A field equation,
discrete operator, solved value, and observation remain separate records.

There is one equation authority and one solve authority. Resolved operations, access plans,
lowering reports, native signatures, and execution plans are derived records. They cannot replace
the authored model or introduce an invisible solve, use-latest fallback, arbitrary owner, silent
mean subtraction, duplicate exchange, or inferred inverse closure. A shared native application may
reuse source work, but mathematical multiplicity and owner identity remain distinct.

Source, representation, and registered tests answer different questions from native execution,
numerical correctness, and performance. No source witness, type, conformance case, or placeholder
matrix closes an execution criterion. Unsupported and unavailable cells must retain their phase,
reason, and exact disposition.

## Seven evidence levels

All 15 leaf packages start with all seven levels `pending`. The external machine-readable
crosswalk repeats these placeholders for every leaf so agents can fill them without changing the
acceptance contract.

| Level | Required record before a pass can be claimed |
| --- | --- |
| `representable` | Typed authoring or IR witness with stable identity or serialization, including the relevant tuple, support, dependency, effect, boundary, or transition records. |
| `validated` | Exact validation command, exit code, and accepted diagnostic or plan. |
| `resolved` | Resolved report with source mapping, dependencies, effects, coverage, disposition, and any required compatibility or support map. |
| `emitted` | Generated/native source or typed operation record plus authenticated artifact and build provenance. |
| `executed` | Exact runtime command, configuration, exit code, and retained output or artifact hashes. |
| `numerically_checked` | Declared equation/oracle, norms, tolerance, comparison method, refinement or restart policy, and result. |
| `performance_characterized` | Equivalent numerical conditions, warmups/repetitions, device and MPI completion, raw metrics, MAD or other uncertainty summary, and scope limits. |

The status vocabulary is `pending`, `pass`, `fail`, `unsupported`, or `unavailable`. A structured
refusal is evidence for an intentional unsupported route; it is not a successful execution. An
unavailable dimension, backend, rank, layout, or device is not silently substituted by a different
cell.

## Bounded matrix declaration

No M3-M6 dimension, backend, MPI-rank, layout, level, block, method, restart, output, or tolerance
cell is declared by this document. Each agent must fill the matrix from the executed package and
its supplied receipt before claiming a numerical or performance result.

| Axis | Required agent-supplied declaration |
| --- | --- |
| Dimension | Exact native and scientific dimensions; unavailable dimensions and reason. |
| Backend | Serial, OpenMP/Kokkos, MPI, GPU, or other selected backend with toolchain. |
| MPI ranks | Exact ranks and communicator, including rank-parity or collective requirements. |
| Layout | Uniform, AMR, multiblock, or mapped support with storage and transfer policy. |
| Levels and blocks | Exact hierarchy, refinement ratios, active blocks, ghost/halo requirements, and topology state. |
| Method | Equation, discretization, temporal method, coefficients, restrictions, solver strategy, and derivative route. |
| Restart and output | Fresh/checkpoint/regrid state, histories, dense output, HDF5/ParaView/diagnostic outputs, and reopen policy. |
| Tolerances | Norms, residual/error limits, conservation or inventory tolerances, reproducibility mode, and performance uncertainty. |

## Repaired execution order

The manifest's dependency graph is repaired so the shared solve and transaction foundations arrive
before field observation and later implicit routes:

```text
M3.1 -> M6.1 -> M3.2 -> M6.2 -> M3.3 -> M3.4
                                      |
                                      +-> M4.1 -> M4.2 -> M4.3 -> M4.4
                                      |
                                      +-> M5.1 -> M5.2
                                      |          |
                                      |          +-> M5.3 -> M5.4 -> M5.5
                                      |
                                      +-> M5.3
```

The diagram shows the principal order; each leaf's exact manifest dependencies remain authoritative
below. M4.1 and M5.1 may proceed in parallel once their M1/M2 prerequisites are closed. M6.3
depends on M6.2, M3.3, M4.3, and M5.4; M6.4 depends on M6.2 and M3.3. The M6 phase therefore
requires M6.3 and M6.4 after the two foundation leaves in this document.

## M3 - Unify field problems and solve results

### M3.1 - Define joint field problems, physical boundaries and independent storage

**Scope:**
- Generalize the existing field facade into one authoritative FieldProblem with unknown/equation tuples, typed dependencies, compatibility, branch and joint normalization data.
- Normalize physical boundary/interface relations separately from numerical enforcement, discretization and solver choices; retain existing convenient constructors as adapters.
- Give fields their own storage binding and quantity ownership instead of selecting an arbitrary species as conceptual owner.

**Acceptance (all criteria):**

1. Required implementation tests construct scalar and joint two-unknown problems without duplicating an equation or authoring provider tables.
2. Qualified field identities and storage ownership remain unambiguous when model instances reuse local names and when two states contribute to the load.
3. Physical boundary relations survive validation as physical data; equivalent combined constructors normalize into the same separate physical/numerical records.
4. Joint compatibility/gauge/branch conditions belong to the whole problem; an individual field cannot silently add an independent gauge.
5. Invalid unknown/equation/type or boundary ownership combinations fail with structured diagnostics; the new representation alone is not reported as an executed numerical route.

**Dependencies:** `M1.1`, `M1.2`, `M2.1`.

**Related issue references:** ADC-669, ADC-705, ADC-727.

**Source and code:** Specification sections 4.1, 6.2, 7.1, 10.3, 13/M3; PDF sections 4.2.1-4.2.3 and 5.1-5.2. Paths: `python/pops/physics/_board_elliptic.py`; `python/pops/fields/operator.py`; `python/pops/fields/discretization.py`; `tests/python/unit/fields/test_field_operator_contract.py`; `tests/python/unit/fields/test_field_discretization_contract.py`.

**Evidence status:** all seven levels `pending`; matrix declaration `pending`.

### M3.2 - Execute variable-coefficient scalar fields through the general resolved contract

**Scope:**
- Emit scalar field problems directly from resolved operations and derive native inputs from all typed equation leaves, including multiple qualified states and coefficients.
- Preserve the existing Poisson and screened-Poisson fast paths as numerical realizations of the same contract.
- Add a bounded, explicitly documented variable-coefficient scalar route with its supported dimension/backend/MPI/layout/boundary configuration.

**Acceptance (all criteria):**

1. Required lifecycle tests validate, resolve, compile, bind and execute a scalar load assembled directly from two qualified states without manual per-species provider composition.
2. A nonconstant-coefficient manufactured problem converges on the complete declared qualification matrix; unexecuted or unavailable configurations remain explicitly unqualified.
3. Existing Poisson and screened-Poisson cases retain their numerical method and pass replacement-equivalence checks under the selected reproducibility mode.
4. The general route does not force every equation into operator='poisson'; unsupported numerical forms receive an explicit lowering disposition.
5. Coefficient/boundary data and mandatory compatibility or selection conditions participate in the problem and access plan; no mean subtraction silently changes an incompatible physical load.

**Dependencies:** `M3.1`, `M2.2`, `M2.3`, `M6.1`.

**Related issue references:** ADC-772, ADC-670, ADC-740.

**Source and code:** Specification sections 2.2, 6.2, 8.1-8.3, 9.2, 13/M3; PDF sections 3.2.7, 4.2.3, 5.2 and 12.4.4. Paths: `python/pops/physics/_board_elliptic.py`; `python/pops/fields/operator.py`; `python/pops/codegen/module_lowering.py`; `tests/python/unit/codegen/test_module_lowering.py`; `tests/python/unit/codegen/test_module_lowering_coverage.py`.

**Evidence status:** all seven levels `pending`; matrix declaration `pending`.

### M3.3 - Bind stage field observations and validate precise field reuse

**Scope:**
- Bind explicit Program field solves to inferred qualified stage dependencies, using value tuples when identities disambiguate roles and explicit mappings only when needed.
- Expose observations of consumed solve results to both source and transport-flux consumers with their correct cell/face/reconstruction sampling.
- Separate field-problem, discrete-operator, solved-value and observation identities, and implement accuracy-aware provenance/reuse policies.

**Acceptance (all criteria):**

1. Required native lifecycle cases execute field-coupled transport and an Euler-Poisson source use with visible stage solves and explicit SolveOutcome.consume failure disposition.
2. An omitted dependency or ambiguous formal role fails explicitly; no hidden solve or 'use latest field' fallback fills missing data.
3. Two stages at the same physical timestamp with different state versions remain distinct, and a field result from a rejected/provisional scope cannot be read from an incompatible scope.
4. A momentum-only change can reuse a density-dependent potential when every actual dependency and accuracy/use condition still agrees; changing a coefficient, boundary, geometry, gauge or relevant parameter invalidates the appropriate object.
5. Changing only an observation's differentiation/reconstruction rule invalidates that observation without falsely identifying it with the potential; face and cell samples are not interchangeable.
6. Conformance instrumentation records actual solve counts and rejected work; reuse assertions are backed by executed tests rather than timestamp or cache-key inspection alone.

**Dependencies:** `M3.2`, `M1.4`, `M2.2`, `M6.1`, `M6.2`.

**Related issue references:** ADC-669, ADC-759.

**Source and code:** Specification sections 6.3, 8.2-8.3, 9.2-9.3, 9.5, 12, 14; PDF sections 6.1.3-6.1.4 and 7.2. Paths: `python/pops/fields/operator.py`; `python/pops/time/_program/solve.py`; `python/pops/time/solve_outcome.py`; `tests/python/unit/time/test_field_solve_outcome.py`; `tests/python/examples/final/test_multiphysics_core_example.py`; `examples/final/EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py`.

**Evidence status:** all seven levels `pending`; matrix declaration `pending`.

### M3.4 - Execute a joint multi-field problem with one shared normalization

**Scope:**
- Provide one bounded native realization of a coupled problem with multiple field unknowns and a shared kernel, compatibility condition and normalization.
- Use the same general solve/result contract for joint solution tuples, observations and failure disposition; preserve a single equation authority across block/nested solver choices.
- Qualify the two-field lambda-coupled Neumann example from the mathematical note as an adversarial joint-gauge case.

**Acceptance (all criteria):**

1. Required lifecycle tests execute -Delta(phi1)+lambda(phi1-phi2)=f1 and -Delta(phi2)+lambda(phi2-phi1)=f2 for lambda>0 on the explicitly supported configuration matrix.
2. For constant f1=c0 and f2=-c0, the solution admits individual means c0/(2lambda) and -c0/(2lambda) under one joint zero-mean normalization; two independent zero-mean constraints are rejected as an inappropriate added restriction.
3. The compatible load condition uses the integral of f1+f2 and the shared constant kernel; incompatible total load fails coherently instead of being repaired by arbitrary normalization.
4. Every returned field and observation is read only after explicit outcome consumption, and the tuple retains the same solve application and provisional scope.
5. Numerical residual/solution-error evidence and supported backend/layout limits are recorded separately from early joint-field representability; completion requires the executed route.

**Dependencies:** `M3.1`, `M3.2`, `M6.1`.

**Related issue references:** ADC-670.

**Source and code:** Specification sections 6.2, 7.1, 9.2-9.3, 12, 13/M3; PDF section 4.2.3, p.11. Paths: `python/pops/fields/operator.py`; `python/pops/fields/discretization.py`; `python/pops/codegen/module_lowering.py`; `python/pops/time/_program/solve.py`; `tests/python/unit/fields/test_field_operator_contract.py`; `tests/python/unit/time/test_field_solve_outcome.py`.

**Evidence status:** all seven levels `pending`; matrix declaration `pending`.

## M4 - Generalize interactions and typed native calls

### M4.1 - Execute heterogeneous interactions and joint constitutive outputs

**Scope:**
- Generalize interaction and constitutive-law applications to heterogeneous input/output tuples with dependency signatures derived from the union of all output expressions.
- Lower separate output projections into one joint application per stage, iterate, spatial location and numerical sampling context.
- Retain specialized existing collision implementations as selectable realizations of the general interaction contract.

**Acceptance (all criteria):**

1. Required native tests execute an interaction between unequal state sizes and deliver correctly shaped outputs to their own qualified balance targets.
2. Instrumented separate projections share one joint evaluation for identical evaluation context; different stages, iterates or sampling contexts remain distinct evaluations.
3. Using a projected contribution twice preserves twice its mathematical contribution even when the joint native call is shared.
4. A joint constitutive flux can return heterogeneous species/energy outputs with declared collective constraints; outputs are not implemented by assuming opposite equal-length arrays. Include y1+y2=1 with unequal independently chosen D1/D2 as an adversarial collective-law case: conservation of each scalar must not falsely certify preservation of the collective constraint.
5. Existing specialized collision cases retain an explicit numerical realization and pass the appropriate regression/equivalence checks without becoming the only supported vocabulary.

**Dependencies:** `M1.3`, `M2.1`, `M2.2`, `M2.3`.

**Related issue references:** ADC-690.

**Source and code:** Specification sections 6.4, 7.1, 10.1, 11.1, 13/M4; PDF sections 4.1.2-4.1.3 and 8.2.2. Paths: `python/pops/codegen/module_lowering.py`; `include/pops/core/model/physical_model.hpp`; `examples/final/EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py`; `tests/python/examples/final/test_multiphysics_core_example.py`.

**Evidence status:** all seven levels `pending`; matrix declaration `pending`.

### M4.2 - Add authenticated typed native calls without core model kinds

**Scope:**
- Extend authenticated native-component inputs with a typed native-call expression and explicit signature, read footprint, output structure, execution domain, effects and build inputs.
- Expose a symbolic model-library wrapper that produces immutable native calls during Python authoring and direct specialized C++/Kokkos calls during execution.
- Define toolchain/interface compatibility and host/device target requirements without claiming a universal stable C++ ABI.

**Acceptance (all criteria):**

1. A required imported constitutive or closure primitive validates, resolves, compiles, binds and executes in a case without adding a model-specific core operator kind or editing PoPS core for that imported operator.
2. Generated cell/face computation uses statically selected native call targets and typed output projections, with no runtime Python callback, string lookup or physical-type dispatch.
3. Build provenance authenticates the component tree and declared build inputs; incompatible toolchain/interface or unsupported execution-target combinations fail explicitly.
4. A host-only shared-library function is never reported device callable merely because it loads on the host; every advertised device route is compiled/linked for that target and executed in its qualification matrix.
5. Opaque input reads are conservatively complete unless a valid component read footprint is declared; capture distinguishes numerical constants from runtime parameter references.

**Dependencies:** `M1.3`, `M2.2`, `M2.3`.

**Related issue references:** ADC-681, ADC-719, ADC-694.

**Source and code:** Specification sections 4.2, 6.5, 8.2, 11.2-11.3, 13/M4; PDF section 12.4.3. Paths: `python/pops/native_components.py`; `python/pops/codegen/module_lowering.py`; `tests/python/integration/native_loader/test_external_component_package.py`; `include/pops/core/model/physical_model.hpp`.

**Evidence status:** all seven levels `pending`; matrix declaration `pending`.

### M4.3 - Preserve native failure domains and derivative contracts in solves

**Scope:**
- Carry native input-domain and failure-status contracts into generated evaluations and the early general solve protocol.
- Provide explicit exact derivative, approximate Jacobian and selected finite-difference routes where a nonlinear solver requires them.
- Expose local evaluation status at the planned synchronization boundary before ranks branch into collective solves or reductions.

**Acceptance (all criteria):**

1. Required tests inject a native domain failure at one cell and verify a structured category/status reaches the solve/attempt failure disposition without producing readable invalid outputs.
2. On each advertised MPI route, one-rank failure is aggregated before dependent collective control flow; the test terminates coherently and does not leave other ranks waiting in a solve or reduction.
3. A nonlinear solve requiring derivatives refuses an unsupported route; exact, approximate and explicitly selected finite differences remain distinct in the resolved inspection output.
4. Differentiation never silently follows a lagged residual or unsupported limiter; guarded invalid operands are not eagerly evaluated or hoisted outside their guard.
5. Completion records executed configurations and failure-injection outcomes; arbitrary process loss or external-library faults remain outside ordinary transactional rollback claims.

**Dependencies:** `M4.2`, `M2.2`, `M2.4`, `M6.1`, `M6.2`.

**Related issue references:** ADC-757, ADC-750.

**Source and code:** Specification sections 4.2, 8.2, 9.2-9.3, 10.4, 11.2; PDF sections 6.2.4, 7 and 8.1. Paths: `python/pops/native_components.py`; `python/pops/codegen/module_lowering.py`; `python/pops/time/_program/solve.py`; `python/pops/time/solve_outcome.py`; `tests/python/integration/native_loader/test_external_component_package.py`.

**Evidence status:** all seven levels `pending`; matrix declaration `pending`.

### M4.4 - Qualify joint interaction inventories and native-call costs

**Scope:**
- Define physical inventory/moment maps for heterogeneous interaction outputs and verify exchanges in the actual accepted update with its temporal quadrature.
- Add bounded conformance cases for joint source/constitutive projections and an imported native primitive alongside the complete declared numerical qualification matrix.
- Measure joint-evaluation counts, rejected work, materialization/fusion effects and native-call performance under equivalent numerical conditions.

**Acceptance (all criteria):**

1. Required numerical tests verify the declared common physical inventory maps for unequal states; they do not infer conservation from opposite arrays or mismatched velocity increments.
2. A momentum-exchange case includes the stated thermal/energy bookkeeping, and compatible accepted quadrature preserves the declared inventory within its specified tolerance.
3. Projections of one application use the same input context; differing recipient quadrature weights are rejected or exposed as an explicit nonconservative method rather than certified by the local law alone.
4. Measured evaluation counts distinguish shared applications from repeated mathematical use; opaque functions are not assumed to eliminate unused outputs without a specialized entry point or measured/compiler evidence.
5. Performance comparison uses the same equations, discretization, timestep policy, tolerances and output frequency, with device completion included; unexecuted numerical or backend cells remain explicitly unqualified.

**Dependencies:** `M4.1`, `M4.2`, `M4.3`, `M1.4`, `M6.1`, `M6.2`.

**Related issue references:** ADC-690, ADC-769.

**Source and code:** Specification sections 6.4, 10.1, 11.4-11.5, 12, 13/M4, 14; PDF sections 4.1.2-4.1.4 and 8.2.2. Paths: `examples/final/EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py`; `tests/python/examples/final/test_multiphysics_core_example.py`; `tests/python/integration/native_loader/test_external_component_package.py`; `benchmarks/manifest.toml`.

**Evidence status:** all seven levels `pending`; matrix declaration `pending`.

## M5 - Generalize diffusion and accepted exchanges

### M5.1 - Execute scalar diffusion with explicit gradient and boundary semantics

**Scope:**
- Add constitutive diffusive-flux authoring with explicit gradient variables and signed inclusion in the selected balance.
- Resolve and emit a bounded scalar gradient/constitutive/face/divergence realization with declared sampling and composed stencil/halo requirements.
- Normalize physical diffusive boundary relations separately from gradient, trace and flux enforcement while preserving one physical boundary law.

**Acceptance (all criteria):**

1. A required scalar diffusion case traverses validate, resolve, compile, bind and run from ddt(U)==div(Fd), using the same scientific declaration later consumed by explicit and implicit temporal routes.
2. The gradient variables and discrete realization are visible in the resolved plan; a diffusive law is not interpreted as an additional hyperbolic Riemann flux.
3. Manufactured-solution convergence and physical boundary balance are executed for the full declared scalar configuration matrix with predeclared norms, tolerances and refinements.
4. Adding diffusion to an inflow/outflow transport case requires a valid diffusion boundary closure; missing or independently overconstrained value/normal-flux data fail precisely.
5. Nonlinear gradient-variable changes retain the chosen discrete construction; no unproved discrete chain-rule substitution or silent sign loss is introduced.

**Dependencies:** `M1.2`, `M2.1`, `M2.2`, `M2.3`.

**Related issue references:** ADC-682, ADC-749.

**Source and code:** Specification sections 4.4, 6.1, 7.2, 8.4, 10.3, 13/M5; PDF sections 4.2.1 and 5.6. Paths: `python/pops/physics/_board_rate.py`; `python/pops/codegen/module_lowering.py`; `include/pops/core/model/physical_model.hpp`; `tests/python/unit/codegen/test_module_lowering.py`.

**Evidence status:** all seven levels `pending`; matrix declaration `pending`.

### M5.2 - Extend diffusion to variable coefficients and a bounded tensor route

**Scope:**
- Extend scalar diffusion first to nonconstant scalar coefficients, then a precisely selected tensor case rather than claiming arbitrary anisotropic support.
- Retain coefficient and gradient-variable dependencies, conormal boundary meaning and pairing/dissipation assumptions in the resolved construction.
- Publish the exact supported tensor structure, dimensions, backend, MPI, layout, boundary and temporal combinations.

**Acceptance (all criteria):**

1. Required manufactured solutions demonstrate the declared spatial convergence for nonconstant scalar coefficients and the selected tensor route on their complete qualification matrix.
2. Variable coefficients remain inside the intended divergence; tests distinguish a(x)div(F) from div(a(x)F) and reject any transformation that omits the additional term.
3. Boundary tests use the constitutive conormal flux, including the selected anisotropy, rather than replacing it by a raw normal derivative.
4. Coefficient/domain/shape changes produce correct dependency invalidation or structured unsupported-route diagnostics, never a silently scalarized tensor realization.
5. Conservation, admissibility and dissipation claims have separate checked/assumed scope; a positive-semidefinite tensor declaration alone is not reported as a proof for the discrete coupled update.

**Dependencies:** `M5.1`, `M2.2`, `M2.4`.

**Related issue references:** ADC-772.

**Source and code:** Specification sections 6.1, 7.4, 8.2-8.4, 13/M5; PDF sections 4.2.1, 5.6 and 10.3. Paths: `python/pops/physics/_board_rate.py`; `python/pops/codegen/module_lowering.py`; `include/pops/core/model/physical_model.hpp`; `tests/python/unit/codegen/test_module_lowering_coverage.py`.

**Evidence status:** all seven levels `pending`; matrix declaration `pending`.

### M5.3 - Enforce combined explicit restrictions and weighted diffusive exchanges

**Scope:**
- Compute timestep restrictions from the actual selected numerical update and active temporal partition, including the combined transport/diffusion condition.
- Produce oriented diffusive face-exchange records with measure, evaluation context and the actual accepted temporal quadrature.
- Derive macro-step restrictions from every AMR level and local subcycling ratio; expose the exchange contract required for subsequent coupled AMR reflux.

**Acceptance (all criteria):**

1. For the specified first-order 1D update, required tests enforce c+2r<=1 and exhibit rejection of a step that passes independent advection and diffusion bounds but violates their combined positivity condition.
2. Switching the implicit/explicit partition changes only the applicable restriction; accuracy or coupling conditions are not automatically removed by an implicit label.
3. AMR restriction tests use actual local h and substep duration, including a ratio-two hierarchy where two substeps do not alone establish explicit diffusive stability.
4. Executed conservative cases reconcile the accepted state change with signed transport and diffusion exchanges using their own temporal weights, orientation and measure exactly once.
5. Repeated stage evaluations are distinguished from accepted quadrature contributions; rejected/iterative work cannot be counted as extra physical exchange, and unsupported affine-weight propagation is diagnosed explicitly.

**Dependencies:** `M5.1`, `M1.4`, `M2.1`, `M2.2`, `M6.1`, `M6.2`.

**Related issue references:** ADC-677, ADC-756.

**Source and code:** Specification sections 9.4, 10.1-10.2, 13/M5; PDF sections 8.2.1-8.2.3 and 10.2-10.3. Paths: `python/pops/time/_program/authoring.py`; `python/pops/time/_program/commit_validation.py`; `python/pops/codegen/module_lowering.py`; `examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py`; `docs/tuto/scalar_advection/06_openmp_amr_explicit_ssprk2.py`.

**Evidence status:** all seven levels `pending`; matrix declaration `pending`.

### M5.4 - Execute spatial implicit diffusion while preserving accumulation

**Scope:**
- Use the general solve protocol for an evolved-state implicit diffusion stage with spatially coupled unknowns, frozen previous-stage data, a separate seed and explicit failure disposition.
- Retain the discrete accumulation equation Q_kappa(q+)-Un=tau R_kappa and its representation/quadrature; preserve current-iterate coefficient/gradient dependencies where advertised.
- Reuse suitable native elliptic algorithms as numerical realizations without redefining the temporal stage as a physical field or cell-local collision solve.
- Deliver the scalar implicit route first. Variable/tensor implicit variants require their M5.2 realization and are claimed only when separately executed; they do not block scalar R1.

**Acceptance (all criteria):**

1. Required lifecycle cases execute explicit and implicit diffusion from the same physical declaration and demonstrate the declared spatial/temporal manufactured-solution convergence.
2. The implicit solve couples the required neighboring degrees of freedom and is never routed through a cell-local collision solver without an explicitly different justified method.
3. A nonidentity accumulation case preserves Q_kappa(q+)-Un; the H(T)=T+T^2, Tn=0, tau P=1 conformance check returns the energy-consistent solution rather than the chain-rule value T+=1/2 with energy 3/4.
4. For I-tau nu Delta, the identity term determines the constant mode from the old state: a nonzero-mean right-hand side is valid and is not rejected by physical Poisson zero-mean compatibility or gauge rules.
5. Frozen data and seed are distinct; any advertised nonlinear route retains current-iterate dependence and an explicit derivative strategy, while lagging inside a preconditioner is distinguished from changing the residual.
6. Failure leaves unreadable or rejected stage results provisional and contributes no accepted exchanges; all advertised configurations have executed evidence, with unavailable cells explicitly unqualified.

**Dependencies:** `M5.1`, `M5.3`, `M1.2`, `M2.3`, `M6.1`, `M6.2`.

**Related issue references:** ADC-670, ADC-750.

**Source and code:** Specification sections 4.3, 9.1-9.3, 10.1, 12, 13/M5; PDF sections 3.2.3, 5.1 and 6.2.4, especially p.17 footnote 26. Paths: `python/pops/time/_program/solve.py`; `python/pops/time/solve_outcome.py`; `python/pops/codegen/module_lowering.py`; `python/pops/physics/_board_rate.py`; `tests/python/unit/time/test_solve_outcome_contract.py`; `tests/python/unit/time/test_field_solve_outcome.py`.

**Evidence status:** all seven levels `pending`; matrix declaration `pending`.

### M5.5 - Execute a joint fitted drift-diffusion construction with exact coverage

**Scope:**
- Implement one bounded joint drift-diffusion numerical construction that covers the signed transport and diffusion occurrences together while preserving the original physical decomposition.
- Use explicit field observations and face sampling, a stable Bernoulli-function evaluation at vanishing potential jump, and the construction's declared boundary/time policy.
- Reject requests for separately evaluated IMEX partitions when this realization exposes only the joint total flux.

**Acceptance (all criteria):**

1. A required native one-dimensional fitted-flux case executes from the selected physical balance and reports one joint coverage set containing each consumed signed occurrence exactly once.
2. The local Scharfetter-Gummel conformance test yields zero flux for nR=nL*exp(-(psiR-psiL)); zero potential jump yields the declared centered diffusive limit without division-by-zero or cancellation failure.
3. Adding an additional diffusion realization to the already covered joint flux is rejected as double counting; an uncovered physical term is rejected as incomplete coverage.
4. A separate IMEX transport/diffusion request either selects an explicitly decomposable alternative realization or fails precisely; the total fitted flux is not silently split.
5. Manufactured convergence, boundary balance and accepted face-time exchange accounting are executed for the declared numerical matrix; the local equilibrium identity alone is not reported as a universal positivity, convergence or entropy proof.

**Dependencies:** `M5.1`, `M5.3`, `M3.3`, `M2.1`, `M2.3`.

**Related issue references:** ADC-772.

**Source and code:** Specification sections 6.1, 7.2, 9.4, 10.1-10.2, 12, 13/M5; PDF sections 5.8 and 8.2.3. Paths: `python/pops/physics/_board_rate.py`; `python/pops/codegen/module_lowering.py`; `python/pops/fields/operator.py`; `tests/python/unit/codegen/test_module_lowering_coverage.py`; `examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py`.

**Evidence status:** all seven levels `pending`; matrix declaration `pending`.

## M6 - General solve and accepted transactions

### M6.1 - Define and execute the general solve request and result protocol

**Scope:**
- Extend existing Program SSA, nested regions and SolveOutcome.consume with a typed Problem request: unknown tuple, frozen equation inputs, independent seeds, branch/gauge selection, requested outputs, residual/error interpretation, solver strategy and explicit failure disposition.
- Provide the early shared protocol consumed by general fields and spatial implicit diffusion. Keep physical equations separate from solver coordinates and preserve explicit accumulation Q(q+)-U_n in residual requests.
- Classify dependency cycles: an explicit cyclic evaluation is rejected; a declared implicit problem supplies the unknowns and residual boundary. Declare exact, approximate or explicitly selected finite-difference derivative availability without treating every opaque call as differentiable.
- Adapt one existing supported native solve through the new protocol end to end, retaining the old public consume spelling and scalar authoring behavior. This child establishes the protocol; later children integrate new numerical routes.

**Acceptance (all criteria):**

1. A supported existing solve passes validate -> resolve -> compile -> bind -> run through the new request and produces an explicitly consumed typed result tuple with source mappings and lowering disposition.
2. Changing only a seed leaves frozen equation inputs and the defined residual unchanged; changing a frozen coefficient changes the problem identity. Tests distinguish equation data from iterative initialization.
3. Attempting to read an outcome without an explicit failure action, supplying missing/duplicate unknown bindings, or using a result outside its legal region produces a structured diagnostic before publication.
4. Joint-result projection preserves each output identity and one solve authority; tuple representation supports multiple field unknowns without claiming their native realization is complete.
5. An undeclared explicit dependency cycle is rejected; a supported declared implicit problem resolves it with stated derivative provenance. Unsupported derivatives or nonlinear forms fail explicitly.
6. Record representable, validated, resolved, emitted, executed, numerically checked and performance-characterized status separately for the adapted route, with exact configuration and commands; no unexecuted capability is reported as working.

**Dependencies:** `M2.3`.

**Related issue references:** ADC-665, ADC-700.

**Source and code:** Migration specification sections 7.3, 8.5, 9.2-9.3, 13 M6; attached note section 6.2.4 for accumulation and implicit diffusion constant mode. Paths: `python/pops/time/_program/solve.py`; `python/pops/time/_program/region_validation.py`; `python/pops/time/solve_outcome.py`; `tests/python/unit/time/test_solve_outcome_contract.py`; `tests/python/unit/time/test_field_solve_outcome.py`; `include/pops/runtime/program/residual_operator.hpp`.

**Evidence status:** all seven levels `pending`; matrix declaration `pending`.

### M6.2 - Enforce nested attempt publication and accepted exchange transactions

**Scope:**
- Implement scoped provisional continuation for state, fields, accepted exchange records and retained temporal objects. A nested substep may publish only into its enclosing provisional scope; authority changes only when the enclosing window accepts.
- Define failure categories, explicit retry/reject/abort policies and finite retry budgets for the general solve protocol. Rejected stages and nonlinear iterates contribute no accepted exchanges.
- Propagate native/device status to one consistent distributed decision before subsequent required collectives, preserving buffer and object lifetimes through asynchronous work and rollback.
- Use existing Program commit and region validation plus runtime transactions; retain the existing consumer requirement for SolveOutcome and avoid a second commit authority.
- Expose reusable failure aggregation and accepted-ledger services early, before native-call and implicit-diffusion vertical paths are qualified.

**Acceptance (all criteria):**

1. Failure injection after a nested solve or substep demonstrates that rejected state, fields, histories and exchange records are absent from the accepted continuation; accepted enclosing windows publish each object once.
2. A successful retry starts from the declared accepted/frozen inputs and permitted new seed, and yields the same accepted result as the equivalent successful attempt without leaked rejected contributions.
3. Iteration and rejected-stage counts can increase without increasing the accepted physical exchange; the final ledger matches the actual accepted temporal quadrature and mathematical occurrence multiplicity.
4. An exhausted finite retry budget and unrecoverable status terminate with explicit diagnostic evidence rather than looping or silently accepting a failed solve.
5. On every supported declared MPI/device configuration, a one-rank or device-local injected failure reaches a coherent decision without deadlock, use-after-free or partial authoritative publication; unavailable configurations remain unqualified.
6. Record exact executed configuration, failure position, accepted-state comparison and seven-stage capability/evidence status; a source-level transaction contract alone does not satisfy execution acceptance.

**Dependencies:** `M6.1`, `M2.1`.

**Related issue references:** ADC-666, ADC-702, ADC-757.

**Source and code:** Migration specification sections 9.3, 10.1, 10.4, 13 M6. Paths: `python/pops/time/_program/authoring.py`; `python/pops/time/_program/solve.py`; `python/pops/time/_program/commit_validation.py`; `python/pops/time/_program/region_validation.py`; `python/pops/time/solve_outcome.py`; `include/pops/runtime/program`; `tests/python/examples/final/test_multiphysics_core_example.py`.

**Evidence status:** all seven levels `pending`; matrix declaration `pending`.

## Cross-milestone dependencies and R1 unblocking

The four entry lanes are independent only through their declared M1/M2 prerequisites:

- M3.1 starts the field-problem lane. M6.1 then supplies the shared solve-request protocol before
  M3.2, M3.3, and M3.4 can close.
- M4.1 starts the heterogeneous-interaction lane after M1.3, M2.1, M2.2, and M2.3. Its typed
  native, failure, and cost leaves follow in order.
- M5.1 starts the diffusion lane after M1.2, M2.1, M2.2, and M2.3. M5.2 may precede the exchange
  and implicit leaves when its coefficient route is selected.
- M6.1 starts the solve and result lane after M2.3. M6.2 is its transaction successor and is also
  required by later failure-domain and accepted-exchange paths.

The repaired principal order is M3.1 -> M6.1 -> M3.2 -> M6.2 -> M3.3 -> M3.4, with
M4.1 -> M4.2 -> M4.3 -> M4.4 and M5.1 -> M5.2 -> M5.3 -> M5.4 -> M5.5 joining when their
exact dependencies are satisfied. The individual manifest dependency lists remain authoritative
where this compact order has parallel branches.

Two remaining M6 packages are outside the 15-leaf crosswalk but remain phase dependencies:

| Package | Scope index | Dependencies | Related issue references | Status |
| --- | --- | --- | --- | --- |
| M6.3 | Execute checked explicit, implicit, IMEX, and splitting method expansions | M6.2, M3.3, M4.3, M5.4 | ADC-663, ADC-668, ADC-720 | pending |
| M6.4 | Make histories, dense output, restart, and topology transitions explicit | M6.2, M3.3 | ADC-667, ADC-678, ADC-717, ADC-742 | pending |

The M6 phase cannot close until both M6.3 and M6.4 close with their own receipts. The package
labels above are execution-index labels; the manifest remains the source of their full acceptance
criteria.

R1 is the first integrated release gate and remains pending until M0, M2, M3.2, M3.3, M4, M5.3,
M5.4, and M6 are complete. Its eight representative native paths are scalar AMR, field-coupled
transport, Euler-Poisson source, heterogeneous interaction, explicit diffusion, implicit diffusion,
nonconstant-coefficient scalar field, and an imported native primitive. The full R1 acceptance
remains:

1. Execute every path through validate -> resolve -> compile -> bind -> run.
2. Run the complete M0-frozen numerical matrix for those paths with accepted conservation, exchange,
   and failure/retry evidence; reduced conformance cases are insufficient.
3. Compare unchanged numerical methods, reproducibility policy, and performance metrics against
   authenticated M0 baselines, with any changed invalid behavior carrying an explicit corrected
   contract.
4. Report all seven evidence levels and the exact platform, layout, and restart scope for every
   path. Do not advertise unexecuted joint-field, high-rank, or asynchronous-AMR capabilities.
5. Keep examples equation-oriented, without provider tables, duplicated equations, or
   model-specific core edits.

R1 related issue references are [ADC-695](https://linear.app/romain7522/issue/ADC-695),
[ADC-690](https://linear.app/romain7522/issue/ADC-690),
[ADC-691](https://linear.app/romain7522/issue/ADC-691),
[ADC-692](https://linear.app/romain7522/issue/ADC-692), and
[ADC-694](https://linear.app/romain7522/issue/ADC-694). The existing parent issue links remain
[ADC-903](https://linear.app/romain7522/issue/ADC-903),
[ADC-904](https://linear.app/romain7522/issue/ADC-904),
[ADC-905](https://linear.app/romain7522/issue/ADC-905), and
[ADC-906](https://linear.app/romain7522/issue/ADC-906).

## Receipt requirements and scope limits

A future package receipt must identify the exact source revision and worktree state, environment and
toolchain, selected matrix cell, setup/build/validation/execute/compare commands, exit codes,
retained artifact and output hashes, and each of the seven evidence statuses. The bounded matrix
must state dimension, backend, MPI ranks and communicator, layout, levels and blocks, method,
restart and output policy, and tolerances. Cache-affected observations must be labeled and cannot
stand in for a cold or equivalent comparison.

This document deliberately leaves all M3-M6 package, evidence, and matrix statuses pending. It
records the frozen contract and execution order; it does not claim implementation, native
execution, numerical qualification, performance characterization, or R1 completion. The M0-M2
report and contract remain the source for that earlier phase:

- migration_m0_m2_contract.md
- migration_m0_m2_results.md

The supplied migration manifest is the authority for any acceptance wording or dependency not
repeated here. No child Linear issue is implied by the execution-index labels.
