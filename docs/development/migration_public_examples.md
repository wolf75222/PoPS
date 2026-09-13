# R1 public migration examples

This guide documents the bounded public dispatcher in
[`examples/migration/r1_workflows.py`](../../examples/migration/r1_workflows.py). It gives
one runnable command surface for the eight R1 workflow shapes required by ADC-924:
`Model -> Case -> DiscretizationPlan -> Program -> validate -> resolve -> compile -> bind -> run`.
The examples are lifecycle demonstrations and a compact representative witness set for these
public capability shapes. They do not by themselves qualify numerical or scientific behavior,
execution-environment coverage, or performance, and this page contains no runtime or performance
result. Test selection and evidence reuse follow
[`migration_verification_scope.md`](migration_verification_scope.md). The current status and
qualification boundary are tracked in [`migration_m3_m8_results.md`](migration_m3_m8_results.md).

## Install and select a native artifact

Use an installed Python wheel built for the exact dimension used by the example. The
repository baseline requires C++20, CMake 3.21 or newer, Kokkos 4.4.01, Python 3.12, and numpy.
The recommended Dim=2 setup is:

```bash
bash scripts/setup_env.sh
bash scripts/build_python.sh --dim 2
```

The script has no `--mpi` workflow option. An MPI wheel is a separate exact artifact:

```bash
bash scripts/build_python.sh --dim 2 --mpi
mpiexec -n 2 env POPS_NATIVE_DIM=2 python examples/migration/r1_workflows.py field-transport --cells 16 --steps 1
```

Use the MPI command only when the installed artifact manifest names `MPI_COMM_WORLD`; the
launcher, wheel, and MPI runtime must agree, and the native runtime requires `MPI_THREAD_MULTIPLE`
(with parallel HDF5 for the MPI build). Serial artifacts use no execution resource. The
dispatcher accepts the communicator reported by the artifact and does not silently change dimensions.

## Command surface

```bash
POPS_NATIVE_DIM=2 python examples/migration/r1_workflows.py --list
POPS_NATIVE_DIM=2 python examples/migration/r1_workflows.py <workflow> --cells 16 --steps 1 --work-dir outputs/migration-r1
```

`workflow` is optional only with `--list`. The choices are the eight names below.
`--cells` defaults to `128` for `scalar-amr` and `16` for every other workflow, and must be at
least `4`. `--steps` defaults to `1` for non-AMR workflows; when omitted for `scalar-amr`, the
route derives `steps=round(0.2/dt)` with `dt=min(0.01,0.2/cells)`. A supplied `--steps` is used
by every workflow. For `scalar-amr`, `--scheme inline|imported` selects the inline
`inline_ssprk2` or imported `ssprk2` definition. `--work-dir` defaults to `outputs/migration-r1`.
The direct workflows print an indented JSON summary. `accepted_steps` reports the initial
accepted run and `final_time` the route's final runtime clock; for `scalar-amr`, the clock includes
the continuous and restarted continuation. `artifact` identifies the compiled artifact, and state
summaries report shape, component means, minimum, and maximum. Diagnostics such as `mean_change`,
`pair_mean_defect`, `gradient_l2`, and `potential_l2` are descriptive values without qualification
thresholds.
The `scalar-amr` route also writes a checkpoint, restores it strictly, and compares continuous
and restarted continuation before returning.

## Eight routes

| Selector | Model and boundary used | Output and claim limit |
| --- | --- | --- |
| `scalar-amr` | Unit-square scalar advection with velocity `(1.0,0.25)`, a Gaussian initial condition, inflow/outflow boundaries, MUSCL VanLeer reconstruction, and a `ScalarUpwind` solver. A two-level AMR hierarchy uses threshold tagging, buffering, conservative transfer, scheduled regridding, and subcycling. `--scheme` selects the inline `inline_ssprk2` or imported `ssprk2` method. | Runs the accepted trajectory, writes a checkpoint, restores it strictly, and compares continuous versus restarted continuation; returns checkpoint, restart status, levels, patches, regrids, state, and output path. These lifecycle checks do not claim scientific qualification or performance. |
| `field-transport` | Dim=2 unit square, uniform `(cells,cells)`, periodic axes. A two-component fluid consumes a solved periodic Poisson potential and its gradient; `dt=0.125`. | Returns the `fluid` state summary, component mean changes, and gradient norm. These values show a consumed-field data path, not the field accuracy rows. |
| `euler-poisson` | Same periodic unit square. A three-component `(rho,mx,my)` Euler state receives an electric source from a separate frozen-load Poisson block; `dt=0.125`. | Returns the fluid state, mean changes, and gradient norm. The example does not claim Euler-Poisson accuracy or conservation. |
| `heterogeneous-interaction` | Periodic unit square with unequal species states `(p,E)` and `(p,E,m)`, one joint exchange, and `dt=1e-3`. | Returns left/right state summaries and pair mean defects for `p` and `E`; those defects have no pass threshold here. |
| `explicit-diffusion` | Periodic unit square scalar heat equation with `0.1*grad(u)`, Forward Euler, and `dt=0.05/(0.1*cells*cells)`. | Returns a heat state summary and mean change. The small route is a lifecycle example, not an explicit diffusion convergence row. |
| `implicit-diffusion` | The same periodic scalar heat equation with an implicit diffusion stage, backward Euler, a Newton solve, and `dt=1e-4`. | Returns a heat state summary and mean change. It does not claim implicit spatial or temporal order. |
| `variable-coefficient-field` | Periodic unit square; two load/reference blocks feed `-DivCoeffGrad(potential, coefficient)`, with coefficient `1.5+0.25*cos(2*pi*x)` and a shared mean gauge; `dt=0.125`. | Returns coefficient range, potential mean, and potential L2. These are field lifecycle diagnostics, not a manufactured-solution result. |
| `imported-native-primitive` | Two independent periodic blocks on the unit square, with distinct initial profiles. Python defines the velocities, conservation equation, finite-volume method, first-order reconstruction, and Rusanov solver; an imported `NativeFunction` supplies scalar multiplication through `migration_arithmetic::multiply`; `dt=1e-3`. | Stages `native-arithmetic/arithmetic.hpp` below `--work-dir` as a header-only component and returns component, artifact, and binary identities plus left/right state summaries. Identities authenticate provenance; they do not prove native qualification or performance. |

All direct routes construct the unit-square `CartesianGrid` at the requested cell count and
call `pops.run` with the requested `t_end` and `max_steps`. Periodicity is explicit in the
field, interaction, diffusion, and imported-primitive routes. The scalar AMR route owns its
geometry, Gaussian initial condition, transport boundaries, AMR controls, and time-method
definition.

The imported primitive additionally needs a C++20 compiler, PoPS component headers, and an
authenticated installed Dim=2 native artifact. It stages the header-only
`migration_arithmetic` component with `PreparedNativeComponent` and compiles it for the
`NativeFunction` calls; set `POPS_INCLUDE` when the installed header location is not
discoverable. Its implementation and native header are
[`imported_native_primitive.py`](../../examples/migration/scientific/imported_native_primitive.py)
and [`arithmetic.hpp`](../../examples/migration/scientific/arithmetic.hpp).

## Normative qualification routes

The following test modules are scientific qualification entrypoints, not user examples. Select
relevant coherent witnesses for each change under the verification policy and retain their
declared dimensions, rows, oracles, and tolerances when making a qualification claim.

| Example route | Qualification modules and row families |
| --- | --- |
| Scalar AMR | [`test_amr_transport_diffusion_qualification.py`](../../tests/python/integration/runtime/test_amr_transport_diffusion_qualification.py): `explicit-amr-periodic-r1`, `explicit-amr-physical-r1`, `explicit-amr-retry-r1`, corresponding `r2` rows, and `explicit-amr-supplement`; implicit controls are in [`test_amr_implicit_diffusion.py`](../../tests/python/integration/runtime/test_amr_implicit_diffusion.py): `implicit-amr-reference`, `implicit-amr-physical`, `implicit-amr-controls`. |
| Field transport and Euler-Poisson | [`test_public_field_consumers.py`](../../tests/python/integration/runtime/test_public_field_consumers.py): `field-consumers` (transport and Euler branches), `joint-gradient`; field problem/reuse modules cover `field-solve`, `field-incompatible`, `field-stage`, and `field-accuracy-invalidation`. |
| Heterogeneous interaction | [`test_native_interaction_matrix.py`](../../tests/python/unit/codegen/test_native_interaction_matrix.py): `interaction-numerical`, `interaction-cost`; [`test_native_interaction_solve.py`](../../tests/python/unit/codegen/test_native_interaction_solve.py): `interaction-solve`; [`test_native_constitutive_faces.py`](../../tests/python/unit/codegen/test_native_constitutive_faces.py): `constitutive-face-matrix`; [`test_collective_diffusion_constraint.py`](../../tests/python/unit/codegen/test_collective_diffusion_constraint.py): `collective-constraint`; [`test_native_repeated_interaction_balance.py`](../../tests/python/unit/codegen/test_native_repeated_interaction_balance.py): `repeated-balance-interactions-serial`, `repeated-balance-interactions-solve-mpi`. Joint codegen is a source control without a numerical row family. |
| Explicit diffusion | [`test_public_diffusion_matrix.py`](../../tests/python/integration/runtime/test_public_diffusion_matrix.py): `explicit-spatial`, `explicit-face-ledger`, `nonlinear-gradient-variable`, `combined-stability-dim2`, `diffusion-authority`; [`test_combined_diffusion_exchange_ledger.py`](../../tests/python/integration/runtime/test_combined_diffusion_exchange_ledger.py): `combined-ledger`, `eb-refusal`; the separate Dim=1 fitted-flux rows are in [`test_public_drift_diffusion_matrix.py`](../../tests/python/integration/runtime/test_public_drift_diffusion_matrix.py). |
| Implicit diffusion | [`test_implicit_diffusion_lifecycle.py`](../../tests/python/integration/runtime/test_implicit_diffusion_lifecycle.py): `implicit-lifecycle-parity`, `implicit-lifecycle-supplement-spatial`, `implicit-lifecycle-controls`, `original-m5-be-spatial`, `original-m5-be-temporal`, `original-m5-uniform-budget`, `original-m5-uniform-be-ledger`; AMR rows are in [`test_amr_implicit_diffusion.py`](../../tests/python/integration/runtime/test_amr_implicit_diffusion.py). |
| Variable-coefficient field | [`test_public_field_problem.py`](../../tests/python/integration/runtime/test_public_field_problem.py) and [`test_public_field_reuse.py`](../../tests/python/integration/runtime/test_public_field_reuse.py): `field-solve`, `field-incompatible`, `field-stage`, `field-accuracy-invalidation`, `field-immutable-authority-replacements`, `field-authority-atomic-refusals`; variable branches also occur in the explicit and implicit diffusion modules above. |
| Imported native primitive | [`test_native_call_compiled.py`](../../tests/python/unit/codegen/test_native_call_compiled.py) and [`test_external_component_package.py`](../../tests/python/integration/native_loader/test_external_component_package.py): external component group. These tests have no R1 numerical row family and are not substituted for the dispatcher. |

The Gaussian control remains a distinct implicit AMR scientific row:
[`test_unperiodized_gaussian_conserves_against_reference_without_an_order_claim`](../../tests/python/integration/runtime/test_amr_implicit_diffusion.py)
records family `implicit-amr-unperiodized-gaussian-control`, N=16/32/64 against uniform
2N, four steps at `dt=0.00025`, finite L2 only, and no order claim. This guide does not
execute or promote that row.

There is one dispatcher and eight scientific modules under
[`examples/migration/scientific/`](../../examples/migration/scientific/); the dispatcher routes
each selector to its module. These examples describe executable lifecycle paths and their
invocation summaries; they do not establish numerical or scientific qualification,
execution-environment coverage, or performance. Consult the verification scope for test
selection, evidence reuse, and claim boundaries, and the results ledger for recorded receipts.
Update those records from authenticated source/native/wheel runs before any release or
performance statement.
