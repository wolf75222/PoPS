# PoPS M3-M8 and R1 current results

**Evidence refresh: 2026-09-09.** This is the concise current ledger for the
candidate-v6 evidence set. It is a qualification record, not a final-HEAD claim:
the integrated source/native candidate is `9164272546e0074d825603a1ea299990b5ee9ace`,
M0-M2 remain preserved, M3-M7/R1 remain in progress, and M8 remains backlog.

The preserved M0-M2 index is [`migration_m0_m2_results.md`](migration_m0_m2_results.md),
with the machine-readable [M0 index](migration_evidence/m0/index.json) and
[M1-M2 index](migration_evidence/m1_m2/index.json). The older
[`migration_m3_m6_progress.md`](migration_m3_m6_progress.md) and its
[foundation ledger](migration_evidence/m3_m6_foundation/index.json) retain historical
receipts; this page does not relabel those ledgers as current closure.

Every claim below keeps `validate -> resolve -> compile -> bind -> run` separate
from numerical checking and performance characterization. The full declared
matrices remain required. No smoke or reduced case, tolerance relaxation, timeout
increase, loaded test duration, or source-level pass is a substitute for them.

## Frozen candidate and execution configuration

| Item | Current receipt |
| --- | --- |
| Source and freeze | `9164272546e0074d825603a1ea299990b5ee9ace`; clean MAIN/FROZEN equality is recorded in [candidate-final-v6-source-freeze.json](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/candidate-final-v6-source-freeze.json). |
| Native CMake cache | Release, `POPS_NATIVE_DIM=2`, MPI and HDF5 enabled, tests enabled, Python build disabled; exact configure/build commands are [configure](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/candidate-final-v6-native-current-configure-command.json) and [build](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/candidate-final-v6-native-current-build-command.json). |
| Host/toolchain | Python 3.12, AppleClang 21, Kokkos 5.2.1, OpenMP plus Serial, MPICH 4.1.2, parallel HDF5 1.14.3, 8 CPU/16 GiB, `OMP_NUM_THREADS=2`, CMake parallelism 2. |
| Native execution | Exact [CTest command](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/candidate-final-v6-native-current-ctest-command.json), [summary](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/candidate-final-v6-native-current-summary.json), and [JUnit](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/candidate-final-v6-native-current-ctest.xml). |
| Python source contract batch | Exact [command](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/candidate-final-v6-source-repair-batch-command.json), [result](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/candidate-final-v6-source-repair-batch-result.json), and [JUnit](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/candidate-final-v6-source-repair-batch.xml). |

## Current verified results

| Evidence | Result and limit |
| --- | --- |
| Full native candidate | `1143` inventory cases: `1124` passed, `19` conditional skips, `0` failures, `0` errors; CTest wall time `1463.6758 s`. Source and native artifacts were unchanged before/after. This is structural/native execution evidence, not full scientific qualification. |
| Skip disposition | The [exact skip review](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/candidate-final-v6-ctest-skip-disposition.json) identifies `17` duplicate serial discoveries covered by exact named MPI2 registrations and `2` required cells that did not execute: the hyperbolic-boundary MPI2 registration and the Dim1 `PreparedBz` control. They remain open rows. |
| Static and private wheels | [Static gates](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/candidate-final-v6-static-gates.json) pass M2/M3/M4 check-only, catalog, packaging, Ruff, and Pyright (`662` files, `0` errors, `2` warnings). The [Dim1](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/candidate-final-v6-dim1-private-proof.json) and [Dim2](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/candidate-final-v6-dim2-private-proof.json) private proofs pass. These establish artifacts and contracts, not numerical science. |
| Source contract batch | The exact rebuilt-package batch contains `169` passing tests with no failure/error/skip in its JUnit. The [result](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/candidate-final-v6-source-repair-batch-result.json) proves source/native proof integrity; it is not an M3-M8 scientific matrix. |
| Original tutorials | [Tutorial conformance](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/candidate-final-v6-launcher-fixed-tutorial-conformance/summary.json) passes both unchanged profiles: OpenMP at one rank and MPI at two ranks. The private [MPI launcher repair](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/candidate-final-v6-private-prefix-mpiexec-repair.json) only links the selected private prefix to the installed MPICH launcher; it changes no production or wheel bytes. Tutorial completion does not establish convergence, performance, portability, or accuracy beyond tutorial assertions. |
| PMPI inventory | The [v6 review](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/candidate-final-v6-pmpi-inventory-review.md) and [receipt](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/candidate-final-v6-pmpi-inventory-receipt.json) find `14` wrapped communication APIs, `20` explicit non-payload APIs, and `0` unknown/unclassified APIs. This is static inventory only: no new PMPI measurement or MPI replay was performed. |
| Baseline boundary | The [M0 report](migration_m0_m2_results.md) and [baseline index](migration_evidence/m0/index.json) retain the five original example failures before accepted science. No old/new accepted numerical comparison is inferred from them. |

## Phase ledger

The issue links below are the complete remaining M3-M8/R1 set from the pinned
[migration manifest](/Users/romaindespoulain/Documents/Codex/2026-09-07/infer-the-intended-scope-from-the/notes/migration-manifest.json).

| Scope | Linear issues | Current disposition and remaining boundary |
| --- | --- | --- |
| M3: unify field problems and solve results | [ADC-903](https://linear.app/romain7522/issue/ADC-903); M3.1 [ADC-925](https://linear.app/romain7522/issue/ADC-925), M3.2 [ADC-926](https://linear.app/romain7522/issue/ADC-926), M3.3 [ADC-927](https://linear.app/romain7522/issue/ADC-927), M3.4 [ADC-928](https://linear.app/romain7522/issue/ADC-928) | Static/source contracts and the 169-test batch are recorded. Field accuracy, invalidation, and complete variable/joint field numerical rows remain open. The 21-file/27-site bind-context inventory is preflight only; see [bind preflight](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/candidate-final-v6-bind-context-preflight.json). |
| M4: interactions and typed native calls | [ADC-904](https://linear.app/romain7522/issue/ADC-904); M4.1 [ADC-929](https://linear.app/romain7522/issue/ADC-929), M4.2 [ADC-930](https://linear.app/romain7522/issue/ADC-930), M4.3 [ADC-931](https://linear.app/romain7522/issue/ADC-931), M4.4 [ADC-932](https://linear.app/romain7522/issue/ADC-932) | Native inventory and typed-contract evidence are present. Complete heterogeneous interaction, derivative/failure-domain, MPI, and cost qualification remains open. |
| M5: transport-diffusion execution paths | [ADC-905](https://linear.app/romain7522/issue/ADC-905); M5.1 [ADC-933](https://linear.app/romain7522/issue/ADC-933), M5.2 [ADC-934](https://linear.app/romain7522/issue/ADC-934), M5.3 [ADC-935](https://linear.app/romain7522/issue/ADC-935), M5.4 [ADC-936](https://linear.app/romain7522/issue/ADC-936), M5.5 [ADC-937](https://linear.app/romain7522/issue/ADC-937) | The complete M5 plan is declared in the [diffusion qualification plan](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/diffusion-qualification-plan.md), but current full numerical rows are not complete. Test-only additions are pending for mean-preserving BE `N=16/32/64` at `t=0.05`, the `N=32` temporal row, nonlinear accumulation at `2e-11`, insufficient-budget rollback, and accepted implicit exchanges. |
| M6: temporal problems and accepted continuation | [ADC-906](https://linear.app/romain7522/issue/ADC-906); M6.1 [ADC-938](https://linear.app/romain7522/issue/ADC-938), M6.2 [ADC-939](https://linear.app/romain7522/issue/ADC-939), M6.3 [ADC-940](https://linear.app/romain7522/issue/ADC-940), M6.4 [ADC-941](https://linear.app/romain7522/issue/ADC-941) | Solve/result and transaction contracts have source evidence. Complete explicit, implicit, IMEX, splitting, history, dense-output, restart, and topology-transition matrices remain open; no current temporal result is promoted from an older candidate. |
| M7: coupled supports, layouts, and AMR | [ADC-907](https://linear.app/romain7522/issue/ADC-907); M7.1 [ADC-942](https://linear.app/romain7522/issue/ADC-942), M7.2 [ADC-943](https://linear.app/romain7522/issue/ADC-943), M7.3 [ADC-944](https://linear.app/romain7522/issue/ADC-944), M7.4 [ADC-945](https://linear.app/romain7522/issue/ADC-945) | AMR and support controls are present in the current native inventory, but complete AMR transport/diffusion, composite field, distributed exchange, and transition qualification is open. Physical mapping is bounded to the host Dim2 one-rank `1x1v-to-1x` patch; different mapping extensions refuse. |
| R1: first integrated release | [ADC-921](https://linear.app/romain7522/issue/ADC-921) | The eight-path release matrix and its complete numerical, restart/output, collective, and failure/retry evidence are not complete. Passing tutorials and contract batches do not close R1. |
| M8: retire obsolete routes | [ADC-908](https://linear.app/romain7522/issue/ADC-908); M8.1 [ADC-922](https://linear.app/romain7522/issue/ADC-922), M8.2 [ADC-923](https://linear.app/romain7522/issue/ADC-923), M8.3 [ADC-924](https://linear.app/romain7522/issue/ADC-924) | M8 has no closure. M8.1 replacement-specific equivalence is pending. M8.2 deletion is deferred for exactly seven orphan verification tests in the [current inventory](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/m8-orphan-campaign-current-inventory.json); active `FieldOperator`/`_b_field_operator`, `PhysicalModelFor`, and `HierarchyTensorSolverProvider` consumers remain retained. M8.3 diagnostics and completion documentation wait on the preceding evidence. |

## Pending rows for root integration

These rows are intentionally explicit so later execution can replace the status
without rewriting historical receipts:

| Row | Required update |
| --- | --- |
| Native conditional coverage | Execute the two required cells identified by the [skip disposition](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/candidate-final-v6-ctest-skip-disposition.md), or preserve an evidence-backed refusal/unavailable disposition. Reconcile the full CTest count after any registration change. |
| M3 field accuracy/reuse | Finish the declared [field plan](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260908-evidence/fields-qualification-plan.json), including accuracy and invalidation cases, with one exact source/artifact/configuration pin. |
| M5 normative matrix | Add and run the full original rows and report norms, rates, mass, nonlinear residual, rollback, and accepted-exchange oracles. The plan's tolerances remain fixed. |
| M6/M7/R1 science | Run the complete temporal, AMR, distributed, restart/output, and eight-path matrices. Classify unavailable GPU, public Dim3, and multinode cells explicitly. |
| Performance | Run the prepared paired protocols in a quiet window with device completion and raw metrics; do not derive performance from loaded test times or contract batches. |
| M8 equivalence/deletion | Complete M3/M5/M7/R1 prerequisites, record row-by-row M8.1 equivalence, then review the seven-file M8.2 candidate separately. No deletion is authorized by the inventory. |

No tests or builds were run for this documentation-only change. The source,
configuration, command, artifact, and evidence paths above are retained for the
root agent to update after the remaining qualification work and before any final
commit or Linear closure.
