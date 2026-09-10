# PoPS M3-M8 and R1 results — Latest executed frozen candidate: v22

**Evidence snapshot: 2026-09-10.** The latest executed frozen candidate is v22 at source
`75636f29fe01d96af8a2d49a12a58418b257c16f`. This page records implementation, source/static checks,
package checks, native execution, numerical qualification, and performance separately. It is not a
closure claim: M3-M7 and R1 remain in progress and M8 remains backlog. The prior v21 candidate at
`50b7953e45edb959b3fdcf17244db4d3a543d2ca` remains a bounded record with a failed five-test
critical group (2 PASS, 3 FAIL), not the latest executed frozen candidate.

The v22 freeze receipt `candidate-final-v22-source-freeze.json` has SHA-256
`1e4ab000ae6587443dc97f2719a7795e54f6f5c95d24131ddd478ade20fd411c` over 2,376 byte-identical
tracked files; its primary review SHA-256 is
`6e0432b3653c05b75853c0d46da7147476a203afb4d69f8954a1f0b2e1f84ae1`.

The former v6 ledger is preserved verbatim in
[`migration_m3_m8_results_v6_historical.md`](migration_m3_m8_results_v6_historical.md). Its receipts
remain historical and are not relabeled as v22 evidence. The M0-M2 indexes are unchanged:
[`migration_m0_m2_results.md`](migration_m0_m2_results.md),
[`migration_evidence/m0/index.json`](migration_evidence/m0/index.json), and
[`migration_evidence/m1_m2/index.json`](migration_evidence/m1_m2/index.json).

## How to read this ledger

Source contracts and static inventories establish represented interfaces. A build or installed-wheel
proof does not establish native runtime behavior; native execution does not establish numerical
qualification or performance. Every accepted claim requires the declared configuration, exact
command, source and artifact identity, retained output, and the original oracle. No smoke case,
tolerance relaxation, timeout increase, or loaded test duration substitutes for the declared matrix.

## v22 evidence

| Area | Established evidence | Current boundary |
| --- | --- | --- |
| Frozen source | The v22 freeze and primary-review receipts above pin clean source `75636f29fe01d96af8a2d49a12a58418b257c16f` and byte equality for 2,376 tracked files. | This is provenance for the candidate, not numerical or performance qualification. |
| v22 Dim2 package | `candidate-final-v22-wheel-dim2-result.json` records a 16.7805265 s wheel build. The prefix receipt `candidate-final-v22-prefix-dim2.json` (SHA-256 `22dde1c4f8694041c00adcb1b6daec39f3b33f448fff52fa24736749710393f0`) records source `75636f29fe01d96af8a2d49a12a58418b257c16f`; its venv, install, prove, and doctor result files exit 0. The private proof `candidate-final-v22-dim2-private-proof.json` (SHA-256 `73bc3d09271de6019db54b993eca8c75d1e6bff6012ab51f2fe0517b2c452a0b`) authenticates wheel SHA-256 `65ea922fab30182e75f889655eb938b33e3633775d7b7c07dce7c741ca1617ae`, native leaf SHA-256 `d998bb11673dfd2d5dd6b444f391d6f2bff1fa1c7cf76af9c4ca4e59aa08ec60`, and headers SHA-256 `aa341ffbb9ea819052cc560b31a2b80f777c6db89a4aabfbcaf0230949ebe6fc`. | Only Dim2 is packaged. These checks do not qualify changed Python runtime behavior; Dim1/Dim3 packaging and science remain open. |
| Prior v21 Dim2 package | `candidate-final-v21-wheel-dim2-result.json` records the prior v21 wheel build in 117.970 s. | This package receipt belongs to source `50b7953e45edb959b3fdcf17244db4d3a543d2ca` and is not v22 evidence. |
| Focused v22 AMR transfer gate | At source `75636f29fe01d96af8a2d49a12a58418b257c16f`, configure completes in 11.126 s and the `test_amr_layout_transfer` target builds in 93.885 s. `generic-amr-sync-75636f2-r1-ctest.result.json` records 7 selectors, 0 failures, and 0 skips in 30.341075 s: six serial cases plus one two-rank MPI suite with six cases per rank; source HEAD and clean status are unchanged before and after. | This focused C++ execution is not the full v22 matrix. The v22 collection is not execution credit, and full native map/AMR groups remain open. |
| v22 default native build | `generic-v22-full-native-build-r1.result.json` reports a successful default native build in 268.0126235 s at source `75636f29fe01d96af8a2d49a12a58418b257c16f`, with clean source status before and after. | This is build evidence only. It does not substitute for the focused CTest, the matrix groups, or numerical qualification. |
| v22 critical AMR field-map group | The five-test serial group reports 2 PASS and 3 FAIL in 250.864989 s. Binding completes; all three failures occur before the first time step with `RuntimeError: per-layout temporal state diverged at program_schedule`, so no update or rollback assertion is reached. The runner result `candidate-final-v22-matrix-v18-amr-field-map-continuations-dim2-result.json` has SHA-256 `10d08b016016fbbd0a5a30a028dc08853268c98f071c3aaf23f2824069fb7b7a`; its group receipt has SHA-256 `51d7985d440a54437b5d2dd15d382d51c9594e00e3d8d68cb8b185e9a1f5cb24`. | This is incomplete v22 native coverage. The remaining declared groups, including the critical physical-map and MPI lanes, remain unexecuted. |
| v22 declared and collected matrix | The 35-group manifest spans the generic migration, including M7: 25 serial and 10 MPI groups. R1 is tracked separately as eight workflows. Collection records enumerate 20 Dim2 serial groups (170 cases) and 10 MPI groups (67 cases per rank). | Collection is not execution credit. Full field/map, AMR, MPI, restart, numerical, failure/retry, and performance qualification remains open. |
| Prior v21 source/static control | At source `50b7953e45edb959b3fdcf17244db4d3a543d2ca`, `generic-v21-source-integrated-r1.result.json` reports 790 PASS, 0 failures, 0 errors, and 0 skips in 166.709654 s with unchanged source; Pyright reports 0 errors and 2 existing warnings. Its external-import census found 47 unique PoPS imports across 6 generated DSOs and no missing imports. | This is a v21 control, not a v22 source gate or runtime qualification. |
| Prior v21 critical execution | The v21 five-test serial group reports 2 PASS and 3 FAIL in 252.263385 s. All three failures stop during bind with `ValueError: AMR physical synchronization identity is unknown`; no update or rollback assertion executes after bind. Group receipt SHA-256 is `730f161888cf72d3d8b219fb4f3a3b5eda746176a8ba4eaef4e8d1f65edf489c`; runner result SHA-256 is `c2d599a19d76e3b9f8008bb5cc9384a2206a8bc3dc109e5a4038c7e1a3e0bcce`. | This failed v21 group is retained as prior-candidate evidence; it is not a v22 result. |
| Prior native controls | The v19 full C++ gate at `13dc8bb` reports 1,212 PASS, 0 FAIL, and 21 unchanged conditional SKIP across 1,233 entries. The old v17 uniform-map comparator covered one module with nine tests: 9 PASS/0 FAIL/0 SKIP in 158.857867625 s, with six native and nine reference arrays captured. | These controls do not qualify the changed v22 Python runtime; current-candidate Gaussian and generic-map replacement comparisons remain incomplete. |

## Preserved M0-M2 and M8 boundaries

The M0 report retains the five original public profiles that failed before accepted science; no old
R1 arrays are inferred from them. Seven retained verification test files are present at both baseline
and the v22 source; their campaign assets were already absent at baseline. This is pre-existing
infrastructure debt and does not establish replacement equivalence.

The M8 decision in `generic-v19-m8-equivalence-next-actions.json` is 8 retained-needed, 2
replacement-qualification-pending, and 1 baseline-orphan group. Legacy interfaces remain retained;
v22 replacement comparisons for Gaussian and generic maps are incomplete. No M8
removal or equivalence claim follows from these receipts; removal remains conditional on a qualified
replacement and review.

## Phase ledger

| Phase | Represented in the v22 source | Qualification boundary |
| --- | --- | --- |
| M3 — ADC-903, ADC-925–928 | Public Python arbitrary-N fields, cross-component modes, axes/support/state contracts, and one solve/publication authority are represented. | Field accuracy, invalidation, joint-field numerics, MPI behavior, and reuse remain unqualified. |
| M4 — ADC-904, ADC-929–932 | Generic reactions/kernels, typed calls, and DSO-backed component routes are represented. | Complete heterogeneous interaction, derivative/failure-domain, MPI, and cost evidence remains open. |
| M5 — ADC-905, ADC-933–937 | Dim1/2/3 full-spatial-SPD diffusion and reusable RHS/resource contracts are represented; only Dim2 has a package proof. | No v22 numerical diffusion matrix or convergence/oracle run is qualified. |
| M6 — ADC-906, ADC-938–941 | Inline/imported schemes, lifecycle barriers, retry/rollback, and interstage state contracts are represented. | The v22 critical group fails before its first step on per-layout `program_schedule` state divergence. Native continuation, restart, topology transition, and numerical time matrices remain open; unsupported asynchronous coupling may be an explicit refusal. |
| M7 — ADC-907, ADC-942–945 | Generic axes, domains, measures, cardinality, support/state contracts, field maps, and interstage AMR requirements are represented. | Full v22 serial/MPI map and AMR qualification remains open after the focused transfer check above. |
| R1 — ADC-921 | Eight R1 workflows are tracked separately from the 35-group migration manifest; R1 does not require M7 closure. | Complete native, numerical, restart/output, collective, and failure/retry evidence is absent. M8 requires both R1 and M7 evidence, without imposing an ordering between them. |
| M8 — ADC-908, ADC-922–924 | Retention and replacement decisions are recorded above. | The two replacement comparisons and any removal review remain pending. |

The temporal Python patch sequence `dd931464` then `ba95dd3f` is integrated as commits
`37f7c40854b922f0357dce4201c5ec984a17db14` and `faeeee3dfa67601b7e8b7dbc27d0d020470f3d88`
after root diff review. Native execution for this change remains pending.

This documentation change runs no tests or builds and changes no CI, Linear, or PR state.
