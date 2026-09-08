# Migration verification scope

**Decision recorded 2026-09-08:** retire the stale public promise of the detached
`verification/` campaign as active support. The seven orphan tests remain in the tree as a
historical record. Their archive or deletion is deferred until the current replacement matrix
has qualified. This is a scope and documentation decision; it is not numerical qualification,
M8 closure, or permission to invent missing assets.

**Milestone status:** current M3-M8 replacement qualification remains in progress. Historical
M0 evidence and its limitations are preserved; source/package authentication, targeted diagnostics,
and invalidated runs do not turn an unqualified matrix cell green.

This page describes the executable repository-owned routes and their evidence boundary. It is
scoped to ADC-922 equivalence, ADC-923 deletion prerequisites, and ADC-924 completion evidence.

## Boundary

| Area | Recorded scope |
| --- | --- |
| Detached campaign | The former `verification/` manifest, runner, checker, case package, and JSON schemas are absent. The README no longer presents that campaign as an active route. |
| Seven orphan tests | Retain all seven `tests/python/verification/test_*.py` files for now. Do not register them in a new suite, delete them, or add fake assets. A later cleanup may archive or remove them only after the replacement matrix and ADC-922/923 evidence qualify. |
| Numerical qualification | Unavailable until the root integration run completes the full declared M3-M8 matrices at one exact source, native artifact, wheel, and configuration provenance. A source check, unit result, smoke run, targeted diagnostic, or failed attempt does not close a cell. |
| Supported configurations | This decision changes documentation scope only. Existing dimension, backend, MPI-rank, level, block, restart, output, and refusal envelopes remain unchanged until each replacement route has its own evidence. |
| GPU and cluster claims | No local CPU or MPI result establishes GPU, multi-node, or cluster support. Such cells remain unavailable unless their own environment and receipt are present. |

## Baseline and current absence

The baseline `51bcdc2e2399dbc922c58c3008ae30a11332f1d8` and the later root source snapshot
`cd9173f8543d49c550eb49686818afc33bb0e5a9` contain the same seven test files. Neither tree
contains any of the following paths:

```
verification/
verification/manifest.toml
scripts/run_verification.py
scripts/check_verification_manifest.py
schemas/verification_*.json
```

A reachable-history scan for these test paths returns only the baseline `51bcdc2` entry; no
committed campaign assets were found. The current workflow and `scripts/ci_select_tests.py`
select manifest-owned suites and contain no registration for `tests/python/verification`. The
project-wide `pyproject.toml:testpaths = ["tests/python"]` can still discover those files, so
collection visibility does not make them an active verification suite.
The manifest-driven selector has no verification suite: `python scripts/ci_select_tests.py python
--changed-files /dev/null --force-all` reports 511/511 manifest-owned Python files, and none of the
seven paths is selected. This confirms the files are outside the current normative selector
without deleting their historical records.

The seven files and their missing dependencies are:

| Retained file | Missing dependency referenced by the file | Classification |
| --- | --- | --- |
| `tests/python/verification/test_run_verification.py` | `scripts/run_verification.py`, `verification/manifest.toml`, and the former `verification/cases/...` package | Historical planner/runner contract |
| `tests/python/verification/test_check_verification_manifest.py` | `scripts/check_verification_manifest.py` and `verification/manifest.toml` | Historical manifest-checker contract |
| `tests/python/verification/test_reference_errors.py` | `verification.pops_verify.reference_errors` | Historical oracle helper |
| `tests/python/verification/test_verification_report_schema.py` | `schemas/verification_report.v1.json` | Historical report schema contract |
| `tests/python/verification/test_verification_provenance_schema.py` | `schemas/verification_provenance.v1.json` | Historical provenance schema contract |
| `tests/python/verification/test_verification_metrics_schema.py` | `schemas/verification_metrics.v1.json` | Historical metrics schema contract |
| `tests/python/verification/test_verification_manifest_schema.py` | `schemas/verification_manifest.v1.json`, `verification/manifest.toml`, and former case paths | Historical manifest schema contract |

The historical M0 contract remains unchanged. Its statements that the normative campaign is
unavailable and that a smoke or reduced test cannot substitute for it are still the evidence
boundary. Other historical documents that mention the former paths require a separate
documentation reconciliation; this scope change does not rewrite their frozen receipts.

## Current executable routes

The repository-owned routes and historical gate labels are these:

| Route | Source of selection | Executable entry points | Evidence boundary |
| --- | --- | --- | --- |
| Source and package contract | `include/pops_headers.manifest`, tracked `python/pops` sources | `python scripts/check_packaging_manifest.py` | Proves manifest/source consistency. It does not prove runtime or numerical behavior. |
| Historical ADC-672--678 and ADC-679--687 gates | `tests/gates/m3_amr_multilayout.toml` and `tests/gates/m4_runtime_io.toml` | `python scripts/run_m3_gate.py --check-only` and `python scripts/run_m4_gate.py --check-only`, or their full historical runners | These are older AMR/multi-layout and runtime/I/O contracts. Their source checks do not define current migration M3 fields or M4 interactions and cannot qualify those milestones. |
| Current migration M3-M4 fixtures | `docs/development/migration_m3_m6_contract.md`, current Python unit/integration fixtures, and `tests/test_manifest.toml` | M3 field-problem/observation fixtures such as `test_public_field_problem.py`, `test_public_field_reuse.py`, and `test_public_field_consumers.py`; M4 interaction/typed-native fixtures such as `test_joint_interaction_codegen.py`, `test_native_interaction_matrix.py`, `test_native_call_compiled.py`, and `test_external_component_package.py` | These are the current migration contracts. They require execution with the matching native artifact and declared oracles; source presence or manifest selection is not qualification. |
| Native C++ | CMake presets and C++ rows in `tests/test_manifest.toml` | `cmake --preset serial`, `cmake --build --preset serial`, `ctest --preset serial --output-on-failure`; use the matching `mpi` preset only for declared MPI/HDF5 cells | Proves only the configured CTest selection and its actual result. A skipped, missing, or unbuilt target remains unavailable. |
| Python native artifact | Explicit `--dim` and optional `--mpi` build choice | `bash scripts/build_python.sh --dim N` with `N` set to `1`, `2`, or `3`; add `--mpi` only when MPI and parallel HDF5 are declared | The build invokes installed-wheel/native checks and `doctor()`. This authenticates an artifact; it does not qualify every runtime or scientific cell. |
| M5-M8 and final integration | Rows and rank declarations in `tests/test_manifest.toml`, plus the migration integration matrix | Root-owned exact-source/native/wheel matrix runs and retained receipts; `scripts/run_final_gate.py` is the available selected-dimension release-gate entry point | A receipt must identify source SHA, native/wheel identity, dimensions, backend, ranks, levels/blocks, restart/output cells, and pass/fail/skip reasons. No unrun row is inferred green. |
| Performance measurement | `benchmarks/manifest.toml` and its protocol | The benchmark harness under `benchmarks/` with its declared warmups, repetitions, fences, barriers, and equation pairing | Operation counts, source checks, and unit tests are not performance evidence. GPU/device and multi-node claims require their own declared environment and receipt. |

For a release-gate replay, the command has to write evidence outside the checkout and name the
exact wheel dimensions:

```bash
python scripts/run_final_gate.py \
  --evidence /tmp/pops-final-gate.json \
  --dim 2 \
  --wheel-dim 2
```

That entry point is a release gate for the selected configuration. It does not convert omitted
M3-M8 matrix cells into passes or recreate the detached `verification/` campaign.

## Current receipt status

The evidence ledger in
[`migration_m8_replacement_inventory.md`](migration_m8_replacement_inventory.md) retains the
full hashes and provenance. Elapsed durations are receipt-recorded values and were not rerun or
independently confirmed in this documentation lane. The relevant status boundary is:

| Receipt family | Exact source/native context | Result | Interpretation |
| --- | --- | --- | --- |
| V7 targeted MPI | `5766bdd` source/native; three targeted entry points | History and output passed; generated FE field-identity control failed; remaining matrix rows were not run | Historical targeted evidence, not full M7 qualification |
| V8b Python MPI | `d114cb42096ec6da3549ae3cb62035cf8aeb6df6` with wheel `5b6581e252dcacc5c9ee82feeb27acd6e2d6d143ebb8d5b8099f62722acf8f67` | 7/10 passed; regrid-on-restart, external AMR field topology, and strict Uniform history reader failed | Executed attempt; pending repairs and native rerun |
| V8b temporal | Same `d114cb` source/native and wheel | 9/10 passed; split-failure envelope test failed; extracted metrics are marked `operation_counts_only_no_performance_claim` | Temporal attempt, not complete qualification |
| V8b physical support | Same `d114cb` source/native and wheel | 0/6 passed; all binds lacked required `sample_grad_x` | No physical lifecycle or restart result was executed |
| V8b continuation | Same `d114cb` source/native and wheel | 0/2 passed; strict Uniform reader rejected continuation archive members | No continuation equivalence result was executed |
| V9 native expanded (15 targets) | `candidate-v9-native-expanded-result.json` and matching XML; source `cd9173f8543d49c550eb49686818afc33bb0e5a9`; source unchanged during run | 119/123 passed, 2 failed, 2 skipped; 234.972361s | Valid source-unchanged receipt at `cd9173f`, but two fixture fixes landed later; rerun is required for current qualification |
| V10 temporal (10) | `candidate-v10-temporal-ten-result.json`; source `bb109c2f7695cfee97b73079607a06cc16578156`; source/header/wheel proofs unchanged | 10/10 passed, 0 skipped; 176.490187s | Successful temporal workflow slice; not full M3-M8 qualification |
| V10 continuation (2) | `candidate-v10-continuation-two-result.json`; same `bb109c2` source/native/wheel provenance | 2/2 passed, 0 skipped; 110.448827s | Successful continuation slice; restart/output qualification remains open |
| V10 physical support (6) | `candidate-v10-physical-six-result.json`; same `bb109c2` source/native/wheel provenance | 5/6 passed, 1 failed, 0 skipped; 218.594530s | The rejected-solve case failed with `solve_linear failed: iteration_limit`; later `RejectAttempt` policy repair is not natively validated |
| V10 explicit AMR (6) | `candidate-v10-explicit-amr-six-result.json`; same `bb109c2` source/native/wheel provenance | 0/6 passed, 6 failed, 0 skipped; 214.280457s | Pending repair for the flux-producing AMR expression budget; rerun required |
| V10 shared-interface (8) | `candidate-v10-shared-eight-result.json`; same `bb109c2` source/native/wheel provenance; source/header/wheel proofs unchanged | 6/8 passed, 2 failed, 0 skipped; 338.233341s | Two restart-interface fragments exceed the artifact budget; repair is in progress |
| V10 native expanded (17) | `candidate-v10-native-expanded-result.json`; source `bb109c2`; original receipt records `source_unchanged=false` | 130/133 passed, 1 failed, 2 skipped; 429.883088s | Invalidated diagnostic only after a formatter accident changed source during the run. Preserve the receipt; the main source was later restored by `candidate-v10-main-format-restoration.json`. |
| FAC targeted diagnostic | `fac-physical-ghost-probe/result.json`; commit `70de36c`, integrated in staging `ec2aa6a`; MPI2 targeted run | 8 targeted cases passed in 54.102s with original assertions/tolerances | Targeted diagnostic only; it is not a full integrated FAC/native/MPI gate or M3-M8 qualification |

The V9 native15 receipt is a valid source-unchanged execution at `cd9173f`, but two fixture fixes
landed after that run, so it needs a fresh execution before it can support current qualification.
The V10 receipts are immutable workflow evidence at `bb109c2`, with source/header/wheel proof
identities unchanged: temporal and continuation are
successful slices, while physical, explicit AMR, and shared-interface rows still have recorded
failures. The V10 native17 result remains diagnostic only because its original receipt records
`source_unchanged=false`; the restoration receipt confirms the later source restoration but does
not retroactively validate that run. The FAC result is a bounded targeted diagnostic, not a full
gate. A later clean Dim2 wheel build/prove/doctor result is artifact evidence only; it does not
replace the full migration matrix.

## Retirement gate

The seven historical tests become removable only when all of the following are recorded in the
replacement evidence set:

1. The full declared M3-M8 feature/configuration matrix has run at one current exact source and
   native/wheel provenance, including numerical, restart, output, collective, and refusal
   oracles.
2. Every promised capability that the former campaign covered has a current owner in
   `tests/test_manifest.toml`, a gate manifest, or an explicitly documented unavailable/refused
   cell. No smoke or reduced case fills a missing cell.
3. ADC-922 records old-to-new equivalence or a scoped changed-behavior oracle; ADC-923 names the
   exact obsolete files and updates collection, packaging, and CI references; ADC-924 retains
   diagnostics, examples, and the completion record.
4. Archive/removal is performed as a separate change after the receipts are immutable. If the
   product scope reopens the former campaign promise, restore its complete runner, package,
   manifest, schemas, and cases before attempting qualification.

Until those conditions hold, the correct status is **historical orphan assets retained;
replacement qualification open**.
