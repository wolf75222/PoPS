# Migration verification scope

**Decision recorded 2026-09-08:** retire the stale public promise of the detached
`verification/` campaign as active support. The seven orphan tests remain in the tree as a
historical record. Their archive or deletion is deferred until the current replacement matrix
has qualified. This is a scope and documentation decision; it is not numerical qualification,
M8 closure, or permission to invent missing assets.

This page describes the executable repository-owned routes and their evidence boundary. It is
scoped to ADC-922 equivalence, ADC-923 deletion prerequisites, and ADC-924 completion evidence.

## Boundary

| Area | Recorded scope |
| --- | --- |
| Detached campaign | The former `verification/` manifest, runner, checker, case package, and JSON schemas are absent. The README no longer presents that campaign as an active route. |
| Seven orphan tests | Retain all seven `tests/python/verification/test_*.py` files for now. Do not register them in a new suite, delete them, or add fake assets. A later cleanup may archive or remove them only after the replacement matrix and ADC-922/923 evidence qualify. |
| Numerical qualification | Unavailable until the root integration run completes the full declared M3-M7 matrices at one exact source, native artifact, wheel, and configuration provenance. A source check, unit result, smoke run, or failed attempt does not close a cell. |
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

The active repository-owned source and native routes are these:

| Route | Source of selection | Executable entry points | Evidence boundary |
| --- | --- | --- | --- |
| Source and package contract | `include/pops_headers.manifest`, tracked `python/pops` sources | `python scripts/check_packaging_manifest.py`; `python scripts/run_m3_gate.py --check-only`; `python scripts/run_m4_gate.py --check-only` | Proves manifest/source consistency and gate-manifest structure. It does not prove runtime or numerical behavior. |
| Native C++ | CMake presets and C++ rows in `tests/test_manifest.toml` | `cmake --preset serial`, `cmake --build --preset serial`, `ctest --preset serial --output-on-failure`; use the matching `mpi` preset only for declared MPI/HDF5 cells | Proves only the configured CTest selection and its actual result. A skipped, missing, or unbuilt target remains unavailable. |
| Python native artifact | Explicit `--dim` and optional `--mpi` build choice | `bash scripts/build_python.sh --dim N` with `N` set to `1`, `2`, or `3`; add `--mpi` only when MPI and parallel HDF5 are declared | The build invokes installed-wheel/native checks and `doctor()`. This authenticates an artifact; it does not qualify every runtime or scientific cell. |
| M3-M4 migration gates | `tests/gates/m3_amr_multilayout.toml` and `tests/gates/m4_runtime_io.toml` | `python scripts/run_m3_gate.py --manifest tests/gates/m3_amr_multilayout.toml --build-dir build-mpi`; `python scripts/run_m4_gate.py --manifest tests/gates/m4_runtime_io.toml --build-dir build-mpi` | The full gate must actually run with its native and MPI prerequisites. `--check-only` is a source check only. |
| M5-M7 and final integration | Rows and rank declarations in `tests/test_manifest.toml`, plus the migration integration matrix | Root-owned exact-source/native/wheel matrix runs and retained receipts; `scripts/run_final_gate.py` is the available selected-dimension release-gate entry point | A receipt must identify source SHA, native/wheel identity, dimensions, backend, ranks, levels/blocks, restart/output cells, and pass/fail/skip reasons. No unrun row is inferred green. |
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
M3-M7 matrix cells into passes or recreate the detached `verification/` campaign.

## Current receipt status

The evidence ledger in
[`migration_m8_replacement_inventory.md`](migration_m8_replacement_inventory.md) retains the
full hashes and provenance. The relevant status boundary is:

| Receipt family | Exact source/native context | Result | Interpretation |
| --- | --- | --- | --- |
| V7 targeted MPI | `5766bdd` source/native; three targeted entry points | History and output passed; generated FE field-identity control failed; remaining matrix rows were not run | Historical targeted evidence, not full M7 qualification |
| V8b Python MPI | `d114cb42096ec6da3549ae3cb62035cf8aeb6df6` with wheel `5b6581e252dcacc5c9ee82feeb27acd6e2d6d143ebb8d5b8099f62722acf8f67` | 7/10 passed; regrid-on-restart, external AMR field topology, and strict Uniform history reader failed | Executed attempt; pending repairs and native rerun |
| V8b temporal | Same `d114cb` source/native and wheel | 9/10 passed; split-failure envelope test failed; extracted metrics are marked `operation_counts_only_no_performance_claim` | Temporal attempt, not complete qualification |
| V8b physical support | Same `d114cb` source/native and wheel | 0/6 passed; all binds lacked required `sample_grad_x` | No physical lifecycle or restart result was executed |
| V8b continuation | Same `d114cb` source/native and wheel | 0/2 passed; strict Uniform reader rejected continuation archive members | No continuation equivalence result was executed |
| Later root source | Caller scope inspected at `cd9173f`; later native gate had no attached receipt when this page was written | Open | Do not transfer V7/V8b results to `cd9173f` or claim M8 closure |

The pending repairs reported for the V8b failures are separate source states and need native
reruns before they can affect this status. A later clean Dim2 wheel build/prove/doctor result is
artifact evidence only; it does not replace the full migration matrix.

## Retirement gate

The seven historical tests become removable only when all of the following are recorded in the
replacement evidence set:

1. The full declared M3-M7 feature/configuration matrix has run at one current exact source and
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
