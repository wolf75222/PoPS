# Verification

**Plan (v1.3):** [PoPS_VERIFICATION_VALIDATION_BENCHMARK_PLAN_MONOREPO_v1.3.md](PoPS_VERIFICATION_VALIDATION_BENCHMARK_PLAN_MONOREPO_v1.3.md)

## What this tree is

This directory is the repo-local scientific campaign. It is not installed in the
`pops` wheel (`wheel.packages = ["python/pops"]` in `pyproject.toml`).

It is distinct from the fast-test catalogue (`tests/test_manifest.toml`) and
from the performance harness (`benchmarks/manifest.toml`). Those three manifests
keep separate authority; do not copy resolutions, oracles, or thresholds between
them.

`verification/manifest.toml` catalogues the scientific cases (PH, TR, EU, PO,
TM, CP, AM, GE, RB, IF, PF). Native `pops.run` is compiler/Kokkos-gated. One
native artifact compiles exactly one spatial dimension. Timed PF work delegates
to `benchmarks/manifest.toml`; this tree does not invent a second harness.

## Layout

```text
verification/
├── README.md                 # this file
├── PoPS_VERIFICATION_VALIDATION_BENCHMARK_PLAN_MONOREPO_v1.3.md
├── __init__.py               # repo-local package marker; not in the wheel
├── manifest.toml             # scientific campaign source of truth
├── cases/                    # PH, TR, EU, PO, TM, CP, AM, RB
├── machines/                 # leftover ROMEO helpers; not the campaign runner
└── pops_verify/              # private post-process helpers
```

`pops_verify` modules (filename -> role):

| File | Role |
|---|---|
| `__init__.py` | Package marker for the private oracle-comparison helpers |
| `campaign.py` | Expand selected cases into single-dimension jobs |
| `case_authoring.py` | Shared Case/Program construction for catalogued cases |
| `cell_averages.py` | Analytic cell averages of an external oracle on Cartesian cells |
| `conservation.py` | Discrete conservation residual from an already-reduced balance |
| `convergence.py` | Observed order from a resolution series of already-computed errors |
| `interface_error.py` | Coarse-fine interface-band errors from an already-sampled field |
| `leaf_reference_errors.py` | AMR leaf-only oracle norms |
| `native_diagnostics.py` | Attach public `pops.diagnostics` reductions to a `ConsumerGraph` |
| `official_benchmark.py` | Invoke `benchmarks/manifest.toml` (`arith_halo`, `scalar_mg`) |
| `phase.py` | Phase and frequency diagnostics from an already-sampled probe |
| `provenance.py` | Build and validate a per-run `pops.verification.provenance.v1` document |
| `reference_errors.py` | Volume-weighted errors of a numerical field against an external oracle |
| `report.py` | Campaign Markdown/CSV/JSON renderer from an already-built summary |
| `symmetry.py` | Symmetry diagnostics from an already-sampled field |
| `visualization/` | Phase 8 figures, visual manifests, and completeness gates |

## Manifest

`verification/manifest.toml` is the scientific source of truth. Schema
`pops.verification.manifest.v1`. `max_nodes = 2`. Current capabilities include
`exact_native_dimension = true`, `cartesian_system_runtime = true`,
`polar_system_runtime = false`, and an AMR baseline of 3 levels with ratios
`[2, 2]`.

Validate with:

```bash
python scripts/check_verification_manifest.py
```

The checker loads `schemas/verification_manifest.v1.json` and does not compile,
bind, run cases, or launch jobs.

## Runner

`scripts/run_verification.py` validates the manifest, expands selected cases
into single-dimension jobs, and writes `plan.json` under `--output`. With
`--execute` it calls each job's public `run_native` in-process. It does not
spawn MPI ranks: launch the same script under `srun`/`mpiexec` when the
artifact proves `MPI_COMM_WORLD`.

```bash
python scripts/run_verification.py \
  --suite pr \
  --dimensions 1 \
  --max-nodes 2 \
  --output build/verification/pr-<sha>

python scripts/run_verification.py \
  --suite pr \
  --dimensions 1 \
  --max-nodes 2 \
  --output build/verification/pr-<sha> \
  --execute
```

`--max-nodes > 2` is refused.

Allowed `--suite` values are `pr`, `nightly`, `weekly`, `release`, and
`two_node`. `--dimensions` is a comma-separated list of `1`, `2`, and/or `3`.

## Native dimension

One native artifact compiles exactly one spatial dimension. `POPS_NATIVE_DIM`
is `1`, `2`, or `3`. There is no fallback to another native extension.

Dim1, Dim2, and Dim3 reports stay distinct. A campaign that requests dimensions
other than the loaded artifact is refused.

## Case folder contract

Each directory under `verification/cases/` starts with a `README.md` contract
block. The scientific manifest stays autonomous; it does not reuse the
fast-test catalogue.

| Field | Required content |
|---|---|
| Identifier | Stable case id |
| `verification_kind` | `code-verification`, `solution-verification`, `physical-validation`, `robustness`, or `infrastructure` |
| `evidence_status` | `required`, `capability-gated`, `reproduction-candidate`, or `established-reproduction` |
| Equations | System actually solved, conserved variables, and sources |
| Oracle | Analytic formula, MMS, dispersion, invariant, published reference, or converged numerical solution |
| Domain and boundaries | Dimension, bounds, periodic / Dirichlet / Neumann / reflecting |
| Parameters | Values, units, or `dimensionless` |
| Native dimensions | Required `POPS_NATIVE_DIM` values; one run uses exactly one |
| Required capabilities | Cartesian/polar, uniform/AMR, total levels, subcycling, Poisson, MPI, HDF5, Kokkos space |
| Configurations | Resolutions, blocks, CFL, integrator, flux, reconstruction, AMR, interface placement |
| Diagnostics | Norms, conservation, phase, frequency, symmetry, residual, positivity, cost |
| Thresholds | Tolerances and minimum order, with justification |
| Proves | What the assertions actually establish |
| Does not prove | What remains out of scope |
| Resources | Resolutions, MPI ranks, threads, GPU, nodes, memory, and wall-time limit |
| Provenance | Unique monorepo SHA, dirty flag, PoPS version, catalog/header/native-leaf digests, compiler, Kokkos, MPI, machine, and Slurm job |

Typical files:

```text
README.md
run.py
exact.py
analyze.py
case.toml
configs/
reference/
```

`exact.py` never reads PoPS output to build its reference. `analyze.py` never
mutates simulation fields.

## Outputs

Generated fields, snapshots, and figures go to
`build/verification/<case-id>/<run-id>/`. Do not store full fields under
`verification/`.

Campaign artifacts, when rendered, are:

- `REPORT.md`
- `summary.json`
- `coverage.csv`
- `failures.csv`

The runner today writes only `plan.json`.

## Schemas

Five versioned contracts live under `schemas/`:

| File | Instance |
|---|---|
| `schemas/verification_manifest.v1.json` | parsed `verification/manifest.toml` |
| `schemas/verification_metrics.v1.json` | per-run `pops.verification.metrics.v1` |
| `schemas/verification_provenance.v1.json` | per-run `pops.verification.provenance.v1` |
| `schemas/verification_report.v1.json` | campaign `pops.verification.report.v1` |
| `schemas/verification_visuals.v1.json` | Phase 8 visual contracts and `visual_manifest.json` |

An incompatible change requires a new schema version.

`jsonschema` is a test/dev extra (`[project.optional-dependencies] test`). It
is not part of the `pops` wheel. Matplotlib is an optional `viz` extra used
only by `scripts/render_verification_visuals.py`.

```bash
python scripts/check_verification_visuals.py
python scripts/render_verification_visuals.py \
  --run build/verification/<case-id>/<run-id> \
  --formats png,pdf,svg \
  --strict
python scripts/render_verification_visuals.py \
  --examples build/verification/phase8-fixtures \
  --formats png,pdf,svg
```

Generated fixture plots belong under gitignored `build/verification/`. Do not
commit PNG/PDF/SVG duplicates. Every fixture figure is labeled
`DETERMINISTIC FIXTURE` and uses a `fixture:` provenance SHA.

## Helpers

`verification/pops_verify/` post-processes already-sampled or already-reduced
arrays. It does not replace `pops.diagnostics`. Accepted numerical state is
reduced by the public descriptors; these modules compare oracles, derive
orders, residuals, phase, symmetry, and provenance after that reduction.

There is no solver in these modules. They do not call `pops.run`.

## ROMEO

Launch profiles live under `verification/machines/`. Scientific one- and
two-node jobs remain capability-gated by the machine and the compiled native
dimension. Manifest planning and helper tests run locally.
