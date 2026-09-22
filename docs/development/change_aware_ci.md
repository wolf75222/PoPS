# Change-aware CI routing

The CI workflow plans a bounded PR run from the complete change set and then executes that plan.
It reduces repeated work when the dependency impact is known, while keeping an explicit complete
fallback whenever the impact cannot be proved. A plan is a routing decision; it is not a test
result, a performance measurement, or scientific qualification.

## One diff and one plan

The `changes` job records `ci-input/changed-files.txt`. For a pull request,
[`scripts/ci_plan.py`](../../scripts/ci_plan.py) first computes the merge base of the PR base and
head revisions, then runs the equivalent of:

```text
git diff --name-status -z --find-renames <merge-base> <head>
```

The NUL-delimited status stream preserves unusual path names. A rename or copy contributes both
the old and new paths, and a deletion contributes the deleted path. If the base or head cannot be
resolved, the diff fails, or a path contains a newline or carriage return that cannot safely enter
the newline-delimited selector inputs, the planner writes an unresolved sentinel and selects the
complete matrix. Non-PR events do not need a path diff: they are complete plans by policy.

`set-mode` consumes that one input and publishes both a `ci-plan` artifact and step outputs. The
artifact contains the JSON plan, the C++ and Python selector inputs, and the architecture-test
input; the C++ shards, Python shards, and architecture gate download those files. MPI, OpenMP,
and the aggregate gate consume the published route outputs, including required-job booleans,
stable matrices, and selected native dimensions, and use their own manifest projections. Prewarm,
build, and cache jobs also branch on these outputs and publish or consume their own authenticated
build artifacts. No job independently rediscovers the PR diff.

## What owns each decision

The separation below keeps job policy from test ownership. A new route rule belongs in the policy;
a new test, label, rank, or dependency belongs in the manifest or graph authority.

| Question | Authority | Result |
| --- | --- | --- |
| Should a path force the complete plan, MPI, architecture checks, or metadata-only handling? | [`scripts/ci_components.toml`](../../scripts/ci_components.toml) and `ci_plan.py` | Component flags and a full-fallback reason |
| Is an input intrinsically broad, such as a binding adapter or executable example? | The central component policy | Complete plan, even if a lower-level selector can project a label group |
| Which C++ suites and Python suite roots exist, and which tests own MPI ranks or orchestrators? | [`tests/test_manifest.toml`](../../tests/test_manifest.toml) | Manifest-owned targets, files, labels, and MPI contracts |
| Which tests import a changed Python module? | [`scripts/ci_import_closure.py`](../../scripts/ci_import_closure.py), used by `ci_select_tests.py` | Reverse Python import and cross-test closure |
| Which C++ suites include a changed header? | [`scripts/ci_include_graph.py`](../../scripts/ci_include_graph.py), used by `ci_select_tests.py` | Per-suite include impact, including generated-code consumers |
| Which tests consume a changed runtime translation unit? | `src/CMakeLists.txt`, `tests/CMakeLists.txt`, and `ci_select_tests.py` | Runtime object-library owners and their transitive linked test consumers |
| Which native-loader tests are affected by a code-generation emitter? | Python emitter/import closure plus the C++ include graph | Codegen and native-loader label groups and their manifest tests |

The policy file therefore does not duplicate a test list, and the workflow does not become a second
manifest. `ci_select_tests.py` reads the existing manifest and graph authorities, validates their
inventories, and records per-file reasons in the plan for review.

## PR selection and conservative boundaries

For a normal pull request, C++ impact is compositional: every changed file contributes its own
mapped impact and the selected targets are the union. A project header uses its include closure;
a runtime C++ source uses the object libraries declared by CMake and their linked consumers; a
binding adapter matches the central build-and-CI policy and forces the complete plan; a codegen
emitter selects native-loader consumers; and a manifest-owned C++ test selects its target
directly. The selector's binding-label projection remains useful for dedicated label jobs, but it
does not narrow a normal plan. Python selection combines direct manifest
tests, reverse import closure for `python/pops/*.py`, area labels, native-header dependencies,
and cross-test closure. A small mapped change also receives the configured smoke backstop.

The narrow cases are deliberate:

- Markdown documentation, tutorials, and other metadata changes have no functional C++ or Python
  test selection. Executable documentation inputs such as `docs/*.py`, `docs/*.sh`, `docs/*.cpp`,
  `docs/*.hpp`, `tutorials/*.py`, and `tutorials/*.sh` match the executable-documentation rule
  and force the complete plan, even though the documentation rule also matches their directories.
  A direct edit to one `tests/python/architecture/test_*.py` file selects that source check for
  the source-only architecture job.
- A mapped functional test edit selects its manifest owner. The architecture gate remains
  source-only; it does not import the functional suite.
- Changes to native or Python component source areas marked `architecture` select the complete
  architecture directory. These are cross-cutting source fences, even when the functional test
  selection is otherwise bounded.

The planner fails safe to the complete matrix for build-system and workflow inputs, shared core or
support inputs, an unknown or unmapped path, an unreadable import/include graph, a runtime source
that is absent from the central object-library map, or a deleted or missing non-metadata source.
An unresolved dependency in either language escalates the whole plan; it cannot suppress the other
language or a capability lane. This preserves coverage when a rename, deletion, new build input,
or incomplete graph makes a subset untrustworthy.

The existing large migration PR is expected to select the complete matrix because the CI planner,
policy, selector, and workflow files are themselves broad build-and-CI inputs. The optimization
benefits later focused PRs after those controls are established. It uses the accumulated PR diff
against the merge base, not merely the last commit.

## Representative plan shapes

These are verified planning outputs against the current source tree. Counts describe selected
files or C++ targets and route flags; they do not say that any test ran or that CI passed.

| Changed input | C++ targets | Python files | Architecture files | Route details |
| --- | ---: | ---: | ---: | --- |
| `tests/python/unit/runtime/test_capacity_limits.py` | 0 | 1 | 0 | MPI and OpenMP off |
| `tests/cpp/unit/mesh/test_box.cpp` | 1 | 0 | 0 | C++ prewarm off; MPI and OpenMP off |
| `tests/python/architecture/test_ci_shard_binpack.py` | 0 | 0 | 1 | Native jobs 0; source-only architecture check |
| `include/pops/numerics/elliptic/polar/polar_tensor_operator.hpp` | 6 (3 consumers + 3 shared smoke backstop) | 0 | 98 | MPI and OpenMP off; full source-only architecture fence |
| `tests/python/unit/runtime/test_capacity_limits.py` plus `unmapped/input.dat` | 191 | 550 | 98 | Complete plan; MPI and OpenMP on |

## Shards, prewarming, and the dedicated cache check

The planner keeps fixed partition identities and publishes only nonempty matrix entries. C++
targets are packed into the fixed thirteen-way partition, and Python files into the fixed thirty
eight-way partition, using deterministic duration-weighted bin packing. Empty bins are omitted from
the GitHub matrix, but a nonempty bin keeps its original index; indices are never renumbered to
fit one PR. Python's verification step reconstructs the same partition and fails unless every
selected file appears exactly once, with the explicitly excluded dedicated file accounted for.
C++ shards authenticate their selected `cpp-target:*` CTest labels before running cases.

The synthetic AMR Program loader's nine cases consume six source-built fixture variants. CMake
builds each variant once with the same native compiler, ABI, Kokkos and MPI contract, then each
case copies and authenticates its binary in fresh local files before constructing fresh runtime
state. This removes eleven compiler invocations from CTest without dropping cases or changing
the seven-minute test watchdog. The target's build weight includes all six fixture builds;
its updated build and test weights remain explicit estimates until complete CI receipts arrive.

The `python_dimensions` output drives the serial native package build and prewarm matrices. They
contain the selected declared dimensions, limited to native Dim1 and Dim2. With the current
manifest, a selection with no Dim1 file uses Dim2; selecting a Dim1 file adds Dim1. Shards then
download only the package artifacts for those dimensions.

Standalone CTest contracts, including packaging and build checks, run only on shard 0 of a complete
C++ plan. Generated component-catalog checks and C++ duration/catalog integrity checks remain
independent source-only architecture checks.

The compile-cache test is excluded from ordinary Python shards and runs once in its own job only
when it is selected. The job proves the codegen, manifest, cache-ABI, and generated-program
compile/bind path with its isolated semantic cache; it is not duplicated into every shard. The
Python native module is built once per selected dimension and downloaded by the shards, while
runtime loader tests retain their own cache use.

C++ prewarming is conditional on an actual selected target consuming one of the shared runtime
object libraries. If no selected C++ target is an object-library consumer, the C++ prewarm job is
skipped and the final shard gate accepts that explicit, planned skip. When it runs, prewarming
publishes only content-addressed compiler-cache entries and an exact compile contract. Each final
shard reconfigures the official graph, authenticates the contract, compiles its own executables,
links them, and runs its own tests. A prewarm cache is not a linked binary and cannot hide a missing
object, changed ABI, or failed target build.

## MPI and complete overrides

MPI is a coherent capability lane. The planner requests it when the plan is complete, when a
changed component is marked `mpi`, or when a changed Python module's import closure reaches a
manifest-owned MPI entrypoint or MPI orchestrator. The lane then uses the manifest's exact C++
rank launches, Python MPI entrypoints, serial orchestrators that create their own rank worlds, and
collective I/O checks. An unrelated mapped change does not create a partial MPI fragment.

The `collective-field-protocols` policy covers the elliptic `interface`, `linear`, and `amr`
header families and requests MPI for their collective contracts. The isolated
`include/pops/numerics/elliptic/polar/polar_tensor_operator.hpp` path is intentionally outside
that MPI policy, so its bounded native selection does not start the MPI lane.

The planner selects the complete suite for every non-PR event, for a PR carrying `ci-full` or
`ci-kokkos`, and for the `force_full` workflow-call override. This covers master pushes, nightly
schedule runs, manual dispatch, and release workflow calls according to their event policy.
The PR workflow still runs its required routing and aggregate checks for metadata-only changes;
expensive jobs may be skipped only when the shared plan says they are optional.

The `gate` aggregate is fail-closed: `changes` and `set-mode` must succeed, every routed job must
succeed, and an optional job may be skipped only as the result of that successful routing decision.
The presence of a plan, an empty shard, a cache hit, or a local selector output never claims that
tests ran or that GitHub CI is green.

## Reproduce a plan locally

Run the planner from the repository root with temporary outputs outside the checkout. Replace
`origin/master` with the exact PR base ref when needed:

```bash
plan_tmp="$(mktemp -d "${TMPDIR:-/tmp}/pops-ci-plan.XXXXXX")"
trap 'rm -rf "$plan_tmp"' EXIT

python3 scripts/ci_plan.py changes \
  --event-name pull_request \
  --base origin/master \
  --head HEAD \
  --output-file "$plan_tmp/changed-files.txt"

python3 scripts/ci_plan.py plan \
  --event-name pull_request \
  --changed-files "$plan_tmp/changed-files.txt" \
  --output-dir "$plan_tmp/ci-plan" \
  --github-output "$plan_tmp/github-output"
```

This reproduces the changed-path and selection artifacts only. It does not configure, compile,
link, or execute any test, and its counts are not CI or performance evidence.
