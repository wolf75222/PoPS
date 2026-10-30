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

The MPI native build runs two sequential phases with separate 28-minute watchdogs inside a
60-minute native-build cap; the complete job remains capped at 180 minutes. The manifest projection puts
`test_program_runtime`, `test_amr_synthetic_program_loader_transaction`,
`test_mpi_system_gather_scatter`, and `test_mpi_system_io_gather` in a strictly sequential compiler
phase. The second phase builds the exact complement with parallelism four and the preset's
unchanged single-slot heavy-TU pool. New manifest targets enter the complement automatically;
missing anchors or a non-disjoint/incomplete partition fail planning. Python module linking and
M4 compilation retain their separate 14-minute and 10-minute watchdogs. Their enclosing steps have
20-minute and 12-minute caps, so the three build steps together permit at most 92 minutes within
the unchanged job cap; previously all build commands shared one 60-minute step.

This partition addresses the PR #680 `4adfddf` run, which reached 164 of 173 build tasks and linked
80 of 84 targets before its single 28-minute watchdog expired. The program-runtime completion
window was about 408 seconds, with sampled host memory headroom falling to 452 MiB. The loader's
720-second build weight includes its six newly precompiled fixtures. Reserving 180 seconds for
each of the four unfinished targets gives scheduling estimates of roughly 25 and 26 minutes for
the two phases. That reserve is an explicit assumption, about twice the largest observed ordinary
MPI completion window; it is not a measured duration. The earlier run benefited from compiler
caches, and completion windows can overlap. These estimates do not establish cold-build safety or
CI success: the complete build and unchanged 121-launch MPI test inventory still need an actual
GitHub receipt.

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

## Master's tutorials (MG8)

`mg8-tutorials` runs on every plan that needs the Python lane, complete plans included, and is
required then: a change of a route that a tutorial uses runs it (MG8-1). Its three lanes, `serial-dim2`,
`mpi-dim2` (two ranks) and `serial-dim3`, each hold one installed PoPS: master's MPI module compiles
`MPI_COMM_WORLD` artifacts only, and tutorial 16 needs Dim3. A lane builds master `bb9aea0c`
(cached by commit) and this tree, then `scripts/run_mg8_tutorials.py run` executes each of the 17
tutorials of `docs/tutorials/scalar_advection` from a copy of the directory, once with master's
text against master and once with this tree's text against this tree, on the same runner. A probe in the tutorial process
records, after every `pops.run` call, the clock, the hierarchy and every state level, through the
public methods that the tutorials call. `compare` then checks the return codes, those states
bitwise, the printed and console-monitor lines, and the outputs (NumPy archives, ParaView series,
figures), under `tests/gates/mg8_master_tutorials.toml`. Only identity digests and the run
directory are not compared. The reference is rebuilt on each runner because NumPy's and glibc's
elementary functions dispatch on CPU features: arrays stored on another machine would compare two
machines, not two trees.

A difference of a result is recorded in the contract, by its exact name. Each `[[cause]]` names
one change of this tree, the commits that introduced it, the issue that owns it and the evidence
that attributes it; each `[[divergence]]` lists, for one tutorial, the exact checks that differ,
the causes they come from and its attribution: `reproduced`, with the verdict of a build of this
tree with master's semantics for those causes that gives master's bits, or `inferred`. Only results
can be recorded: a state, the clock, the hierarchy or the report after a `pops.run` call, a printed
or monitor line, an output file or series. A failure never is: a tutorial that fails, a missing
run, an observation that the probe could not make, a different number of `pops.run` calls or
printed lines, and a missing, different or unlisted set of output files always fail the job, and no
tutorial is skipped (CF14).

The list only shrinks (MG6). `compare` fails on a difference that no divergence lists and on a
listed check that no longer differs. On every event, the `changes` job runs `run_mg8_tutorials.py
check --check-base <base> --verify-commits`: it fails when the ledger records a check, a cause or a
pair of tutorial and cause that the base does not, when a cause changes its issue, when an output
stops being compared, when identity lines grow or when anything outside the ledger changes, and
when a cited commit is not in the history or changes no code that the tutorials execute
(`include/`, `src/`, `python/`, `cmake/`, `CMakeLists.txt`; the full history is checked out there;
a shallow clone's boundary commit is refused, its changes are unknown). The commits of a cause may
be corrected, since they admit no difference: each one is checked the same way. A slice that
changes a result that the ledger does not record restores master's result or does not land; it
never records it. A slice that changes a route that a tutorial uses keeps the old spelling until
the route is removed (MG1).

A recorded check is a limit of MG8, not a guard. An entry names a check, not its value: the value
depends on the runner's CPU, the reason the reference is rebuilt on each runner, so it cannot be
pinned in the contract. A recorded check therefore explains every later value of that check: while
the ledger records a state, a slice that changes that state again passes MG8 unseen, unless it also
changes something that is not recorded (the number of steps, the hierarchy, a printed or monitor
line). MG8 guards a result only once its cause is fixed and its entry removed. The fixes of ROM-101
to ROM-104, which empty the ledger, therefore land before any slice that changes a route that a
tutorial uses (landing plan of N5, section 6.2).

Each run step has a budget and each tutorial a watchdog under it (`--watchdog`), and the run report
is written after every tutorial. Tutorial 14 runs to t = 8 on two ranks and this tree takes about
four times master's time on it (ROM-105), so it runs in a step of its own (`--exclude 14`, then
`--tutorial 14`). The comparison runs after a step that failed or timed out, so its summary names
any tutorial that did not finish, and the job's limit covers the sum of its steps.

String arrays of ParaView outputs are compared, except `pops_output_manifest`, a JSON manifest of
versioned identities. The JIT flags differ between master (`-O3 -DNDEBUG`) and this tree
(`-fno-fast-math -ffp-contract=off` in addition); neither the JIT nor the native modules pass
`-march`, so GCC targets the x86-64 baseline, which has no FMA: `-ffp-contract=off` has nothing to
contract, and a build of this tree with master's semantics gives master's bits (CC7). Each run
report records the CPU model and features, NumPy's runtime (SIMD extensions) and the versions of
NumPy and Matplotlib, so that a change of runner shows in the summary.

The reference of MG8-0 (master's runs in a container, three lanes) is archived outside the code
tree: `/mnt/project-files/landing/mg8/mg8-master-bb9aea0c.tar.xz` (SHA-256 `8528f7d842a1bbb9...`),
with the manifest of every state and output. CI never reads it: it rebuilds the reference on each
runner.

Run one lane locally with two packages built from the presets (copy each build tree's
`python/pops` with `cp -rL`):

```sh
python3 scripts/run_mg8_tutorials.py run --phase reference --lane serial-dim2 \
  --source <master checkout> --python "$(command -v python3)" --pythonpath <master package> \
  --include <master checkout>/include --kokkos-root <Kokkos prefix> --output /tmp/mg8/reference
python3 scripts/run_mg8_tutorials.py run --phase candidate --lane serial-dim2 \
  --source . --python "$(command -v python3)" --pythonpath <package of this tree> \
  --include include --kokkos-root <Kokkos prefix> --output /tmp/mg8/candidate
python3 scripts/run_mg8_tutorials.py compare --lane serial-dim2 \
  --reference /tmp/mg8/reference --candidate /tmp/mg8/candidate
```

The `mpi-dim2` lane also needs `--mpiexec "$(command -v mpiexec)"`. Before pushing a change of the
contract, `python3 scripts/run_mg8_tutorials.py check --check-base <base> --verify-commits` runs the
check of the `changes` job (it needs the history back to the cited commits).

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
