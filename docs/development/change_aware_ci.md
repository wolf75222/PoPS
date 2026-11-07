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

The MPI native build runs three sequential phases with separate 28-minute watchdogs inside a
60-minute native-build cap; the complete job remains capped at 180 minutes. The manifest projection puts
`test_program_runtime`, `test_amr_synthetic_program_loader_transaction`,
`test_mpi_system_gather_scatter`, and `test_mpi_system_io_gather` in an isolated phase. That phase
first compiles the seven synthetic loader fixture DSOs of the loader test
(`pops_synthetic_loader_fixtures`) two at a time, then its four targets with one compiler, under
its single watchdog. Two further phases split the exact complement by the build costs of
`tests/cpp/mpi_build_durations.json` and build it with parallelism four. The MPI job configures the
heavy-TU pool with two slots (`-DPOPS_HEAVY_TEST_TU_POOL=2`; the presets keep one): every test that
links the System or AMR runtime compiles at `-O0` in that pool. New manifest targets enter the
complement automatically; missing anchors or a non-disjoint/incomplete partition fail planning. Python module linking and
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

The two-slot pool addresses the PR #688 `a3a03d2` run (job 123904651562). That PR changed two
AMR History metadata headers, so ccache served 17 of 126 compiles. The isolated phase took 25.7
minutes and the first complement phase 15.4 minutes; the second complement phase was stopped by
the 60-minute step cap after 18.9 minutes, with two heavy units still to compile. In both
complement phases the last heavy units compiled one at a time (load average near 1.1 on four
cores) while 8.5 to 14.5 GiB stayed available. Compiled alone with their exact `ci-mpi` commands
(GCC 13.3, `-O0`), the 38 heavy units of the complement phases peak at 0.4 to 3.9 GiB, except
`test_generated_amr_system_block` at 6.3 GiB; the largest pair within one phase needs about 10.2
GiB, and the 56 other units of these phases peak at 1.2 GiB or less. Two heavy and two other
units together therefore stay near 12.2 GiB, against the 14.5 GiB available with no compiler
running. The isolated phase keeps one compiler: `test_program_runtime` alone left 82 MiB
available in that run.

The fixture order addresses the PR #704 `c494c61` run (job 124154515435), whose isolated phase
used 1509 of its 1680 seconds: `test_program_runtime` compiled alone for 600 seconds, with 88 MiB
available at worst, then the seven fixtures one at a time for 813 seconds while 11.7 to 14.2 GiB
stayed available. Each fixture is one `-O2` compile and link of a generated Program by
`tests/cpp/support/native_dso_compiler.hpp`, which calls the compiler directly, without ccache, so
every build compiles all seven. Compiled alone with their exact commands (GCC 13.3, Xeon 2.1 GHz),
they peak at 3.6 to 3.7 GiB and take 134 to 165 seconds of compiler CPU each, 1025 seconds in
series. The isolated-phase command replayed on that machine compiled them two at a time in 586
seconds, each fixture within 6 percent of its time alone, with a sampled peak of 7.3 GiB for the
two compilers; the runner has 14.5 GiB available with no compiler running. The fixture rules are
in the heavy-TU pool, whose two slots bound them at any parallelism, and `test_program_runtime`
still compiles only after them, alone. At that ratio the fixtures of the #704 run take about 480
instead of 813 seconds, which brings its isolated phase near 1170 seconds, 70 instead of 90
percent of the watchdog. No watchdog, step cap or job cap changes.

The MPI Python matrix gives each ranked entrypoint one 1200-second launch and each pair of
independent worlds two 900-second orchestrator launches whose child keeps its 840-second
watchdog. These limits do not change. A group whose measured cost exceeds its budget is split by
cost into declared partitions instead, each its own matrix group. A ranked entrypoint lists
`partitions = [{ id, nodes }]`, naming every case of its pytest receiver exactly once; a world
pair lists one parent node per partition, closed by `MPI_ORCHESTRATOR_PARTITION_CONTRACTS` and
the support module that owns the fixture inventory. Every partition launch still collects the
whole receiver, or every function of its world inventory, and
[`mpi_partition_units.py`](../../tests/python/support/mpi_partition_units.py) refuses any missing,
added, renamed or duplicated case against the declared inventory before it runs the partition's
units only. The receipts must then name exactly those units. The partitions of one selection are
disjoint and their union is the inventory, so the lane still runs every case.

Run 41037321460 of PR #705 (`cf2f510`) stopped eight ranked groups at 1200 seconds and four world
groups at their 840-second child watchdog. Its serial shards give each case's cost; the cases each
MPI group finished before its limit bound the ratio R of MPI to serial cost. Partitions aim at 60%
of the budget at the upper bound of R. A single case cannot be split further.
`relaxed_multiblock_native` failed its first case in that run, which hid the other two (fixed by
PR #721). Its three cases cost 917 seconds serial (shard 45); its R is bounded by its relaxed
siblings in the same run (0.66 to 1.16) and by its local np2 to one-rank ratio with that fix (1.22).

| MPI group | Cases | R | Partitions | Largest partition, predicted |
| --- | --- | --- | --- | --- |
| `implicit_derivatives` (ranked) | 10 | 0.97 to 1.05 | 2 | 650 s (54%) |
| `v14_distributed_derivatives` (ranked) | 10 | 1.03 to 1.09 | 2 | 673 s (56%) |
| `coupled_methods_native` (ranked) | 7 | 0.81 to 1.53 | 4 | `mprk22_network[1.5-True]` alone, 796 to 1504 s |
| `original_field_recovery_native` (ranked) | 7 | 1.38 to 1.72 | 4 | 602 s (50%) |
| `original_field_typed_recovery_native` (ranked) | 4 | 1.29 to 1.72 | 4 | 400 s (33%) |
| `relaxed_multiblock_native` (ranked) | 3 | 0.66 to 1.22 | 2 | 705 s (59%) |
| `amr_original_field`, saved restart (worlds) | 3 | at most 1.43 | 3 | 553 s (66%) |
| `captured_diffusion` (worlds) | 4 | 1.38 to 1.88 | 4 | about 300 s (36%) |
| `evolved_original_stage` (worlds) | 8 | at most 1.35 | 4 | 422 s (50%) |
| `spatial_interaction` (worlds) | 8 | 1.58 to 1.90 | 3 | 405 s (48%) |
| `amr_profile`, tag selection (worlds) | 8 | MPI2 launch 828 s | 2 | about 414 s (49%) |
| `amr_profile`, stationary regrids (worlds) | 5 | MPI2 launch 761 s | 2 | 464 s (55%) |

Partitioning cannot fix a single case that costs more than its budget: `gyrokinetic_nd` (two
cases, 1547 and 1866 seconds serial), `v14_decision_assess_nested_reject` (one case, 1340 seconds
serial) and the depth-160 case of `finite_structural_depth_native` need their cost reduced, and so
does `mprk22_network[1.5-True]` if R exceeds about 1.2. These predictions are scheduling estimates;
the matrix receipts of the partitioned groups decide.

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
any tutorial that did not finish, and the job's limit covers the sum of its steps. This tree's steps
run after a step of master that failed or timed out too, so that a failure of master's run does not
hide this tree's results. A tutorial runs with its standard input closed, and each of its ranks
compiles into a JIT cache of its own (`POPS_CACHE_DIR/rankN`, set by the probe before
`import pops`): master's facade passes the cache path to the compiler explicitly, which bypasses the
cache lock that b6 holds since `3a0b04f5d7`, so master's two ranks compiled the same model into one
file at the same time; a rank that hashed the file while the other rank's linker rewrote it raised
at bind ("compiled binary differs from the authenticated facade artifact") and the other rank waited
in a collective until the watchdog, the stall of master's run on #706. When a tutorial outlives its
watchdog, the runner sends SIGTERM, then SIGKILL, to its launcher and to every process under it:
Open MPI 4 starts each rank in a process group of its own, and killing `mpiexec` alone would leave
the ranks running on into the next tutorials. On SIGTERM, the probe of each rank first writes the
Python stack of each of its threads (`faulthandler`). A tutorial that fails prints, in the job log,
the verdict of each rank, those stacks and the end of its logs, with workflow commands paused: the
job log is readable wherever the published run is not.

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
