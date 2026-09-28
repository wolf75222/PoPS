# Integration checkpoint — 2026-09-29

The requested migration is **in progress**. This checkpoint records actual code,
failures and next actions; it is not acceptance of the complete specification.

## Source and installed artifact

Worktree: `work/PoPS`, branch `codex/api-040-native-20260928`.
Base: `3a93ba7f1b3fc46ee85a06d9d3482c9b4ece7feb`.
First integrated commit: `75aa789ab5f6a32b36fa5c3e215a546718fcb18b`.
The worktree also contains uncommitted C09 control-expression work, native
Uniform path evaluation, C01 field capability correction, source-authored
reconstruction, parameter routing/binding corrections and acceptance tests.
Do not discard those changes or commit the extracted handoff/prototype.

The last healthy installed package is PoPS 1.1.0 in conda `pops-api040`,
Dim=2, Kokkos OpenMP CPU, MPICH and parallel HDF5. Its native SHA-256 is
`e1070cb63a8e733b0a6032ac4d2b8ad003756668c100e057e5f8edd5b9c01280`;
SDK signature is
`aa301ce673e78b6833d4cd25c3dfe283d3eb8e6506795f7a50dc5b88a6feff2f`.
The latest build authenticates 1,062 shipped source files. The targeted native
campaign is running in `outputs/installed-c01-user-targeted` in the parent
workspace; its source manifest fixes exactly which code it tests. The original
shared `pops` environment and existing PR 680 worktrees were preserved.

## Observed results

* Installed first-tranche campaign: 60 checks, 57 passed and 3 failed, zero
  skipped. All three failures expose the absent Uniform native path route.
  They remain in `outputs/installed-acceptance` in the parent workspace.
* Installed C09/Uniform campaign: 110 checks, 101 passed, nine failed, zero
  skipped. Six failures expose a fabricated implicit Poisson dependency, one
  exposes failure to canonicalize a qualified authoring parameter at bind, and
  two expose a generated C++ local-name collision. Source corrections and
  independent tests are now present; installed reception awaits the next build.
  The campaign authenticated 1,060 shipped Python/SDK files, not the prototype.
  Receipts are in the parent workspace `outputs/installed-c09-uniform`.
* Independent native AMR path reception: three passed, covering full-array
  oracle on one level, fine-array oracle and composite conservation on two
  levels, and exact rollback after a real interfacial CFL violation. Receipts
  are in this worktree's `outputs/astra-symbolic-amr-*` and the report
  `outputs/astra-symbolic-path-amr.md`.
* Corrected C++ campaign: 92 CTest invocations, 91 passed, zero failed and one
  Dim=1-only case skipped in the Dim=2 build. The preceding failed campaign is
  retained: two rollback assertions assumed root-gathered state on every rank.
  They now compare each rank's accepted projection. The new C01 capability
  tests and stronger preflight marker were added after that successful campaign
  and still need execution. Parent workspace `outputs/native-c09-uniform.*`.
* C01 runtime campaign: 49 CTest invocations, 47 passed, one failed and one
  single-process skip for a rank-divergence test. Four new C01 CFL scenarios
  passed on two ranks. The first MPI fixture failed before its capability
  assertion because it queried world size before MPI initialization. The
  fixture now initializes the world before constructing its distribution;
  rebuild/reception is pending. Preserve `outputs/native-c01-runtime.*`.
* Prepared-resource replacement passed serial and two-rank constructor-failure
  tests, including preservation of the old Kokkos allocation and retry.
* Prototype reference results are separate: 57 current groups and 51 legacy
  records. They do not qualify any production backend.

## Active integration

Only GPT-6 Astra/high and GPT-6 Sol/high workers are used. Earlier 5.6 agents
were stopped when the user specified GPT-6 only. The healthy reviewed build is
logged in parent workspace `outputs/integrated-c01-user-reviewed-build.log`.
Independent reviews found and repaired masked nonfinite stencil leaves, eager
primitive conversion and frozen-container authentication. Three installed
source-reconstruction oracles now pass; the targeted campaign is still running.
Principal unknown grouping C11 remains a reproduced gap, with ten authoring
counterexamples, and is being isolated for a separate implementation tranche.

The primary owns `program_emit_path.py`, parameter routing, package builds and
final integration. The native path uses existing finite-volume face fields,
Cartesian operators and transactional System publication. Its numerical model
is authored in Python and lowered as general expressions. It is not a copy of
the reference mini-runtime.

## Next actions

1. Complete the targeted installed campaign against native e1070cb, including
   all nine prior failures, C09 consumers and checkpoint/restart.
2. Rebuild the six affected C++ targets from the reviewed snapshot, including
   the primitive lazy stencil test and MPI fixture correction, then run CTest.
3. Run `run_installed_checks.py`; it authenticates all tracked Python and SDK
   source files against the installed package. Include independent two-,
   three- and four-component path oracles, runtime parameter rebinding, C09
   inactive/active branch failures, AMR and existing retry regressions.
4. Rebuild the affected C++ targets and replay the MPI rollback tests. Save
   exact hashes, commands, XML and failures before further repairs.
5. Run the full M01 cell-average example from saved native states; its eight
   mathematical witness checks alone do not qualify an execution backend.
6. Continue the remaining contract/corpus gaps. The current registry is an
   inventory, not proof of all C01–C40 or M01–M28/W01–W12 obligations. GPU,
   other dimensions and full scientific qualification remain unproven.

Use README reproduction commands. `scripts/setup_env.sh` already ran once in
this worktree. Do not run setup again or replace the preserved shared install.
