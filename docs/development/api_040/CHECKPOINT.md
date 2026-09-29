# Integration checkpoint — 2026-09-29

The requested migration is **in progress**. This checkpoint records actual code,
failures and next actions; it is not acceptance of the complete specification.

## Source and installed artifact

Worktree: `work/PoPS`, branch `codex/api-040-native-20260928`.
Base: `3a93ba7f1b3fc46ee85a06d9d3482c9b4ece7feb`.
First integrated commit: `75aa789ab5f6a32b36fa5c3e215a546718fcb18b`.
Second integrated commit: `022f5acbb9181eb85a6d9cd05e30a92aad1c4872`:
C09 control expressions, native Uniform paths, C01 field capability correction,
source-authored reconstruction, parameter routing/binding corrections and tests.
Uncommitted scientific examples, independent oracles and evidence remain in the
main worktree. Do not discard them or commit the extracted handoff/prototype.

The last healthy installed package is PoPS 1.1.0 in conda `pops-api040`,
Dim=2, Kokkos OpenMP CPU, MPICH and parallel HDF5. Its native SHA-256 is
`e1070cb63a8e733b0a6032ac4d2b8ad003756668c100e057e5f8edd5b9c01280`;
SDK signature is
`aa301ce673e78b6833d4cd25c3dfe283d3eb8e6506795f7a50dc5b88a6feff2f`.
The latest build authenticates 1,062 shipped source files. The targeted native
campaign passed in `outputs/installed-c01-user-targeted` in the parent
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
  independent tests are present and all nine failures passed installed reception
  on native e1070cb. Preserve the original failed campaign.
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
  and passed in the reviewed campaign below. Parent workspace
  `outputs/native-c09-uniform.*` preserves the earlier result.
* C01 runtime campaign: 49 CTest invocations, 47 passed, one failed and one
  single-process skip for a rank-divergence test. Four new C01 CFL scenarios
  passed on two ranks. The first MPI fixture failed before its capability
  assertion because it queried world size before MPI initialization. The
  fixture now initializes the world before constructing its distribution;
  corrected rebuild/reception passed. Preserve `outputs/native-c01-runtime.*`.
* Reviewed installed campaigns on native e1070cb: 167 unit tests, 16 targeted
  runtime tests and 23 complementary runtime tests passed, no failures/skips.
  These are disjoint test groups (206 total), not independent simulations.
  Receipts: parent `outputs/installed-c01-user-{unit,targeted,complement}`.
* Reviewed C++ campaign: 98 CTest invocations, 96 passed, no failures, two
  explicit skips (Dim=1-only test in Dim=2; rank-divergence test without a peer).
  The latter test passed in its actual two-rank invocation. Three MPI two-rank
  invocations are included. Parent `outputs/native-c01-user-reviewed.{log,xml}`.
* M01/W01: all N=40/80/160/320, t=.125 runs passed with native OpenMP 1 and
  4 threads and MPICH 2 ranks. Observed L1 orders 2.07545/2.07374/2.05989;
  independent saved-state integral oracle and hashes verified. This is full
  Dim=2 with y-invariant data, not a Dim=1/Dim=3 or AMR qualification.
* M03 ideal EOS: N=100/200/400, t=.15 passed with OpenMP 1 thread, authenticated
  before and after execution. Density L1 .02232/.01572/.01056. The stiffened
  EOS (gamma=1.4, p_inf=.3) with renamed components initially failed the
  constant-boundary-flux assumption at N=100: numerical tails reach Outflow.
  Preserve `outputs/m03-stiffened-renamed-openmp1`; no thresholds, EOS or
  boundaries were relaxed. Receipt schema 2 reconstructs the incoming SSPRK2
  boundary integral independently from every saved accepted native state.
  The complete stiffened case now passes in
  `outputs/m03-stiffened-boundary-budget`: density L1 .03115/.02185/.01530,
  mass/energy/momentum budget defects below 4.7e-16. The original far-field
  approximation error is still reported separately. Seven pure boundary-oracle
  tests include transverse variation. The ideal case needs a schema-2 rerun.
* M04 full-case execution failed at N=32: a global maximum x-wave speed was
  also charged in the zero-velocity y direction, rejecting a valid combined
  transport/diffusion step. Sol's isolated correction propagates actual face
  stability frequencies. The independent W02 unstable combined-step witness
  passed installed rejection/rollback reception (one test); it does not qualify
  the still-failing stable M04 case. Preserve both campaigns.
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
source-reconstruction oracles now pass, as do the targeted and complementary
campaigns. Active isolated implementations, all based on 022f5ac:

* Astra semantics: `../PoPS-principal-group`, branch `codex/api040-principal-group`;
  principal-group signatures, sampling and native packed flux evaluation.
* Sol native: `../PoPS-numerical-bodies`, branch `codex/api040-numerical-bodies`;
  live reconstruction parameters and authored face bodies through Uniform/AMR.
* Astra protocols: `../PoPS-resource-lifetime`, branch `codex/api040-resource-lifetime`;
  real native resource leases, completion and cancellation lifetime.
* Sol reference: `../PoPS-degenerate-diffusion`, branch `codex/api040-degenerate-diffusion`;
  exact zero coefficients on ordinary diagonal diffusion axes.

Setup ran once per checkout in separate `pops-api040-*` environments. These
branches have source checks, not installed native acceptance. Root owns heavy
builds, cross-review and integration. The managed worktree tool could not find
the nested repository from the chat root; these checkouts used Git's fallback.

The primary owns `program_emit_path.py`, parameter routing, package builds and
final integration. The native path uses existing finite-volume face fields,
Cartesian operators and transactional System publication. Its numerical model
is authored in Python and lowered as general expressions. It is not a copy of
the reference mini-runtime.

## Next actions

1. Integrate and receive the M04 directional-frequency correction. Preserve the
   failed stable case and the successful W02 negative witness independently.
2. Complete the independent schema-2 M03 budget review, retaining its previous
   failure. M02 Godunov has 33 passing pure-oracle tests; native reception awaits
   User face bodies with an explicit numerical stability function. Independent
   native tests cover captures, widths, AMR, nonfinite values and a highly
   dissipative flux whose physical wave speed alone is an insufficient bound.
3. Review and integrate the four isolated branches, rebuild one coherent main
   snapshot, authenticate installed sources and rerun affected native tests.
4. Use `run_scientific_checks.py` for subsequent scientific receipts: it checks
   installed source/native identity before and after, exact full-case status,
   ranks/threads, source hashes and fresh output directories.
5. Continue the remaining contract/corpus gaps. The current registry is an
   inventory, not proof of all C01–C40 or M01–M28/W01–W12 obligations. GPU,
   other dimensions and full scientific qualification remain unproven.

Use README reproduction commands. `scripts/setup_env.sh` already ran once in
this worktree. Do not run setup again or replace the preserved shared install.
