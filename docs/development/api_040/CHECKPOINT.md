# Integration checkpoint — 2026-09-29

The requested migration is **in progress**. This checkpoint records actual code,
failures and next actions; it is not acceptance of the complete specification.

## Current reception: source 38faddb, native be89992b (in progress)

The incremental MPI/Dim2 OpenMP wheel build succeeded. The installed native SHA256 is
`be89992bb2a67a5e3b04dfeb2552d5954f4c7e8d8f1c2ba094e8f0452b4d466c`;
header signature `88f64c8a329859e1fb91169a34eb52386b61b84847458c0c1fb986077f457c2c`.
Installed authentication matches all 1,079 shipped Python/header files, manifest SHA256
`7a96401c0d681a40a4ca3b27ab21501a57e9419fbb95272b8175c48e52e2ed1f`.
The receipt observed HEAD 6d556cd (only tests/docs after the production snapshot).
Parent-workspace receipts: `outputs/build-38faddb-openmp-mpi.log` and
`outputs/identity-38faddb/`. This authenticates the package; native acceptance is pending.

Additional integrated fixes: condensation consumes every inverse failure collectively;
row-equilibrated block inversion handles extreme scales without partial publication
(numerical decision version 1). Independent O2 and ASan/UBSan checks each passed 465
host binary64 assertions. M08 publication now stops provenance traversal at the current
SSA State/solve frontier. Independent current-conflict and stale-field tests pass.
M08 scientific reception now must distinguish a deliberately stale-field trajectory;
the original 3e-7 comparison threshold was insufficient and was tightened to 3e-11
before native execution. M15 names precisely the supplied axial B.1 order-four closure;
M16 retains its realizable oblique nonhyperbolic counterexample.

The integrated source suite at 79db292 passed 151 tests. The 38faddb native C++ target
build is running; root owns all heavy builds. Tests were moved into existing manifest
suites, and the three independent T4 witnesses were added to the actual MPI2 filter.
The standalone binary64 inverse-review harness is a reproducible evidence source,
not a claimed CTest target. Seven pre-existing verification tests remain byte-pinned
as unavailable by the repository catalogue; they were not silently activated.

Further independent audits are active in isolated GPT-6 worktrees: LocalNewton can
currently declare convergence from a small update despite a failed original residual
(Astra reproduced R=z^2-2, z~1.5, residual~.25 with tolerance1e-12); Sol is examining
multi-parent restart regrid history preservation. These are open findings, not accepted
properties of the installed artifact. A separate Sol task is preparing exact M09 cases,
and Astra reviews the bounded HyQMOM library case. No remote job, push or merge occurred.

## Integration before reconstruction: 79db292 (historical)

Main worktree `work/PoPS`, branch `codex/api-040-native-20260928`. The installed
Dim2 artifact still belongs to the previous 9e9a0e5 reception below. Its green
receipts do not qualify the new source. No push, PR update or remote HPC job
was performed. Existing worktrees and the user's original `pops` environment
are preserved; the dedicated environment is `pops-api040`.

Integrated since that reception:

* T3 original LocalResidual evaluation domains; concurrent consumer publication
  cursor authority and compensation handling; C11 exact StateStorage install.
* AMR completion votes use the durable RuntimeInstance lane after scheduled
  regridding. The previous invalid-communicator abort has a native regression.
* C11 joint principal fluxes use actual AMR stage traces, halos and reflux.
  Joint primitive coordinates have explicit inverses, domain checks, per-row
  parameters, arbitrary component packing and a common required halo radius.
  ModuleManifest is version 10; version 9 is explicitly refused. Source and
  complete generated C++ syntax checks pass; native tests remain pending.
* Native System/Program solve outcomes carry owner/attempt authority across
  move, restart, rejection and destruction. Pending results block transitions;
  rejected internal steps revoke escaped results before snapshot restore.
  The restart finalizer remains nonthrowing after external publication.
* Path stability version 2 checks the actual explicit consumer's rational
  weights. Fixed/external steps, diagnostic RHS, guarded values and retained
  Butcher-form SSP stages have independent tests. A rate-only accepted state
  cannot bypass the missing convex-state-budget refusal. The original source
  failure and the no-state counterexample are preserved.
* A public AMR local-box diagnostic and a two-rank failure witness distinguish
  actual cell ownership from merely launching two MPI processes. Reception is
  pending. C04 now classifies only proven blockages; an absent Riemann provider
  is deliberately not a proof of mathematical incompatibility. C26/C27 map to
  production ABI/SDK authorities, not the mini-runtime's C export names.

Source-only evidence includes 51 primitive/group tests, independent Sol reviews,
15 independent public SSP/path tests and 17 common stability tests. Native
syntax checks are compilation evidence only. The root owns all heavy builds.

Scientific receptions completed on the **previous** native `220b48d3…`:

* M02 full shock and rarefaction passed on N=100/200/400 to t=.2 with the
  Python-authored Godunov flux; receipts `m02-{shock,rarefaction}-220b-openmp1`.
* M04 Forward Euler still fails its unchanged first-order threshold. The
  separately declared SSPRK2 variant passes N=32/64/128 to t=.1 with observed
  orders .9781404/.9888494 and discrete Fourier error at most 1.33e-15.
* Full M17 compiled the generic 15-moment physical/path bodies but accepted
  zero steps because the old route wrongly required a step_cfl Courant for
  FixedDt. `m17-220b-openmp1` remains failed, without native saved-state proof.
  The mechanism above corrects this gap; M17 must be rerun after reconstruction.
* MPI2 User/W02 reception passed five tests and skipped one per rank. The
  skipped Uniform witness had no cells on rank 1. Its aggregate status remains
  failed; the new AMR ownership witness must supply the missing proof.

M06 native enthalpy diagnostics, M13 reaction chain and an independently authored
Maxwell–Cattaneo structural case are integrated with separate exact oracles,
fixed criteria, saved-state verification and reproduction commands. They have
not yet run natively. `run_scientific_checks.py` authenticates source/native
identity before and after each campaign and records actual MPI size per run.

Active isolated work, all GPT-6 Astra/high or GPT-6 Sol/high: a further T3
condensation counterexample (ignored inverse status and determinant overflow),
M08 coupled field/stage reception, and the mathematically supplied HyQMOM B.1
variant/M16 oblique counterexample. No multi-order HyQMOM closure is invented
from the incomplete notes. See each agent's report for precise scope.

Next: integrate the independently checked inverse/condensation correction,
run packaging checks, freeze the build input, incrementally rebuild with
`scripts/build_python.sh --dim 2 --mpi`, authenticate the installed package,
then run the prepared C++/MPI, Python rollback and full scientific campaigns.
Current source receipts cannot replace any of those native executions.

## Previous reception: 9e9a0e5 (historical)

The integrated source now includes principal groups C11, source-authored face
and reconstruction bodies T2, nonnegative diagonal diffusion (including exact
zero axes), actual directional face frequencies, native resource lifetimes C36,
per-substep collective rejection, and the symbolic-path sampling resolution fix.
Current HEAD at the start of this reception is
`9e9a0e5bd56ebca6e0f678485f02f1d2b7ecfa78`.

The current installed native SHA-256 is
`220b48d3264f413aa5ccd8bba58a8b0d1a6b31bf5bade6faac904b1b7ef8cc87`,
SDK signature
`47299dd6ba34f1d97464e06e06fb2fe3c4aab75958eb516cc0e57732437c39f6`.
Authentication compared all 1,075 shipped source files with the actual installed
package under `pops-api040`; it did not import the prototype or `ROOT/python`.

New observed results, with evidence under the parent workspace `outputs/`:

* `native-c11-t2-c36`: 114 CTest invocations, 108 passed, no failures and six
  explicit skips. Two need CUDA; four rank-divergence tests skip in their
  single-process invocation. Three actual MPI2 invocations are included.
* `installed-c11-t2-c36-runtime`: stopped after two genuine installed C11
  failures. Vector models run, but separate-state principal groups fail during
  native installation because the Python spatial descriptor contradicts
  `program_only_storage`. Isolated fix `98eb431` has 22 passing source tests
  and awaits integration/rebuild; the native guard remains intact.
* `installed-t2-diffusion-w02-runtime`: eight uniform diffusion tests passed
  before AMR aborted inside MPICH with an invalid communicator. There is no
  final XML and the campaign is failed, not eight accepted tests. The isolated
  AMR repetition with capture disabled reproduces the same abort; both logs
  are preserved. Its native lifecycle diagnosis is in progress.
* `installed-t2-uniform-w02`: nine tests passed, zero skips: independent
  multi-block/vector User flux oracles, parameter rebinding, lazy singular
  branches, failed active bodies/bounds, dissipative numerical CFL and both
  exact-x/isotropic W02 rollback witnesses.
* `mpi-runner-native-identity`: the new rank-separated runner authenticates
  the same installed library on two actual MPI ranks. Its one pure-oracle
  test only validates the harness; it is not a distributed PDE qualification.
* `m04-xonly-220b-openmp1`: all grids reach `t=.1`, but Forward Euler fails
  the unchanged first-order threshold (.60635 < .7). Independent discrete
  Fourier analysis agrees with actual saved states within 4.45e-16. Preserve
  the failed receipt. A separate SSPRK2 variant with unchanged physical data,
  grids, dt and criteria is being received; see `m04_temporal_variants.md`.

Ready isolated changes not yet qualified in the installed package:
`cc93a82` preserves original LocalResidual evaluation domains;
`a2c8dd8` protects concurrent consumer publications with shared cursor authority,
checkpoint CAS and terminal handling of failed compensation;
`98eb431` corrects principal storage installation. Independent source reviews
and tests are recorded separately from native acceptance. M17 canonical/reverse
source and independent oracle are ready; native reception remains outstanding.
The new structural Maxwell–Cattaneo case is authored independently in the
resource-lifetime worktree, with an exact Fourier oracle, but not run natively.

Root owns integration and heavy builds. Astra now extends joint principal AMR
trace/reflux and diagnoses the invalid communicator; Sol implements live solve
result authority and prepares M06/M13. All four workers are GPT-6 Astra/high or
GPT-6 Sol/high. Never infer full C01–C40/corpus completion from these tranches.

## Earlier source and installed artifact (historical)

Worktree: `work/PoPS`, branch `codex/api-040-native-20260928`.
Base: `3a93ba7f1b3fc46ee85a06d9d3482c9b4ece7feb`.
First integrated commit: `75aa789ab5f6a32b36fa5c3e215a546718fcb18b`.
Second integrated commit: `022f5acbb9181eb85a6d9cd05e30a92aad1c4872`:
C09 control expressions, native Uniform paths, C01 field capability correction,
source-authored reconstruction, parameter routing/binding corrections and tests.
Uncommitted scientific examples, independent oracles and evidence remain in the
main worktree. Do not discard them or commit the extracted handoff/prototype.

The earlier healthy installed package was PoPS 1.1.0 in conda `pops-api040`,
Dim=2, Kokkos OpenMP CPU, MPICH and parallel HDF5. Its native SHA-256 is
`e1070cb63a8e733b0a6032ac4d2b8ad003756668c100e057e5f8edd5b9c01280`;
SDK signature is
`aa301ce673e78b6833d4cd25c3dfe283d3eb8e6506795f7a50dc5b88a6feff2f`.
The latest build authenticates 1,062 shipped source files. The targeted native
campaign passed in `outputs/installed-c01-user-targeted` in the parent
workspace; its source manifest fixes exactly which code it tests. The original
shared `pops` environment and existing PR 680 worktrees were preserved.

## Earlier observed results

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

## Earlier integration inventory

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

## Earlier action list (superseded by current reception above)

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
