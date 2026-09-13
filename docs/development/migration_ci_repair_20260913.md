# PoPS CI repair consolidation: 2026-09-13

This page records the CI repair and portability salvage boundary around the migration.
Historical migration qualification remains pinned to source
`175b0883f9f0752f5710ef3db2c7d7d6802e400a`; CI repair work starts at
`95b10ae78589e805af6c40b5f9e6d326cfd1f132`, and local receipts are pinned individually
by their own source and receipt records. The current representative migration ledger is
[`migration_m3_m8_results.md`](migration_m3_m8_results.md). This page records local CI
evidence and review decisions; it does not claim a full matrix rerun.

The CI evidence (CIE) is retained outside this checkout at
`/Users/romaindespoulain/dev/tmp/PoPS-ci-fixes-20260913-evidence/pr677-678-semantic-overlap.md`
and
`/Users/romaindespoulain/dev/tmp/PoPS-ci-fixes-20260913-evidence/portability-salvage-validation.json`.
These paths are provenance references, not repository links.

## Disposition

PR678 remains the active repair line; PR677 is closed as a superseded alternative.
PR677 was not merged and is not full-equivalent to PR678. Independent portability salvage
is recorded at source `d717035`. The overlap review is a selection and provenance record,
not another qualification result.

The review explicitly leaves these semantics unadopted: ABI v5 single-entrypoint; detached
services with owner-last unload rather than current process-retained DSOs; accepted and
provisional typed tokens; global allocation-free behavior; wire versions; and old
qualification. PR677's Uniform v9, AMR v12, and POPSAND5 variants are not adopted.
PR678's Uniform v8, AMR v11, and POPSAND7 variants remain the preserved current choices.
These are separate compatibility decisions and are not implied by the local witnesses below.

The original CI run `34760671991` had 31 failing Python selectors, including three architecture
checks; all are mapped to implemented repairs in CIE `python-failure-reconciliation.md`.
Eleven C++ shards shared an unbuilt-discovery inventory failure, and the MPI checkpoint lane
exposed a parser capture defect. Completed local witnesses retain their execution scope below;
the serial Linux fixtures and long-running subprocesses still require the new GitHub run.

## Local witnesses

| Lane | Observed evidence | Boundary |
| --- | --- | --- |
| Checkpoint | One MPI2 CTest completed in `79.59 s`. | A single checkpoint witness; no total CI claim. |
| Face provider | Two face-provider C++ checks completed in `0.74 s`. | Targeted local evidence. |
| Euler helper | Source `7e1e9e7`, rebuilt production `9e68`, and CIE `euler-poisson-v2.json` plus its log: `N=16`, 12 steps, two engines, and `10.70 s`; independent inward-force, zero-first-work, and positive-second-work checks pass with all-state, mass, and patch parity at `1e-12`. | Targeted local numerical witness. |
| Integrated CI | Three integrated CI checks at source `ebfeb40` completed in `0.24 s`. | Targeted local evidence. |
| Standalone and fixtures | Four standalone CMake checks, 11 fixture checks, 7 label tests, checkpoint stream regression, and scheduler `6+1` completed. | An inventory of local witnesses, not a full suite. |
| Documentation | 66 historical receipts were checked exactly. | Documentation integrity only. |
| Maps | The continuation/restart/repeated-call selector passed in the `142.34 s` combined run, CIE `targeted-runtime-v1.xml`. | The same run's original projection failure is retained. |
| Projection | Rebuilt production source `47b6f07`; the existing two-block, two-level AMR projection test passes in the `56.26 s` invocation, CIE `projection-runtime-v2.json`. | Targeted native execution; no whole-matrix claim. |

## Remaining gates

The projection fix republishes dirty auxiliary values before accepting bootstrap, after every
source is materialized and while the rollback snapshot is still retained. An independent Astra
review checked collective consistency, the initial evaluation point, rollback coverage, and
avoidance of duplicate field solves. The Euler helper receipt predates this change; its field
bootstrap already clears the dirty set, so the new conditional adds no work to that path.

The original failed receipts are retained. PR CI is pending and unverified; local CMake, CTest,
fixture, and documentation checks do not establish GitHub CI. No full local validation campaign
was repeated. The Linux/Serial paths, original `N=48` Euler fixture, scoped subprocess deadlines,
and all broad platform combinations remain for GitHub verification.

No total matrix or requalification claim follows from this consolidation. The qualified
source provenance remains `175b0883f9f0752f5710ef3db2c7d7d6802e400a`, and the current
repair evidence must be read with the explicit limitations above.

## Follow-up: old CI receipt and final repair state

Run `34765485865` is the retained GitHub receipt from head `4f5450a`. All 11 C++
shards and the `ubuntu-latest / Kokkos Serial (C++)` aggregate passed; the
documentation gate and Python compile-cache gate also passed independently. This
is an old receipt, not a new green CI run. New GitHub CI on the active PR678 line
remains pending and unverified. Evidence is retained under
`/Users/romaindespoulain/dev/tmp/PoPS-ci-fixes-20260913-evidence/round2`.

| Area | Final evidence and boundary |
| --- | --- |
| Architecture and routing | The old capture-signature assertion has a passing exact-source witness at `227121a` (`1`, `0.03 s`). Commits `68d3b98` and `17d11eb` route native work per dimension with explicit headers; seven integrated source checks pass in `0.19 s`, and the actual Dim1 combined native case passes in `27.72 s`. |
| Interaction guard | `0118f4b` keeps the common exact-bound context and MPI-1/MPI-2 guard for the 12-case interaction matrix; the focused unit passes in `0.73 s`, with independent Astra review. |
| Native/public contracts | `3bcc820` uses the full installer and preserves authored `FixedDt` instead of falling back to `1e300`; independent engines match exactly over 12 steps at `n=16` (`61.90 s`). The active writer is `POPSAND7`; AND7 adds four history-sample `u64` values per slot. |
| AMR and fields | The admissible conservative/primitive Roe pair (`n=48`, 12 steps) passes in `140.79 s`. The external-field hold v3 run uses the final `n=16` partial-fine fixture (the earlier fixture was `n=8` full-fine) and passes in `25.07 s` with full phase counts and active-cell coarse/fine semantics. |
| Checkpoint bytes | The old MPI job passed its C++ groups (1, 20, 64, 4, and 26 tests), then its Python capture inspector rejected the current format. Two actual Dim2 AND7 images, including four history slots, pass extraction and byte-preservation checks; five synthetic parser cases pass. |
| Python capacity | `b439d98` selects 32 Python shards and five indivisible long-file bins weighted 90/60/60/60/40 minutes; ordinary limits remain 35-minute modeled, 45-minute test, and 50-minute job. Minmod and multistage retain 380-second scheduling estimates. `cc5b645` preserves all 17 AMR implicit cases (8 matrix, 9 lifecycle); the prior 9-case receipt passes and is reused. |
| Cold compilation | The unchanged Minmod native case passes in `235.74 s`; six compiler calls account for 97.4% of that time. The prior profiler CI receipt (`270.35 s`) is reused. Multistage completed SSPRK2 before its old 300-second deadline; the remaining RK4 phase awaits the corrected CI budget. |
| V3 and focused checks | `2aac72c` splits V3 into the existing A-C groups (7 artifacts) and D capabilities group (5 artifacts) without dropping checks; the D-only native run passes in `164.90 s` with five cold compiles, and prior A-C CI passes remain preserved. Named-flux exact-source validation passes in `2.45 s`; the joint species/energy native face case passes in `2.17 s` through `pops_include`. |

Temporal repair `5e81752` establishes the exact stage before seven local operator consumers
and qualifies implicit IMEX rates and nested residual sources. Generic apply/transform
operations with ambiguous coordinates remain explicitly refused. The focused source gate
passes 22 tests in `13.99 s`; an independent Astra review found no further issue. The
installed Dim2 package was rebuilt and checked against both changed Python sources. Its
native ARS222 row passes in `21.78 s` at `16 x 16` cells with 8, 16, 32, and 64 steps;
observed orders are `2.067`, `2.033`, and `2.016` (expected 2). The source checks prove
stage-before-provider ordering in Uniform and AMR emission. This autonomous native
row does not qualify time-dependent native callbacks or AMR auxiliary publication policies.
Receipts: `imex-stage-native.result.json`, `imex-stage-native.xml`, and
`temporal-installed-source.json` in the round2 evidence directory.
PR677 remains closed and preserved as a superseded, non-equivalent alternative;
all new repairs stay on PR678. These receipts do not establish package-level or
scientific full requalification and contain no CUDA/GPU proof.

## Follow-up: remaining CI failures on 20136cff

R3 tracks repair of [GitHub CI run `34769170454`](https://github.com/wolf75222/PoPS/actions/runs/34769170454), which executed head `20136cff`. The run is a failed/incomplete receipt. The evidence below records bounded repairs and witnesses; it does not establish a new green CI run or full qualification.

| Failure group | Repair and evidence boundary |
| --- | --- |
| Architecture/V3 (`103755607451`) | The V3 architecture helper moved while the architecture check still looked at its former location. The affected architecture file passes all 28 checks; this is not a full architecture rerun. |
| Production ABI (`103756161869`) | Commit `91613d7` changes the invalid binary-token fixture to actual wrong-ABI bytes/hash. `production-abi-native.result.json` and `.log` record exact native refusal passing in `105.6497 s`; this is a targeted guard witness. |
| dt-bounds (`103756161907`) | Commit `1e46115` splits the 25 AST-identical assertions into 14 global-AMR and 11 compiled-stability checks, retaining the `600 s` per-process deadline. The later aggregate update below includes these files. `dt-bounds-split-preservation.json` confirms the split; the v2 native receipt passes in `66.9685 s` for B5 source-frequency (`dt=.008`) and B4 AMR (`dt=.0001`) only, without repeating passed phases. `dt-split-ci-capacity-v2.xml` passes in `0.15 s`. |

The temporal failures share two causes: manual IMEX construction lacked the implicit evaluation partition, and schedule lowering assumed the output scratch was the first emitted line before the stage prelude. Commit `1f2cb0a` preserves qualified IMEX evaluations through scheduled lowering; `d1e78d3` exposes temporal-partition authoring through the public API. Manual/factory graph equivalence remains preserved. The explicit output-setup boundary keeps stage, provider, and kernel work inside the guarded branch while hoisting storage; the AMR hold remains an existing unsupported refusal. `temporal-source-focused.log` records 79 source checks passing in `40.20 s`, and `temporal-normalized-consumer.log` records normalized graph/source equality without native execution.

At `d1e78d3`, the fresh Dim2 build completes in `10.219 s`; separate install/source-proof/doctor checks pass and match the source hashes in `temporal-installed-source.json`. The native retry/checkpoint and IMEX-only witnesses pass in `73.598 s` and `31.019 s`. The first local HyQMOM projection receipt (`hyqmom-projection-native.result.json`/`.xml`) took `170.778 s` and failed only because the private environment lacked optional `h5py`; CI already installs it. After installing `h5py 3.16.0` privately, the exact selector passes in `125.0966 s` (`hyqmom-projection-native-v2.result.json`/`.xml`) and reaches native projection after the IMEX fix. This is a single projection witness, not a full HyQMOM example campaign. The public-export worker report records one pass in `0.29 s`; no XML receipt was captured for that check.

Commit `43bf8a1` splits seven test files (five new) while preserving 39 AMR and nine vacuum assertions. Its AMR ABI hash fix follows the same wrong-ABI cause as the uniform fixture. The first native AMR ABI receipt (`amr-abi-native.result.json`/`.log`) fails in `120.557 s` earlier, because the detached `CompiledModel` omits provider fields and its HLLC flag disagrees with exact HLLC provider evidence. Commit `419889d` preserves all 34 constructor arguments, including exact HLLC/Roe provider evidence. The corrected native ABI refusal passes in `117.364 s` (`amr-abi-native-v2.result.json`/`.log`). The unchanged missing-boundary rollback phase passes in `88.082 s` (`vacuum-boundary-native.result.json`/`.xml`), refusing the invalid step without state or clock publication.

Commit `361aa81` refreshes weights from 31 downloaded current-CI timing artifacts covering 539 files with phase evidence. It raises 155 underweighted rows using a `1.25 ×` observed-phase lower bound rounded up to 5 seconds only when above the old weight, and never reduces unchanged rows. Split aggregate weights are separate estimates; the unfinished HyQMOM module has an explicit `1500 s` estimate. The result is 550 files across 37 shards, with an ordinary maximum of `2046.5 s` under the `2100 s` model cap. The existing 45-minute test and 50-minute job caps remain, as do five long-science caps. The OpenMP explicit list includes all three new AMR leaves, with no lost coverage. `final-ci-capacity.xml` records two checks passing in `0.16 s`; `ci-workflow-fence.xml` records one check passing in `0.14 s`.

Commit `f0ce902` records failed-test tracebacks immediately in JSONL so a job timeout cannot erase earlier failures. `ci-failure-receipts.xml` records two synthetic subprocess checks passing in `0.36 s`. In the latest old-run snapshot, only shard 0 remains pending; shards 11 and 31 are accounted as failed, and all other failures are diagnosed. The current C++ shard and aggregate jobs pass, but the raw MPI checkpoint lane passes 23 tests before the manual IMEX example fails with the shared temporal cause; the MPI lane as a whole is not green.

Final review also updates the exact public facade inventory for the new `evaluation_partition` export. Its single architecture check passes in `0.14 s` (`time-public-inventory.xml`). No product source changed after the native build at `d1e78d3`; subsequent corrections concern tests, CI and this evidence record.

PR677 is closed and preserved at head `64df61b`; PR678 is the only integration branch, with no backport to the incompatible alternative. The existing native Dim2 local MPI/OpenMP build is not Linux/Serial proof, and no full validation or MPI campaign rerun follows from these receipts. CI after the next push remains pending; this appendix makes no all-new-CI-green claim.

Evidence is retained externally under `/Users/romaindespoulain/dev/tmp/PoPS-ci-fixes-20260913-evidence/round3`.

## Follow-up: MPI lifecycle and checkpoint deadline on 000ffbf

[CI run `34772786865`](https://github.com/wolf75222/PoPS/actions/runs/34772786865) records 69 successful checks, two skipped checks and three failures. The aggregation correctly reflects two leaf failures: MPI script import and the checkpoint process deadline. All C++ shards and 36 of 37 Python shards pass. The MPI job's M4 pytest groups pass 24 and 18 checks before the direct MPI script fails.

Commit `9cd4256` restores direct-script imports and argument forwarding in the M4, workflow and CTest native-dimension bootstraps. Executable regressions reproduced the missing sibling import at all three sites. The worker reports 37 affected architecture checks passing, then three refined path/argument checks passing; these are tool-output receipts. The checkpoint scenario remains AST-identical except for its explicit process budget, increased from 300 to 600 seconds because an earlier successful cold CI run already took 290.78 seconds. The complete cold local checkpoint check passes in 174.244 seconds (`checkpoint-native.result.json`); no scenario or assertion was removed.

Replaying only the failed two-rank MPI entrypoint exposed further defects. The external topology fixture rejected the required coarse-only bootstrap before fine-level creation. Its moving AMR case now matches the serial case at `n=16`, with coarsening enabled and `dt=0.02`, preserving all accepted-step, regrid, failure and retry positions. Global provider facts must agree across ranks, while local patch reports must form a unique, complete partition of the actual ownership inventory.

Commit `4a6ee97` also restores the two missing layout identities in the AMR Python report. Persistent field materialization identities exclude the live boundary evaluation point; accepted/bootstrap snapshots restore that point, and each solve still authenticates and refreshes its current boundary context. This fixes patch-identity drift after rollback with unchanged geometry. Consumer-free failed runs may reuse their deterministic identity only after every MPI rank proves complete restoration, successful cleanup and an unchanged effect fence. Output/observer lifecycles remain sealed. Twenty-two affected lifecycle unit checks pass against an isolated installed-package overlay whose loaded source hashes are recorded in `retry-lifecycle/affected-unit.log`.

The incremental Dim2 MPI/OpenMP build at `4a6ee97` passes in 61.861 seconds, recompiling both the AMR runtime and its binding; fresh wheel installation, package proof and runtime doctor pass separately. The exact failing M4 entrypoint then passes through the actual corrected launcher with two ranks and two OpenMP threads in 27.508 seconds (`m4-mpi-native-v3.result.json`/`.log`). All ten checks pass: two-level distributed ownership, common provider facts, a real layout-changing regrid, collective and rank-local non-finite failures, exact field/state/clock/topology rollback, and both retries. Earlier failed and diagnostic receipts are retained. No product code changes follow this successful native run.

The requested historical [Docs push](https://github.com/wolf75222/PoPS/actions/runs/32068289713) and [scheduled Docs run](https://github.com/wolf75222/PoPS/actions/runs/34750435497) test unchanged master `51bcdc2e` and fail on four missing verification links. Those links are already corrected in this PR; its Docs PR lint passes at `000ffbf`. Both Docs workflow summaries now quote literal Markdown safely; executing their actual summary bodies passes without invoking `_pops` (`docs-summary-proof.json`). Advisory lint remains unchanged: 140 Markdown findings are in seven untouched files; codespell suggestions concern existing French conventions, valid identifiers or exact evidence bytes. No repository-wide documentation rewrite or lint suppression is included.

PR677 remains closed at `64df61b`; PR678 is the integration line. Full validation and unrelated successful checks were not repeated. New PR CI remains pending until GitHub executes the pushed changes; these targeted repairs do not establish new scientific or performance qualification. Round 4 receipts are retained under `/Users/romaindespoulain/dev/tmp/PoPS-ci-fixes-20260913-evidence/round4`.
