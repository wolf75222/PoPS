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

Retain PR678 as the active repair line and close PR677 as a superseded alternative.
PR677 was not merged and is not full-equivalent to PR678. Independent portability salvage
is recorded at source `d717035`. The overlap review is a selection and provenance record,
not another qualification result.

The review explicitly leaves these semantics unadopted: ABI v5 single-entrypoint; detached
services with owner-last unload rather than current process-retained DSOs; accepted and
provisional typed tokens; global allocation-free behavior; wire versions; and old
qualification. PR677's Uniform v9, AMR v12, and POPSAND5 variants are not adopted.
PR678's Uniform v8, AMR v11, and POPSAND6 variants remain the preserved current choices.
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
