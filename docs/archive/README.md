# Documentation archive

Documents kept for the record but OUTSIDE the main navigation: these are
either planning and vision notes (the actual state is described by
[`../ARCHITECTURE.md`](../ARCHITECTURE.md), [`../ALGORITHMS.md`](../ALGORITHMS.md) and
[`../GPU_RUNTIME_PORT.md`](../GPU_RUNTIME_PORT.md)), or application notes whose
executable content now lives in the [`adc_cases`](https://github.com/wolf75222/adc_cases) repository.

They are not kept up to date. In case of divergence, the non-archive documentation is authoritative.

## Planning and vision

| File | Content |
|---|---|
| `ROADMAP.md` | living list of intentions, done / to do |
| `TODO.md` | tasks of the `PhysicalModel` -> multi-species system work item |
| `WORK_TODO.md` | work list of the extensible aux / AMR parity / runtime work item |
| `ETAT_DES_LIEUX.md` | synthesis of cross audits (architecture, numerics, robustness, perf, HPC) |
| `ARCHITECTURE_CIBLE.md` | vision doc (north star), prior to the current state |
| `DESIGN_MULTISPECIES.md` | design of the multi-species target (whiteboard session) |
| `PLAN_VARIABLES_EPM.md` | plan of the Variables + EPM level work item |

## Closed audits

One-shot audit reports kept as a snapshot record, not a backlog. The audit METHOD and the
live index stay at [`../AUDIT.md`](../AUDIT.md); actionable findings live in Linear.

| File | Content |
|---|---|
| `CODEBASE_AUDIT.md` | maintainability audit (2026-06-06 snapshot) |
| `STYLE_CONFORMANCE_AUDIT.md` | conformance to the C++ standards (2026-06-12) |
| `COMMENTS_AUDIT.md` | accuracy of code comments (2026-06-12) |
| `CONFORMANCE_AUDIT.md` | spec and requirement conformance (ADC-188) |
| `STL_BOOST_AUDIT.md` | STL idiom and Boost decision (ADC-192) |
| `DEAD_CODE_AUDIT.md` | one-shot cppcheck `unusedFunction` sweep |
| `DOC_REFONTE_AUDIT.md` | Phase-1 documentation truth matrix |
| `TOOLCHAIN_ROBUSTESSE_AUDIT_2026-06-10.md` | toolchain and install robustness audit |

## Closed roadmaps and delivered design notes

Plans and design notes whose feature has shipped or whose items are done / tracked in Linear.

| File | Content |
|---|---|
| `PAPER_ROADMAP.md` | Hoffart reproduction roadmap (arXiv:2510.11808) |
| `FULL_MODEL_VALIDATION_ROADMAP.md` | full-model reproduction roadmap (supersedes `PAPER_ROADMAP.md`) |
| `BUILD_UX_ROADMAP_2026-06-10.md` | build and install UX audit and roadmap |
| `PERF_SCALING_TODO.md` | closed performance-scaling TODO |
| `RESEARCH_BACKLOG.md` | non-auto-completable research backlog |
| `SAMRAI_BACKEND_PLAN.md` | technical plan for a SAMRAI AMR backend (not started) |
| `AMR_CONDENSED_SCHUR_DESIGN.md` | AMR condensed-Schur implementation design (largely delivered) |

## Application notes (scenarios, runs)

The `adc_cpp` core is model-agnostic: these notes describe SCENARIOS or
measurement campaigns that live on the application side.

| File | Content |
|---|---|
| `two_fluid_ap.md` | method note of the two-fluid AP scheme (scenario, lives in `adc_cases/two_fluid_ap/`) |
| `DIOCOTRON_GROWTH_RATE.md` | reproduction of the diocotron growth rate vs Hoffart arXiv:2510.11808 |
| `HERO_RUN_AMR.md` | design of the distributed AMR hero-run on ROMEO |
| `ROMEO.md` | log of the ROMEO runs (GH200 + EPYC) |
