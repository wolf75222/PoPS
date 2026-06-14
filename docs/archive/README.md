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

## Application notes (scenarios, runs)

The `adc_cpp` core is model-agnostic: these notes describe SCENARIOS or
measurement campaigns that live on the application side.

| File | Content |
|---|---|
| `two_fluid_ap.md` | method note of the two-fluid AP scheme (scenario, lives in `adc_cases/two_fluid_ap/`) |
| `DIOCOTRON_GROWTH_RATE.md` | reproduction of the diocotron growth rate vs Hoffart arXiv:2510.11808 |
| `HERO_RUN_AMR.md` | design of the distributed AMR hero-run on ROMEO |
| `ROMEO.md` | log of the ROMEO runs (GH200 + EPYC) |
