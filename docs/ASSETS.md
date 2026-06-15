# Asset manifest (adc_cpp)

> **Update (ADC-255, 2026-06-15):** the 21 assets marked `delete-orphan` below were removed
> from `docs/` (no live reference; their only mentions were in this audit record). The rows
> are kept as the historical record of what was deleted.


This document lists the images tracked by git under `docs/` of the `adc_cpp`
repository, their actual reference surface, their known producer and a
management decision. It exists because almost all of these assets were produced
outside their committed path and **carry no recorded provenance**
(`adc_cpp` SHA, backend, resolution, generation command). The only set of
assets *with* traceable provenance is that of the canonical tutorial
`docs/sphinx/tutorials/_assets/`, documented in the final section.

## Scope

The glob `docs/*.png` + `docs/*.gif` (root `docs/`, excluding `docs/_build/`,
excluding `docs/sphinx/tutorials/_assets/`) counts **33 images**: **20 PNG + 13
GIF**. The "Referenced by" column comes from a `grep` of the `.md` files
(excluding `docs/_build/`). `docs/DOC_REFONTE_AUDIT.md` is the audit document
that *catalogs* all these files; it is not a live doc surface
and is therefore not counted as a display reference below.

## Provenance status

**None** of the 33 images in `docs/` carries any recorded provenance. For
each one, the following are unknown: the `adc_cpp` SHA at generation time, the
backend (prototype / aot / production), the grid resolution, the number of
steps, and the exact command that produced it. The `tut_*` figures and the
`fig_*`/`anim_*` gallery were committed as artifacts coming out of a
local pipeline, not rebuilt at their repository path.

The *live* doc surface references for display only **two** of these
33 files:

- `anim_romeo_diocotron_amr3.gif`, HTML embed of the `README.md:12` hero;
- `fig_openmp_scaling.png`, markdown embed `docs/PERFORMANCE.md:99`.

Everything else is either archive-only (`docs/archive/*.md`), or orphaned
(referenced only in the audit document, or no longer at all).

## Decision legend

- **keep**: asset of a live surface; to be kept. Provenance to be
  recorded (without which it remains non-reproducible).
- **regenerate-with-provenance**: to be rebuilt via a versioned script
  emitting a `provenance.json`, if the asset is to return to the doc.
- **move-to-archive**: asset only useful to the archive pages; to be
  kept with the archive (ideally under `docs/archive/assets/`).
- **delete-orphan**: no live reference left; candidate for deletion.

## GIF (13)

| File | Referenced by (excl. `_build`, excl. audit) | Producer | Decision |
|---|---|---|---|
| `anim_romeo_diocotron_amr3.gif` | `README.md` (hero, l.12) | unknown, assumed ROMEO/GH200 run, not documented | **keep**, only GIF of the live surface; ROMEO provenance to be recorded |
| `anim_magnetic_diocotron.gif` | `docs/archive/ROADMAP.md` | unknown | **move-to-archive** |
| `anim_diocotron.gif` | none (audit only) | unknown, ex dead Sphinx gallery | **delete-orphan** or regenerate-with-provenance if reused |
| `anim_diocotron_column.gif` | none (audit only) | unknown, ex dead gallery | **delete-orphan** |
| `anim_diocotron_amr3.gif` | none (audit only) | unknown, ex dead gallery | **delete-orphan** |
| `anim_diocotron_multipatch.gif` | none (audit only) | unknown, ex dead gallery | **delete-orphan** |
| `anim_diocotron_amr.gif` | none | unknown | **delete-orphan** |
| `anim_diocotron_mpi.gif` | none | unknown | **delete-orphan** |
| `anim_python_amr.gif` | none | unknown | **delete-orphan** |
| `tut_diocotron_py.gif` | none | unknown, ex Sphinx tutorials (removed, commit 194c63f) | **delete-orphan** or regenerate-with-provenance |
| `tut_diocotron_ring.gif` | none | unknown, ex tutorials | **delete-orphan** or regenerate-with-provenance |
| `tut_ep_collapse.gif` | none | unknown, ex tutorials | **delete-orphan** or regenerate-with-provenance |
| `tut_tfap_field.gif` | none | unknown, ex tutorials | **delete-orphan** or regenerate-with-provenance |

## PNG (20)

| File | Referenced by (excl. `_build`, excl. audit) | Producer | Decision |
|---|---|---|---|
| `fig_openmp_scaling.png` | `docs/PERFORMANCE.md` (l.99) | `scripts/plot_bench_scaling.py` (cited in PERFORMANCE.md:92) | **keep**, only PNG of a live surface; follows the PERFORMANCE.md decision, provenance to be recorded |
| `fig_diocotron_amr_vs_uniforme.png` | `docs/archive/ROADMAP.md` | unknown | **move-to-archive** |
| `fig_diocotron_conv_modes.png` | `docs/archive/DIOCOTRON_GROWTH_RATE.md` | unknown | **move-to-archive** |
| `fig_diocotron_highorder.png` | `docs/archive/DIOCOTRON_GROWTH_RATE.md` | unknown | **move-to-archive** |
| `fig_diocotron_invariants.png` | `docs/archive/DIOCOTRON_GROWTH_RATE.md` | unknown | **move-to-archive** |
| `fig_diocotron_ml_convergence.png` | `docs/archive/ROADMAP.md` | unknown | **move-to-archive** |
| `fig_diocotron_reproduction.png` | `docs/archive/ROADMAP.md` | unknown | **move-to-archive** |
| `romeo_amr_efficiency.png` | `docs/archive/ROMEO.md` | unknown, assumed ROMEO run | **move-to-archive** |
| `romeo_growth_mode4.png` | `docs/archive/ROMEO.md` | unknown, assumed ROMEO run | **move-to-archive** |
| `romeo_highorder_convergence.png` | `docs/archive/ROMEO.md` | unknown, assumed ROMEO run | **move-to-archive** |
| `fig_diocotron_growth.png` | none (audit only) | unknown, ex dead gallery | **delete-orphan** |
| `fig_diocotron_modes.png` | none (audit only) | unknown, ex dead gallery | **delete-orphan** |
| `fig_diocotron_column_growth.png` | none | unknown | **delete-orphan** |
| `fig_diocotron_theory.png` | none | unknown | **delete-orphan** |
| `tut_diocotron_growth.png` | none | unknown, ex tutorials (commit 194c63f) | **delete-orphan** or regenerate-with-provenance |
| `tut_diocotron_sequence.png` | none | unknown, ex tutorials | **delete-orphan** or regenerate-with-provenance |
| `tut_euler_poisson.png` | none | unknown, ex tutorials | **delete-orphan** or regenerate-with-provenance |
| `tut_plasma.png` | none | unknown, ex tutorials | **delete-orphan** or regenerate-with-provenance |
| `tut_poisson_backends.png` | none | unknown, ex tutorials | **delete-orphan** or regenerate-with-provenance |
| `tut_tfap_ap.png` | none | unknown, ex tutorials | **delete-orphan** or regenerate-with-provenance |

## Summary

- **2 keep**: `anim_romeo_diocotron_amr3.gif` (README), `fig_openmp_scaling.png`
  (PERFORMANCE.md). Live surface; provenance to be recorded.
- **10 move-to-archive**: figures `fig_diocotron_*` and `romeo_*` plus
  `anim_magnetic_diocotron.gif`, referenced only by `docs/archive/*.md`.
- **21 orphans**: the 10 `tut_*` files (ex pool of the Sphinx tutorials
  moved to `adc_cases`, removal of `tutorials/` commit 194c63f) plus the
  ex images of the dead gallery and other `anim_*`/`fig_*` without reference. For
  each: **delete-orphan**, or **regenerate-with-provenance** if the asset is to
  return to the new gallery/tutorial.

The `tut_*` have **no** provenance and are no longer referenced anywhere:
they are already fully orphaned independently of any overhaul.

## Canonical tutorial assets (with provenance)

Unlike the above, the A->Z tutorial lives under
`docs/sphinx/tutorials/` and **embeds its provenance**. The script
`docs/sphinx/tutorials/diocotron_tutorial.py` regenerates its 4 images and writes
`docs/sphinx/tutorials/_assets/provenance.json` at each execution.

Common provenance (extracted from `provenance.json`):

- script: `docs/sphinx/tutorials/diocotron_tutorial.py`
- command: `python diocotron_tutorial.py --n 96 --steps 60`
- `adc_cpp` SHA: `e58b513d2245c9258a8720b91830b9ee95cafde9`
- compilation backend: `aot`
- execution backend: `serial` (default; cf. getting_started for Kokkos/MPI)
- resolution: `96x96`, `steps=60`, `cfl=0.4`, Python `3.12.2`
- control metrics: `growth_factor=1.5212313128`,
  `mass_drift=1.81e-16`, `amr_uniform_max_delta=0.0717869334`

| File | Dimensions | Provenance |
|---|---|---|
| `docs/sphinx/tutorials/_assets/diocotron_growth.png` | 1104x432 | `provenance.json` (key `assets`) |
| `docs/sphinx/tutorials/_assets/diocotron_cover.png` | 456x432 | `provenance.json` (key `assets`) |
| `docs/sphinx/tutorials/_assets/diocotron.gif` | 380x360 | `provenance.json` (key `assets`) |
| `docs/sphinx/tutorials/_assets/diocotron_uniform_vs_amr.png` | 912x432 | `provenance.json` (key `assets`) |

The folder also contains the compiled `.so` files associated with the run
(`diocotron_aot.so`, `diocotron_production.so`), artifacts of the same pipeline.

This set is the model to follow for any regeneration of the assets above
marked **regenerate-with-provenance**: a versioned script, a reproducible
command, and a `provenance.json` committed next to the images.
