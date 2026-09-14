# Numerical snapshots and rendering

Run `03_render_results.py` on output from tutorial 01 or 02. Each argument is one physical trajectory; the renderer searches its segment subdirectories recursively. Pass separate trajectory directories to compare models, resolutions or modes 3/4/5. For a restarted trajectory, pass their common output parent so the original near-zero potential sample remains available.

```sh
env -u PYTHONPATH python docs/tutorials/diocotron/03_render_results.py \
  /path/to/euler-mode5/results /path/to/hyqmom15-mode5/results \
  --output /path/to/new-render-directory
```

Use the `pops` Python environment with NumPy, Matplotlib and Pillow. Rendering imports no native PoPS extension. The output directory must be new. `--help` lists fixed color scales, resolution and GIF duration options. Read completed, immutable snapshots; do not point the renderer at a file still being written.

Use `--context-label "Integration smoke only"` when rendering short qualification runs. This scope label appears on every figure and GIF frame and is preserved in the manifest. It does not change the saved data, requested paper times, fit windows or normalization rules.

## Snapshot contract

Each `snapshot-<macro_step>.npz` contains plain arrays and scalar JSON strings; loading uses `allow_pickle=False`.

| Member | Meaning |
|---|---|
| `q0_levelL` | `float64` shape `(ntheta*2**L, nr*2**L)`, computational cell average of `q0=r*rho`. Validity is determined by the patch table. |
| `psi_levelL` | Same shape; potential `psi=phi/alpha` at the stored source midpoint. Absent only for the true initial state. |
| `metadata` | JSON with `time`, `potential_time`, `macro_step`, `mass`, `initial_mass`, `levels`, and run diagnostics. `time` belongs to density; `potential_time` belongs to potential and is initially `null`. |
| `parameters` | The complete authored case parameter dictionary, including model, mode, geometry, numerical settings and native artifact identity. |
| `patches` | Unmodified `simulation.amr.patch_table().to_dict()`. |

The depth-two named potential ring rotates at the end of each accepted step. Snapshots read raw slot **1**, which holds the newest accepted source potential, and timestamp it using `time - history_slot_dt("plasma.potential", 0, 1)/2`. The parameter `potential_history_slot=1` records this convention. Earlier qualification archives without this parameter used raw slot 0 and do not have qualified non-initial potential timestamps; their density timestamps and complete native checkpoint arrays remain usable.

`progress.json` is atomically replaced after each successful public `pops.run` chunk. It records `time`, `macro_step`, `n_levels`, and `elapsed_seconds`. It proves that chunk returned successfully; only a completed checkpoint proves durable restart state.

`PatchReport.per_level[L].boxes` contains flattened **inclusive** tuples `(lo_r, lo_theta, hi_r, hi_theta)`. Its base entry has no boxes and covers the entire domain. Fine entries provide the actual global patch boxes. The renderer checks dimensions, census, nested coverage and complete parent-cell alignment. It masks cells covered by the next finer level and ignores invalid fine-array entries. It recomputes composite mass from active `q0*dr*dtheta` cells and records the difference from the native diagnostic.

Overlapping segment snapshots are merged only if their state/potential arrays, patch tables, physical timestamps and mass agree exactly. Segment-local `initial_mass` and elapsed wall time may differ. Conflicting data or different physical/numerical/artifact identities inside one trajectory are refused.

## Density, schlieren and actual time panels

Physical density is `rho=q0/r_center`. The renderer plots native polar cell edges in physical Cartesian coordinates, masking covered coarse cells. It does not interpolate images or fill missing simulation times.

The physical gradient is `sqrt((d_r rho)^2 + (d_theta rho/r)^2)`. Differences use available same-level neighbors, with periodic angle and explicit one-sided differences at patch/radial edges. Zero-filled invalid fine cells never enter a stencil. This is a visualization diagnostic; a patch-edge derivative has lower order than a centered interior derivative.

The default schlieren transform is

```text
S = 1 - exp(-strength * |grad rho| / gradient_scale)
strength = 1
gradient_scale = (mean_ring + perturbation - background) / radius
```

Both quantities stay fixed throughout a trajectory. Colors span `[0,1]`; density colors default to `[background, mean_ring+perturbation]`. The manifest records these choices and counts densities outside the color range and schlieren values above `0.999`. Color saturation does not alter numerical data. `--gradient-scale`, `--schlieren-strength`, `--density-min` and `--density-max` allow explicit alternative displays.

The paper panels request density times `0.1, 1.25, 2.5, 3.75, 5, 6.25, 7.5, 8.75, 10`, with reference `t_f=10`. A state must lie within `--time-tolerance` (default `1e-8`) of its requested time. A complete run produces nine panels; a partial run produces only available panels, named `*-partial`, and lists every missing time. No matching state means no panel figure. A separate latest-state image always displays the latest actual density snapshot.

The GIF contains only saved density/schlieren states, with their actual times and accepted steps visible. Display durations are proportional to physical intervals, rounded to 10 ms with a 10 ms minimum; the final frame is held for one second. There is one fixed 256-color palette and no interpolated frames. Fewer than two snapshots means no GIF.

## Potential modes and fixed fits

At `r=6`, the renderer uses periodic bilinear interpolation of actual cell-center potential values. At each angle it selects the finest level with all four interpolation corners valid, otherwise the next valid coarser level. Angular samples use one fixed uniform grid at the finest level present anywhere in that trajectory. The manifest records how many angular samples came from each level.

The coefficient is `sum(psi(r,theta_j)*exp(-i*mode*theta_j))/Ntheta`. Its modulus is normalized by the first positive-modulus potential sample with actual `potential_time <= 1e-8`. A later restart segment alone cannot supply this normalization; its normalized curve is omitted explicitly. `phi/alpha` and `phi` have the same normalized modulus. All curves use actual midpoint timestamps, including the normalization time.

The fixed exponential-fit windows from [Hoffart et al., Figure 5.4](https://arxiv.org/abs/2510.11808) are mode 3 `[0.4,0.7]`, mode 4 `[0.6,0.75]`, and mode 5 `[1.15,1.35]`. A fit requires the complete interval and at least three positive samples within it. The fit is linear least squares in log amplitude; its residuals and slope standard error are saved. Windows never move to suit the data.

The dashed reference is the nominal-density, vacuum-annulus linear drift-limit theory of [Davidson and Felice, equations 26–28](https://ccap.hep.ph.ic.ac.uk/trac/raw-attachment/wiki/Research/LhARA/GaborLens/Literature/1998_Davidson.pdf), using the recorded `alpha/abs(omega)` and geometry. It is anchored at the measured normalization time and evaluated only at observed times. The manifest separately records nominal-density theory, mean-ring-density theory and the paper's printed rates; these are not fitted PoPS results. The reference does not model the finite background or finite perturbation.

`render-manifest.json` preserves snapshot hashes, timestamps, masks' cell census, transforms, mass comparisons, Fourier samples, fits and missing data. CSV files preserve actual complex coefficients and their timestamps. An empty input produces a missing-data manifest and a nonzero exit without scientific figures. Successful rendering by itself does not qualify the simulation.
