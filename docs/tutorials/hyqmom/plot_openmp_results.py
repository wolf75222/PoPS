#!/usr/bin/env python3
"""Trace les etats grossiers initial et final des sept cas HyQMOM AMR."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
CASES = (
    "01_openmp_diocotron_hll",
    "02_openmp_constant_hll",
    "03_openmp_fluid_wave_hll",
    "04_openmp_electrostatic_wave_hll",
    "05_openmp_magnetic_wave_hll",
    "06_openmp_shock_tube_hll",
    "07_openmp_crossing_jets_hll",
)


def load_case(name):
    path = RESULTS / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run the corresponding MPI tutorial first"
        )
    with np.load(path, allow_pickle=False) as result:
        initial = np.asarray(result["initial"], dtype=np.float64)
        final = np.asarray(result["final"], dtype=np.float64)
    if initial.shape != final.shape or initial.shape[0] != 15:
        raise RuntimeError(f"{path} does not contain two 15-moment states")
    if not np.isfinite(initial).all() or not np.isfinite(final).all():
        raise RuntimeError(f"{path} contains a non-finite value")
    return initial, final


FIGURES.mkdir(parents=True, exist_ok=True)
overview, overview_axes = plt.subplots(
    len(CASES), 2, figsize=(8, 3 * len(CASES)), constrained_layout=True,
)

for row, name in enumerate(CASES):
    initial, final = load_case(name)
    density = final[0]
    safe_density = np.maximum(np.abs(density), np.finfo(np.float64).tiny)
    speed = np.hypot(final[1] / safe_density, final[5] / safe_density)
    change = density - initial[0]

    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    fields = (
        (initial[0], "M00 initial"),
        (density, "M00 final"),
        (change, "Variation de M00"),
        (speed, "Norme de la vitesse moyenne"),
    )
    for axis, (field, title) in zip(axes.flat, fields, strict=True):
        image = axis.imshow(field.T, origin="lower", cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("indice x du niveau grossier")
        axis.set_ylabel("indice y du niveau grossier")
        axis.set_aspect("equal")
        figure.colorbar(image, ax=axis)
    destination = FIGURES / f"{name}.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    print("Figure written to %s" % destination)

    for axis, field, title in (
        (overview_axes[row, 0], density, "M00 final"),
        (overview_axes[row, 1], change, "Variation de M00"),
    ):
        image = axis.imshow(field.T, origin="lower", cmap="viridis")
        axis.set_title(f"{name}\n{title}")
        axis.set_xticks(())
        axis.set_yticks(())
        overview.colorbar(image, ax=axis)

overview_path = FIGURES / "hyqmom_amr_overview.png"
overview.savefig(overview_path, dpi=160)
plt.close(overview)
print("Overview written to %s" % overview_path)
