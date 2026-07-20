#!/usr/bin/env python3
"""Trace les champs ecrits par 01_openmp_diocotron_hll.py."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULT_FILE = HERE / "results" / "01_openmp_diocotron_hll.npz"
FIGURE_FILE = HERE / "figures" / "hyqmom15_diocotron.png"

with np.load(RESULT_FILE, allow_pickle=False) as result:
    initial = np.asarray(result["initial"], dtype=np.float64)
    final = np.asarray(result["final"], dtype=np.float64)
    potential = np.asarray(result["last_stage_potential"], dtype=np.float64)
    x = np.asarray(result["x"], dtype=np.float64)
    y = np.asarray(result["y"], dtype=np.float64)

density = final[0]
if not np.isfinite(final).all() or not np.isfinite(potential).all():
    raise RuntimeError("the saved HyQMOM result contains a non-finite value")
if np.any(density <= 0.0):
    raise RuntimeError("the saved HyQMOM density is not strictly positive")
velocity_x = final[1] / density
velocity_y = final[5] / density
speed = np.hypot(velocity_x, velocity_y)
extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))

figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
fields = (
    (initial[0], "Densite initiale"),
    (density, "Densite finale"),
    (potential, "Potentiel au debut du dernier pas"),
    (speed, "Vitesse moyenne finale"),
)

for axis, (field, title) in zip(axes.flat, fields, strict=True):
    image = axis.imshow(field, origin="lower", extent=extent, cmap="viridis")
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal")
    figure.colorbar(image, ax=axis)

FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(FIGURE_FILE, dpi=180)
print("Figure written to %s" % FIGURE_FILE)
