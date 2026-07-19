#!/usr/bin/env python3
"""Genere les figures a partir des deux executions OpenMP du tutoriel."""

from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
LIBRARY_RESULT = HERE / "results" / "01_openmp_preset_ssprk2.npz"
EXPLICIT_RESULT = HERE / "results" / "02_openmp_explicit_ssprk2.npz"
FIGURE_DIR = HERE / "figures"

if not LIBRARY_RESULT.is_file() or not EXPLICIT_RESULT.is_file():
    raise FileNotFoundError(
        "run 01_openmp_preset_ssprk2.py and 02_openmp_explicit_ssprk2.py "
        "before generating figures"
    )

with np.load(LIBRARY_RESULT, allow_pickle=False) as data:
    initial = np.asarray(data["initial"], dtype=np.float64)[0]
    library_final = np.asarray(data["final"], dtype=np.float64)[0]
    nx = int(data["nx"])
    ny = int(data["ny"])
    ax = float(data["ax"])
    ay = float(data["ay"])
    t_end = float(data["t_end"])

with np.load(EXPLICIT_RESULT, allow_pickle=False) as data:
    explicit_final = np.asarray(data["final"], dtype=np.float64)[0]

if initial.shape != (ny, nx) or library_final.shape != initial.shape:
    raise ValueError("the saved tutorial fields do not match the authored grid")
if explicit_final.shape != library_final.shape:
    raise ValueError("the preset and explicit runs use different field layouts")

difference = explicit_final - library_final
max_difference = float(np.max(np.abs(difference)))

x = (np.arange(nx, dtype=np.float64) + 0.5) / nx
y = (np.arange(ny, dtype=np.float64) + 0.5) / ny
expected_center = (0.30 + ax * t_end, 0.35 + ay * t_end)
cut_index = int(np.argmin(np.abs(y - expected_center[1])))

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
color_limits = (float(initial.min()), float(initial.max()))

image = axes[0].imshow(
    initial,
    origin="lower",
    extent=(0.0, 1.0, 0.0, 1.0),
    vmin=color_limits[0],
    vmax=color_limits[1],
    cmap="viridis",
)
axes[0].set_title("Initial condition")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

axes[1].imshow(
    library_final,
    origin="lower",
    extent=(0.0, 1.0, 0.0, 1.0),
    vmin=color_limits[0],
    vmax=color_limits[1],
    cmap="viridis",
)
axes[1].set_title("PoPS at t = %.2f" % t_end)
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

difference_scale = max(max_difference, np.finfo(np.float64).eps)
axes[2].imshow(
    difference,
    origin="lower",
    extent=(0.0, 1.0, 0.0, 1.0),
    vmin=-difference_scale,
    vmax=difference_scale,
    cmap="coolwarm",
)
axes[2].set_title("Explicit RK2 - preset\nmax = %.3e" % max_difference)
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")

figure.colorbar(image, ax=axes[:2], shrink=0.88, label="u")
field_figure = FIGURE_DIR / "scalar_advection_fields.png"
figure.savefig(field_figure, dpi=180)
plt.close(figure)

figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
axis.plot(x, initial[cut_index], "--", linewidth=2.0, label="initial")
axis.plot(x, library_final[cut_index], linewidth=2.0, label="SSPRK2 preset")
axis.plot(
    x,
    explicit_final[cut_index],
    ":",
    linewidth=2.0,
    label="explicit SSPRK2",
)
axis.axvline(expected_center[0], color="0.5", linewidth=1.0, label="expected center")
axis.set_title("Horizontal cut at y = %.3f" % y[cut_index])
axis.set_xlabel("x")
axis.set_ylabel("u")
axis.grid(alpha=0.25)
axis.legend()

cut_figure = FIGURE_DIR / "scalar_advection_cut.png"
figure.savefig(cut_figure, dpi=180)
plt.close(figure)

print("wrote %s" % field_figure)
print("wrote %s" % cut_figure)
print("max |explicit - preset| = %.16e" % max_difference)
