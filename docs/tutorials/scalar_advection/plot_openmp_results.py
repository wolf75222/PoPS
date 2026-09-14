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
RELATIVE_L2_TOLERANCE = 0.10

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
    far_field = float(data["far_field"])
    gaussian_amplitude = float(data["gaussian_amplitude"])
    gaussian_beta = float(data["gaussian_beta"])
    gaussian_center_x = float(data["gaussian_center_x"])
    gaussian_center_y = float(data["gaussian_center_y"])
    t_end = float(data["t_end"])

with np.load(EXPLICIT_RESULT, allow_pickle=False) as data:
    explicit_final = np.asarray(data["final"], dtype=np.float64)[0]
    explicit_metadata = {
        name: float(data[name])
        for name in (
            "ax",
            "ay",
            "far_field",
            "gaussian_amplitude",
            "gaussian_beta",
            "gaussian_center_x",
            "gaussian_center_y",
            "t_end",
        )
    }

if initial.shape != (ny, nx) or library_final.shape != initial.shape:
    raise ValueError("the saved tutorial fields do not match the authored grid")
if explicit_final.shape != library_final.shape:
    raise ValueError("the preset and explicit runs use different field layouts")
reference_metadata = {
    "ax": ax,
    "ay": ay,
    "far_field": far_field,
    "gaussian_amplitude": gaussian_amplitude,
    "gaussian_beta": gaussian_beta,
    "gaussian_center_x": gaussian_center_x,
    "gaussian_center_y": gaussian_center_y,
    "t_end": t_end,
}
if explicit_metadata != reference_metadata:
    raise ValueError("the preset and explicit runs describe different physical cases")

difference = explicit_final - library_final
max_difference = float(np.max(np.abs(difference)))

x = (np.arange(nx, dtype=np.float64) + 0.5) / nx
y = (np.arange(ny, dtype=np.float64) + 0.5) / ny
xx, yy = np.meshgrid(x, y, indexing="xy")
departure_x = xx - ax * t_end
departure_y = yy - ay * t_end
inside_initial_domain = (
    (departure_x >= 0.0)
    & (departure_x <= 1.0)
    & (departure_y >= 0.0)
    & (departure_y <= 1.0)
)
analytic = np.full_like(library_final, far_field)
analytic[inside_initial_domain] = (
    far_field
    + gaussian_amplitude
    * np.exp(
        -gaussian_beta
        * (
            (departure_x[inside_initial_domain] - gaussian_center_x) ** 2
            + (departure_y[inside_initial_domain] - gaussian_center_y) ** 2
        )
    )
)

analytic_error = library_final - analytic
cell_area = 1.0 / (nx * ny)
l1_error = float(cell_area * np.sum(np.abs(analytic_error)))
l2_error = float(np.sqrt(cell_area * np.sum(analytic_error**2)))
linf_error = float(np.max(np.abs(analytic_error)))
analytic_perturbation_l2 = float(
    np.sqrt(cell_area * np.sum((analytic - far_field) ** 2))
)
relative_l2_error = l2_error / analytic_perturbation_l2

expected_center = (
    gaussian_center_x + ax * t_end,
    gaussian_center_y + ay * t_end,
)
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
axis.plot(x, analytic[cut_index], "--", linewidth=2.0, label="analytic")
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

figure, axes = plt.subplots(2, 2, figsize=(10.4, 8.0), constrained_layout=True)
field_limits = (float(analytic.min()), float(analytic.max()))

numerical_image = axes[0, 0].imshow(
    library_final,
    origin="lower",
    extent=(0.0, 1.0, 0.0, 1.0),
    vmin=field_limits[0],
    vmax=field_limits[1],
    cmap="viridis",
)
axes[0, 0].set_title("PoPS at t = %.2f" % t_end)

axes[0, 1].imshow(
    analytic,
    origin="lower",
    extent=(0.0, 1.0, 0.0, 1.0),
    vmin=field_limits[0],
    vmax=field_limits[1],
    cmap="viridis",
)
axes[0, 1].set_title("Analytic solution")

error_image = axes[1, 0].imshow(
    np.abs(analytic_error),
    origin="lower",
    extent=(0.0, 1.0, 0.0, 1.0),
    vmin=0.0,
    cmap="magma",
)
axes[1, 0].set_title(
    "Absolute error\n"
    r"$L_1=%.2e,\ L_2=%.2e,\ L_\infty=%.2e$"
    % (l1_error, l2_error, linf_error)
)

axes[1, 1].plot(
    x,
    analytic[cut_index],
    "--",
    linewidth=2.2,
    label="analytic",
)
axes[1, 1].plot(
    x,
    library_final[cut_index],
    linewidth=2.0,
    label="PoPS",
)
axes[1, 1].set_title(
    "Cut at y = %.3f\nrelative L2 = %.2e" % (
        y[cut_index],
        relative_l2_error,
    )
)
axes[1, 1].set_ylabel("u")
axes[1, 1].grid(alpha=0.25)
axes[1, 1].legend()

for axis in axes.flat:
    axis.set_xlabel("x")
for axis in axes[:, 0]:
    axis.set_ylabel("y")
figure.colorbar(numerical_image, ax=axes[0, :], shrink=0.86, label="u")
figure.colorbar(error_image, ax=axes[1, 0], shrink=0.86, label="|PoPS - analytic|")

analytic_figure = FIGURE_DIR / "scalar_advection_analytic_verification.png"
figure.savefig(analytic_figure, dpi=180)
plt.close(figure)

print("wrote %s" % field_figure)
print("wrote %s" % cut_figure)
print("wrote %s" % analytic_figure)
print("max |explicit - preset| = %.16e" % max_difference)
print("analytic L1 error        = %.16e" % l1_error)
print("analytic L2 error        = %.16e" % l2_error)
print("analytic Linf error      = %.16e" % linf_error)
print("analytic relative L2     = %.16e" % relative_l2_error)

if relative_l2_error > RELATIVE_L2_TOLERANCE:
    raise RuntimeError(
        "analytic verification failed: relative L2 %.6e exceeds %.6e"
        % (relative_l2_error, RELATIVE_L2_TOLERANCE)
    )
