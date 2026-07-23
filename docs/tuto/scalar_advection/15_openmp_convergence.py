#!/usr/bin/env python3
"""Etude de convergence de l'advection scalaire 2D vers la solution analytique.

Le meme probleme est resolu sur quatre maillages uniformes. Le nombre CFL reste
constant : le pas de temps diminue donc avec la taille des cellules. Les erreurs
spatiales et temporelles sont ainsi raffinees ensemble.
"""

# ruff: noqa: E402

from pathlib import Path

import matplotlib
import numpy as np

import pops

matplotlib.use("Agg")

# La configuration OpenMP precede toute initialisation native de Kokkos.
pops.set_threads(7)

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Inflow, Outflow
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import SSPRK2
from pops.math import ddt, div
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import AdaptiveCFL


RESOLUTIONS = np.asarray((32, 64, 128, 256), dtype=np.int64)
AX = 1.0
AY = 0.25
FAR_FIELD = 0.05
GAUSSIAN_AMPLITUDE = 0.95
GAUSSIAN_BETA = 120.0
GAUSSIAN_CENTER_X = 0.30
GAUSSIAN_CENTER_Y = 0.35
CFL = 0.45
T_END = 0.20
MAX_STEPS = 10_000

HERE = Path(__file__).resolve().parent
RESULT_FILE = HERE / "results" / "15_openmp_convergence.npz"
FIGURE_FILE = HERE / "figures" / "scalar_advection_convergence.png"


def gaussian(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Valeur initiale analytique aux points (x, y)."""
    return FAR_FIELD + GAUSSIAN_AMPLITUDE * np.exp(
        -GAUSSIAN_BETA
        * (
            (x - GAUSSIAN_CENTER_X) ** 2
            + (y - GAUSSIAN_CENTER_Y) ** 2
        )
    )


def analytic_solution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solution exacte a T_END obtenue en remontant les caracteristiques."""
    departure_x = x - AX * T_END
    departure_y = y - AY * T_END
    inside = (
        (departure_x >= 0.0)
        & (departure_x <= 1.0)
        & (departure_y >= 0.0)
        & (departure_y <= 1.0)
    )
    exact = np.full_like(x, FAR_FIELD)
    exact[inside] = gaussian(departure_x[inside], departure_y[inside])
    return exact


# 1. Le domaine, le modele et la discretisation sont identiques pour les quatre grilles.
domain = Rectangle(
    "unit_square",
    lower=(0.0, 0.0),
    upper=(1.0, 1.0),
).tag("fluid")

frame = domain.frame(Cartesian2D())
x_axis, y_axis = frame.axes

model = pops.Model("scalar_advection", frame=frame)
U = model.state(
    "U",
    components=("u",),
    representation=Conservative(),
    space=CellState(frame=frame),
)
(u,) = U

velocity = model.vector(
    "a",
    frame=frame,
    components={x_axis: AX, y_axis: AY},
)
physical_flux = model.flux(
    "advection_flux",
    frame=frame,
    state=U,
    components={x_axis: (AX * u,), y_axis: (AY * u,)},
    waves={x_axis: (AX,), y_axis: (AY,)},
)
advection_rate = model.rate(
    "advection_rate",
    equation=ddt(U) == -div(physical_flux),
)

finite_volume = FiniteVolume(
    flux=physical_flux,
    variables=variables.Conservative(U),
    reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
    riemann=riemann.ScalarUpwind(velocity=velocity),
)
numerics = DiscretizationPlan()
numerics.rates.add(advection_rate, finite_volume)

case = pops.Case("tutorial_scalar_advection_convergence")
tracer = case.block("tracer", model=model)
tracer_U = tracer[U]

boundaries = frame.boundaries
numerics.boundaries.add(
    TransportBoundarySet({
        boundaries.x_min: Inflow(state=tracer_U, value=FAR_FIELD),
        boundaries.x_max: Outflow(state=tracer_U),
        boundaries.y_min: Inflow(state=tracer_U, value=FAR_FIELD),
        boundaries.y_max: Outflow(state=tracer_U),
    })
)
case.numerics(numerics, block=tracer)

program = SSPRK2(tracer_U, rate=advection_rate)
# La grande borne max_dt laisse la CFL choisir dt a toutes les resolutions.
program.step_strategy(AdaptiveCFL(cfl=CFL, max_dt=1.0))
case.program(program)
validated = pops.validate(case)


# 2. Chaque resolution suit exactement validate -> resolve -> compile -> bind -> run.
l1_errors: list[float] = []
l2_errors: list[float] = []
linf_errors: list[float] = []
relative_l2_errors: list[float] = []

for resolution in RESOLUTIONS:
    n = int(resolution)
    coordinates = (np.arange(n, dtype=np.float64) + 0.5) / n
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")

    initial_u = gaussian(xx, yy)
    initial_state = np.ascontiguousarray(
        initial_u[np.newaxis, :, :],
        dtype=np.float64,
    )

    grid = CartesianGrid(frame=frame, cells=(n, n))
    resolved = pops.resolve(validated, layout=Uniform(grid))
    artifact = pops.compile(resolved)
    simulation = pops.bind(
        artifact,
        initial_state={"tracer": initial_state},
    )
    pops.run(simulation, t_end=T_END, max_steps=MAX_STEPS)

    numerical = np.asarray(
        simulation.state_global("tracer"),
        dtype=np.float64,
    ).reshape(initial_state.shape)[0]
    exact = analytic_solution(xx, yy)
    error = numerical - exact
    cell_area = 1.0 / float(n * n)

    l1 = cell_area * np.sum(np.abs(error))
    l2 = np.sqrt(cell_area * np.sum(error**2))
    linf = np.max(np.abs(error))
    perturbation_l2 = np.sqrt(
        cell_area * np.sum((exact - FAR_FIELD) ** 2)
    )

    l1_errors.append(float(l1))
    l2_errors.append(float(l2))
    linf_errors.append(float(linf))
    relative_l2_errors.append(float(l2 / perturbation_l2))

l1_errors_array = np.asarray(l1_errors)
l2_errors_array = np.asarray(l2_errors)
linf_errors_array = np.asarray(linf_errors)
relative_l2_errors_array = np.asarray(relative_l2_errors)


# 3. Pour un raffinement par deux, p = log(E_h/E_h/2) / log(2).
def observed_orders(errors: np.ndarray) -> np.ndarray:
    return np.log(errors[:-1] / errors[1:]) / np.log(2.0)


l1_orders = observed_orders(l1_errors_array)
l2_orders = observed_orders(l2_errors_array)
linf_orders = observed_orders(linf_errors_array)

for name, errors in (
    ("L1", l1_errors_array),
    ("L2", l2_errors_array),
    ("Linf", linf_errors_array),
):
    if not np.all(np.diff(errors) < 0.0):
        raise RuntimeError(
            f"convergence verification failed: {name} does not decrease"
        )


# 4. Les valeurs brutes et les ordres restent disponibles pour une analyse ulterieure.
RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    RESULT_FILE,
    resolutions=RESOLUTIONS,
    l1=l1_errors_array,
    l2=l2_errors_array,
    linf=linf_errors_array,
    relative_l2=relative_l2_errors_array,
    l1_orders=l1_orders,
    l2_orders=l2_orders,
    linf_orders=linf_orders,
    cfl=CFL,
    t_end=T_END,
)


# 5. Figure log-log des erreurs et ordre mesure entre deux maillages successifs.
FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)
figure, (error_axis, order_axis) = plt.subplots(
    1,
    2,
    figsize=(12.0, 4.8),
    constrained_layout=True,
)

for label, errors, marker in (
    (r"$L^1$", l1_errors_array, "o"),
    (r"$L^2$", l2_errors_array, "s"),
    (r"$L^\infty$", linf_errors_array, "^"),
):
    error_axis.loglog(
        RESOLUTIONS,
        errors,
        marker=marker,
        linewidth=2.0,
        markersize=6.0,
        label=label,
    )

resolution_ratio = RESOLUTIONS / RESOLUTIONS[0]
error_axis.loglog(
    RESOLUTIONS,
    l2_errors_array[0] * resolution_ratio.astype(np.float64) ** -1,
    linestyle=":",
    color="0.45",
    label=r"reference $N^{-1}$",
)
error_axis.loglog(
    RESOLUTIONS,
    l2_errors_array[0] * resolution_ratio.astype(np.float64) ** -2,
    linestyle="--",
    color="0.25",
    label=r"reference $N^{-2}$",
)
error_axis.set(
    xlabel=r"Cells per direction $N$",
    ylabel="Error",
    title="Convergence toward the analytic solution",
)
error_axis.set_xticks(RESOLUTIONS, labels=[str(n) for n in RESOLUTIONS])
error_axis.xaxis.set_major_formatter(ScalarFormatter())
error_axis.xaxis.set_minor_formatter(NullFormatter())
error_axis.grid(True, which="both", alpha=0.25)
error_axis.legend()

fine_resolutions = RESOLUTIONS[1:]
for label, orders, marker in (
    (r"$L^1$", l1_orders, "o"),
    (r"$L^2$", l2_orders, "s"),
    (r"$L^\infty$", linf_orders, "^"),
):
    order_axis.plot(
        fine_resolutions,
        orders,
        marker=marker,
        linewidth=2.0,
        markersize=6.0,
        label=label,
    )

order_axis.axhline(1.0, linestyle=":", color="0.45", label="order 1")
order_axis.axhline(2.0, linestyle="--", color="0.25", label="order 2")
order_axis.set(
    xlabel="Fine-grid cells per direction",
    ylabel="Observed order",
    title=r"$p=\log(E_N/E_{2N})/\log(2)$",
)
order_axis.set_xticks(
    fine_resolutions,
    labels=[str(n) for n in fine_resolutions],
)
order_axis.grid(True, alpha=0.25)
order_axis.legend()

figure.savefig(FIGURE_FILE, dpi=180)
plt.close(figure)


print("\nScalar-advection convergence study")
print("  N             L1             L2           Linf       rel. L2")
for index, resolution in enumerate(RESOLUTIONS):
    print(
        f"{resolution:4d}  "
        f"{l1_errors_array[index]:13.6e}  "
        f"{l2_errors_array[index]:13.6e}  "
        f"{linf_errors_array[index]:13.6e}  "
        f"{relative_l2_errors_array[index]:10.4%}"
    )

print("\nObserved orders (coarse -> fine)")
print("  pair          p(L1)     p(L2)   p(Linf)")
for index, fine_resolution in enumerate(fine_resolutions):
    coarse_resolution = RESOLUTIONS[index]
    print(
        f"{coarse_resolution:4d}->{fine_resolution:<4d}  "
        f"{l1_orders[index]:8.4f}  "
        f"{l2_orders[index]:8.4f}  "
        f"{linf_orders[index]:8.4f}"
    )

print(f"\n  result : {RESULT_FILE}")
print(f"  figure : {FIGURE_FILE}")
