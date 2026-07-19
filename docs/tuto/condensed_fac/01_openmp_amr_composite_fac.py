#!/usr/bin/env python3
"""Source implicite condensee sur AMR avec un solve composite FAC natif.

Ce parcours avance elimine deux moments couples, construit un operateur tensoriel scalaire,
puis le resout une seule fois sur toute la hierarchie AMR. Python ne fait qu'authorer le graphe ;
les assemblages, les iterations FAC et la reconstruction s'executent en C++/Kokkos OpenMP.
"""

# ruff: noqa: E402

import numpy as np
import pops

pops.set_threads(7)

from pops.amr import (
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    AMRTransfer,
    Buffer,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    Tag,
)
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Outflow
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import BindArray, Gaussian
from pops.linalg import LinearProblem
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.physics import Density, Momentum
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.runtime_environment import runtime_environment_report
from pops.solvers import CompositeTensorFAC, Hierarchy
from pops.spaces import CellState
from pops.time import FailRun, FixedDt, every


N = 16
ROTATION_RATE = 3.0
DT = 1.0e-2


# 1. Le domaine non periodique fournit une fermeture elliptique de Dirichlet.
domain = Rectangle(
    "unit_square",
    lower=(0.0, 0.0),
    upper=(1.0, 1.0),
).tag("fluid")
frame = domain.frame(Cartesian2D())
x_axis, y_axis = frame.axes
grid = CartesianGrid(frame=frame, cells=(N, N))


# 2. Etat physique : densite et deux moments couples par une rotation locale.
model = pops.Model("condensed_rotation", frame=frame)
U = model.state(
    "U",
    components=("density", "east_momentum", "north_momentum"),
    representation=Conservative(),
    space=CellState(frame=frame),
    roles={
        "density": Density(),
        "east_momentum": Momentum(axis=x_axis),
        "north_momentum": Momentum(axis=y_axis),
    },
)
density, east_momentum, north_momentum = U

zero_flux = (
    0.0 * density,
    0.0 * east_momentum,
    0.0 * north_momentum,
)
physical_flux = model.flux(
    "inert_flux",
    frame=frame,
    state=U,
    components={x_axis: zero_flux, y_axis: zero_flux},
    waves={x_axis: (0.0, 0.0, 0.0), y_axis: (0.0, 0.0, 0.0)},
)
inert_rate = model.rate("inert_rate", equation=ddt(U) == -div(physical_flux))

implicit_rotation = model.operator(
    "implicit_rotation",
    returns=model.local_linear_operator(
        "rotation_matrix",
        on=U,
        matrix=(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, ROTATION_RATE),
            (0.0, -ROTATION_RATE, 0.0),
        ),
    ),
)


# 3. Un marqueur scalaire immobile porte uniquement le critere de raffinement.
marker_model = pops.Model("mesh_marker", frame=frame)
marker_state = marker_model.state(
    "U",
    components=("marker",),
    representation=Conservative(),
    space=CellState(frame=frame),
    roles={"marker": Density()},
)
(marker,) = marker_state
marker_flux = marker_model.flux(
    "inert_marker_flux",
    frame=frame,
    state=marker_state,
    components={x_axis: (0.0 * marker,), y_axis: (0.0 * marker,)},
    waves={x_axis: (0.0,), y_axis: (0.0,)},
)
marker_rate = marker_model.rate(
    "inert_marker_rate",
    equation=ddt(marker_state) == -div(marker_flux),
)


# 4. Les deux blocs gardent des plans numeriques explicites et independants.
case = pops.Case("tutorial_condensed_fac")
plasma = case.block("plasma", model=model)
mesh_marker = case.block("mesh_marker", model=marker_model)
plasma_U = plasma[U]
marker_U = mesh_marker[marker_state]
boundaries = frame.boundaries

plasma_numerics = DiscretizationPlan()
plasma_numerics.rates.add(
    inert_rate,
    FiniteVolume(
        flux=physical_flux,
        variables=variables.Conservative(U),
        reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.Rusanov(),
    ),
)
plasma_numerics.boundaries.add(TransportBoundarySet({
    boundaries.x_min: Outflow(state=plasma_U),
    boundaries.x_max: Outflow(state=plasma_U),
    boundaries.y_min: Outflow(state=plasma_U),
    boundaries.y_max: Outflow(state=plasma_U),
}))
case.numerics(plasma_numerics, block=plasma)

marker_numerics = DiscretizationPlan()
marker_numerics.rates.add(
    marker_rate,
    FiniteVolume(
        flux=marker_flux,
        variables=variables.Conservative(marker_state),
        reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.Rusanov(),
    ),
)
marker_numerics.boundaries.add(TransportBoundarySet({
    boundaries.x_min: Outflow(state=marker_U),
    boundaries.x_max: Outflow(state=marker_U),
    boundaries.y_min: Outflow(state=marker_U),
    boundaries.y_max: Outflow(state=marker_U),
}))
case.numerics(marker_numerics, block=mesh_marker)


# 5. Condensation : assemblage de A, du second membre, solve FAC, reconstruction.
program = pops.Program("condensed_tensor_fac_step")
q = program.state(plasma_U)
q_marker = program.state(marker_U)
scope = Hierarchy()

coefficients = program.condensed_coeffs(
    "tensor_coefficients",
    state=q.n,
    linear_operator=implicit_rotation,
    subset=(1, 2),
    c=program.dt * program.dt,
    th_dt=program.dt,
    c_rho=0,
)
previous_potential = program.history(
    "plasma.potential",
    lag=1,
    ncomp=1,
    block=plasma,
)
rhs_storage = program.scalar_field("tensor_rhs_storage")
rhs = program.condensed_rhs(
    rhs_storage,
    previous_potential,
    q.n,
    linear_operator=implicit_rotation,
    subset=(1, 2),
    th_dt=program.dt,
    g=program.dt,
)

operator = program.matrix_free_operator("condensed_tensor", scope=scope)
program.set_apply(
    operator,
    lambda builder, _out, value: -1
    * builder.apply_laplacian_coeff(
        builder.scalar_field("tensor_laplacian"),
        value,
        coefficients,
    ),
)

potential = program.solve(
    LinearProblem(
        operator,
        rhs,
        initial_guess=previous_potential,
        scope=scope,
        nullspace=None,
    ),
    solver=CompositeTensorFAC(
        max_iter=30,
        rel_tol=1.0e-9,
        abs_tol=1.0e-12,
    ),
    name="composite_potential",
).consume(action=FailRun())
program.store_history("plasma.potential", potential)

reconstructed = program.condensed_reconstruct(
    "reconstructed_state",
    state=q.n,
    phi=potential,
    linear_operator=implicit_rotation,
    subset=(1, 2),
    th_dt=program.dt,
    c_rho=0,
)
program.commit(
    q.next,
    program.value("accepted_state", reconstructed, at=q.next.point),
)
program.commit(
    q_marker.next,
    program.value("accepted_marker", q_marker.n, at=q_marker.next.point),
)
program.step_strategy(FixedDt(DT))
case.program(program)


# 6. Le vecteur physique est lie explicitement ; le marqueur est analytique.
case.initials.add(InitialCondition(
    state=plasma_U,
    value=BindArray(),
    projection=ConservativeCellAverage(),
))
case.initials.add(InitialCondition(
    state=marker_U,
    value=Gaussian(
        frame=frame,
        center={x_axis: 0.50, y_axis: 0.50},
        background=0.0,
        amplitude=1.0,
        inverse_width=100.0,
    ),
    projection=ConservativeCellAverage(),
))

coordinate = (np.arange(N, dtype=np.float64) + 0.5) / N
x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
density_initial = 1.0 + 0.20 * np.exp(
    -80.0 * ((x - 0.40) ** 2 + (y - 0.55) ** 2)
)
east_initial = density_initial * (
    0.25 + 0.08 * np.sin(2.0 * np.pi * y)
)
north_initial = density_initial * (
    -0.15 + 0.06 * np.cos(2.0 * np.pi * x)
)
initial_state = np.ascontiguousarray(np.stack((
    density_initial,
    east_initial,
    north_initial,
)))


# 7. Deux niveaux synchrones : un solve FAC composite couvre les deux niveaux.
refine_threshold = case.param(RuntimeParam("refine_marker", default=0.20))
tagging = AMRTagging(
    rules=(
        Tag(ValueExpr(marker_U) > case.value(refine_threshold)),
        Buffer(cells=1),
    ),
    hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
    conflict_policy=ConflictPolicy.REFINE_WINS,
)

transfer = AMRTransfer()
transfer.state(plasma_U, StateTransfer())
transfer.state(marker_U, StateTransfer())

layout = AMR(
    grid=grid,
    hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
    tagging=tagging,
    regrid=AMRRegrid(schedule=every(100, clock=program.clock)),
    transfer=transfer,
    execution=AMRExecution.synchronous(),
)


# 8. Cycle public final.
validated = pops.validate(case)
resolved = pops.resolve(validated, layout=layout)
artifact = pops.compile(resolved)
simulation = pops.bind(
    artifact,
    initial_values={plasma_U: initial_state},
)
report = pops.run(simulation, t_end=DT, max_steps=1)

final_state = np.asarray(
    simulation.block_level_state_global("plasma", 0),
    dtype=np.float64,
).reshape(initial_state.shape)
patches = simulation.amr.patch_table()

print("PoPS condensed composite-FAC tutorial finished")
print("  solver           : pops.solvers.CompositeTensorFAC")
print("  scope            : pops.solvers.Hierarchy")
print("  Kokkos backend   : %s" % runtime_environment_report()["kokkos_backend"])
print("  requested threads: 7 via pops.set_threads")
print("  accepted steps   : %d" % report.accepted_steps)
print("  final time       : %.6f" % simulation.time())
print("  AMR levels       : %d" % simulation.n_levels())
print("  fine patches     : %d" % patches.n_patches)
print("  initial moment L2: %.12e" % np.linalg.norm(initial_state[1:]))
print("  final moment L2  : %.12e" % np.linalg.norm(final_state[1:]))
