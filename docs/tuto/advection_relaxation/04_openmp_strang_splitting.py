#!/usr/bin/env python3
"""Relaxation, advection, relaxation avec un splitting de Strang explicite.

Le programme compose S(dt/2), T(dt), S(dt/2). Les demi-pas sont portes par
des fractions exactes dans le graphe temporel. Python decrit ce graphe ; le
calcul des cellules est execute en C++/Kokkos.
"""

# ruff: noqa: E402

from fractions import Fraction

import numpy as np
import pops

pops.set_threads(7)

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.representations import Conservative
from pops.runtime_environment import runtime_environment_report
from pops.spaces import CellState
from pops.time import FixedDt, StagePoint, TimePoint


NX = 32
NY = 32
AX = 1.0
AY = 0.25
RELAXATION_RATE = 2.0
DT = 2.0e-3
T_END = 2.0e-2
MAX_STEPS = 100


# 1. Domaine periodique et maillage cartesien uniforme.
domain = Rectangle(
    "unit_square",
    lower=(0.0, 0.0),
    upper=(1.0, 1.0),
).tag("fluid")
frame = domain.frame(Cartesian2D())
x_axis, y_axis = frame.axes
grid = CartesianGrid(
    frame=frame,
    cells=(NX, NY),
    periodic=PeriodicAxes(frame.axes),
)


# 2. Physique : d_t u + div(a u) = -lambda u.
model = pops.Model("advection_relaxation_strang", frame=frame)
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
relaxation_operator = model.operator(
    "relaxation_operator",
    returns=model.local_linear_operator(
        "relaxation_matrix",
        on=U,
        matrix=((-RELAXATION_RATE,),),
    ),
)


# 3. Le transport est discretise par volumes finis et flux upwind.
finite_volume = FiniteVolume(
    flux=physical_flux,
    variables=variables.Conservative(U),
    reconstruction=reconstruction.FirstOrder(),
    riemann=riemann.ScalarUpwind(velocity=velocity),
)
numerics = DiscretizationPlan()
numerics.rates.add(advection_rate, finite_volume)


# 4. Case et etat qualifie.
case = pops.Case("tutorial_advection_relaxation_strang")
tracer = case.block("tracer", model=model)
tracer_U = tracer[U]
case.numerics(numerics, block=tracer)


# 5. Premier demi-pas de relaxation avec RK2 explicite.
program = pops.Program("Strang explicit")
q = program.state(tracer_U)
relaxation_map = relaxation_operator(program=program)

source_rate_0 = program.apply(
    relaxation_map,
    q.n,
    name="source_rate_0",
)
source_predictor_0 = program.value(
    "source_predictor_0",
    q.n + Fraction(1, 2) * program.dt * source_rate_0,
    at=StagePoint(
        "source_predictor_0",
        {
            "source": TimePoint(program.clock, Fraction(1, 2)),
            "explicit": TimePoint(program.clock, 0),
        },
    ),
)
source_rate_1 = program.apply(
    relaxation_map,
    source_predictor_0,
    name="source_rate_1",
)
after_source_half = program.value(
    "after_source_half",
    q.n + Fraction(1, 4) * program.dt * (source_rate_0 + source_rate_1),
    at=StagePoint(
        "after_source_half",
        {
            "source": TimePoint(program.clock, Fraction(1, 2)),
            "explicit": TimePoint(program.clock, 0),
        },
    ),
)


# 6. Pas complet de transport avec SSPRK2.
transport_rate_0 = program.value(
    "transport_rate_0",
    advection_rate(after_source_half),
    at=after_source_half.point,
)
transport_predictor = program.value(
    "transport_predictor",
    after_source_half + program.dt * transport_rate_0,
    at=StagePoint(
        "transport_predictor",
        {
            "source": TimePoint(program.clock, Fraction(1, 2)),
            "explicit": TimePoint(program.clock, 1),
        },
    ),
)
transport_rate_1 = program.value(
    "transport_rate_1",
    advection_rate(transport_predictor),
    at=transport_predictor.point,
)
after_transport = program.value(
    "after_transport",
    after_source_half
    + Fraction(1, 2) * program.dt * (transport_rate_0 + transport_rate_1),
    at=StagePoint(
        "after_transport",
        {
            "source": TimePoint(program.clock, Fraction(1, 2)),
            "explicit": TimePoint(program.clock, 1),
        },
    ),
)


# 7. Second demi-pas de relaxation, lui aussi en RK2 explicite.
source_rate_2 = program.apply(
    relaxation_map,
    after_transport,
    name="source_rate_2",
)
source_predictor_1 = program.value(
    "source_predictor_1",
    after_transport + Fraction(1, 2) * program.dt * source_rate_2,
    at=StagePoint(
        "source_predictor_1",
        {
            "source": TimePoint(program.clock, 1),
            "explicit": TimePoint(program.clock, 1),
        },
    ),
)
source_rate_3 = program.apply(
    relaxation_map,
    source_predictor_1,
    name="source_rate_3",
)
next_state = program.value(
    "strang_step",
    after_transport
    + Fraction(1, 4) * program.dt * (source_rate_2 + source_rate_3),
    at=q.next.point,
)
program.commit(q.next, next_state)
program.step_strategy(FixedDt(DT))
case.program(program)


# 8. Une bosse gaussienne permet de voir simultanement transport et amortissement.
x = (np.arange(NX, dtype=np.float64) + 0.5) / NX
y = (np.arange(NY, dtype=np.float64) + 0.5) / NY
xx, yy = np.meshgrid(x, y, indexing="xy")
initial_u = 0.05 + 0.95 * np.exp(
    -100.0 * ((xx - 0.30) ** 2 + (yy - 0.35) ** 2)
)
initial_state = np.ascontiguousarray(initial_u[np.newaxis, :, :])


# 9. Validation, compilation, liaison et execution du meme Program.
validated = pops.validate(case)
resolved = pops.resolve(validated, layout=Uniform(grid))
artifact = pops.compile(resolved)
simulation = pops.bind(
    artifact,
    initial_state={"tracer": initial_state},
)
report = pops.run(
    simulation,
    t_end=T_END,
    max_steps=MAX_STEPS,
)

final_state = np.asarray(
    simulation.state_global("tracer"), dtype=np.float64
).reshape(initial_state.shape)

print("PoPS Strang splitting tutorial finished")
print("  substeps       : relaxation(dt/2) -> transport(dt) -> relaxation(dt/2)")
print("  Kokkos backend : %s" % runtime_environment_report()["kokkos_backend"])
print("  accepted steps : %d" % report.accepted_steps)
print("  final time     : %.6f" % simulation.time())
print("  initial L2     : %.12e" % np.linalg.norm(initial_state))
print("  final L2       : %.12e" % np.linalg.norm(final_state))
