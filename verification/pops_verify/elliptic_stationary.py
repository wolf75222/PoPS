"""Author a public stationary Program so elliptic Cases can resolve.

One function, ``author_periodic_poisson``, is the sole authoring path used by
``build_case``, ``resolve_plan``, and native execution. The hyperbolic rate is
a zero-flux holder; it is not a private elliptic solver.
"""
from __future__ import annotations

from typing import Any

import pops
from pops.domain import CartesianDomain
from pops.fields import (
    CellCenteredSecondOrder,
    ConstantNullspace,
    FieldDiscretization,
    FieldOutput,
    GradientOutput,
    MeanValueGauge,
)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Periodic
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.lib.initial import BindArray
from pops.lib.time import ForwardEuler
from pops.math import ddt, div, laplacian
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.physics import Model
from pops.problem import Case
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt

from verification.pops_verify.case_authoring import (
    resolve_case,
    uniform_open_layout,
    uniform_periodic_layout,
)
from verification.pops_verify.native_toolchain import native_unavailable_reason


def author_periodic_poisson(
    *,
    case_name: str,
    model_name: str,
    domain_name: str,
    solver: Any,
    n_cells: int,
    lower=(0.0,),
    upper=(1.0,),
    boundaries=None,
    nullspace=True,
):
    """Author a 1-d Poisson Case with block/numerics/field/program order."""
    count = int(n_cells)
    frame = CartesianDomain(domain_name, lower, upper).frame(Cartesian1D())
    model = Model(model_name, frame=frame)
    state = model.state("U", components=["rhs"])
    (rhs,) = state
    axes = tuple(frame.axes)
    flux = model.flux(
        "stationary",
        frame=frame,
        state=state,
        components={axis: (0.0 * rhs,) for axis in axes},
        waves={axis: (0.0 * rhs,) for axis in axes},
    )
    rate = model.rate("hold", equation=ddt(state) == -div(flux))
    potential = model.field("potential")
    operator = model.field_operator(
        "poisson",
        unknown=potential,
        equation=(-laplacian(potential) == rhs),
        outputs=(
            FieldOutput("potential", potential),
            GradientOutput("electric_field", potential, sign=-1),
        ),
    )
    case = Case(case_name)
    block = case.block("electrostatic", model, states=(state,))
    instance = block[state]
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(numerics, block=block)
    bc = boundaries
    if bc is None:
        bc = (BoundaryCondition(AllPhysicalBoundaries(), Periodic()),)
    kwargs: dict[str, Any] = {
        "method": CellCenteredSecondOrder(),
        "boundaries": bc,
        "solver": solver,
    }
    if nullspace:
        kwargs["nullspace"] = ConstantNullspace()
        kwargs["gauge"] = MeanValueGauge(0.0)
    field = case.field(operator, FieldDiscretization(**kwargs))
    program = ForwardEuler(instance, rate=rate, fields=field)
    program.step_strategy(FixedDt(1.0))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=BindArray(),
            projection=ConservativeCellAverage(),
        )
    )
    return case, instance, frame, count


def run_periodic_poisson_native(
    *,
    case_name: str,
    model_name: str,
    domain_name: str,
    solver: Any,
    n_cells: int,
    rhs,
    t_end: float = 1.0,
    boundaries=None,
    nullspace=True,
):
    """Compile, bind RHS, and return the solved 1-d potential."""
    import numpy as np

    missing = native_unavailable_reason()
    if missing:
        raise RuntimeError(missing)
    case, instance, frame, count = author_periodic_poisson(
        case_name=case_name,
        model_name=model_name,
        domain_name=domain_name,
        solver=solver,
        n_cells=n_cells,
        boundaries=boundaries,
        nullspace=nullspace,
    )
    layout = (
        uniform_periodic_layout(frame, (count,))
        if boundaries is None
        else uniform_open_layout(frame, (count,))
    )
    plan = resolve_case(case, layout=layout)
    artifact = pops.compile(plan)
    initial = np.ascontiguousarray(
        np.asarray(rhs, dtype=np.float64)[np.newaxis, :], dtype=np.float64
    )
    simulation = pops.bind(artifact, initial_values={instance: initial})
    pops.run(simulation, t_end=float(t_end), max_steps=4)
    slots = tuple(simulation.field_provider_slots())
    if not slots:
        raise RuntimeError("native runtime exposed no field-provider slot")
    phi = np.ravel(
        np.asarray(simulation.field_potential_global(slots[0]), dtype=np.float64)
    )[:count]
    return phi, artifact, simulation
