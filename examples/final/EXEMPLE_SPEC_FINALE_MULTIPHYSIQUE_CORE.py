#!/usr/bin/env python3
"""Canonical multi-block Program core: transport, electrostatic field and collisions.

The script is an executable authoring acceptance target.  It builds one typed, owner-qualified
graph containing two transported states, one simultaneous field solve, the electric force read at
that exact field context, and one conservative implicit collision solve.  Every fallible solve has
an explicit action before any result is readable; the final commits are therefore all-or-nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pops
from pops.fields import (
    CellCenteredSecondOrder,
    ConstantNullspace,
    FieldDiscretization,
    FieldOperator,
    FieldOutput,
    FieldProviderContribution,
    FieldProviderPack,
    GradientOutput,
    MeanValueGauge,
)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Periodic
from pops.ir import ValueExpr
from pops.ir.expr import Const, Var
from pops.math import laplacian
from pops.model import Rate, RateBundle, Signature
from pops.solvers.elliptic import GeometricMG
from pops.time import FailRun, RejectAttempt


@dataclass(frozen=True, slots=True)
class MultiphysicsCore:
    """Typed authorities retained for inspection, resolution and execution assembly."""

    module: Any
    case: Any
    electron_space: Any
    ion_space: Any
    field_space: Any
    electron_block: Any
    ion_block: Any
    electron_state: Any
    ion_state: Any
    field: Any
    program: Any
    explicit_electrons: Any
    explicit_ions: Any
    collision: Any


def build_multiphysics_core() -> MultiphysicsCore:
    """Build the complete two-state transactional Program without running Python numerics."""

    module = pops.model.Module("electrostatic_two_fluid")
    electrons = module.state_space(
        "electron_state",
        ("ne", "pex", "pey"),
        roles={"ne": "density", "pex": "momentum_x", "pey": "momentum_y"},
    )
    ions = module.state_space(
        "ion_state",
        ("ni", "pix", "piy"),
        roles={"ni": "density", "pix": "momentum_x", "piy": "momentum_y"},
    )
    fields = module.field_space(
        "electrostatic_fields", ("potential", "electric_x", "electric_y")
    )

    ne, pex, pey = (Var(name, "cons") for name in electrons.components)
    ni, pix, piy = (Var(name, "cons") for name in ions.components)
    electric_x, electric_y = Var("electric_x", "aux"), Var("electric_y", "aux")
    sound_speed_squared = Const(0.2)

    electron_flux = module.operator(
        name="electron_transport_flux",
        signature=Signature((electrons,), Rate(electrons)),
        kind="grid_operator",
        expr={
            "x": [pex, pex * pex / ne + sound_speed_squared * ne, pex * pey / ne],
            "y": [pey, pex * pey / ne, pey * pey / ne + sound_speed_squared * ne],
        },
    )
    ion_flux = module.operator(
        name="ion_transport_flux",
        signature=Signature((ions,), Rate(ions)),
        kind="grid_operator",
        expr={
            "x": [pix, pix * pix / ni + sound_speed_squared * ni, pix * piy / ni],
            "y": [piy, pix * piy / ni, piy * piy / ni + sound_speed_squared * ni],
        },
    )
    electron_force = module.operator(
        name="electron_electric_force",
        signature=Signature((electrons, fields), Rate(electrons)),
        kind="local_source",
        expr=[Const(0.0), -ne * electric_x, -ne * electric_y],
    )
    ion_force = module.operator(
        name="ion_electric_force",
        signature=Signature((ions, fields), Rate(ions)),
        kind="local_source",
        expr=[Const(0.0), ni * electric_x, ni * electric_y],
    )
    explicit_electrons = module.rate_operator(
        "explicit_electrons",
        state_space=module.state_handle(electrons),
        flux=True,
        fluxes=(electron_flux,),
        sources=(electron_force,),
    )
    explicit_ions = module.rate_operator(
        "explicit_ions",
        state_space=module.state_handle(ions),
        flux=True,
        fluxes=(ion_flux,),
        sources=(ion_force,),
    )

    # Each model-owned provider contributes one signed source-density term to the same Poisson RHS.
    electron_charge = module.operator(
        name="electron_charge",
        signature=Signature((electrons,), fields),
        kind="field_operator",
        expr=-ne,
    )
    ion_charge = module.operator(
        name="ion_charge",
        signature=Signature((ions,), fields),
        kind="field_operator",
        expr=ni,
    )

    exchange_x = Const(2.0) * (pix - pex)
    exchange_y = Const(2.0) * (piy - pey)
    collision = module.operator(
        name="implicit_collision",
        signature=Signature(
            (electrons, ions),
            RateBundle({"electrons": Rate(electrons), "ions": Rate(ions)}),
        ),
        kind="coupled_rate",
        expr={
            "electrons": [Const(0.0), exchange_x, exchange_y],
            "ions": [Const(0.0), -exchange_x, -exchange_y],
        },
    )

    case = pops.Case("two_fluid_transport_field_collision")
    electron_block = case.block("electrons", module)
    ion_block = case.block("ions", module)
    electron_state = electron_block[module.state_handle(electrons)]
    ion_state = ion_block[module.state_handle(ions)]

    # The field declaration is model-owned; the Case qualification chooses the electron block as
    # its storage anchor.  The physical RHS below remains a genuine two-block provider pack.
    unknown = electron_block[module.field_handle(fields)]
    field_operator = FieldOperator(
        "electrostatic",
        unknown=unknown,
        equation=-laplacian(ValueExpr(unknown)) == (-ne + ni),
        providers=FieldProviderPack(
            (
                FieldProviderContribution(electron_block[electron_charge]),
                FieldProviderContribution(ion_block[ion_charge]),
            )
        ),
        outputs=(
            FieldOutput("potential", unknown),
            GradientOutput("electric", unknown, sign=-1),
        ),
    )
    field_discretization = FieldDiscretization(
        method=CellCenteredSecondOrder(),
        boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Periodic()),),
        solver=GeometricMG(),
        nullspace=ConstantNullspace(),
        gauge=MeanValueGauge(0.0),
    )
    field = case.field(field_operator, field_discretization)

    program = pops.Program("transport_field_collision")
    electron_time = program.state(electron_state)
    ion_time = program.state(ion_state)

    exact_fields = program.solve_fields_from_blocks(
        (electron_time.n, ion_time.n), field=field, name="electrostatic_at_n"
    ).consume(action=FailRun())
    electron_rate = program.call(
        explicit_electrons, electron_time.n, exact_fields, name="electron_explicit_rate"
    )
    ion_rate = program.call(
        explicit_ions, ion_time.n, exact_fields, name="ion_explicit_rate"
    )
    electron_predictor = program.value(
        "electron_predictor",
        electron_time.n + program.dt * electron_rate,
        at=electron_time.next.point,
    )
    ion_predictor = program.value(
        "ion_predictor",
        ion_time.n + program.dt * ion_rate,
        at=ion_time.next.point,
    )

    collided = program.solve_implicit(
        collision,
        (electron_predictor, ion_predictor),
        method="newton",
        tol=1.0e-11,
        max_iter=12,
        fd_eps=1.0e-7,
        at={
            electron_block: electron_time.next.point,
            ion_block: ion_time.next.point,
        },
        name="collision_at_next",
    ).consume(action=RejectAttempt())
    program.commit_many(
        {
            electron_time.next: collided[electron_block],
            ion_time.next: collided[ion_block],
        }
    )
    case.program(program)

    return MultiphysicsCore(
        module=module,
        case=case,
        electron_space=electrons,
        ion_space=ions,
        field_space=fields,
        electron_block=electron_block,
        ion_block=ion_block,
        electron_state=electron_state,
        ion_state=ion_state,
        field=field,
        program=program,
        explicit_electrons=explicit_electrons,
        explicit_ions=explicit_ions,
        collision=collision,
    )


def main() -> None:
    """Build and validate the immutable Program graph; native compilation is a later phase."""

    core = build_multiphysics_core()
    graph = core.program.to_graph()
    if len(graph.nodes) == 0:
        raise RuntimeError("the multiphysics Program graph is unexpectedly empty")
    print("PoPS multiphysics Program graph: %d nodes" % len(graph.nodes))


if __name__ == "__main__":
    main()
