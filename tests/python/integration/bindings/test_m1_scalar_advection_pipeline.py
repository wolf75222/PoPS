"""ADC-661 M1 gate: the final scalar-advection shape crosses every typed phase."""
from __future__ import annotations

from fractions import Fraction

import numpy as np

import pops
from pops.math import ddt, div
from pops.mesh.layouts import Uniform
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
from pops.time import StagePoint, TimePoint


def test_scalar_advection_completes_typed_phase_pipeline():
    model = pops.physics.Model("scalar_advection")
    state = model.state("U", components=("u",), roles={"u": "density"})
    (u,) = state
    flux = model.flux(
        "F",
        on=state,
        x=(u,),
        y=(0 * u,),
        waves={"x": (1,), "y": (0,)},
    )
    rate = model.rate("A", ddt(state) == -div(flux))

    case = pops.Problem(name="scalar-advection")
    tracer = case.add_block(
        "tracer",
        model,
        spatial=pops.FiniteVolume(
            reconstruction=FirstOrder(),
            riemann=Rusanov(),
        ),
    )

    program = pops.Program("ssprk2").bind_operators(model.module)
    temporal = program.state(tracer, state)
    k0 = rate(temporal.n, name="k0")
    stage_handle = temporal.stage(
        "predictor",
        point=StagePoint(
            "predictor", {"explicit": TimePoint(program.clock, Fraction(1, 1))}
        ),
    )
    stage = program.define(stage_handle, temporal.n + program.dt * k0)
    k1 = rate(stage, name="k1")
    next_state = program.linear_combine(
        "q_next",
        Fraction(1, 2) * temporal.n
        + Fraction(1, 2) * (stage + program.dt * k1),
        at=temporal.next.point,
    )
    program.commit(temporal.next, next_state)
    case.time(program)

    validated = pops.validate(case)
    assert validated is case and case.frozen
    resolved = pops.resolve(
        validated,
        layout=Uniform(pops.CartesianMesh(n=16, L=1.0, periodic=True)),
    )
    assert type(resolved) is pops.ResolvedSimulationPlan
    artifact = pops.compile(resolved)
    assert type(artifact) is pops.CompiledSimulationArtifact
    assert type(artifact.plan).__name__ == "CompiledPlanRecord"
    assert artifact.program.model is None
    assert artifact.program.program._compiled_detached is True
    assert type(artifact.program.program_graph) is pops.time.ProgramGraph

    initial = np.ones((1, 16, 16), dtype=np.float64)
    simulation = pops.bind(
        artifact,
        pops.BindInputs(initial_state={"tracer": initial}),
    )
    assert type(simulation).__name__ == "BoundSimulation"
