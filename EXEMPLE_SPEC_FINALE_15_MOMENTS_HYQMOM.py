#!/usr/bin/env python3
"""Final HyQMOM15 authoring example over the canonical PoPS contracts.

It exercises the complete public lifecycle, including the manifest-sized 15x15 implicit magnetic
solve, runtime installation, stepping, diagnostics and restartable checkpoint consumers.
"""
from __future__ import annotations

import json
import tempfile

import numpy as np
import pops
from pops.codegen import BindInputs
from pops.diagnostics import Integral, MinMax
from pops.lib.models.moments import HyQMOM15
from pops.lib.time import IMEX
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.numerics import DiscretizationPlan, FiniteVolume
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import VanLeer
from pops.numerics.riemann import HLL
from pops.numerics.riemann.waves import ExplicitPair
from pops.numerics.variables import Conservative
from pops.output import Checkpoint, HDF5, ScientificOutput
from pops.params import ConstParam
from pops.runtime import ConsumerGraph
from pops.time import AdaptiveCFL, every


def build_case() -> tuple[pops.Case, object]:
    # Compile-time physical constants select the native zero-copy production route. Applications
    # that calibrate them between runs may pass RuntimeParam declarations to the same factory.
    physics = HyQMOM15.vlasov_lorentz(
        q_over_m=ConstParam("q_over_m", -1.0),
        omega_c=ConstParam("omega_c", 1.0),
    )
    U = physics.states["U"]
    F = physics.fluxes["transport"]
    explicit_rate = physics.operators["transport"]
    implicit_source = physics.operators["magnetic_rotation"]
    case = pops.Case("hyqmom15-vlasov")
    plasma = case.block("plasma", physics)
    state = plasma[U]

    numerics = DiscretizationPlan()
    numerics.rates.add(
        explicit_rate,
        FiniteVolume(
            flux=F,
            variables=Conservative(U),
            reconstruction=MUSCL(VanLeer()),
            # The model computes the signed pair from its flux Jacobian; the compiled ABI exports
            # that already-materialized pair to HLL through the ExplicitPair runtime route.
            riemann=HLL(waves=ExplicitPair()),
        ),
    )
    case.numerics(numerics, block=plasma)

    program = IMEX(
        state,
        explicit_operator=explicit_rate,
        implicit_operator=implicit_source,
    )
    program.step_strategy(
        AdaptiveCFL(0.4),
        staged_effects=(
            "state", "fields", "flux_ledgers", "histories", "schedules", "consumers"),
        guards=("moments.realizable(order=4)",),
        projections=("moments.smooth_floor(M00,C20,C02)",),
    )
    case.program(program)

    output_schedule = every(20, clock=program.clock)
    particle_number = Integral(
        block=plasma, role="M00", cadence=output_schedule)
    realizability_bounds = MinMax(
        block=plasma, role="M00", cadence=output_schedule)
    case.consumers(ConsumerGraph.from_consumers((
        ScientificOutput(
            format=HDF5(),
            schedule=output_schedule,
            fields=(state,),
            diagnostics=(particle_number, realizability_bounds),
            target="outputs/hyqmom15",
        ),
        Checkpoint(
            schedule=every(100, clock=program.clock),
            target="checkpoints/hyqmom15",
            bit_identical=True,
        ),
    )))
    return case, physics


def main() -> None:
    case, physics = build_case()
    validated = pops.validate(case)
    mesh_n = 8
    resolved = pops.resolve(
        validated,
        layout=Uniform(CartesianMesh(n=mesh_n, L=1.0, periodic=True)),
    )
    artifact = pops.compile(resolved)

    components = physics.states["U"].components
    initial = np.zeros((len(components), mesh_n, mesh_n), dtype=np.float64)
    component = {name: index for index, name in enumerate(components)}
    # Isotropic centred Gaussian moments: a realizable, spatially uniform equilibrium.
    for name, value in {
        "M00": 1.0,
        "M20": 0.1,
        "M02": 0.1,
        "M40": 0.03,
        "M22": 0.01,
        "M04": 0.03,
    }.items():
        initial[component[name], :, :] = value
    bind_inputs = BindInputs(initial_state={"plasma": initial})
    simulation = pops.bind(artifact, bind_inputs)
    steps = simulation.run(t_end=1.0e-5, max_steps=1)
    final_state = np.asarray(simulation.get_state("plasma"))

    # Restart authenticates the bind/run identities before restoring every moment and clock.
    with tempfile.TemporaryDirectory(prefix="pops-hyqmom15-") as directory:
        checkpoint = simulation.checkpoint("%s/restart" % directory)
        restarted = pops.bind(artifact, bind_inputs)
        restarted.restart(checkpoint)
        restart_equal = np.array_equal(
            np.asarray(restarted.get_state("plasma")), final_state)

    transaction = case._time.transaction_plan()
    print(json.dumps({
        "artifact": artifact.artifact_identity.token,
        "bound": type(simulation).__name__,
        "checkpoint_restart_bit_identical": restart_equal,
        "model": physics.name,
        "moments": list(components),
        "n_moments": len(components),
        "finite": bool(np.isfinite(final_state).all()),
        "realizability": {"order": HyQMOM15.order},
        "program_hash": case._time.to_graph().graph_hash,
        "runtime_steps": steps,
        "runtime_time": simulation.time(),
        "transaction": transaction.to_data(),
        "resolved_blocks": [block.name for block in resolved.blocks],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
