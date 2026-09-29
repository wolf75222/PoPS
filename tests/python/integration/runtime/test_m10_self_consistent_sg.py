"""M10: a fitted flux consumes the Poisson field solved from its own stage density.

This periodic, neutral-background variant tests a closed one-dimensional mesh.
It does not claim the separate physical-wall boundary variant or a global
positivity/entropy theorem.
"""
from __future__ import annotations

import json
import math as pymath
import os
from pathlib import Path

import numpy as np
import pops
import pytest
from pops import math
from pops.codegen import Production
from pops.domain import CartesianDomain
from pops.fields import FieldBoundary, FieldDiscretization, FieldProblem, SharedMeanGauge, bcs
from pops.fields.methods import CellCenteredSecondOrder
from pops.frames import Cartesian1D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model import Handle, OwnerPath
from pops.model.provider_pack import ProviderPack
from pops.numerics import DiscretizationPlan, ScharfetterGummel, StateStorage
from pops.solvers import CG
from pops.time import FailRun, FixedDt, Program
from tests.python.support.native_execution_context import artifact_execution_context


N = 32
D = 0.1
AMPLITUDE = 0.4
PERTURBATION = 0.08
DT = 1 / (16 * N * N)


def _data(n=N):
    x = (np.arange(n) + 0.5) / n
    psi = AMPLITUDE * np.cos(2 * np.pi * x)
    equilibrium = np.exp(-psi)
    lambda_one = 4 * n * n * np.sin(np.pi / n) ** 2
    background = equilibrium - lambda_one * psi
    mode = np.cos(4 * np.pi * x)
    lambda_two = 4 * n * n * np.sin(2 * np.pi / n) ** 2
    density = equilibrium + PERTURBATION * mode
    potential = psi + PERTURBATION / lambda_two * mode
    return equilibrium, background, density, potential, psi


def _bernoulli(z):
    if abs(z) < 1e-8:
        return 1 - z / 2 + z * z / 12
    return z / pymath.expm1(z)


def _face_flux(density, potential):
    right_n = np.roll(density, -1)
    jump = np.roll(potential, -1) - potential
    return np.asarray([D * len(density) * (
        _bernoulli(-float(delta)) * float(n_r) -
        _bernoulli(float(delta)) * float(n_l))
        for delta, n_l, n_r in zip(jump, density, right_n, strict=True)])


def _one_step(density, potential, dt=DT):
    # PreparedDiffusion publishes +div(D grad(n) + D n grad(psi))=-div(j).
    faces = _face_flux(density, potential)
    return density + dt * len(density) * (faces - np.roll(faces, 1))


def _case(n=N, dt=DT):
    frame = CartesianDomain("M10-periodic", (0.,), (1.,)).frame(Cartesian1D())
    case = pops.Case("M10-self-consistent-SG")

    background_model = pops.Model("fixed-background", frame=frame)
    background = background_model.state("U", components=("b",))
    frozen = background_model.source("fixed", on=background, value=(0 * background[0],))
    background_rate = background_model.rate(
        "zero-rate", equation=math.ddt(background) == frozen)
    background_block = case.block("background", background_model)
    background_numerics = DiscretizationPlan()
    background_numerics.rates.add(background_rate, StateStorage())
    case.numerics(background_numerics, block=background_block)

    density_model = pops.Model("density", frame=frame)
    state = density_model.state("U", components=("n",))
    potential = density_model.aux("potential")
    drift = density_model.drift_flux("drift", state=state, mobility=D, potential=potential)
    diffusion = density_model.diffusive_flux(
        "diffusion", state=state, value=D * math.grad(state))
    rate = density_model.rate("joint-rate", equation=math.ddt(state) ==
                              -math.div(drift) + math.div(diffusion))
    density_block = case.block("density", density_model)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, ScharfetterGummel(drift=drift, flux=diffusion))
    case.numerics(numerics, block=density_block)

    phi = Handle("phi", kind="field", owner=OwnerPath.model("M10-Poisson"))
    problem = FieldProblem(
        "M10-Poisson", unknowns=(phi,),
        equations=(-math.laplacian(phi) == state[0] - background[0],),
        boundaries=(FieldBoundary(phi, bcs.BoundaryCondition(
            bcs.AllPhysicalBoundaries(), bcs.Periodic())),),
        gauge=SharedMeanGauge((phi,)))
    field = case.field(problem, FieldDiscretization(
        method=CellCenteredSecondOrder(), boundaries=(),
        solver=CG(max_iter=4000, rel_tol=1e-12, abs_tol=1e-13)))

    program = Program("M10-one-stage")
    density_time = program.state(density_block[state])
    background_time = program.state(background_block[background])
    point = program.stage("potential", c=0)
    solved = field.observe(program.solve(field, values={
        density_block[state]: density_time.n,
        background_block[background]: background_time.n,
    }, at=point).consume(action=FailRun()))
    module = density_model.module
    carrier = density_block[module.field_handle(module.field_spaces()["fields"])]
    observed = solved[field[phi]]
    context = solved.publish({(carrier, "potential"): observed},
                             states={density_block[state]: density_time.n})
    rhs = rate(density_time.n, context)
    end = program.value("density-end", density_time.n + program.dt * rhs,
                        at=density_time.next.point)
    unchanged = program.value("background-end", 1 * background_time.n,
                              at=background_time.next.point)
    program.commit_many({density_time.next: end, background_time.next: unchanged})
    program.store_history("stage-potential", observed, depth=1)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(n,),
                                   periodic=PeriodicAxes(frame.axes)))
    return case, layout, density_model, rate, program, problem


def test_m10_source_has_one_joint_face_construction_and_live_poisson_dependencies():
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph

    case, layout, model, rate, program, problem = _case()
    resolved = pops.resolve(pops.validate(case), layout=layout, backend=Production())
    view = model.balance_contract(rate)
    assert [(row.kind, row.coefficient) for row in view.occurrences] == [
        ("drift", -1), ("diffusion", 1)]
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    code = emit_cpp_program(program, model_graph=graph)
    assert ".apply_fitted(" in code
    assert code.count(".stage_accepted_exchanges(") == 1
    assert "joint-occurrences:0,1" in code
    assert "ctx.neg_div_flux_default_into(" not in code
    assert len(lower_and_validate(model)) >= 1
    assert len(problem.dependencies()) == 2
    load = next(node for node in program._values if node.op == "field_problem_load")
    assert len(load.attrs["field_dependencies"]) == 2
    equilibrium, background, density, potential, stale = _data()
    discrete_minus_laplacian = -N * N * (
        np.roll(potential, -1) - 2 * potential + np.roll(potential, 1))
    np.testing.assert_allclose(discrete_minus_laplacian, density - background,
                               rtol=0, atol=1e-12)
    assert np.max(np.abs(_face_flux(equilibrium, stale))) < 1e-13
    assert np.max(np.abs(_one_step(density, potential) - _one_step(density, stale))) > 1e-8


@pytest.mark.compiler
@pytest.mark.native_loader
def test_m10_native_self_consistent_stage_field_and_joint_flux(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    equilibrium, background, density, potential, stale = _data()
    case, layout, _, _, _, _ = _case()
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout,
                                         backend=Production()))
    runtime = pops.bind(artifact, initial_state={
        "density": np.ascontiguousarray(density[None]),
        "background": np.ascontiguousarray(background[None]),
    }, resources={"execution_context": artifact_execution_context(artifact)})
    report = pops.run(runtime, t_end=DT, max_steps=1, console=False)
    assert report.accepted_steps == 1
    assert runtime.macro_step() == 1 and runtime.time() == DT
    actual = np.asarray(runtime.state_global("density")).reshape(-1)
    observed = np.asarray(runtime.history_global("stage-potential", 1)).reshape(-1)
    expected = _one_step(density, potential)
    stale_result = _one_step(density, stale)
    assert np.max(np.abs(expected - stale_result)) > 1e-8
    np.testing.assert_allclose(observed, potential, rtol=0, atol=2e-11)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-11)
    assert np.max(np.abs(actual - stale_result)) > 1e-8
    np.testing.assert_array_equal(np.asarray(runtime.state_global("background")).reshape(-1),
                                  background)
    records = runtime._executor._program_exchange_records()
    assert len(records) == 2 * N  # two oriented cell incidences for each periodic face
    assert all("joint-occurrences:0,1" in row["occurrence_identity"] for row in records)
    exchanged = np.zeros(N)
    for row in records:
        cell = int(row["quadrature_identity"].split("/")[0].split(":")[1])
        exchanged[cell] += row["integrated_amount"]
    np.testing.assert_allclose(exchanged, (actual - density) / N, rtol=0, atol=2e-11)
    destination = os.environ.get("POPS_M10_EVIDENCE_DIR")
    if destination:
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        (target / "self-consistent-periodic.json").write_text(json.dumps({
            "artifact": artifact.artifact_identity.token,
            "accepted_steps": report.accepted_steps,
            "potential_error": float(np.max(np.abs(observed - potential))),
            "state_error": float(np.max(np.abs(actual - expected))),
            "stale_state_gap": float(np.max(np.abs(actual - stale_result))),
            "face_records": len(records),
        }, indent=2) + "\n")
