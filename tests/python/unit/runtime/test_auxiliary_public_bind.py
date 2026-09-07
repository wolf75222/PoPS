"""Real public artifacts preserve InputAux identity and reject competing field uploads."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

import pops
from pops.codegen import Production
from pops.domain import Rectangle
from pops.fields import AuxiliaryBoundary, DerivedAux
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model.provider_pack import ComponentKey, ProviderPack
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.time import FixedDt
from tests.python.support.native_execution_context import artifact_execution_context


def _auxiliary_pack(artifact):
    return ProviderPack.from_data(
        artifact.plan.blocks[0].resolved_operations.to_data()["provider_evidence"]["auxiliary"])


@pytest.fixture(scope="module")
def full_multiphysics_artifact():
    path = Path(__file__).resolve().parents[4] / (
        "examples/final/EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py")
    spec = importlib.util.spec_from_file_location("_public_aux_multiphysics", path)
    example = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = example
    spec.loader.exec_module(example)
    target, artifact = example.compile_final_case()
    assert example.DEFAULT_CELLS == 8
    assert set(block.name for block in artifact.blocks) == {"electrons", "ions"}
    return example, target, artifact


def test_full_multiphysics_binds_without_uploading_native_field_outputs(
    full_multiphysics_artifact,
):
    example, _target, artifact = full_multiphysics_artifact
    inputs = example.build_initial_state()
    first = example._bind_artifact(artifact, initial_state=inputs)
    second = example._bind_artifact(artifact, initial_state=inputs)
    assert first.bind_identity == second.bind_identity
    assert first.bound_snapshot.to_dict()["aux_evidence"] == {}
    assert first.field_provider_slots() == ("electrostatic",)
    field_plan, = artifact.plan.field_plans.values()
    assert field_plan.native_install_data()["output_route"]["owner_block"] == "electrons"
    for simulation in (first, second):
        for name, expected in inputs.items():
            np.testing.assert_array_equal(simulation.state_global(name), expected)
        potential = np.asarray(simulation.field_potential_global("electrostatic"))
        assert potential.size == example.DEFAULT_CELLS**2
        np.testing.assert_array_equal(potential, np.zeros_like(potential))


@pytest.mark.parametrize("kind", ("bare_name", "field_output", "foreign_owner"))
def test_full_multiphysics_refuses_duplicate_or_foreign_upload_authority(
    full_multiphysics_artifact, kind,
):
    example, _target, artifact = full_multiphysics_artifact
    field_plan, = artifact.plan.field_plans.values()
    key = ComponentKey(**field_plan.native_install_data()["output_route"]["component_keys"][0])
    if kind == "bare_name":
        key = key.component
        error, match = TypeError, "exact pops.model.ComponentKey"
    else:
        if kind == "foreign_owner":
            key = replace(key, owner_qid=key.owner_qid + "/foreign")
        error, match = ValueError, "only declared exact InputAux"
    with pytest.raises(error, match=match):
        example._bind_artifact(
            artifact, initial_state=example.build_initial_state(),
            aux={key: np.zeros((example.DEFAULT_CELLS, example.DEFAULT_CELLS))})


@pytest.fixture(scope="module")
def input_artifact():
    frame = Rectangle("aux-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("typed_external_projection", frame=frame)
    state = model.state("U", components=("q",))
    (q,) = state
    imposed = model.aux("imposed")
    flux = model.flux(
        "stationary_flux", frame=frame, state=state,
        components={x_axis: (0.0 * q,), y_axis: (0.0 * q,)},
        waves={x_axis: (0.0 * q,), y_axis: (0.0 * q,)})
    rate = model.rate("stationary", equation=ddt(state) == -div(flux))
    model.projection((imposed,))
    module = model.module
    derived = module.aux_field("derived")
    module.aux_provider(DerivedAux(
        module.aux_handle(derived),
        2.0 * ValueExpr(module.field_handle(module.field_spaces()["fields"])),
        boundary=AuxiliaryBoundary(width=1, kind="foextrap")))
    case = pops.Case("typed_external_projection_case")
    block = case.block("scalar", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, FiniteVolume(
        flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case.numerics(numerics, block=block)
    program = pops.Program("external_projection_step")
    current = program.state(block[state])
    candidate = program.value(
        "candidate", current.n + program.dt * rate(current.n), at=current.next.point)
    program.commit(current.next, program.project(candidate))
    program.step_strategy(FixedDt(0.125))
    case.program(program)
    grid = CartesianGrid(frame=frame, cells=(24, 24), periodic=PeriodicAxes(frame.axes))
    return pops.compile(pops.resolve(pops.validate(case), layout=Uniform(grid), backend=Production()))


def _bind_input(artifact, values):
    return pops.bind(
        artifact, initial_state={"scalar": np.ones((1, 24, 24))}, aux=values,
        resources={"execution_context": artifact_execution_context(artifact)})


def test_public_input_bind_preserves_exact_key_array_and_native_projection_value(input_artifact):
    pack = _auxiliary_pack(input_artifact)
    key, = (key for key in pack if pack.declared_entry(key).producer == "runtime_input")
    values = np.full((24, 24), 2.0)
    simulation = _bind_input(input_artifact, {key: values})
    evidence = simulation.bound_snapshot.to_dict()["aux_evidence"]["components"]
    assert len(evidence) == 1
    assert evidence[0]["key"] == key.to_data()
    assert evidence[0]["array"]["shape"] == [24, 24]
    report = pops.run(simulation, t_end=0.125, max_steps=1, console=False)
    assert report.accepted_steps == 1
    np.testing.assert_array_equal(simulation.state_global("scalar"), np.full((1, 24, 24), 2.0))


@pytest.mark.parametrize("kind", ("bare_name", "foreign_owner", "derived", "missing"))
def test_public_input_bind_refuses_missing_or_inauthentic_component(input_artifact, kind):
    pack = _auxiliary_pack(input_artifact)
    key, = (key for key in pack if pack.declared_entry(key).producer == "runtime_input")
    error, match = ValueError, "only declared exact InputAux"
    if kind == "bare_name":
        key = key.component
        error, match = TypeError, "exact pops.model.ComponentKey"
    elif kind == "foreign_owner":
        key = replace(key, owner_qid=key.owner_qid + "/foreign")
    elif kind == "derived":
        key, = (key for key in pack if pack.declared_entry(key).producer.startswith("derived"))
    values = {key: np.full((24, 24), 2.0)}
    if kind == "missing":
        values = {}
        match = "missing declared InputAux"
    with pytest.raises(error, match=match):
        _bind_input(input_artifact, values)
