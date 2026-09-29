"""Installed Dim2 coupled-gradient Fourier and accepted-face witness.

The NumPy stencil and face oracle are independent of the native tensor-face
implementation. Each run uses a fresh public Program and initial state.
"""
import json
import re

import numpy as np
import pops
import pytest

from pops import math
from pops._native_collectives import allgather_value
from pops._native_selector import select_native_dimension
from pops.domain import CartesianDomain
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.time import SSPRK2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import CoupledGradient, DiscretizationPlan
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.collective_checks import collective_call, collective_check, state_snapshots
from tests.python.support.native_execution_context import artifact_execution_context


pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
NX, NY = 8, 6
LX, LY = 1., 2.
HX, HY = LX / NX, LY / NY
DT = 1.e-5
WEIGHTS = (.5, 1.5)
D = np.array(((.25, .05), (.05, .10)))
R = np.array(((0., -.3), (.3, 0.)))
B = D + R


def _case(order):
    frame = CartesianDomain("periodic", lower=(0., 0.), upper=(LX, LY)).frame(Cartesian2D())
    model = pops.Model("dim2_coupled_gradient", frame=frame)
    labels = ("first", "second")
    state = model.state("u", components=tuple(labels[i] for i in order))
    permute = lambda matrix: tuple(tuple(float(matrix[i, j]) for j in order) for i in order)
    flux = model.coupled_gradient_flux("law", state=state,
        dissipative=permute(D), reversible=permute(R))
    rate = model.rate("evolution", equation=math.ddt(state) ==
        WEIGHTS[0] * math.div(flux) + WEIGHTS[1] * math.div(flux))
    case = pops.Case("dim2_coupled_gradient")
    block = case.block("mixture", model, states=(state,))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, CoupledGradient(flux=flux))
    case.numerics(numerics, block=block)
    program = SSPRK2(block[state], rate=rate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                      projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(NX, NY),
                                  periodic=PeriodicAxes(frame.axes)))
    return pops.resolve(pops.validate(case), layout=layout), block[state]


def _initial():
    x = (np.arange(NX) + .5) / NX
    y = (np.arange(NY) + .5) / NY
    phase = 2 * np.pi * (x[None, :] + 2 * y[:, None])
    return np.stack((1 + .1 * np.cos(phase) + .03 * np.cos(4 * np.pi * y[:, None]) * np.ones((1, NX)),
                     -.2 + .07 * np.sin(phase) + .02 * np.sin(2 * np.pi * x)[None, :]))


def _laplacian(values):
    return ((np.roll(values, 1, axis=2) - 2 * values + np.roll(values, -1, axis=2)) / HX**2 +
            (np.roll(values, 1, axis=1) - 2 * values + np.roll(values, -1, axis=1)) / HY**2)


def _rate(values):
    return sum(WEIGHTS) * np.einsum("ab,byx->ayx", B, _laplacian(values))


@pytest.mark.parametrize("order", ((0, 1), (1, 0)))
def test_dim2_fourier_faces_and_component_permutation(isolated_native_cache, native_cxx,
        kokkos_root, tmp_path, order):
    del isolated_native_cache, native_cxx, kokkos_root
    native = select_native_dimension(2)
    world = native.mpi_world()
    resolved, subject = collective_call(world, lambda: _case(order))
    artifact = compile_resolved_plan_once(world, resolved,
        route="dim2 coupled gradient", compile_artifact=pops.compile)
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    initial = collective_call(world, _initial)
    bound = collective_call(world, lambda: np.ascontiguousarray(initial[list(order)]))
    runtime = collective_call(world, lambda: pops.bind(artifact,
        initial_values={subject: bound}, resources={"execution_context": context}))
    collective_call(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
    actual = state_snapshots(runtime, world, ("mixture",))[0]
    local_records = collective_call(world, runtime._executor._program_exchange_records)
    batches = allgather_value(world, local_records)
    with collective_check(world):
        records = [row for batch in batches for row in batch]
        assert runtime.time() == DT and runtime.macro_step() == 1
        assert len(records) == 2 * NX * NY * 2 * 2 * 2 * 2
        if world.rank == 0:
            predictor = initial + DT * _rate(initial)
            expected = .5 * (initial + predictor + DT * _rate(predictor))
            actual = actual.reshape(2, NY, NX)[np.argsort(order)]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=4.e-12)
            contexts = tuple(dict.fromkeys(row["evaluation_context"] for row in records))
            assert len(contexts) == 2
            stages = {}
            for context_name in contexts:
                names = re.findall(r"StagePoint\(name='ssprk2_stage_([01])',", context_name)
                assert len(names) == 1, context_name
                stages[context_name] = (initial, predictor)[int(names[0])]
            assert len({id(value) for value in stages.values()}) == 2
            increments = np.zeros_like(initial)
            incidences = set()
            for row in records:
                cell_token, axis_token, side_token, component_token = row["quadrature_identity"].split("/")
                i, j = map(int, cell_token.split(":")[1:])
                axis = int(axis_token.split(":")[1])
                side = int(side_token.split(":")[1])
                component = int(component_token.split(":")[1])
                occurrence = int(row["occurrence_identity"].rsplit(":", 1)[1])
                incidence = (i, j, axis, side, component, occurrence, row["evaluation_context"])
                assert incidence not in incidences
                incidences.add(incidence)
                assert 0 <= i < NX and 0 <= j < NY
                assert axis in (0, 1) and side in (0, 1) and component in (0, 1)
                assert occurrence in (0, 1)
                assert row["occurrence_identity"] == row["operation_identity"] + "/occurrence:" + str(occurrence)
                assert row["orientation"] == (-1 if side == 0 else 1)
                assert row["multiplicity"] == 1
                h, area = (HX, HY) if axis == 0 else (HY, HX)
                assert row["face_measure"] == area
                assert row["temporal_weight"] == DT / 2 * WEIGHTS[occurrence]
                physical_component = order[component]
                w = np.einsum("ab,byx->ayx", B, stages[row["evaluation_context"]])
                left = [j, i]
                right = [j, i]
                coordinate = 1 if axis == 0 else 0
                left[coordinate] = (left[coordinate] - (side == 0)) % (NX if axis == 0 else NY)
                right[coordinate] = (right[coordinate] + (side == 1)) % (NX if axis == 0 else NY)
                flux = (w[physical_component, *right] - w[physical_component, *left]) / h
                assert abs(row["numerical_flux"] - flux) <= 3.e-12
                amount = row["orientation"] * flux * area * row["temporal_weight"]
                assert abs(row["integrated_amount"] - amount) <= 3.e-13
                increments[physical_component, j, i] += row["integrated_amount"]
            np.testing.assert_allclose(increments, (actual - initial) * HX * HY,
                                       rtol=0, atol=4.e-13)
            np.testing.assert_allclose(increments.sum(axis=(1, 2)), 0, rtol=0, atol=4.e-13)
            np.savez_compressed(tmp_path / "coupled_gradient_dim2.npz",
                                initial=initial, final=actual, increments=increments,
                                order=order, dt=DT)
            (tmp_path / "coupled_gradient_dim2_ledger.json").write_text(
                json.dumps(records, indent=2) + "\n")
