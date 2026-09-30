"""Installed Dim3 coupled-gradient reception with saved native states and ledgers.

No fallback dimension or native skip is authored here. Run the required native
lane with POPS_REQUIRE_NATIVE_TESTS=1 so repository prerequisite gates fail closed.
The NumPy face/stencil oracle is separate from the compiled implementation.
"""

import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pops
import pytest

from pops import math
from pops._native_collectives import allgather_bytes, allgather_value
from pops._native_selector import select_native_dimension
from pops.domain import CartesianDomain
from pops.frames import Cartesian3D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.time import SSPRK2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import CoupledGradient, DiscretizationPlan
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.collective_checks import (
    collective_call,
    collective_check,
    state_snapshots,
)
from tests.python.support.native_execution_context import artifact_execution_context


pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
CELLS = (6, 4, 3)
LENGTHS = (1.0, 2.0, 3.0)
SPACING = tuple(length / cells for length, cells in zip(LENGTHS, CELLS, strict=True))
VOLUME = float(np.prod(SPACING))
DT = 1.0e-5
WEIGHTS = (0.5, 1.5)
D = np.array(((0.25, 0.05), (0.05, 0.10)))
R = np.array(((0.0, -0.3), (0.3, 0.0)))
B = D + R
STATE_ATOL, FLUX_ATOL, AMOUNT_ATOL = 4.0e-12, 3.0e-12, 4.0e-13
EXPECTED_RECORDS = int(np.prod(CELLS)) * 3 * 2 * 2 * 2 * 2


def _case(order):
    frame = CartesianDomain("periodic", lower=(0.0, 0.0, 0.0), upper=LENGTHS).frame(Cartesian3D())
    model = pops.Model("dim3_coupled_gradient", frame=frame)
    labels = ("first", "second")
    state = model.state("u", components=tuple(labels[index] for index in order))

    def permute(matrix):
        return tuple(tuple(float(matrix[i, j]) for j in order) for i in order)

    flux = model.coupled_gradient_flux(
        "law", state=state, dissipative=permute(D), reversible=permute(R)
    )
    rate = model.rate(
        "evolution",
        equation=math.ddt(state) == WEIGHTS[0] * math.div(flux) + WEIGHTS[1] * math.div(flux),
    )
    case = pops.Case("dim3_coupled_gradient")
    block = case.block("mixture", model, states=(state,))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, CoupledGradient(flux=flux))
    case.numerics(numerics, block=block)
    program = SSPRK2(block[state], rate=rate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=block[state], value=BindArray(), projection=ConservativeCellAverage()
        )
    )
    layout = Uniform(CartesianGrid(frame=frame, cells=CELLS, periodic=PeriodicAxes(frame.axes)))
    return pops.resolve(pops.validate(case), layout=layout), block[state]


def _initial():
    nx, ny, nz = CELLS
    x = (np.arange(nx) + 0.5)[None, None, :] / nx
    y = (np.arange(ny) + 0.5)[None, :, None] / ny
    z = (np.arange(nz) + 0.5)[:, None, None] / nz
    phase = 2 * np.pi * (x + y + z)
    # Exact Cartesian cell means; arrays use component,z,y,x with x varying fastest.
    oblique = float(np.prod(np.sinc(1 / np.asarray(CELLS))))
    return np.stack(
        (
            1
            + 0.1 * oblique * np.cos(phase)
            + 0.03 * np.sinc(1 / ny) * np.cos(2 * np.pi * y)
            + 0.02 * np.sinc(1 / nz) * np.sin(2 * np.pi * z),
            -0.2
            + 0.07 * oblique * np.sin(phase)
            + 0.02 * np.sinc(1 / nx) * np.sin(2 * np.pi * x)
            + 0.015 * np.sinc(1 / nz) * np.cos(2 * np.pi * z),
        )
    )


def _rate(values):
    laplacian = sum(
        (np.roll(values, 1, axis=3 - axis) - 2 * values + np.roll(values, -1, axis=3 - axis)) / h**2
        for axis, h in enumerate(SPACING)
    )
    return sum(WEIGHTS) * np.einsum("ab,bzyx->azyx", B, laplacian)


def _check_initial(initial):
    assert initial.shape == (2, *reversed(CELLS)) and np.isfinite(initial).all()
    np.testing.assert_array_equal(D, D.T)
    np.testing.assert_array_equal(R, -R.T)
    assert np.linalg.eigvalsh(D).min() > 0
    variation = [
        [
            float(
                np.max(np.abs(np.roll(initial[component], 1, axis=2 - axis) - initial[component]))
            )
            for axis in range(3)
        ]
        for component in range(2)
    ]
    assert np.min(variation) > 1.0e-3, variation
    return variation


def _ledger_oracle(records, order, initial, predictor):
    assert len(records) == EXPECTED_RECORDS
    assert len({row["operation_identity"] for row in records}) == 1
    contexts = tuple(dict.fromkeys(row["evaluation_context"] for row in records))
    assert len(contexts) == 2
    stages = {}
    for context in contexts:
        names = re.findall(r"StagePoint\(name='ssprk2_stage_([01])',", context)
        assert len(names) == 1, context
        stages[context] = int(names[0])
    assert set(stages.values()) == {0, 1}
    constitutive = tuple(np.einsum("ab,bzyx->azyx", B, stage) for stage in (initial, predictor))
    increments = np.zeros_like(initial)
    incidences, observed_axes, observed_occurrences = set(), set(), set()
    flux_error, amount_error = 0.0, 0.0
    for row in records:
        assert np.isfinite(
            [
                row[name]
                for name in (
                    "face_measure",
                    "numerical_flux",
                    "temporal_weight",
                    "integrated_amount",
                )
            ]
        ).all()
        cell_token, axis_token, side_token, component_token = row["quadrature_identity"].split("/")
        i, j, k = map(int, cell_token.split(":")[1:])
        axis = int(axis_token.split(":")[1])
        side = int(side_token.split(":")[1])
        component = int(component_token.split(":")[1])
        occurrence = int(row["occurrence_identity"].rsplit(":", 1)[1])
        incidence = (i, j, k, axis, side, component, occurrence, stages[row["evaluation_context"]])
        assert incidence not in incidences
        incidences.add(incidence)
        assert all(
            0 <= coordinate < size for coordinate, size in zip((i, j, k), CELLS, strict=True)
        )
        assert axis in (0, 1, 2) and side in (0, 1) and component in (0, 1)
        assert occurrence in (0, 1)
        observed_axes.add(axis)
        observed_occurrences.add(occurrence)
        assert row["occurrence_identity"] == row["operation_identity"] + "/occurrence:" + str(
            occurrence
        )
        assert row["orientation"] == (-1 if side == 0 else 1) and row["multiplicity"] == 1
        area = float(np.prod([h for tangent, h in enumerate(SPACING) if tangent != axis]))
        assert row["face_measure"] == area
        assert row["temporal_weight"] == DT / 2 * WEIGHTS[occurrence]
        left, right = [k, j, i], [k, j, i]
        coordinate = 2 - axis
        left[coordinate] = (left[coordinate] - (side == 0)) % CELLS[axis]
        right[coordinate] = (right[coordinate] + (side == 1)) % CELLS[axis]
        physical_component = order[component]
        w = constitutive[stages[row["evaluation_context"]]]
        flux = (w[(physical_component, *right)] - w[(physical_component, *left)]) / SPACING[axis]
        flux_error = max(flux_error, abs(row["numerical_flux"] - flux))
        amount = row["orientation"] * flux * area * row["temporal_weight"]
        amount_error = max(amount_error, abs(row["integrated_amount"] - amount))
        increments[physical_component, k, j, i] += row["integrated_amount"]
    assert observed_axes == {0, 1, 2} and observed_occurrences == {0, 1}
    return increments, {
        "record_count": len(records),
        "unique_incidences": len(incidences),
        "max_flux_error": flux_error,
        "max_amount_error": amount_error,
    }


@pytest.mark.parametrize("order", ((0, 1), (1, 0)))
def test_dim3_fourier_faces_and_component_permutation(
    isolated_native_cache, tmp_path, record_property, order
):
    del isolated_native_cache
    package = Path(pops.__file__).resolve()
    assert package.is_relative_to(Path(sys.prefix).resolve()), f"installed PoPS required: {package}"
    native = select_native_dimension(3)
    assert native.__native_dimension__ == 3
    world = native.mpi_world()
    resolved, subject = collective_call(world, lambda: _case(order))
    artifact = compile_resolved_plan_once(
        world, resolved, route="dim3 coupled gradient", compile_artifact=pops.compile
    )
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    requested = collective_call(world, _initial)
    with collective_check(world):
        variation = _check_initial(requested)
    bound = collective_call(world, lambda: np.ascontiguousarray(requested[list(order)]))
    runtime = collective_call(
        world,
        lambda: pops.bind(
            artifact, initial_values={subject: bound}, resources={"execution_context": context}
        ),
    )
    before = state_snapshots(runtime, world, ("mixture",))[0]
    directory = tmp_path / ("coupled-gradient-dim3-order-" + "".join(map(str, order)))
    with collective_check(world):
        if world.rank == 0:
            directory.mkdir()
            initial = before.reshape(2, *reversed(CELLS))[np.argsort(order)]
            np.savez_compressed(
                directory / "initial.npz",
                initial=initial,
                requested_initial=requested,
                bound_input=bound,
                order=order,
                cells=CELLS,
                lengths=LENGTHS,
            )
            np.testing.assert_allclose(initial, requested, rtol=0, atol=2.0e-14)
    collective_call(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
    after = state_snapshots(runtime, world, ("mixture",))[0]
    local_records = collective_call(world, runtime._executor._program_exchange_records)
    record_batches = allgather_value(world, local_records)
    local_wire = collective_call(world, runtime._executor._checkpoint_program_exchanges)
    wire_batches = allgather_bytes(world, local_wire)
    native_time, native_step = collective_call(
        world, lambda: (runtime.time(), runtime.macro_step()))
    with collective_check(world):
        assert native_time == DT and native_step == 1
        # This graph has no IntegralState or moving geometry declaration.
        assert all(wire.startswith(b"POPSEX01") for wire in wire_batches)
        if world.rank == 0:
            actual = after.reshape(2, *reversed(CELLS))[np.argsort(order)]
            predictor = initial + DT * _rate(initial)
            expected = 0.5 * (initial + predictor + DT * _rate(predictor))
            records = [row for batch in record_batches for row in batch]
            # Save native evidence before the numerical oracle can reject it.
            for rank, wire in enumerate(wire_batches):
                (directory / f"ledger-rank-{rank:04d}.bin").write_bytes(wire)
            (directory / "native-ledger.json").write_text(
                json.dumps(record_batches, indent=2, allow_nan=False) + "\n"
            )
            np.savez_compressed(
                directory / "accepted.npz",
                initial=initial,
                final=actual,
                state_increment=actual - initial,
                oracle_predictor=predictor,
                oracle_final=expected,
                order=order,
                dt=DT,
                cells=CELLS,
                lengths=LENGTHS,
                spacing=SPACING,
                cell_volume=VOLUME,
                dissipative=D,
                reversible=R,
                occurrence_weights=WEIGHTS,
            )
            increments, metrics = _ledger_oracle(records, order, initial, predictor)
            np.savez_compressed(
                directory / "increments.npz",
                native_face_amounts=increments,
                state_amounts=(actual - initial) * VOLUME,
            )
            metrics.update(
                max_state_error=float(np.max(np.abs(actual - expected))),
                max_balance_error=float(np.max(np.abs(increments - (actual - initial) * VOLUME))),
                max_global_amount=float(np.max(np.abs(increments.sum(axis=(1, 2, 3))))),
                initial_energy=float(np.sum(initial**2) * VOLUME),
                final_energy=float(np.sum(actual**2) * VOLUME),
            )
            receipt = {
                "native_dimension": native.__native_dimension__,
                "ranks": int(world.size),
                "package_path": str(package),
                "native_path": str(Path(native.__file__).resolve()),
                "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
                "native_abi": str(native.abi_key()),
                "native_time": native_time,
                "native_macro_step": native_step,
                "artifact_identity": artifact.artifact_identity.token,
                "platform_manifest": artifact.platform_manifest.to_data(),
                "fixture_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "storage": "canonical physical components,z,y,x",
                "component_order": list(order),
                "axis_variation": variation,
                "cells": list(CELLS),
                "lengths": list(LENGTHS),
                "dt": DT,
                "dissipative": D.tolist(),
                "reversible": R.tolist(),
                "occurrence_weights": list(WEIGHTS),
                "ledger_sha256_by_rank": [
                    hashlib.sha256(wire).hexdigest() for wire in wire_batches
                ],
                "records_by_rank": [len(batch) for batch in record_batches],
                "metrics": metrics,
                "criteria": {
                    "state_atol": STATE_ATOL,
                    "flux_atol": FLUX_ATOL,
                    "amount_atol": AMOUNT_ATOL,
                    "expected_records": EXPECTED_RECORDS,
                    "initial_bound_atol": 2.0e-14,
                    "minimum_energy_drop": 1.0e-9,
                },
            }
            (directory / "receipt.json").write_text(
                json.dumps(receipt, indent=2, allow_nan=False) + "\n"
            )
            np.testing.assert_allclose(actual, expected, rtol=0, atol=STATE_ATOL)
            assert metrics["max_flux_error"] <= FLUX_ATOL
            assert metrics["max_amount_error"] <= AMOUNT_ATOL
            np.testing.assert_allclose(
                increments, (actual - initial) * VOLUME, rtol=0, atol=AMOUNT_ATOL
            )
            np.testing.assert_allclose(increments.sum(axis=(1, 2, 3)), 0, rtol=0, atol=AMOUNT_ATOL)
            assert metrics["final_energy"] < metrics["initial_energy"] - 1.0e-9
    record_property("coupled_gradient_dim3_receipts", str(directory))
    record_property("artifact_identity", artifact.artifact_identity.token)
