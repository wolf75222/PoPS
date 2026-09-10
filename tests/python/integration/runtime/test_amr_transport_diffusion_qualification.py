"""ADC-942 declared Dim2 CPU AMR transport-diffusion qualification matrix."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math as pmath
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pops
import pytest
from pops import math
from pops.analytic import cos as analytic_cos
from pops.analytic import sin as analytic_sin
from pops.analytic import x as analytic_x
from pops.analytic import y as analytic_y
from pops.amr import (
    AMRClockRelation,
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
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import Analytic
from pops.lib.time import ForwardEuler
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import Diffusion, DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.physics.diffusion import DiffusiveBoundary
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt, every
from tests.python.support.amr_snapshots import (
    composite_active_block_state,
    composite_active_mask,
    level_valid_mask,
)
from tests.python.support.native_execution_context import artifact_execution_context


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
ROOT = Path(__file__).resolve().parents[4]
REFINEMENTS = (16, 32, 64)
VELOCITY = (0.35, -0.2)
DIFFUSIVITY = 0.05
FINAL_TIME = 1.0e-2


def _stable_dt(n):
    fine_n = 2 * n
    frequency = 4 * DIFFUSIVITY * fine_n**2 + sum(map(abs, VELOCITY)) * fine_n
    return 0.4 / frequency


def _author(n, dt, *, physical=False, cxx=None):
    frame = Rectangle("qualified-amr-diffusion", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D()
    )
    model = pops.Model("qualified-amr-diffusion-model", frame=frame)
    state = model.state("U", components=("u",))
    (u,) = state
    boundaries = None
    if physical:
        boundaries = tuple(
            DiffusiveBoundary(axis, side, "value", 1.0, (1.0, 0.0))
            if axis == 0
            else DiffusiveBoundary(axis, side, "conormal", 0.0)
            for axis in range(2)
            for side in ("lower", "upper")
        )
    diffusion = model.diffusive_flux(
        "diffusion", state=state, value=DIFFUSIVITY * math.grad(u), boundaries=boundaries
    )
    transport = None
    equation = div(diffusion)
    method = None
    if not physical:
        transport = model.flux(
            "transport",
            frame=frame,
            state=state,
            components={
                axis: (speed * u,) for axis, speed in zip(frame.axes, VELOCITY, strict=True)
            },
            waves={axis: (speed,) for axis, speed in zip(frame.axes, VELOCITY, strict=True)},
        )
        equation = -div(transport) + equation
        method = FiniteVolume(
            flux=transport,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        )
    rate = model.rate("transport-diffusion", equation=ddt(state) == equation)
    case = pops.Case("qualified-amr-diffusion-case")
    block = case.block("heat", model)
    block_state = block[state]
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, Diffusion(flux=diffusion, transport=method))
    case.numerics(numerics, block=block)
    program = ForwardEuler(block_state, rate=rate)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    if physical:
        profile = 1.0 + analytic_x(frame)
        threshold_value = 1.5
    else:
        profile = 1.0 + 0.2 * analytic_sin(2 * pmath.pi * analytic_x(frame)) * analytic_cos(
            2 * pmath.pi * analytic_y(frame)
        )
        threshold_value = 1.04
    case.initials.add(
        InitialCondition(
            state=block_state,
            value=Analytic(frame=frame, components=(profile,)),
            projection=ConservativeCellAverage(),
        )
    )
    threshold = case.param(RuntimeParam("refine-threshold", default=threshold_value))
    transfer = AMRTransfer()
    transfer.state(block_state, StateTransfer())
    layout = AMR(
        grid=CartesianGrid(
            frame=frame, cells=(n, n), periodic=None if physical else PeriodicAxes(frame.axes)
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(Tag(ValueExpr(block_state) > case.value(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(4, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
    )
    compile_options = {"include": str(ROOT / "include")}
    if cxx is not None:
        compile_options["cxx"] = cxx
    return pops.resolve(
        pops.validate(case), layout=layout, backend=Production(), compile_options=compile_options
    )


def _bind(artifact):
    return pops.bind(
        artifact,
        resources={"execution_context": artifact_execution_context(artifact)},
    )


def _compile(resolved, *, route):
    """Publish once when this same matrix is launched under two MPI ranks."""
    from pops import _pops

    comm = _pops.mpi_world()
    if int(comm.size) == 1:
        return pops.compile(resolved)
    from tests.python.integration.mpi._compile_once import compile_resolved_plan_once

    return compile_resolved_plan_once(comm, resolved, route=route, compile_artifact=pops.compile)


def _collective_path(path):
    from pops import _pops

    comm = _pops.mpi_world()
    if int(comm.size) == 1:
        return path
    from pops._native_collectives import broadcast_value

    shared = broadcast_value(comm, str(path) if int(comm.rank) == 0 else None, root=0)
    return Path(shared)


def _evidence(artifact, n, payload, *, record_property, name):
    from pops import _pops

    platform = artifact.platform_manifest.to_data()
    row = {
        "schema": "pops.adc942.amr-transport-diffusion.v1",
        "grid": {"dimension": 2, "base_cells": [n, n], "levels": 2, "ratio": 2},
        "mpi_ranks": int(_pops.mpi_world().size),
        "backend": platform["backend"],
        "target": platform["target"],
        "device": platform["device"],
        **payload,
    }
    encoded = json.dumps(row, sort_keys=True)
    record_property(name, encoded)
    destination = os.environ.get("POPS_DIFFUSION_QUALIFICATION_EVIDENCE_DIR")
    if destination and int(_pops.mpi_world().rank) == 0:
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        (path / (name + ".json")).write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")


def _exchange_level(context):
    prefix = "pops.exchange.frame.v1/"
    assert context.startswith(prefix)
    size, encoded = context[len(prefix) :].split(":", 1)
    clock_size = int(size)
    assert encoded[clock_size] == "/"
    fields = encoded[clock_size + 1 :].split("/", 3)
    return int(fields[1])


def _quadrature(record):
    cell_token, axis_token, side_token = record["quadrature_identity"].split("/")
    cell = tuple(map(int, cell_token.split(":")[1:]))
    return cell, int(axis_token.split(":")[1]), int(side_token.split(":")[1])


def _assert_physical_linear_profile(runtime, level, *, n):
    level_n = n * 2**level
    valid = level_valid_mask(runtime, level, refinement_ratio=2)
    state = np.asarray(
        runtime.block_level_state_global("heat", level), dtype=np.float64
    ).reshape(level_n, level_n)
    exact_x = 1.0 + (np.arange(level_n, dtype=np.float64) + 0.5) / level_n
    exact = np.broadcast_to(exact_x, state.shape)
    np.testing.assert_allclose(state[valid], exact[valid], rtol=0.0, atol=3.0e-12)
    # The public dense snapshot uses zero only to represent cells without a patch.
    np.testing.assert_array_equal(state[~valid], 0.0)
    return valid, float(np.max(np.abs(state[valid] - exact[valid])))


def _coarse_fine_basis_oracle(runtime, providers=("provider/1", "provider/4")):
    rows = tuple(tuple(map(str, row)) for row in runtime._executor.program_flux_ledger_manifest())
    assert all(len(row) == 17 for row in rows)
    assert runtime.n_levels() == 2
    coarse_active = composite_active_mask(runtime, 0, refinement_ratio=2)
    if not coarse_active.any():
        # A fully covered coarse level has no coarse/fine interface to reconcile.
        # Authenticate that geometry and require the corresponding empty ledger.
        assert composite_active_mask(runtime, 1, refinement_ratio=2).all()
        assert not rows
        return {"no_coarse_fine_faces": "full_fine_cover", "ledger_count": 0}
    result = {}
    for provider in providers:
        stage_prefix = "pops.program-flux-expression.v1/" + provider + "/rhs/"
        selected = tuple(row for row in rows if row[2].startswith(stage_prefix))
        assert selected
        provider_result = {}
        axes = tuple(
            axis for axis in ("x", "y") if any(row[10].startswith(axis) for row in selected)
        )
        assert axes
        for axis in axes:
            coarse = tuple(row for row in selected if row[10] == axis + "_coarse")
            fine = tuple(row for row in selected if row[10] == axis + "_fine")
            assert coarse and fine

            def weighted_measure(group):
                return sum(
                    float(Fraction(int(row[8]), int(row[9]))) * float(row[11]) * float(row[12])
                    for row in group
                )

            coarse_measure = weighted_measure(coarse)
            fine_measure = weighted_measure(fine)
            defect = abs(coarse_measure - fine_measure)
            assert defect <= 5.0e-14 * max(1.0, coarse_measure, fine_measure)
            provider_result[axis] = {
                "coarse_count": len(coarse),
                "fine_count": len(fine),
                "coarse_weighted_measure": coarse_measure,
                "fine_weighted_measure": fine_measure,
                "signed_reconciliation_defect": defect,
            }
        result[provider] = provider_result
    return result


def _mass(runtime, n):
    return sum(
        float(composite_active_block_state(runtime, "heat", level, refinement_ratio=2).sum())
        / (n * 2**level) ** 2
        for level in range(runtime.n_levels())
    )


def _periodic_l2(runtime, n, time):
    squared = 0.0
    measure = 0.0
    amplitude = 0.2 * np.exp(-8 * np.pi**2 * DIFFUSIVITY * time)
    for level in range(runtime.n_levels()):
        level_n = n * 2**level
        coordinates = (np.arange(level_n) + 0.5) / level_n
        xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
        average = np.sinc(1 / level_n) ** 2
        exact = 1.0 + amplitude * average * np.sin(2 * np.pi * (xx - VELOCITY[0] * time)) * np.cos(
            2 * np.pi * (yy - VELOCITY[1] * time)
        )
        actual = np.asarray(runtime.block_level_state_global("heat", level), dtype=np.float64)
        actual = actual.reshape((1, level_n, level_n))[0]
        active = composite_active_mask(runtime, level, refinement_ratio=2)
        cell_measure = 1.0 / level_n**2
        squared += float(np.sum((actual[active] - exact[active]) ** 2)) * cell_measure
        measure += float(active.sum()) * cell_measure
    assert abs(measure - 1.0) < 2.0e-14
    return pmath.sqrt(squared)


@pytest.mark.parametrize("n", REFINEMENTS)
def test_periodic_full_refinement_restart_and_face_inventory(
    isolated_native_cache,
    native_cxx,
    kokkos_root,
    tmp_path,
    n,
    record_property,
):
    del isolated_native_cache, kokkos_root
    dt = _stable_dt(n)
    resolved = _author(n, dt, cxx=native_cxx)
    artifact = _compile(resolved, route="periodic-n%d" % n)
    continuous = _bind(artifact)
    restarted_source = _bind(artifact)
    initial_mass = _mass(continuous, n)
    steps = pmath.ceil(FINAL_TIME / dt)
    half_steps = steps // 2
    # Match the native binary64 clock's additions so intermediate run boundaries
    # preserve the exact accepted dt sequence on both sides of the restart.
    step_times = [0.0]
    for _ in range(steps - 1):
        step_times.append(step_times[-1] + dt)
    half_time = step_times[half_steps]
    before_last_time = step_times[-1]
    first_report = pops.run(continuous, t_end=before_last_time, max_steps=steps, console=False)
    mass_before_last = _mass(continuous, n)
    final_report = pops.run(continuous, t_end=FINAL_TIME, max_steps=2, console=False)
    pops.run(restarted_source, t_end=half_time, max_steps=half_steps, console=False)
    checkpoint = restarted_source.checkpoint(_collective_path(tmp_path / ("diffusion-n%d" % n)))
    restarted = _bind(artifact)
    restarted.restart(checkpoint)
    pops.run(restarted, t_end=FINAL_TIME, max_steps=steps + 1, console=False)
    final_mass = _mass(continuous, n)
    restarted_mass = _mass(restarted, n)
    continuous_mass_drift = final_mass - initial_mass
    restarted_mass_drift = restarted_mass - initial_mass
    assert abs(continuous_mass_drift) < 8.0e-11
    assert abs(restarted_mass_drift) < 8.0e-11
    l2_error = _periodic_l2(continuous, n, FINAL_TIME)
    assert l2_error < 0.025 / n
    restart_hashes = []
    for level in range(continuous.n_levels()):
        continuous_level = np.ascontiguousarray(continuous.block_level_state_global("heat", level))
        restarted_level = np.ascontiguousarray(restarted.block_level_state_global("heat", level))
        np.testing.assert_array_equal(
            continuous_level,
            restarted_level,
        )
        restart_hashes.append(hashlib.sha256(continuous_level.tobytes()).hexdigest())
    exchanges = continuous._executor._program_exchange_records()
    assert exchanges and max(abs(row["numerical_flux"]) for row in exchanges) > 0
    signed_exchange = sum(row["integrated_amount"] for row in exchanges)
    accepted_mass_change = final_mass - mass_before_last
    assert abs(signed_exchange - accepted_mass_change) < 8.0e-11
    basis = _coarse_fine_basis_oracle(continuous)
    _evidence(
        artifact,
        n,
        {
            "case": "periodic_fourier",
            "final_time": FINAL_TIME,
            "dt": dt,
            "accepted_steps": continuous.macro_step(),
            "first_run_accepted": first_report.accepted_steps,
            "final_run_accepted": final_report.accepted_steps,
            "rejected_steps": first_report.rejected_steps + final_report.rejected_steps,
            "l2_error": l2_error,
            "l2_bound": 0.025 / n,
            "continuous_mass_drift": continuous_mass_drift,
            "restarted_mass_drift": restarted_mass_drift,
            "restart_bit_exact": True,
            "restart_level_sha256": restart_hashes,
            "last_step_signed_exchange": signed_exchange,
            "last_step_mass_change": accepted_mass_change,
            "last_step_exchange_mass_defect": signed_exchange - accepted_mass_change,
            "accepted_exchange_count": len(exchanges),
            "coarse_fine_basis": basis,
        },
        record_property=record_property,
        name="adc942-periodic-n%d" % n,
    )


def test_physical_diffusion_boundary_is_steady_across_refinement_and_subcycling(
    isolated_native_cache,
    native_cxx,
    kokkos_root,
    record_property,
):
    del isolated_native_cache, kokkos_root
    n = REFINEMENTS[0]
    resolved = _author(n, _stable_dt(n), physical=True, cxx=native_cxx)
    artifact = _compile(resolved, route="physical-n%d" % n)
    runtime = _bind(artifact)
    before_boxes = tuple(runtime.patch_boxes())
    before_profiles = tuple(
        _assert_physical_linear_profile(runtime, level, n=n)
        for level in range(runtime.n_levels())
    )
    initial_mass = _mass(runtime, n)
    accepted_step_topologies = []
    accepted_controller_step = type(runtime)._accepted_controller_step

    def observe_accepted_step(simulation, *args, **kwargs):
        topology_before = tuple(simulation.patch_boxes())
        step_report = accepted_controller_step(simulation, *args, **kwargs)
        accepted_step_topologies.append((topology_before, tuple(simulation.patch_boxes())))
        return step_report

    with patch.object(type(runtime), "_accepted_controller_step", observe_accepted_step):
        report = pops.run(runtime, t_end=FINAL_TIME, max_steps=128, console=False)
    after_boxes = tuple(runtime.patch_boxes())
    after_profiles = tuple(
        _assert_physical_linear_profile(runtime, level, n=n)
        for level in range(runtime.n_levels())
    )
    final_mass = _mass(runtime, n)
    support_changes = tuple(
        int(np.count_nonzero(before[0] != after[0]))
        for before, after in zip(before_profiles, after_profiles, strict=True)
    )
    assert before_boxes != after_boxes
    assert any(support_changes)
    assert report.accepted_steps > 0
    assert len(accepted_step_topologies) == report.accepted_steps
    last_step_topology_before, last_step_topology_after = accepted_step_topologies[-1]
    assert last_step_topology_before == last_step_topology_after == after_boxes
    state_error = max(
        error
        for profiles in (before_profiles, after_profiles)
        for _valid, error in profiles
    )
    active_by_level = tuple(
        composite_active_mask(runtime, level, refinement_ratio=2)
        for level in range(runtime.n_levels())
    )
    valid_by_level = tuple(
        level_valid_mask(runtime, level, refinement_ratio=2)
        for level in range(runtime.n_levels())
    )
    physical_raw = {(axis, side): [] for axis in range(2) for side in range(2)}
    physical = {(axis, side): [] for axis in range(2) for side in range(2)}
    covered_coarse = {(axis, side): [] for axis in range(2) for side in range(2)}
    for record in runtime._executor._program_exchange_records():
        cell, axis, side = _quadrature(record)
        level = _exchange_level(record["evaluation_context"])
        assert 0 <= level < runtime.n_levels()
        level_n = n * 2**level
        assert all(0 <= coordinate < level_n for coordinate in cell)
        on_physical_face = (side == 0 and cell[axis] == 0) or (
            side == 1 and cell[axis] == level_n - 1
        )
        if not on_physical_face:
            continue
        physical_raw[axis, side].append(record)
        if active_by_level[level][tuple(reversed(cell))]:
            physical[axis, side].append(record)
            continue
        assert level + 1 < runtime.n_levels()
        child_cell = tuple(reversed(cell))
        child_region = tuple(
            slice(coordinate * 2, (coordinate + 1) * 2) for coordinate in child_cell
        )
        assert valid_by_level[level + 1][child_region].all()
        covered_coarse[axis, side].append(record)
    boundary_evidence = {}
    net_boundary_amount = 0.0
    last_dt = runtime._executor.program_last_dt()
    for (axis, side), records in physical.items():
        assert records
        raw_records = physical_raw[axis, side]
        excluded_records = covered_coarse[axis, side]
        assert len(raw_records) == len(records) + len(excluded_records)
        expected_flux = DIFFUSIVITY if axis == 0 else 0.0
        raw_flux_defect = max(abs(row["numerical_flux"] - expected_flux) for row in raw_records)
        assert raw_flux_defect < 3.0e-13
        flux_defect = max(abs(row["numerical_flux"] - expected_flux) for row in records)
        assert flux_defect < 3.0e-13
        raw_weighted_area = sum(
            row["face_measure"] * row["temporal_weight"] for row in raw_records
        )
        weighted_area = sum(row["face_measure"] * row["temporal_weight"] for row in records)
        assert abs(weighted_area - last_dt) < 3.0e-13
        amount = sum(row["integrated_amount"] for row in records)
        expected_amount = (1 if side else -1) * expected_flux * last_dt
        assert abs(amount - expected_amount) < 3.0e-13
        net_boundary_amount += amount
        boundary_evidence["axis%d_side%d" % (axis, side)] = {
            "raw_count": len(raw_records),
            "composite_count": len(records),
            "covered_coarse_count": len(excluded_records),
            "raw_level_counts": {
                str(level): sum(
                    _exchange_level(row["evaluation_context"]) == level for row in raw_records
                )
                for level in range(runtime.n_levels())
            },
            "expected_flux": expected_flux,
            "raw_max_flux_defect": raw_flux_defect,
            "max_flux_defect": flux_defect,
            "raw_weighted_area": raw_weighted_area,
            "weighted_area": weighted_area,
            "integrated_amount": amount,
            "expected_integrated_amount": expected_amount,
        }
    assert abs(net_boundary_amount) < 3.0e-13
    mass_change = final_mass - initial_mass
    assert abs(net_boundary_amount - mass_change) < 3.0e-13
    _evidence(
        artifact,
        n,
        {
            "case": "physical_linear_steady",
            "final_time": FINAL_TIME,
            "accepted_steps": report.accepted_steps,
            "rejected_steps": report.rejected_steps,
            "max_state_error": state_error,
            "initial_patch_boxes": before_boxes,
            "final_patch_boxes": after_boxes,
            "last_accepted_step_patch_boxes": {
                "before": last_step_topology_before,
                "after": last_step_topology_after,
            },
            "level_support_changes": support_changes,
            "last_dt": last_dt,
            "physical_boundary": boundary_evidence,
            "net_weighted_boundary_amount": net_boundary_amount,
            "composite_mass_change": mass_change,
            "boundary_mass_defect": net_boundary_amount - mass_change,
            "coarse_fine_basis": _coarse_fine_basis_oracle(runtime, ("provider/4",)),
        },
        record_property=record_property,
        name="adc942-physical-n%d" % n,
    )


def test_combined_local_bound_rejection_leaks_no_accepted_state_and_retry_succeeds(
    isolated_native_cache,
    native_cxx,
    kokkos_root,
    record_property,
):
    del isolated_native_cache, kokkos_root
    n = REFINEMENTS[0]
    stable = _stable_dt(n)
    fine_n = 2 * n
    fine_diffusion_frequency = 4 * DIFFUSIVITY * fine_n**2
    fine_transport_frequency = 2 * max(map(abs, VELOCITY)) * fine_n
    fine_combined_frequency = fine_diffusion_frequency + fine_transport_frequency
    fine_unstable_dt = 0.5 * (1 / fine_combined_frequency + 1 / fine_diffusion_frequency)
    unstable = 2 * fine_unstable_dt  # The fine level takes two substeps per macro step.
    coarse_frequency = 4 * DIFFUSIVITY * n**2 + 2 * max(map(abs, VELOCITY)) * n
    assert unstable * coarse_frequency < 1
    assert fine_unstable_dt * fine_diffusion_frequency < 1
    assert fine_unstable_dt * fine_transport_frequency < 1
    assert fine_unstable_dt * fine_combined_frequency > 1
    resolved = _author(n, unstable, cxx=native_cxx)
    artifact = _compile(resolved, route="rejection-n%d" % n)
    runtime = _bind(artifact)
    before = tuple(
        np.asarray(runtime.block_level_state_global("heat", level)).copy()
        for level in range(runtime.n_levels())
    )
    with pytest.raises(
        RuntimeError, match="combined_transport_diffusion_stability|rejected"
    ) as failure:
        pops.run(runtime, t_end=unstable, max_steps=1, console=False)
    assert runtime.time() == 0 and runtime.macro_step() == 0
    assert not runtime._executor.program_flux_ledger_manifest()
    assert not runtime._executor._program_exchange_records()
    for level, expected in enumerate(before):
        np.testing.assert_array_equal(runtime.block_level_state_global("heat", level), expected)
    report = pops.run(runtime, t_end=stable, max_steps=1, console=False)
    assert report.accepted_steps == 1 and report.rejected_steps == 0
    _evidence(
        artifact,
        n,
        {
            "case": "combined_bound_refusal_retry",
            "stable_dt": stable,
            "refused_dt": unstable,
            "fine_dt": fine_unstable_dt,
            "coarse_combined_bound": unstable * coarse_frequency,
            "fine_diffusion_bound": fine_unstable_dt * fine_diffusion_frequency,
            "fine_transport_bound": fine_unstable_dt * fine_transport_frequency,
            "fine_combined_bound": fine_unstable_dt * fine_combined_frequency,
            "failure": str(failure.value),
            "refused_state_unchanged": True,
            "refused_accepted_exchange_count": 0,
            "retry_accepted_steps": report.accepted_steps,
            "retry_rejected_steps": report.rejected_steps,
        },
        record_property=record_property,
        name="adc942-refusal-retry-n%d" % n,
    )
