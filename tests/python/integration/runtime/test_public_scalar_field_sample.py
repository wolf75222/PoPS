"""Native public scalar projection; this manufactured witness is not the ADC benchmark."""
from __future__ import annotations

import numpy as np

import pops
from tests.python.unit.time.test_field_sampling import manufactured_program
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import native_profile
from tests.python.support.collective_checks import collective_call, collective_check


def initial_declared_fields(plan, mask, potential):
    """Supply the real initial data of this fixture's declared input FieldSpaces.

    FieldSample and the CG solution have separate owned scalar storage. These
    named inputs are initialized because the declaration requires them; neither
    input is used as the matvec mask or as the independent analytical reference.
    """
    from pops.model.provider_pack import ProviderPack
    from pops.codegen.provider_instances import require_instance_contract, runtime_key
    result = {}
    for block in plan.blocks:
        operations = block.resolved_operations
        pack = ProviderPack.from_data(operations.to_data()["provider_evidence"]["auxiliary"])
        instance = require_instance_contract(operations, owner_qid=block.instance_owner_qid)
        for key in pack:
            entry = pack.declared_entry(key)
            if entry.producer != "runtime_input" or not entry.availability or entry.slot is None:
                continue
            assert key.space_kind == "field" and key.space_name == key.component
            assert key.space_name in ("mask", "phi"), "unexpected required input must remain visible"
            value = mask if key.space_name == "mask" else potential
            result[runtime_key(key, instance)] = np.ascontiguousarray(value, dtype=np.float64)
    assert {key.space_name for key in result} == {"mask", "phi"} and len(result) == 2
    return result


def test_public_auxiliary_sample_and_projected_poisson(
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    native, world = native_profile(scope="public typed scalar sample")
    record_property("native_abi_version", native.__abi_version__)
    record_property("native_profile", "MPI" if world is not None else "Serial")
    case, layout, providers = collective_call(world, manufactured_program)
    resolved = collective_call(world, lambda: pops.resolve(
        pops.validate(case), layout=layout, layout_providers=providers))
    # Native packages must be the same publication on every MPI rank; pytest's caches are local.
    from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
    artifact = (pops.compile(resolved) if world is None else compile_resolved_plan_once(
        world, resolved, route="typed scalar sample", compile_artifact=pops.compile))
    n = 16
    coordinate = (np.arange(n, dtype=np.float64)+.5)/n
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    mask = (x-.5)**2+(y-.5)**2 < .2**2
    exact = np.where(mask, np.sin(3*x+2*y)+.1*np.cos(7*x-5*y), 0.)
    lap = (np.roll(exact, 1, 0)+np.roll(exact, -1, 0)
           +np.roll(exact, 1, 1)+np.roll(exact, -1, 1)-4*exact)*n*n
    load = np.where(mask, -lap, 0.)
    initial = np.ascontiguousarray(load[np.newaxis, ...])
    initial_fields = initial_declared_fields(artifact.plan, mask, np.zeros_like(exact))
    runtime = collective_call(world, lambda: pops.bind(artifact, initial_state={"carrier": initial.copy()},
                       aux=initial_fields,
                       resources={"execution_context": artifact_execution_context(artifact)}))
    report = collective_call(world, lambda: pops.run(runtime, t_end=.01, max_steps=1))
    # Read-only native diagnostics after the public lifecycle; no private engine installation.
    diagnostics = collective_call(world, lambda: dict(runtime._executor.program_diagnostics()))
    expected_inf = float(np.abs(exact).max())
    expected_l2 = float(np.linalg.norm(exact))
    state = collective_call(world, lambda: np.asarray(runtime.state_global("carrier")).reshape(initial.shape))
    with collective_check(world):
        assert report.accepted_steps == 1
        assert diagnostics["sampled_mask_cells"] == int(mask.sum())
        assert abs(diagnostics["scalar_solution_norm"]-expected_inf) <= 2e-9
        assert abs(diagnostics["scalar_solution_l2"]-expected_l2) <= 2e-9
        assert diagnostics["scalar_solution_error_inf"] <= 2e-9
        assert np.array_equal(state, initial)
    record_property("field_sample_native", {
        "mask_cells": int(mask.sum()), "solution_inf": diagnostics["scalar_solution_norm"],
        "solution_l2": diagnostics["scalar_solution_l2"],
        "solution_error_inf": diagnostics["scalar_solution_error_inf"],
        "initialized_input_fields": sorted(key.space_name for key in initial_fields),
        "accepted_steps": report.accepted_steps, "adc_reproduction": False,
    })
