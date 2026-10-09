"""Native constrained friction: augmented solve does not authorize force projection."""

import copy
import json

import numpy as np
import pops
import pytest

from tests.python.integration.runtime.test_user_numerical_bodies_runtime import _compile, _root_check
from tests.python.support.m11_w10_case import case_module, oracle
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _snapshot(runtime):
    return tuple(np.asarray(runtime.state_global(name)).copy() for name in ("flux", "multiplier"))


def _run_errors(runtime, world):
    error = None
    try:
        pops.run(runtime, t_end=case_module.DT, max_steps=1, console=False)
    except Exception as exc:
        error = (type(exc).__name__, str(exc))
    if world is None:
        return (error,)
    from pops._native_collectives import allgather_value
    return allgather_value(world, error)


def _all_check(world, operation):
    """Converge a rank-local assertion before the next collective state check."""
    error = None
    try:
        operation()
    except Exception as exc:
        error = (type(exc).__name__, str(exc))
    if world is None:
        errors = (error,)
    else:
        from pops._native_collectives import allgather_value
        errors = allgather_value(world, error)
    assert not any(errors), errors


@pytest.mark.parametrize("permuted", (False, True))
def test_augmented_product_preserves_original_friction_and_refuses_incompatible_force(
        isolated_native_cache, native_cxx, kokkos_root, tmp_path, record_property, permuted):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout, subjects, order = case_module.build_case(permuted=permuted)
    artifact, world = _compile(case, layout, "m11-w10-%s" % permuted)
    context = artifact_execution_context(artifact)

    def bind(force):
        return pops.bind(artifact, initial_values={subjects[0]: force[list(order)].copy(),
                         subjects[1]: np.zeros((1, 4, 4))}, resources={"execution_context": context})

    def check_success(force, label):
        runtime = bind(force)
        errors = _run_errors(runtime, world)
        assert not any(errors), errors
        raw_flux, raw_multiplier = _snapshot(runtime)

        def accepted_clock():
            assert runtime.time() == case_module.DT and runtime.macro_step() == 1
        _all_check(world, accepted_clock)

        def original_equations():
            flux = raw_flux.reshape(3, 4, 4)[np.argsort(order)]
            multiplier = raw_multiplier.reshape(1, 4, 4)
            path = tmp_path / (label + ".npz")
            np.savez_compressed(path, force=force, flux=flux, multiplier=multiplier,
                                time=runtime.time(), macro_step=runtime.macro_step())
            with np.load(path) as saved:
                metrics = oracle.residuals(saved["force"], saved["flux"], saved["multiplier"])
                expected, expected_multiplier = oracle.reference(saved["force"])
                assert metrics["original_residual"] < oracle.THRESHOLD
                assert metrics["constraint"] < oracle.THRESHOLD
                assert metrics["augmented_residual"] < oracle.THRESHOLD
                np.testing.assert_allclose(saved["flux"], expected, rtol=0., atol=oracle.THRESHOLD)
                np.testing.assert_allclose(saved["multiplier"], expected_multiplier,
                                           rtol=0., atol=oracle.THRESHOLD)
                assert float(saved["time"]) == case_module.DT
            record_property(label, json.dumps(metrics, sort_keys=True))
        _root_check(world, original_equations)

    for sample, force in enumerate(oracle.force_samples()):
        check_success(force, "accepted_%d" % sample)

    invalid = oracle.force_samples()[1].copy()
    invalid[0, 1, 2] += .3  # outside range(L), despite solvable augmented equations
    failed = bind(invalid)
    before = _snapshot(failed)
    temporal_before = copy.deepcopy(failed._executor._temporal_restart_state.to_data())
    errors = _run_errors(failed, world)
    assert all(row is not None and row[0] == "RuntimeError" for row in errors), errors
    assert all("original_friction" in row[1] for row in errors), errors
    record_property("incompatible_force_native_diagnostic", repr(errors))
    after = _snapshot(failed)

    def rejected_envelope():
        assert failed.time() == 0. and failed.macro_step() == 0
        assert failed._executor._temporal_restart_state.to_data() == temporal_before
    _all_check(world, rejected_envelope)

    def unchanged():
        for old, new in zip(before, after, strict=True):
            np.testing.assert_array_equal(new, old)
    _root_check(world, unchanged)
    # Same compiled artifact, new bind with a compatible force; no hot edit of failed equations.
    check_success(oracle.force_samples()[2], "artifact_reused_after_refusal")
    record_property("mpi_ranks", 1 if world is None else int(world.size))
