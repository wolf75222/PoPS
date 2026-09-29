"""Installed T3 v2: nonlinear source/apply product, rebinding and atomic failure."""

import copy

import numpy as np
import pops
import pytest

from tests.python.integration.runtime.test_user_numerical_bodies_runtime import _compile, _root_check
from tests.python.support.local_product_operator_case import make_case
from tests.python.support.local_product_operator_oracle import (
    GAINS, WIDTHS, RESIDUAL_TOLERANCE, equation_without_capture,
    incompatible_sum_lower_bound, manufactured_solution, original_residual,
)
from tests.python.support.native_execution_context import artifact_execution_context

from tests.python.support.collective_checks import (
    collective_attempt, collective_call, collective_check, state_snapshots,
)

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _snapshots(runtime, world):
    return state_snapshots(runtime, world, tuple("block_%d" % i for i in range(len(WIDTHS))))


def _failure_rows(world, operation):
    return collective_attempt(world, operation)[1]


@pytest.mark.parametrize("reverse", (False, True))
def test_public_source_apply_product_rebind_and_incompatible_equation_rollback(
        isolated_native_cache, native_cxx, kokkos_root, reverse, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout, subjects, parameters = make_case(
        widths=WIDTHS, derived=True, reverse=reverse, return_parameters=True)
    artifact, world = _compile(case, layout, "local-product-operators-%s" % reverse)
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    with collective_check(world):
        exact = manufactured_solution()

    def bind(captures, gains):
        return collective_call(world, lambda: pops.bind(artifact, params=dict(zip(parameters, gains, strict=True)),
                         initial_values={subject: value.copy() for subject, value in
                                         zip(subjects, captures, strict=True)},
                         resources={"execution_context": context}))

    def accepted(runtime, captures, gains):
        errors = _failure_rows(world, lambda: pops.run(
            runtime, t_end=.01, max_steps=1, console=False))
        assert not any(errors), errors
        after = _snapshots(runtime, world)
        with collective_check(world):
            assert runtime.time() == .01 and runtime.macro_step() == 1

        def equations():
            values = tuple(value.reshape(width, 4, 4)
                           for value, width in zip(after, WIDTHS, strict=True))
            residual = original_residual(values, captures, gains)
            assert all(np.all(np.isfinite(value)) for value in values)
            assert max(np.max(np.abs(row)) for row in residual) < RESIDUAL_TOLERANCE
            for value, target in zip(values, exact, strict=True):
                np.testing.assert_allclose(value, target, rtol=0., atol=1.e-10)
        _root_check(world, equations)

    # Distinct non-default values of the three homonymous, owner-qualified gains;
    # both runs use the exact same compiled artifact and distinct frozen captures.
    for gains in GAINS:
        with collective_check(world):
            captures = equation_without_capture(exact, gains)
        accepted(bind(captures, gains), captures, gains)

    with collective_check(world):
        captures = tuple(value.copy() for value in equation_without_capture(exact, GAINS[0]))
        for value in captures:
            value[:, 1, 2] = -100.
        lower_bound = float(incompatible_sum_lower_bound(captures, GAINS[0])[1, 2])
        assert lower_bound > 600.
        record_property("incompatible_cell_sum_residual_lower_bound", lower_bound)
    failed = bind(captures, GAINS[0])
    before = _snapshots(failed, world)
    # Uniform CPU commits share one clock; include the accepted temporal envelope.
    temporal_before = collective_call(world, lambda: copy.deepcopy(
        failed._executor._temporal_restart_state.to_data()))
    for attempt in range(2):
        errors = _failure_rows(world, lambda: pops.run(
            failed, t_end=.01, max_steps=1, console=False))
        assert all(row is not None and row[2] for row in errors), errors
        assert all("coupled_implicit failed:" in row[1] for row in errors), errors
        with collective_check(world):
            record_property("incompatible_equation_diagnostics_%d" % attempt, repr(errors))
        after = _snapshots(failed, world)
        with collective_check(world):
            assert failed.time() == 0. and failed.macro_step() == 0
            assert failed._executor._temporal_restart_state.to_data() == temporal_before

        def unchanged(after=after):
            for old, value in zip(before, after, strict=True):
                np.testing.assert_array_equal(value, old)
        _root_check(world, unchanged)

    # This is a NEW bind of the SAME artifact with consistent equations. It is
    # intentionally distinct from the repeated refusal on the unchanged runtime.
    with collective_check(world):
        recovered_captures = equation_without_capture(exact, GAINS[1])
    accepted(bind(recovered_captures, GAINS[1]), recovered_captures, GAINS[1])
    with collective_check(world):
        record_property("mpi_ranks", 1 if world is None else int(world.size))
