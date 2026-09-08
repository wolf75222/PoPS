"""Execute precise field dependency versions and current-stage gradient observations."""
from __future__ import annotations

import numpy as np
import pops
import pytest
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.time import FailRun, FixedDt
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.unit.fields.test_program_field_problem import field_case
from tests.python.integration.runtime.test_public_field_problem import _data, _write_evidence

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def _stage_case(n):
    case, field, problem, program, values, point = field_case(component_transforms=True)
    first_key = next(iter(values))
    block = case._block_registry.handles()["first"]
    model = case._block_registry.spec("first")["model"]
    samples = []
    for index, transform in enumerate((None, "momentum_update", "density_update",
                                       "coefficient_update", None)):
        if transform is not None:
            values[first_key] = program.transform(
                values[first_key], transform=block[model.operators[transform]])
        if index == 4:
            point = program.stage("equal timestamp different stage", c=0)
        solution = program.solve(field, values=values, at=point).consume(action=FailRun())
        observed = field.observe(solution)
        samples.append(observed[field[problem.unknowns[0]]])
        if index == 4:
            gradient = observed.gradient(field[problem.unknowns[0]], dimension=2)
            assert gradient.point == point
    for index, sample in enumerate(samples):
        program.store_history("phi%d" % index, sample, depth=1)
    program.store_history("gradient", gradient, depth=1)
    for handle, value in values.items():
        state = program.state(handle)
        program.commit(state.next, program.value("accepted " + value.name, 1 * value,
                                                at=state.next.point))
    program.step_strategy(FixedDt(0.125))
    case.program(program)
    frame = model._frame
    layout = Uniform(CartesianGrid(frame=frame, cells=(n, n), periodic=PeriodicAxes(frame.axes)))
    return case, layout, case.resolve(field).qualified_id


def test_full_component_version_and_gradient_matrix(isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    evidence = {"schema": "pops.m3.field-stage-matrix.v1", "results": []}
    errors = []
    for n in (16, 32, 64):
        first, second, _exact = _data("variable_scalar_two_state_load", n)
        first = np.concatenate((first, np.zeros((1, n, n))))
        second = np.concatenate((second, np.zeros((1, n, n))))
        case, layout, field_identity = _stage_case(n)
        artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
        runtime = pops.bind(artifact, initial_state={"first": first, "second": second},
                           resources={"execution_context": artifact_execution_context(artifact)})
        report = pops.run(runtime, t_end=0.125, max_steps=1)
        assert report.accepted_steps == 1
        actual = [np.asarray(runtime.history_global("phi%d" % index, 0)).reshape(n, n)
                  for index in range(5)]
        for value, scale in zip(actual, (1, 1, 1.5, 0.75, 0.75), strict=True):
            np.testing.assert_allclose(value, scale * actual[0], rtol=0, atol=2e-9)
        np.testing.assert_array_equal(actual[1], actual[0])
        gradient = np.asarray(runtime.history_global("gradient", 0)).reshape(2, n, n)
        discrete_gradient = np.stack(tuple((np.roll(actual[-1], -1, axis) -
            np.roll(actual[-1], 1, axis)) * (n / 2) for axis in (1, 0)))
        np.testing.assert_allclose(gradient, discrete_gradient, rtol=0, atol=2e-11)
        coordinate = (np.arange(n) + 0.5) / n
        x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
        exact_gradient = -0.75 * 2 * np.pi * np.stack((np.sin(2*np.pi*x)*np.cos(2*np.pi*y),
                                                      np.cos(2*np.pi*x)*np.sin(2*np.pi*y)))
        error = gradient - exact_gradient
        l2 = float(np.sqrt(np.sum(error**2) / n**2))
        errors.append(l2)
        expected_first = first.copy()
        expected_first[:2] *= 2
        expected_first[2] += 1
        np.testing.assert_array_equal(np.asarray(runtime.state_global("first")).reshape(first.shape), expected_first)
        np.testing.assert_array_equal(np.asarray(runtime.state_global("second")).reshape(second.shape), second)
        diagnostics = runtime.program_report().diagnostics
        assert diagnostics["field.solves/" + field_identity] == 4
        assert diagnostics["field.reuses/" + field_identity] == 1
        evidence["results"].append({"n": n, "gradient_L2": l2,
            "gradient_Linf": float(np.max(np.abs(error))), "diagnostics": diagnostics,
            "artifact_identity": artifact.artifact_identity.token})
        _write_evidence("fields-stage-matrix-progress", evidence)
    assert min(np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))) > 1.8
    assert errors[-1] < 0.025
    _write_evidence("fields-stage-matrix", evidence)
