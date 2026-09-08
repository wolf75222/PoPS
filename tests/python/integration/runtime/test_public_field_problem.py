"""Full public M3 field matrix: actual native solves, joint gauge and rejected publication.

The predeclared matrix is Uniform Dim2, OpenMP with two threads, MPI1, and N16/32/64.
All norms use the returned native history observations; the residual is independently
recomputed with the selected conservative harmonic-face discretization.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pops
import pytest
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.runtime_environment import runtime_environment_report
from pops.time import FailRun, FixedDt, RejectAttempt
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.unit.fields.test_program_field_problem import field_case


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
REFINEMENTS = (16, 32, 64)
CASES = ("variable_scalar_two_state_load", "joint_neumann_constants", "joint_neumann_smooth")


def _data(kind, n):
    coordinate = (np.arange(n, dtype=np.float64) + 0.5) / n
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    coefficient = 2 + 0.5 * np.sin(2 * np.pi * x)
    if kind == "variable_scalar_two_state_load":
        exact = np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
        load = (8 * np.pi**2 * coefficient * exact
                + 2 * np.pi**2 * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * x)
                * np.cos(2 * np.pi * y))
        return np.stack((0.5 * load, coefficient)), np.stack((0.5 * load, exact)), exact[None]
    if kind == "joint_neumann_smooth":
        mode1 = np.cos(np.pi * x) * np.cos(np.pi * y)
        mode2 = 0.5 * np.cos(2 * np.pi * x) * np.cos(np.pi * y)
        exact = np.stack((0.375 + mode1, -0.375 + mode2))
        coupling = 2 * (exact[0] - exact[1])
        first_load = 2 * np.pi**2 * mode1 + coupling
        second_load = 5 * np.pi**2 * mode2 - coupling
    else:
        first_load = np.full((n, n), 1.5)
        second_load = np.full((n, n), 0 if kind == "joint_incompatible_load" else -1.5)
        exact = np.stack((np.full((n, n), 0.375), np.full((n, n), -0.375)))
    return np.stack((first_load, coefficient)), np.stack((second_load, np.ones_like(x))), exact


def _public_case(kind, n, *, reject=False):
    joint = kind != "variable_scalar_two_state_load"
    case, field, problem, program, values, point = field_case(joint=joint)
    # This provisional diagnostic executes before the failure gate. It must disappear on rejection.
    program.record_scalar("provisional_load_sum", program.sum_component(next(iter(values.values())), 0))
    solved = program.solve(field, values=values, at=point).consume(
        action=RejectAttempt() if reject else FailRun())
    if not reject:
        solved = program.solve(field, values=values, at=point).consume(action=FailRun())
    observations = field.observe(solved)
    for index, unknown in enumerate(problem.unknowns):
        observed = observations[field[unknown]]
        program.record_scalar("phi%d_sum" % index, program.sum_component(observed, 0))
        program.store_history("phi%d" % index, observed, depth=1)
    for handle in values:
        state = program.state(handle)
        program.commit(state.next, program.value(
            "unchanged_" + state.n.name, 1 * state.n, at=state.next.point))
    program.step_strategy(FixedDt(0.125))
    case.program(program)
    frame = case._block_registry.spec("first")["model"]._frame
    layout = Uniform(CartesianGrid(frame=frame, cells=(n, n),
        periodic=None if joint else PeriodicAxes(frame.axes)))
    return case, layout, case.resolve(field).qualified_id


def _negative_divergence(value, coefficient, *, periodic):
    """Independent cell-centered finite-volume residual, including physical face closures."""
    n = value.shape[-1]
    result = np.zeros_like(value)
    for axis in (0, 1):
        high_value, low_value = np.roll(value, -1, axis), np.roll(value, 1, axis)
        high_coef, low_coef = np.roll(coefficient, -1, axis), np.roll(coefficient, 1, axis)
        if not periodic:
            lower, upper = [slice(None), slice(None)], [slice(None), slice(None)]
            lower[axis], upper[axis] = 0, -1
            lower, upper = tuple(lower), tuple(upper)
            low_value[lower], high_value[upper] = value[lower], value[upper]
            low_coef[lower], high_coef[upper] = coefficient[lower], coefficient[upper]
        high_face = 2 * coefficient * high_coef / (coefficient + high_coef)
        low_face = 2 * coefficient * low_coef / (coefficient + low_coef)
        result -= n**2 * (high_face * (high_value - value) - low_face * (value - low_value))
    return result


def _residual(kind, actual, first, second):
    if kind == "variable_scalar_two_state_load":
        rhs = (first[0] + second[0])[None]
        image = _negative_divergence(actual[0], first[1], periodic=True)[None]
    else:
        rhs = np.stack((first[0], second[0]))
        image = np.stack(tuple(_negative_divergence(value, np.ones_like(value), periodic=False)
                               for value in actual))
        coupling = 2 * (actual[0] - actual[1])
        image[0] += coupling
        image[1] -= coupling
    return float(np.linalg.norm(image - rhs) / np.linalg.norm(rhs))


def _write_evidence(name, payload):
    directory = os.environ.get("POPS_FIELD_QUALIFICATION_EVIDENCE_DIR")
    if directory:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / (name + ".json")).write_text(json.dumps(payload, indent=2) + "\n")


def test_full_public_field_mms_and_joint_gauge_matrix(
    isolated_native_cache, native_cxx, kokkos_root, record_property,
):
    del isolated_native_cache, native_cxx, kokkos_root
    evidence = {"schema": "pops.m3.public-field-matrix.v1", "refinements": REFINEMENTS,
                "residual_evidence": "independent discrete residual of native returned observations",
                "results": []}
    for kind in CASES:
        errors = []
        for n in REFINEMENTS:
            first, second, exact = _data(kind, n)
            case, layout, field_identity = _public_case(kind, n)
            validated = pops.validate(case)
            resolved = pops.resolve(validated, layout=layout)
            assert resolved.field_plans == {}
            assert tuple(resolved.program_field_plans) == ("electric",)
            artifact = pops.compile(resolved)
            runtime = pops.bind(artifact, initial_state={"first": first.copy(), "second": second.copy()},
                resources={"execution_context": artifact_execution_context(artifact)})
            report = pops.run(runtime, t_end=0.125, max_steps=1)
            evidence["runtime_environment"] = runtime_environment_report()
            assert report.accepted_steps == 1
            assert runtime.macro_step() == 1 and runtime.time() == 0.125
            actual = np.stack(tuple(np.asarray(runtime.history_global("phi%d" % index, 0))
                                   .reshape(n, n) for index in range(len(exact))))
            # Load states and coefficient state remain frozen throughout the complete solve.
            np.testing.assert_array_equal(np.asarray(runtime.state_global("first")).reshape(first.shape), first)
            np.testing.assert_array_equal(np.asarray(runtime.state_global("second")).reshape(second.shape), second)
            diagnostics = runtime.program_report().diagnostics
            assert diagnostics["field.solves/" + field_identity] == 1
            assert diagnostics["field.reuses/" + field_identity] == 1
            error = actual - exact
            l2 = float(np.sqrt(np.sum(error**2) / n**2))
            linf = float(np.max(np.abs(error)))
            residual = _residual(kind, actual, first, second)
            means = np.mean(actual, axis=(1, 2))
            assert residual < 1e-10
            assert abs(float(np.sum(means))) < 1e-10
            if len(exact) == 2:
                np.testing.assert_allclose(means, (0.375, -0.375), rtol=0, atol=1e-10)
            if kind == "joint_neumann_constants":
                assert linf < 1e-10
            errors.append((l2, linf))
            evidence["results"].append({"case": kind, "n": n, "L2": l2, "Linf": linf,
                "relative_residual": residual, "means": means.tolist(),
                "accepted_steps": report.accepted_steps,
                "diagnostics": diagnostics,
                "artifact_identity": artifact.artifact_identity.token})
            _write_evidence("fields-public-matrix-progress", evidence)
        if kind != "joint_neumann_constants":
            orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
            assert np.min(orders) >= 1.8
            assert errors[-1][0] < 0.002
            evidence.setdefault("orders", {})[kind] = orders.tolist()
    record_property("field_matrix", json.dumps(evidence, sort_keys=True))
    _write_evidence("fields-public-matrix", evidence)


def test_full_public_incompatible_joint_load_rejects_without_publishing(
    isolated_native_cache, native_cxx, kokkos_root, record_property,
):
    del isolated_native_cache, native_cxx, kokkos_root
    rows = []
    for n in REFINEMENTS:
        first, second, _exact = _data("joint_incompatible_load", n)
        case, layout, _field_identity = _public_case("joint_incompatible_load", n, reject=True)
        artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
        runtime = pops.bind(artifact, initial_state={"first": first.copy(), "second": second.copy()},
            resources={"execution_context": artifact_execution_context(artifact)})
        before = runtime.program_report()
        with pytest.raises(Exception, match="incompatible_rhs") as failure:
            pops.run(runtime, t_end=0.125, max_steps=1)
        assert runtime.time() == 0 and runtime.macro_step() == 0
        np.testing.assert_array_equal(np.asarray(runtime.state_global("first")).reshape(first.shape), first)
        np.testing.assert_array_equal(np.asarray(runtime.state_global("second")).reshape(second.shape), second)
        after = runtime.program_report()
        assert after.diagnostics == before.diagnostics
        assert after.histories == before.histories
        rows.append({"n": n, "status": "incompatible_rhs", "error": str(failure.value),
                     "accepted_steps": runtime.macro_step(), "accepted_time": runtime.time()})
    record_property("incompatible_joint_matrix", json.dumps(rows, sort_keys=True))
    _write_evidence("fields-public-incompatible", {"results": rows,
                                                  "runtime_environment": runtime_environment_report()})
