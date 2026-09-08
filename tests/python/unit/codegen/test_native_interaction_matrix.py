"""Declared M4 uniform heterogeneous interaction matrix, imported and transparent paths."""
import ctypes
from pathlib import Path
import statistics
import time

import numpy as np
import pytest
import pops
from pops.layouts import Uniform
from pops.model import Signature
from pops.model.bundles import ProductSpace
from pops.native_calls import NativeDerivative, NativeFunction, NativeInputDomain
from pops.native_components import PreparedNativeComponent
from pops.codegen._orchestration_compile import build_program_model_graph
from pops.codegen.program_codegen import emit_cpp_program
from tests.python.support.layout_plan import cartesian_grid
from tests.python.support.native_execution_context import artifact_execution_context
from test_interaction_inventory_quadrature import interaction_case
from test_native_call_compiled import HEADER


def _imported_factory(directory):
    directory.mkdir()
    (directory / "exchange.hpp").write_text(HEADER + '''
extern "C" long pops_interaction_calls() { return imported::Exchange::calls.load(); }
extern "C" void pops_interaction_reset() { imported::Exchange::calls.store(0); }
''')
    component = PreparedNativeComponent.header_only("matrix.exchange", include_root=directory,
                                                   entry_headers=("exchange.hpp",))

    def captured(left, right):
        function = NativeFunction(component, "imported::Exchange::evaluate",
            Signature((left.space, right.space), ProductSpace({"left": left.space, "right": right.space})),
            reads=((0, 0), (1, 0), (1, 2)),
            domains=(NativeInputDomain(1, 2, lower=0, lower_open=True),),
            derivatives=(NativeDerivative("exact", "imported::Exchange::jacobian"),),
            effects=("fallible", "diagnostic_counter"))
        return function(left, right, occurrence="momentum-energy-exchange")
    return captured


def _initial(n):
    return {"left": np.stack([np.full((n, n), value) for value in (2., 5.)]),
            "right": np.stack([np.full((n, n), value) for value in (1., 3., 2.)])}


def _expected(steps, stages):
    a, b = np.asarray((2., 5.)), np.asarray((1., 3., 2.))
    def rates(left, right):
        force = right[0] / right[2] - left[0]
        power = (left[0] + right[0] / right[2]) * force / 2
        return np.asarray((force, power)), np.asarray((-force, -power, 0))
    for _ in range(steps):
        ra, rb = rates(a, b)
        if stages == 2:
            ra2, rb2 = rates(a + .001 * ra, b + .001 * rb)
            ra, rb = (ra + ra2) / 2, (rb + rb2) / 2
        a, b = a + .001 * ra, b + .001 * rb
    return a, b


def test_native_footprint_preserves_detached_typed_argument_authority(tmp_path):
    from pops.time._program.detach import detach_compiled_program
    factory = _imported_factory(tmp_path / "external")
    case, _program, _model, _maps = interaction_case(n=16, stages=2, native_function=factory)
    resolved = pops.resolve(pops.validate(case), layout=Uniform(cartesian_grid(n=16, periodic=True)))
    graph = build_program_model_graph(resolved)
    source = emit_cpp_program(resolved.time, model_graph=graph)
    detached = detach_compiled_program(resolved.time)
    assert detached._ir_hash() == resolved.time._ir_hash()
    assert emit_cpp_program(detached, model_graph=graph) == source
    assert source.count("imported::Exchange::evaluate(") == 2


@pytest.fixture(scope="module", params=[(n, kind) for n in (16, 32, 64) for kind in ("transparent", "native")],
                ids=["n%d-%s" % (n, kind) for n in (16, 32, 64) for kind in ("transparent", "native")])
def compiled_case(request, tmp_path_factory):
    from pops._native_selector import select_native_dimension
    select_native_dimension(2)
    n, kind = request.param
    directory = tmp_path_factory.mktemp("m4_%d_%s" % (n, kind))
    factory = _imported_factory(directory / "imported") if kind == "native" else None
    case, _program, _model, _maps = interaction_case(n=n, stages=2, native_function=factory)
    resolved = pops.resolve(pops.validate(case), layout=Uniform(cartesian_grid(n=n, periodic=True)))
    source = emit_cpp_program(resolved.time, model_graph=build_program_model_graph(resolved))
    assert source.count("consume_pointwise_evaluation_status") == 2
    assert source.count("ctx.stage_exchange(") == 8
    start = time.perf_counter()
    artifact = pops.compile(resolved)
    cost = time.perf_counter() - start
    native = ctypes.CDLL(str(artifact.program.so_path)) if kind == "native" else None
    if native is not None:
        native.pops_interaction_calls.restype = ctypes.c_long
        native.pops_interaction_reset.restype = None
    return n, kind, artifact, native, cost, len(source.encode())


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.integration
def test_full_interaction_numerical_matrix(compiled_case, record_property):
    n, kind, artifact, native, compile_seconds, source_bytes = compiled_case
    initial = _initial(n)
    simulation = pops.bind(artifact, initial_state=initial,
        resources={"execution_context": artifact_execution_context(artifact)})
    local_cells = sum(np.prod(np.asarray(upper) - np.asarray(lower))
                      for lower, upper in simulation.local_boxes("left"))
    if native is not None:
        native.pops_interaction_reset()
    report = pops.run(simulation, t_end=.1, max_steps=100)
    assert report.accepted_steps == 100 and report.rejected_steps == 0
    for name, expected in zip(("left", "right"), _expected(100, 2), strict=True):
        actual = np.asarray(simulation.state_global(name))
        np.testing.assert_allclose(actual, expected[:, None, None] * np.ones((1, n, n)), rtol=0, atol=1e-11)
    for component, invariant in ((0, 3.), (1, 8.)):
        assert abs(simulation.integral("left", component) + simulation.integral("right", component) - invariant) < 1e-11
    a, b = _expected(100, 2)
    assert a[1] - a[0]**2 / 2 + b[1] - b[0]**2 / (2*b[2]) > 5.75
    rows = simulation._executor_for_block("left")._program_exchange_records()
    assert len(rows) == 8
    for inventory in ("momentum", "total_energy"):
        selected = [row for row in rows if row["occurrence_identity"].endswith(":" + inventory)]
        assert len(selected) == 4
        assert abs(sum(row["integrated_amount"] for row in selected)) < 1e-11
    if native is not None:
        assert native.pops_interaction_calls() == int(local_cells) * 2 * 100
        record_property("actual_native_calls_local", native.pops_interaction_calls())
    record_property("n", n)
    record_property("realization", kind)
    record_property("local_cells", int(local_cells))
    record_property("accepted_steps", report.accepted_steps)
    record_property("joint_stage_evaluations_per_step", 2)
    record_property("accepted_exchange_records_last_step", len(rows))
    record_property("compile_seconds", compile_seconds)
    record_property("program_source_bytes", source_bytes)
    record_property("program_binary_bytes", Path(artifact.program.so_path).stat().st_size)


@pytest.fixture(scope="module", params=[(n, kind) for n in (16, 32, 64) for kind in ("unshared", "shared")],
                ids=["cost-n%d-%s" % (n, kind) for n in (16, 32, 64) for kind in ("unshared", "shared")])
def cost_case(request, tmp_path_factory):
    from pops._native_selector import select_native_dimension
    select_native_dimension(2)
    n, kind = request.param
    factory = _imported_factory(tmp_path_factory.mktemp("m4_cost") / "imported")
    case, _program, _model, _maps = interaction_case(n=n, stages=2, native_function=factory,
                                                   inventories=False, unshared=kind == "unshared")
    resolved = pops.resolve(pops.validate(case), layout=Uniform(cartesian_grid(n=n, periodic=True)))
    source = emit_cpp_program(resolved.time, model_graph=build_program_model_graph(resolved))
    assert source.count("consume_pointwise_evaluation_status") == (4 if kind == "unshared" else 2)
    assert "ctx.stage_exchange(" not in source
    start = time.perf_counter()
    artifact = pops.compile(resolved)
    compile_seconds = time.perf_counter() - start
    native = ctypes.CDLL(str(artifact.program.so_path))
    native.pops_interaction_calls.restype = ctypes.c_long
    native.pops_interaction_reset.restype = None
    return n, kind, artifact, native, compile_seconds, len(source.encode())


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.integration
def test_full_interaction_timing_matrix(cost_case, record_property):
    n, kind, artifact, native, compile_seconds, source_bytes = cost_case
    samples = []
    for repetition in range(9):  # predeclared two warmups, seven measurements
        simulation = pops.bind(artifact, initial_state=_initial(n),
            resources={"execution_context": artifact_execution_context(artifact)})
        simulation.integral("left", 0)  # native reduction/fence before timing
        native.pops_interaction_reset()
        start = time.perf_counter()
        report = pops.run(simulation, t_end=.1, max_steps=100)
        simulation.integral("left", 0)  # synchronize native work before timing ends
        elapsed = time.perf_counter() - start
        assert report.accepted_steps == 100 and report.rejected_steps == 0
        local_cells = sum(np.prod(np.asarray(upper) - np.asarray(lower))
                          for lower, upper in simulation.local_boxes("left"))
        assert native.pops_interaction_calls() == int(local_cells) * (4 if kind == "unshared" else 2) * 100
        for component, invariant in ((0, 3.), (1, 8.)):
            assert abs(simulation.integral("left", component) + simulation.integral("right", component) - invariant) < 1e-11
        if repetition >= 2:
            samples.append(elapsed)
    median = statistics.median(samples)
    record_property("n", n)
    record_property("realization", kind)
    record_property("timing_samples_seconds", samples)
    record_property("timing_median_seconds", median)
    record_property("timing_mad_seconds", statistics.median(abs(sample-median) for sample in samples))
    record_property("timing_scope", "100 full accepted steps and native integral fence; bind/compile excluded")
    record_property("counter_instrumentation", "one atomic increment per native call on both matched routes")
    record_property("accepted_inventory_instrumentation", "omitted on both timing routes; qualified separately")
    record_property("actual_native_calls_local", native.pops_interaction_calls())
    record_property("materialized_rhs_fields_per_stage", 4 if kind == "unshared" else 2)
    record_property("materialized_status_reason_fields_per_stage", 4 if kind == "unshared" else 2)
    record_property("compile_seconds", compile_seconds)
    record_property("program_source_bytes", source_bytes)
    record_property("program_binary_bytes", Path(artifact.program.so_path).stat().st_size)
