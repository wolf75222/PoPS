"""Typed diagnostics execute through the final ConsumerGraph and native runtime."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pops
import pytest
from pops.output import read_paraview
from tests.python.support.native_execution_context import artifact_execution_context


ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("pops_diagnostic_acceptance", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _diagnostic_rows(reopened):
    rows = reopened.manifest["snapshot"]["diagnostics"]
    return {row["key"]["reduction"]: row for row in rows}


def _scalar(row):
    return float.fromhex(row["value"])


def _bind_native_artifact(artifact, **inputs):
    """Bind through the exact serial or MPI resource proven by this artifact."""
    return pops.bind(
        artifact,
        resources={"execution_context": artifact_execution_context(artifact)},
        **inputs,
    )


@pytest.mark.compiler
@pytest.mark.native_loader
def test_typed_diagnostics_execute_as_native_accepted_output(
    tmp_path, isolated_native_cache, native_cxx, kokkos_root,
):
    """All public measures lower exactly once and publish only after an accepted native step."""
    del isolated_native_cache, native_cxx, kokkos_root
    example = _load_example()
    target = example.build_final_case(
        output_root=tmp_path,
        output_mode=example._native_output_mode(),
    )
    validated = pops.validate(target.authoring.case)
    resolved = pops.resolve(validated, layout=target.layout)

    diagnostic_manifest, = tuple(
        node for node in resolved.consumer_graph.nodes
        if node.diagnostic_quantities
    )
    assert len(diagnostic_manifest.diagnostics) == 5
    assert len(diagnostic_manifest.diagnostic_quantities) == 5
    assert all(
        quantity.identity.domain == "consumer-quantity"
        for quantity in diagnostic_manifest.diagnostic_quantities
    )
    executions = tuple(
        quantity.execution for quantity in diagnostic_manifest.diagnostic_quantities)
    assert sum(len(value["operations"]) for value in executions) == 6
    assert {
        operation["name"]
        for execution in executions
        for operation in execution["operations"]
    } == {"integral", "l1", "l2", "linf", "min", "max"}
    assert all(value["conservation"] is None for value in executions)
    assert all(
        quantity.reference == target.authoring.case.resolve(
            target.authoring.tracer_state)
        for quantity in diagnostic_manifest.diagnostic_quantities
    )

    artifact = pops.compile(resolved)
    simulation = example._bind_artifact(
        artifact,
        params=example.build_bind_params(target.authoring),
    )
    report = pops.run(
        simulation,
        t_end=0.11,
        max_steps=1_000,
        output_dir=tmp_path,
    )
    assert report.accepted_steps >= 10

    paths = tuple(sorted(tmp_path.rglob("*.vtu")))
    assert paths, "the accepted diagnostic cadence did not publish its scientific output"
    reopened = read_paraview(paths[-1])
    rows = _diagnostic_rows(reopened)
    assert set(rows) == {
        "integral", "l1", "l2", "linf", "min", "max",
    }

    values = {name: _scalar(row) for name, row in rows.items()}
    assert all(math.isfinite(value) for value in values.values())
    assert values["min"] <= values["max"]
    assert values["l1"] + 1.0e-14 >= abs(values["integral"])
    assert values["linf"] + 1.0e-14 >= max(abs(values["min"]), abs(values["max"]))

    recorded = simulation.inspect().diagnostics
    for row in rows.values():
        key = "%s:%s" % (
            row["key"]["reference"]["qualified_id"], row["key"]["reduction"])
        key = "%s:%s" % (key, row["key"]["state_id"])
        assert recorded[key] == _scalar(row)


def _periodic_conservation_target(
    output_mode, *, declare_case_initial=False, declare_analytic_initial=False,
    declare_analytic_embedded_boundary=False,
):
    """Build one evolving closed system whose integral is a genuine invariant."""
    from pops.diagnostics import ConservationCheck, Integral
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.initial import InitialCondition
    from pops.layouts import Uniform
    from pops.lib.initial import Analytic, Gaussian
    from pops.lib.time import ForwardEuler
    from pops.math import ddt, div
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.output import Checkpoint, ConsumerGraph, ParaView, ScientificOutput
    from pops.projection import ConservativeCellAverage
    from pops.representations import Conservative
    from pops.spaces import CellState
    from pops.time import FixedDt, every

    frame = Rectangle(
        "periodic-diagnostic-domain", lower=(0.0, 0.0), upper=(1.0, 1.0),
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("periodic-diagnostic-advection", frame=frame)
    state = model.state(
        "U", components=("u",), representation=Conservative(),
        space=CellState(frame=frame),
    )
    (u,) = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (0.5 * u,), y_axis: (0.25 * u,)},
        waves={x_axis: (0.5,), y_axis: (0.25,)},
    )
    rate = model.rate("transport-rate", equation=ddt(state) == -div(flux))

    case = pops.Case("periodic-diagnostic-conservation")
    block = case.block("tracer", model=model)
    block_state = block[state]
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(numerics, block=block)
    program = ForwardEuler(block_state, rate=rate)
    dt = 2.5e-3
    program.step_strategy(FixedDt(dt))
    case.program(program)
    if declare_case_initial and declare_analytic_initial:
        raise ValueError("select exactly one Case initial profile")
    if declare_case_initial:
        case.initials.add(InitialCondition(
            state=block_state,
            value=Gaussian(
                frame=frame,
                center={frame.x: 0.35, frame.y: 0.45},
                background=0.2,
                amplitude=0.8,
                inverse_width=80.0,
            ),
            projection=ConservativeCellAverage(),
        ))
    if declare_analytic_initial:
        from pops.analytic import coordinates

        x_coord, y_coord = coordinates(frame)
        density = 0.25 + x_coord * x_coord + 2.0 * y_coord + x_coord * y_coord
        case.initials.add(InitialCondition(
            state=block_state,
            value=Analytic(frame=frame, components=(density,)),
            projection=ConservativeCellAverage(),
        ))
    schedule = every(1, clock=program.clock)
    invariant = ConservationCheck(
        Integral(block=block, cadence=schedule), tolerance=1.0e-10)
    case.consumers(ConsumerGraph.from_consumers((
        ScientificOutput(
            format=ParaView(mode=output_mode),
            schedule=schedule,
            fields=(block_state,),
            diagnostics=(invariant,),
            target="periodic/conservation",
        ),
        Checkpoint(
            schedule=every(100, clock=program.clock),
            target="periodic/restart",
            bit_identical=True,
        ),
    )))
    embedded_boundary = None
    if declare_analytic_embedded_boundary:
        from pops.boundary import ZeroFlux
        from pops.mesh.geometry import Disc, EmbeddedBoundary
        from pops.mesh.masks import Staircase

        outer = Disc(center=(0.5, 0.5), radius=0.4)
        inner = Disc(center=(0.5, 0.5), radius=0.2)
        embedded_boundary = EmbeddedBoundary(
            outer - inner,
            Staircase(),
            ZeroFlux(),
        )
    layout = Uniform(
        CartesianGrid(
            frame=frame,
            cells=(16, 16),
            periodic=PeriodicAxes(frame.axes),
        ),
        embedded_boundary=embedded_boundary,
    )
    coordinates = (np.arange(16, dtype=np.float64) + 0.5) / 16.0
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    initial_state = np.ascontiguousarray(
        (0.2 + 0.8 * np.exp(-80.0 * ((x - 0.35) ** 2 + (y - 0.45) ** 2)))[
            np.newaxis, ...
        ]
    )
    return case, layout, dt, initial_state


def test_uniform_case_initials_resolve_as_the_single_layout_authority():
    """Uniform and AMR layouts consume the same authenticated initial-condition contract."""
    example = _load_example()
    case, layout, _, _ = _periodic_conservation_target(
        example._native_output_mode(), declare_case_initial=True)
    resolved = pops.resolve(pops.validate(case), layout=layout)

    initial_plan = resolved.initial_condition_plan
    assert initial_plan is not None
    binding, = initial_plan.bindings
    assert binding.subject.block_ref.local_id == "tracer"
    assert binding.source.options.to_data()["native_route"] == "gaussian_field"
    assert resolved.bootstrap_plan is None


@pytest.mark.compiler
@pytest.mark.native_loader
def test_generic_analytic_initial_runs_through_the_uniform_native_pipeline(
    isolated_native_cache, native_cxx, kokkos_root,
):
    """A closed-form cell average crosses Python, codegen and the native Kokkos evaluator."""
    del isolated_native_cache, native_cxx, kokkos_root
    example = _load_example()
    case, layout, _, _ = _periodic_conservation_target(
        example._native_output_mode(), declare_analytic_initial=True)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    simulation = _bind_native_artifact(artifact)

    evidence = simulation.bound_snapshot.to_dict()["initial_evidence"]["resolved_plan"]
    assert evidence is not None
    binding, = evidence["bindings"].values()
    assert binding["source"]["options"]["native_route"] == "analytic_expression"
    assert binding["bound_value"] is None

    centers = (np.arange(16, dtype=np.float64) + 0.5) / 16.0
    center_x, center_y = np.meshgrid(centers, centers, indexing="xy")
    dx = 1.0 / 16.0
    expected = (
        0.25
        + center_x * center_x + dx * dx / 12.0
        + 2.0 * center_y
        + center_x * center_y
    )
    actual = np.asarray(simulation.get_state("tracer"), dtype=np.float64)
    assert actual.shape == (1, 16, 16)
    np.testing.assert_allclose(
        np.sort(actual[0], axis=None), np.sort(expected, axis=None), rtol=0.0, atol=2e-15
    )


@pytest.mark.compiler
@pytest.mark.native_loader
def test_public_csg_level_set_binds_one_native_uniform_embedded_boundary(
    isolated_native_cache, native_cxx, kokkos_root, monkeypatch,
):
    """The Python CSG lowers once; native Kokkos owns every mesh-point evaluation."""
    del isolated_native_cache, native_cxx, kokkos_root
    example = _load_example()
    case, layout, _, initial_state = _periodic_conservation_target(
        example._native_output_mode(),
        declare_analytic_embedded_boundary=True,
    )
    resolved = pops.resolve(pops.validate(case), layout=layout)

    normalized, = resolved.layout_plan.layouts
    embedded = normalized.to_data()["options"]["embedded_boundary"]
    root = embedded["level_set"]["expression"]["root"]
    assert root["op"] == "maximum"
    assert embedded["transport"]["mode"] == "staircase"

    artifact = pops.compile(resolved)
    from pops.runtime import _analytic_expression_lowering as expression_lowering

    lowering_calls = 0
    native_lowering = expression_lowering.lower_analytic_components

    def count_lowering(*args, **kwargs):
        nonlocal lowering_calls
        lowering_calls += 1
        return native_lowering(*args, **kwargs)

    monkeypatch.setattr(
        expression_lowering,
        "lower_analytic_components",
        count_lowering,
    )
    simulation = _bind_native_artifact(
        artifact,
        initial_state={"tracer": initial_state},
    )

    assert lowering_calls == 1
    mask = np.asarray(
        simulation._executor.embedded_boundary_mask(), dtype=np.float64,
    ).reshape(16, 16)
    centers = (np.arange(16, dtype=np.float64) + 0.5) / 16.0
    center_x, center_y = np.meshgrid(centers, centers, indexing="xy")
    distance = np.hypot(center_x - 0.5, center_y - 0.5)
    expected = ((distance < 0.4) & (distance > 0.2)).astype(np.float64)
    np.testing.assert_array_equal(mask, expected)
    assert 0 < int(mask.sum()) < mask.size


@pytest.mark.compiler
@pytest.mark.native_loader
def test_conservation_check_tracks_nonzero_baseline_across_evolving_periodic_steps(
    tmp_path, isolated_native_cache, native_cxx, kokkos_root,
):
    """A closed native finite-volume run proves baseline and later drift, not one zero sample."""
    del isolated_native_cache, native_cxx, kokkos_root
    example = _load_example()
    case, layout, dt, initial_state = _periodic_conservation_target(
        example._native_output_mode())
    resolved = pops.resolve(pops.validate(case), layout=layout)
    artifact = pops.compile(resolved)
    simulation = _bind_native_artifact(
        artifact, initial_state={"tracer": initial_state})
    initial = np.asarray(simulation.get_state("tracer"), dtype=np.float64).copy()

    first_report = pops.run(
        simulation,
        t_end=2.0 * dt,
        max_steps=2,
        output_dir=tmp_path,
    )
    assert first_report.accepted_steps == 2
    accepted_registry = simulation.inspect().to_dict()["instance"]["accepted_diagnostics"]
    accepted_diagnostics = {
        key: value for key, value in simulation.inspect().diagnostics.items()
        if key not in {"fallbacks", "solver_events"}
    }
    checkpoint = Path(simulation.checkpoint(tmp_path / "periodic-baseline"))
    with np.load(checkpoint, allow_pickle=False) as stored:
        baseline_state = json.loads(str(stored["runtime_consumer_diagnostics"]))
    assert baseline_state["schema_version"] == 2
    assert len(baseline_state["baselines"]) == 1
    assert len(baseline_state["diagnostics"]) == 1
    assert set(baseline_state["diagnostics"][0]["terms"]) == {
        "quantity", "baseline", "absolute_drift", "tolerance",
    }

    resumed = _bind_native_artifact(
        artifact, initial_state={"tracer": initial_state})
    resumed.restart(checkpoint)
    assert resumed.inspect().to_dict()["instance"]["accepted_diagnostics"] == accepted_registry
    resumed_diagnostics = {
        key: value for key, value in resumed.inspect().diagnostics.items()
        if key not in {"fallbacks", "solver_events"}
    }
    assert resumed_diagnostics == accepted_diagnostics
    second_report = pops.run(
        resumed,
        t_end=3.0 * dt,
        max_steps=1,
        output_dir=tmp_path,
    )
    assert second_report.accepted_steps == 1
    final = np.asarray(resumed.get_state("tracer"), dtype=np.float64)
    assert not np.array_equal(initial, final)

    paths = tuple(sorted(tmp_path.rglob("*.vtu")))
    assert len(paths) >= 2
    samples = []
    for path in paths:
        rows = _diagnostic_rows(read_paraview(path))
        assert set(rows) == {"conservation:integral"}
        row = rows["conservation:integral"]
        terms = {name: float.fromhex(value) for name, value in row["terms"].items()}
        assert set(terms) == {"quantity", "baseline", "absolute_drift", "tolerance"}
        assert terms["absolute_drift"] == abs(_scalar(row))
        assert terms["absolute_drift"] <= terms["tolerance"]
        samples.append(terms)
    assert samples[0]["baseline"] > 0.0
    assert all(row["baseline"] == samples[0]["baseline"] for row in samples[1:])
    assert len(samples[1:]) >= 1
