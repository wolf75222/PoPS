"""The final Production lifecycle has deterministic cache hits and semantic misses.

The test compiles through ``validate -> resolve -> pops.compile`` only.  A second compile of the
same resolved plan must reuse both native packages without invoking the compiler.  Changing one
typed ``ConstParam`` is a semantic change: it produces a different plan/artifact and invokes the
compiler for a new content-addressed package.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pops
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import ConstParam
from pops.physics import Model
from pops.time import FixedDt


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
ROOT = Path(__file__).resolve().parents[4]
# Four deliberate native package compilations on a cold cache can exceed the process-isolation
# default on slower CI runners.  The workflow timeout remains the outer hard bound.
POPS_PROCESS_TIMEOUT = 900


def _resolved_plan(*, speed: float):
    frame = Rectangle(
        "production-cache-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model("production-cache-model", frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    speed_value = model.value(model.param(ConstParam("speed", speed)))
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (speed_value * rho,), y_axis: (0.0 * rho,)},
        waves={x_axis: (speed_value + 0.0 * rho,), y_axis: (0.0 * rho,)},
    )
    rate = model.rate("explicit_rhs", equation=ddt(state) == -div(flux))

    case = pops.Case("production-cache-case")
    block = case.block("tracer", model)
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
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(1.0e-3))
    case.program(program)
    layout = Uniform(
        CartesianGrid(
            frame=frame,
            cells=(8, 8),
            periodic=PeriodicAxes(frame.axes),
        )
    )
    return pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include")},
    )


_IDENTITY_DOMAINS = {
    "semantic_identity": "semantic",
    "artifact_spec_identity": "artifact-spec",
    "binary_identity": "binary",
    "artifact_identity": "artifact",
}


def _component_identities(component) -> dict[str, object]:
    identities = {name: getattr(component, name) for name in _IDENTITY_DOMAINS}
    assert {
        name: identity.domain for name, identity in identities.items()
    } == _IDENTITY_DOMAINS
    assert all(identity.token for identity in identities.values())
    return identities


def _native_evidence(artifact) -> dict[str, dict[str, tuple]]:
    models = {
        block.name: (Path(block.model.so_path), _component_identities(block.model))
        for block in artifact.blocks
    }
    programs = {
        row.layout_id: (
            Path(row.program.so_path),
            _component_identities(row.program),
            row.identity,
        )
        for row in artifact.layout_programs
    }
    assert models and programs
    assert all(path.is_file() for path, _identities in models.values())
    assert all(path.is_file() for path, _identities, _layout_identity in programs.values())
    assert all(row[2].domain == "layout-program" for row in programs.values())
    paths = [row[0] for group in (models, programs) for row in group.values()]
    assert len(paths) == len(set(paths))
    return {"models": models, "programs": programs}


def _native_paths(evidence: dict[str, dict[str, tuple]]) -> tuple[Path, ...]:
    return tuple(
        row[0]
        for kind in ("models", "programs")
        for row in evidence[kind].values()
    )


def _nonuniform_initial_state() -> np.ndarray:
    coordinates = (np.arange(8, dtype=np.float64) + 0.5) / 8.0
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
    field = 1.0 + 0.25 * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    return np.ascontiguousarray(field[np.newaxis, :, :])


def _run_one_step(artifact, initial: np.ndarray) -> np.ndarray:
    runtime = pops.bind(artifact, initial_state={"tracer": initial.copy()})
    report = pops.run(runtime, t_end=1.0e-3, max_steps=1)
    assert report.accepted_steps == 1
    result = np.asarray(runtime.get_state("tracer"), dtype=np.float64).reshape(initial.shape)
    assert np.isfinite(result).all()
    return result.copy()


def test_production_cache_hit_skips_compiler_and_semantic_change_misses(
    isolated_native_cache, native_cxx, kokkos_root, monkeypatch
):
    del native_cxx, kokkos_root  # fixtures make missing native prerequisites explicit skips
    baseline_plan = _resolved_plan(speed=1.0)
    baseline = pops.compile(baseline_plan)
    baseline.verify()
    baseline_evidence = _native_evidence(baseline)
    baseline_paths = _native_paths(baseline_evidence)
    cache_root = Path(isolated_native_cache).resolve()
    assert all(path.resolve().is_relative_to(cache_root) for path in baseline_paths)
    baseline_stats = {
        path: (path.stat().st_mtime_ns, path.stat().st_size) for path in baseline_paths
    }

    import pops.codegen._compile_drivers as compile_drivers

    real_compile = compile_drivers._run_compile
    compile_calls = []

    def observed_compile(command, context):
        compile_calls.append((tuple(command), context))
        return real_compile(command, context)

    monkeypatch.setattr(compile_drivers, "_run_compile", observed_compile)

    hit = pops.compile(baseline_plan)
    hit.verify()
    assert compile_calls == []
    assert hit.semantic_identity == baseline.semantic_identity
    assert hit.artifact_identity == baseline.artifact_identity
    assert _native_evidence(hit) == baseline_evidence
    initial = _nonuniform_initial_state()
    speed_one = _run_one_step(hit, initial)
    assert compile_calls == []
    assert {
        path: (path.stat().st_mtime_ns, path.stat().st_size) for path in baseline_paths
    } == baseline_stats

    changed_plan = _resolved_plan(speed=2.0)
    assert changed_plan.plan_identity != baseline_plan.plan_identity
    changed = pops.compile(changed_plan)
    changed.verify()
    contexts = [str(context) for _command, context in compile_calls]
    assert len(compile_calls) == 2
    assert len([context for context in contexts if "compile_native" in context]) == 1
    assert len([context for context in contexts if "compile_problem" in context]) == 1
    assert changed.semantic_identity != baseline.semantic_identity
    assert changed.artifact_identity != baseline.artifact_identity
    changed_evidence = _native_evidence(changed)
    for kind in ("models", "programs"):
        assert changed_evidence[kind].keys() == baseline_evidence[kind].keys()
        for name, baseline_row in baseline_evidence[kind].items():
            changed_row = changed_evidence[kind][name]
            assert changed_row[0] != baseline_row[0]
            for identity_name in (
                "semantic_identity",
                "artifact_spec_identity",
                "artifact_identity",
            ):
                assert changed_row[1][identity_name] != baseline_row[1][identity_name]
            if kind == "programs":
                assert changed_row[2] != baseline_row[2]

    calls_after_miss = tuple(compile_calls)
    speed_two = _run_one_step(changed, initial)
    assert tuple(compile_calls) == calls_after_miss
    numerical_scale = max(float(np.linalg.norm(initial)), 1.0)
    roundoff = np.finfo(np.float64).eps * numerical_scale
    assert np.linalg.norm(speed_one - initial) > roundoff
    assert np.linalg.norm(speed_two - initial) > roundoff
    assert np.linalg.norm(speed_two - speed_one) > roundoff
