"""Public workflow contracts, with native execution kept as a separate gate."""

from pathlib import Path
import sys

import pytest
import pops

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "examples" / "migration"))
from scientific import (
    euler_poisson,
    explicit_diffusion,
    field_transport,
    heterogeneous_interaction,
    implicit_diffusion,
    imported_native_primitive,
    scalar_amr,
    variable_coefficient_field,
)
from pops.codegen._orchestration_compile import build_program_model_graph
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_kernels import _prepared_native_components


@pytest.mark.parametrize(
    "workflow",
    (
        field_transport,
        euler_poisson,
        heterogeneous_interaction,
        explicit_diffusion,
        implicit_diffusion,
        variable_coefficient_field,
    ),
)
def test_scientific_workflow_public_lifecycle_resolves(workflow):
    case, layout, initial, dt = workflow.build_case(8)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    assert resolved.time is not None
    assert initial and dt > 0


def test_inline_and_imported_user_method_have_identical_temporal_graph():
    inline, layout, _ = scalar_amr.build_case(8, scheme="inline")
    imported, imported_layout, _ = scalar_amr.build_case(8, scheme="imported")
    first = pops.resolve(pops.validate(inline), layout=layout)
    second = pops.resolve(pops.validate(imported), layout=imported_layout)
    assert first.time._ir_hash() == second.time._ir_hash()


@pytest.mark.parametrize("intermediates", [False, True])
def test_native_physical_flux_captures_authenticated_provider_and_real_call(tmp_path, intermediates):
    component = imported_native_primitive.prepare_arithmetic(tmp_path / "native")
    case, layout, _, _ = imported_native_primitive.build_case(
        8, component, intermediates=intermediates)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    assert _prepared_native_components(resolved.time) == (component,)
    graph = build_program_model_graph(resolved)
    from pops.codegen.native_build import model_native_components

    assert model_native_components(graph.model_for_block("left")._m) == (component,)
    source = emit_cpp_program(resolved.time, model_graph=graph)
    assert "arithmetic.hpp" in source
    brick = graph.model_for_block("left")._m.emit_cpp_brick()
    assert brick.count("migration_arithmetic::multiply(") == 2
    assert "flux_evaluation" in brick
    assert brick.index("return {F, native_status_, native_reason_}") < brick.index("F[0] =")
    pure, pure_layout, _, _ = imported_native_primitive.build_case(8, component, imported=False)
    reference = pops.resolve(pops.validate(pure), layout=pure_layout)
    assert _prepared_native_components(reference.time) == ()
    assert resolved.time._ir_hash() != reference.time._ir_hash()


def test_native_flux_arithmetic_remains_joint_with_cse_disabled(tmp_path):
    from pops.codegen.module_emit_helpers import _codegen_exprs
    from pops.model import FieldSpace, Signature
    from pops.native_calls import NativeFunction

    component = imported_native_primitive.prepare_arithmetic(tmp_path / "native")
    scalar = FieldSpace("scalar", components=("value",))
    multiply = NativeFunction(
        component, "migration_arithmetic::multiply", Signature((scalar, scalar), scalar)
    )
    value = multiply((2,), (3,)).value[0]
    lines, outputs, statuses = _codegen_exprs(
        None, (value, 2 * value), False, return_native_statuses=True
    )
    assert "\n".join(lines).count("migration_arithmetic::multiply(") == 1
    assert len(outputs) == 2 and len(statuses) == 1


def test_failed_native_dependencies_guard_shared_arithmetic_and_dependent_calls(tmp_path):
    from pops.codegen.cpp_writer import _cse_emit
    from pops.math import Var
    from pops.model import FieldSpace, Signature
    from pops.native_calls import NativeFunction

    component = imported_native_primitive.prepare_arithmetic(tmp_path / "native")
    scalar = FieldSpace("scalar", components=("value",))
    multiply = NativeFunction(component, "migration_arithmetic::multiply",
                              Signature((scalar, scalar), scalar))
    first = multiply((Var("u", "cons"),), (2,)).value[0]
    shared = first * first
    second = multiply((shared,), (3,)).value[0]
    lines, _, statuses = _cse_emit(
        (shared, shared + second), "pops::Real", "  ", return_native_statuses=True)
    source = "\n".join(lines)
    assert len(statuses) == 2
    # Reusing a failed projection cannot execute its arithmetic or external consumer.
    assert "%s.status == pops::EvaluationStatus::kOk) ?" % statuses[0] in source
    dependent_guard = "if (%s.status == pops::EvaluationStatus::kOk) {" % statuses[0]
    assert dependent_guard in source
    assert source.index(dependent_guard) < source.rindex("migration_arithmetic::multiply(")
    assert "%s.status = pops::EvaluationStatus::kOk;" % statuses[1] in source


def test_guarded_native_dependency_remains_inside_its_selected_branch(tmp_path):
    from pops.codegen.cpp_writer import _cse_emit
    from pops.math import Var
    from pops.model import FieldSpace, Signature
    from pops.native_calls import NativeFunction

    component = imported_native_primitive.prepare_arithmetic(tmp_path / "native")
    scalar = FieldSpace("scalar", components=("value",))
    multiply = NativeFunction(component, "migration_arithmetic::multiply",
                              Signature((scalar, scalar), scalar))
    u = Var("u", "cons")
    first = multiply((u,), (2,)).value[0]
    second = multiply((first * first,), (3,)).value[0]
    predicate = (u > 0) & (second > 1)
    lines, _, statuses = _cse_emit((predicate,), "pops::Real", "  ",
                                  return_native_statuses=True)
    source = "\n".join(lines)
    assert len(statuses) == 2
    branch = source.index("&& ([&]()")
    assert source.index("migration_arithmetic::multiply(") > branch
    assert all(source.index(name + ".status = pops::EvaluationStatus::kOk;") < branch
               for name in statuses)
    assert "if (%s.status == pops::EvaluationStatus::kOk) {" % statuses[0] in source


@pytest.mark.parametrize("difference", ["argument", "target", "contract", "header"])
def test_native_model_hash_authenticates_formula_and_source(tmp_path, difference):
    from dataclasses import replace
    from pops.codegen._compile_emit import model_hash
    from pops.codegen.module_lowering import _module_to_model
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.math import Const
    from pops.model import FieldSpace, Signature
    from pops.native_calls import NativeFunction
    from pops.native_components import PreparedNativeComponent

    component = imported_native_primitive.prepare_arithmetic(tmp_path / "native")
    alternate = component
    if difference == "header":
        root = tmp_path / "alternate"
        root.mkdir()
        (root / "arithmetic.hpp").write_text(
            (tmp_path / "native" / "arithmetic.hpp").read_text() + "\n// changed source authority\n")
        alternate = PreparedNativeComponent.header_only(
            "migration.arithmetic", include_root=root, entry_headers=("arithmetic.hpp",))
    hashes = []
    for changed in (False, True):
        frame = Rectangle("square", lower=(0, 0), upper=(1, 1)).frame(Cartesian2D())
        model = pops.Model("same_model", frame=frame)
        state = model.state("U", components=("u",))
        scalar = FieldSpace("scalar", components=("value",))
        function = NativeFunction(alternate if changed else component,
                                  "migration_arithmetic::multiply",
                                  Signature((state.space, scalar), scalar))
        if changed and difference == "target":
            function = replace(function, target="migration_arithmetic::other")
        if changed and difference == "contract":
            function = replace(function, execution_domains=("host", "device"))
        value = function(tuple(state), (3 if changed and difference == "argument" else 2,)).value[0]
        # Exercise the historically repr-only primitive route as well as the flux inventory.
        recipe = model.primitive("recipe", value)
        primitive = model.primitive("constitutive", recipe)
        model.flux("advection", state=state, frame=frame,
                   components={axis: (primitive,) for axis in frame.axes},
                   waves={axis: (1,) for axis in frame.axes})
        module = model.module
        module.eigenvalues(**{axis.name: (Const(1),) for axis in frame.axes})
        lowered = _module_to_model(module)._m
        assert {"recipe", "constitutive"} <= set(lowered.prim_defs)
        assert "flux_evaluation" in lowered.emit_cpp_brick()
        hashes.append(model_hash(lowered))
    assert hashes[0] != hashes[1]


def test_primitive_expansion_preserves_sharing_and_rejects_cycles():
    from pops._ir.primitive_expansion import expand_primitive_recipes
    from pops.math import Var

    a, b, u = Var("a", "prim"), Var("b", "prim"), Var("u", "cons")
    expanded = expand_primitive_recipes((a, a), {"a": b, "b": u * u})
    assert expanded[0] is expanded[1]
    assert expanded[0].deps() == {"u"}
    with pytest.raises(ValueError, match="primitive recipe cycle: a -> b -> a"):
        expand_primitive_recipes((a,), {"a": b, "b": a})


def test_public_module_composed_rate_captures_native_primitive_recipe(tmp_path):
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.layouts import Uniform
    from pops.math import Const
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.model import FieldSpace, Signature
    from pops.native_calls import NativeFunction
    from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
    from pops.time import FixedDt

    component = imported_native_primitive.prepare_arithmetic(tmp_path / "native")
    frame = Rectangle("domain", lower=(0, 0), upper=(1, 1)).frame(Cartesian2D())
    model = pops.Model("module_physics", frame=frame)
    state = model.state("U", components=("u",))
    scalar = FieldSpace("scalar", components=("value",))
    function = NativeFunction(component, "migration_arithmetic::multiply",
                              Signature((state.space, scalar), scalar))
    primitive = model.primitive("transported", function(tuple(state), (1,)).value[0])
    flux = model.flux("advection", state=state, frame=frame,
                      components={axis: (primitive,) for axis in frame.axes},
                      waves={axis: (1,) for axis in frame.axes})
    module = model.module
    module.eigenvalues(**{axis.name: (Const(1),) for axis in frame.axes})
    state = module.state_handle(module.state_spaces()["U"])
    flux = module.operator_handle(flux.reg_name)
    rate = module.rate_operator("transport", state, fluxes=(flux,), default_flux=flux)
    case = pops.Case("module_native")
    block = case.block("fluid", module)
    plan = DiscretizationPlan()
    plan.rates.add(rate, FiniteVolume(
        flux=(flux,), variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case.numerics(plan, block=block)
    program = pops.Program("advance")
    q = program.state(block[state])
    updated = program.value("updated", q.n + program.dt * rate(q.n), at=q.next.point)
    program.commit(q.next, updated)
    program.step_strategy(FixedDt(.001))
    case.program(program)
    resolved = pops.resolve(pops.validate(case), layout=Uniform(CartesianGrid(
        frame=frame, cells=(8, 8), periodic=PeriodicAxes(frame.axes))))
    assert _prepared_native_components(resolved.time) == (component,)
    graph = build_program_model_graph(resolved)
    assert "flux_evaluation" in graph.model_for_block("fluid")._m.emit_cpp_brick()


def test_native_primitive_missing_exact_derivative_is_refused(tmp_path):
    from pops._ir.expr import Var
    from pops._ir.lowering import diff
    from pops.model import FieldSpace, Signature
    from pops.native_calls import NativeFunction

    component = imported_native_primitive.prepare_arithmetic(tmp_path / "native")
    scalar = FieldSpace("scalar", components=("value",))
    function = NativeFunction(
        component, "migration_arithmetic::multiply", Signature((scalar, scalar), scalar)
    )
    x = Var("x", "cons")
    with pytest.raises(ValueError, match="no declared exact derivative"):
        diff(function((x,), (3,)).value[0], x)


def test_model_and_program_native_staging_use_verified_copies(tmp_path):
    from pops.codegen.native_build import stage_native_components

    component = imported_native_primitive.prepare_arithmetic(tmp_path / "native")
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    flags, _, authorities = stage_native_components((component,), tmp_path / "staged", sdk)
    assert flags == ["-I", authorities[0][1]]
    assert Path(authorities[0][1]).is_relative_to(tmp_path / "staged")
    header = tmp_path / "native" / "arithmetic.hpp"
    header.write_bytes(header.read_bytes() + b"\n// changed source\n")
    with pytest.raises(ValueError, match="changed after registration"):
        stage_native_components((component,), tmp_path / "changed", sdk)


@pytest.mark.compiler
@pytest.mark.native_loader
def test_native_advection_matches_python_law_on_two_distinct_profiles(tmp_path, record_property):
    import hashlib
    import json

    import numpy as np
    from scientific.runtime import _execution_resources

    def file_evidence(path):
        return {"path": str(path), "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    component = imported_native_primitive.prepare_arithmetic(tmp_path / "native")
    outputs, routes = [], []
    for imported in (True, False):
        case, layout, initial, dt = imported_native_primitive.build_case(
            16, component, imported=imported
        )
        initial_image = {name: np.array(value, copy=True) for name, value in initial.items()}
        artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
        runtime = pops.bind(
            artifact, initial_state=initial, resources=_execution_resources(artifact)
        )
        report = pops.run(runtime, t_end=3 * dt, max_steps=3, console=False)
        assert report.accepted_steps == 3
        # The next bind may release/reuse native storage: evidence and comparison own their arrays.
        values = {name: np.array(runtime.state_global(name), copy=True) for name in initial}
        for name, state in values.items():
            assert np.max(np.abs(state - initial[name])) > 1e-6
            assert abs(float(np.mean(state)) - float(np.mean(initial[name]))) < 2e-12
        outputs.append(values)
        arrays = {**{"initial_" + name: value for name, value in initial_image.items()},
                  **{"final_" + name: value for name, value in values.items()}}
        array_path = tmp_path / ("imported-arrays.npz" if imported else "python-arrays.npz")
        np.savez(array_path, **arrays)
        lower, upper = layout.mesh.extent
        cell_volume = float(np.prod((np.asarray(upper) - np.asarray(lower)) / layout.mesh.cells))
        routes.append({
            "imported": imported,
            "artifact_identity": artifact.artifact_identity.token,
            "plan_identity": artifact.plan.plan_identity.token,
            "platform": artifact.platform_manifest.to_data(),
            "model_binaries": {block.name: file_evidence(Path(block.model.so_path))
                               for block in artifact.blocks},
            "configuration": {"cells": list(layout.mesh.cells), "extent": [list(lower), list(upper)],
                              "periodic_axes": [axis.name for axis in layout.mesh.periodic.axes],
                              "dt": dt, "requested_steps": 3, "accepted_steps": report.accepted_steps,
                              "final_time": float(runtime.time()), "cell_volume": cell_volume,
                              "intermediates": False},
            "arrays": file_evidence(array_path),
            "array_members": {name: {"shape": list(value.shape), "dtype": value.dtype.str}
                              for name, value in arrays.items()},
            "block_checks": {
                name: {"initial_mean": float(np.mean(initial_image[name])),
                       "final_mean": float(np.mean(state)),
                       "mean_weight_per_cell": 1.0 / state.size,
                       "maximum_change": float(np.max(np.abs(state - initial_image[name])))}
                for name, state in values.items()},
        })
    for name in outputs[0]:
        np.testing.assert_array_equal(outputs[0][name], outputs[1][name])
    workflow_source = Path(imported_native_primitive.__file__).resolve()
    evidence = {
        "schema": "pops.imported-primitive-conformance.v1",
        "scope": "Existing 16-cell, three-step native/Python conformance; not full R1 science or legacy interface equivalence.",
        "source_files": [file_evidence(path) for path in
                         (Path(__file__).resolve(), workflow_source,
                          workflow_source.with_name("arithmetic.hpp"),
                          workflow_source.with_name("runtime.py"))],
        "prepared_component": {"manifest": component.manifest(),
                               "manifest_sha256": component.manifest_sha256},
        "routes": routes,
        "checks": {"cross_route_full_state": "array_equal",
                   "mean_conservation_absolute_bound": 2e-12,
                   "nonzero_change_lower_bound": 1e-6, "passed": True},
    }
    manifest = tmp_path / "imported-primitive-conformance.json"
    manifest.write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n")
    record_property("imported_primitive_conformance", json.dumps(file_evidence(manifest), sort_keys=True))
