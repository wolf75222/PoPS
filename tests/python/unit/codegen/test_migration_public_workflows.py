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


def test_native_physical_flux_captures_authenticated_provider_and_real_call(tmp_path):
    component = imported_native_primitive.prepare_arithmetic(tmp_path / "native")
    case, layout, _, _ = imported_native_primitive.build_case(8, component)
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
def test_native_advection_matches_python_law_on_two_distinct_profiles(tmp_path):
    import numpy as np
    from scientific.runtime import _execution_resources

    component = imported_native_primitive.prepare_arithmetic(tmp_path / "native")
    outputs = []
    for imported in (True, False):
        case, layout, initial, dt = imported_native_primitive.build_case(
            16, component, imported=imported
        )
        artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
        runtime = pops.bind(
            artifact, initial_state=initial, resources=_execution_resources(artifact)
        )
        report = pops.run(runtime, t_end=3 * dt, max_steps=3, console=False)
        assert report.accepted_steps == 3
        values = {name: np.asarray(runtime.state_global(name)) for name in initial}
        for name, state in values.items():
            assert np.max(np.abs(state - initial[name])) > 1e-6
            assert abs(float(np.mean(state)) - float(np.mean(initial[name]))) < 2e-12
        outputs.append(values)
    for name in outputs[0]:
        np.testing.assert_array_equal(outputs[0][name], outputs[1][name])
