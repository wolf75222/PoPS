"""Generic explicit local-transform authoring and Program code generation."""

from __future__ import annotations

import inspect
import re

import numpy as np
import pytest

from pops.codegen import program_emit_ops
from pops.codegen._resolution import CapabilityResolutionError, _resolve_amr_program
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.frames import Cartesian2D
from pops.physics import Model
from pops.runtime.amr_program_support import AMRProgramSupportContext
from pops.time import Program

from typed_program_support import state_refs


def _canonical_emit_model(model: Model) -> Model:
    """Resolve the fixture through its canonical Module/provider-pack authority."""
    emit_model, source_module = lower_and_validate(model, facade=model)
    assert source_module is model.module
    assert type(emit_model._auxiliary_provider_pack).__name__ == "ProviderPack"
    return emit_model


def _transform_program(*, transform_block_is_second: bool = False, boolean_domain: bool = False):
    model = Model("local_transform_model", frame=Cartesian2D())
    state = model.state("U", components=("q",))
    cached = model.module
    transform = model.local_transform(
        "bounded_shift",
        (state[0] + 1.0,),
        valid_if=((state[0] > 0.0) & (state[0] < 4.0)) if boolean_domain else state[0] > 0.0,
    )
    assert model.module is not cached

    program = Program("local_transform_program")
    if transform_block_is_second:
        first_block, _ = state_refs(program, "first", model=model, state=state)
        first_temporal = program.state(first_block[state])
        first_candidate = program.value(
            "first_candidate", first_temporal.n, at=first_temporal.next.point
        )
        program.commit(first_temporal.next, first_candidate)
    block, _ = state_refs(program, "fluid", model=model, state=state)
    temporal = program.state(block[state])
    candidate = program.value("candidate", temporal.n, at=temporal.next.point)
    transformed = program.transform(candidate, transform=transform, name="transformed_candidate")
    program.commit(temporal.next, transformed)
    return model, state, transform, program


def test_local_transform_is_typed_explicit_and_fresh() -> None:
    model, state, transform, program = _transform_program()
    assert transform.kind == "local_transform"
    values = [value for value in program._values if value.op == "local_transform"]
    assert len(values) == 1
    assert values[0].inputs[0].id != values[0].id
    assert values[0].attrs["transform"] == "bounded_shift"
    assert not any(value.op == "project" for value in program._values)

    assert np.array_equal(
        model.local_transform_value("bounded_shift", np.array([[[2.0, 3.0]]])),
        np.array([[[3.0, 4.0]]]),
    )
    lowered = model.module.to_dsl()
    assert np.array_equal(
        lowered.local_transform_value("bounded_shift", np.array([[[2.0, 3.0]]])),
        np.array([[[3.0, 4.0]]]),
    )
    with pytest.raises(ValueError, match="outside its domain"):
        model.local_transform_value("bounded_shift", np.array([[[-1.0]]]))
    with pytest.raises(FloatingPointError, match="non-finite state"):
        model.local_transform_value("bounded_shift", np.array([[[np.nan]]]))


def test_local_transform_program_emits_one_collective_fail_closed_kernel() -> None:
    model, _, _, program = _transform_program()
    emit_model = _canonical_emit_model(model)
    source = emit_cpp_program(program, model=emit_model)
    assert source.count("transform_failed_") >= 1
    assert "ctx.scratch_state_like(" in source
    install_prelude, step_body = source.split("ctx.install([=](double dt)", 1)
    assert "transform_state_resource_" in install_prelude
    assert "transform_status_resource_" in install_prelude
    assert "ctx.scalar_scratch(" in install_prelude
    assert ", 0, ctx.state(0), 1, 0)" in install_prelude
    assert "ctx.alloc_scalar_field(" not in install_prelude
    assert "ctx.scratch_state_like(" not in step_body
    assert "ctx.alloc_scalar_field(1, 0)" not in step_body
    assert 'require_cartesian_generated_operator(0, "local_transform")' not in source
    assert "ctx.pointwise_active_mask(0," in source
    assert "transform_has_active_mask_" in source
    assert "outA(index, 0) = u0A(index, 0);" in source
    assert "ctx.pointwise_status_max(0," in source
    assert "StepAttemptRejected" in source
    assert "Kokkos::isfinite" in source
    assert "ctx.apply_projection" not in source

    second_block_model, _, _, second_block_program = _transform_program(transform_block_is_second=True)
    second_block_source = emit_cpp_program(
        second_block_program, model=_canonical_emit_model(second_block_model)
    )
    assert ", 0, ctx.state(1), 1, 0)" in second_block_source
    assert "ctx.alloc_scalar_field(1, 0)" not in second_block_source

    flat_context = AMRProgramSupportContext(
        hierarchy_level_count=1,
        frozen_hierarchy=True,
        shared_block_interfaces=False,
        field_routes_validated=True,
    )
    assert _resolve_amr_program("amr", program, context=flat_context)["status"] == "proven"
    refined_context = AMRProgramSupportContext(
        hierarchy_level_count=2,
        frozen_hierarchy=True,
        shared_block_interfaces=False,
        field_routes_validated=True,
    )
    with pytest.raises(CapabilityResolutionError, match="post-synchronization Program phase"):
        _resolve_amr_program("amr", program, context=refined_context)

    amr_source = emit_cpp_program(program, model=emit_model, target="amr_system")
    assert "ctx.pointwise_active_mask(0," in amr_source
    # AMR produces and validates one level at a time; sibling scratch statuses can belong to
    # a prior invocation. The current level still requires the exact collective lane reduction.
    assert amr_source.count("ctx.pointwise_level_status_max(") == 1
    assert re.search(
        r"ctx\.pointwise_level_status_max\(0, transform_status_field_(\d+), "
        r"transform_active_mask_\1, ctx\.prepared_execution_lane\(\)\)",
        amr_source,
    )
    assert "ctx.pointwise_status_max(" not in amr_source
    assert "StepAttemptRejected" in amr_source
    assert "Kokkos::isfinite" in amr_source
    assert "inherit_state_metadata" not in amr_source
    amr_install = amr_source.split('extern "C" void pops_install_program_amr', 1)[1]
    resources, callbacks = amr_install.split("return _PopsAmrLevelProgram{", 1)
    assert "transform_status_resource_" not in resources
    assert "transform_status_resource_" in callbacks
    assert "ctx.prepare_spatial_collectively([&] {" in callbacks
    assert "= &ctx.scalar_scratch(" in callbacks


def test_generated_operator_preflight_keeps_only_unqualified_cartesian_kernels() -> None:
    source = inspect.getsource(program_emit_ops._emit_op)

    def branch(operation: str, next_operation: str) -> str:
        return source.split('elif v.op == "%s":' % operation, 1)[1].split(
            'elif v.op == "%s":' % next_operation, 1
        )[0]

    nonlinear = branch("solve_local_nonlinear", "scalar_field")
    assert 'require_cartesian_generated_operator(%d, %s);' not in nonlinear
    assert 'json.dumps("solve_local_nonlinear")' not in nonlinear

    unqualified = (
        ("coupled_rate", "coupled_rate_out", 'json.dumps("coupled_rate")', "scratch = {}"),
        (
            "solve_coupled_implicit",
            "history",
            'json.dumps("solve_coupled_implicit")',
            "scratch = {}",
        ),
        (
            "rhs",
            "source",
            ('operation = "named_flux"', 'else "named_source"'),
            "ctx.rhs_scratch",
        ),
        ("source", "apply", 'json.dumps("named_source")', "ctx.rhs_scratch"),
        ("apply", "solve_local_linear", 'json.dumps("linear_source_apply")', "ctx.rhs_scratch"),
        (
            "solve_local_linear",
            "solve_local_nonlinear",
            'json.dumps("solve_local_linear")',
            "ctx.scratch_state",
        ),
    )
    for operation, next_operation, identities, allocation in unqualified:
        emitted = branch(operation, next_operation)
        preflight = emitted.index("ctx.require_cartesian_generated_operator")
        if isinstance(identities, str):
            identities = (identities,)
        assert all(identity in emitted for identity in identities)
        assert preflight < emitted.index(allocation)


def test_local_transform_name_collisions_are_rejected() -> None:
    model = Model("local_transform_collision", frame=Cartesian2D())
    state = model.state("U", components=("q",))
    model.local_transform("repair", (state[0],))
    with pytest.raises(ValueError, match="local_transform"):
        model.source("repair", on=state, value=(state[0],))


def test_local_transform_formula_and_domain_are_part_of_module_identity() -> None:
    first = Model("transform_identity", frame=Cartesian2D())
    first_state = first.state("U", components=("q",))
    first.local_transform("repair", (first_state[0] + 1.0,), valid_if=first_state[0] > 0.0)

    changed_formula = Model("transform_identity", frame=Cartesian2D())
    formula_state = changed_formula.state("U", components=("q",))
    changed_formula.local_transform(
        "repair", (formula_state[0] + 2.0,), valid_if=formula_state[0] > 0.0
    )

    changed_domain = Model("transform_identity", frame=Cartesian2D())
    domain_state = changed_domain.state("U", components=("q",))
    changed_domain.local_transform(
        "repair", (domain_state[0] + 1.0,), valid_if=domain_state[0] > 1.0
    )

    identities = {
        first.module.module_hash(),
        changed_formula.module.module_hash(),
        changed_domain.module.module_hash(),
    }
    assert len(identities) == 3


def test_local_transform_observes_guarded_temporaries_after_their_declarations():
    import re
    model, _, _, program = _transform_program(boolean_domain=True)
    source = emit_cpp_program(program, model=_canonical_emit_model(model))
    checks = re.findall(r"if \(!Kokkos::isfinite\(((?:guard[0-9]+_)?cse[0-9]+_)\)\)", source)
    assert checks and any(name.startswith("guard") for name in checks)
    last_evaluation = source.index("&&")
    for name in checks:
        declaration = re.search(r"(?:const )?pops::Real " + name + r" =", source)
        assert declaration is not None
        check = source.index("if (!Kokkos::isfinite(%s))" % name)
        assert declaration.start() < check and last_evaluation < check


@pytest.mark.compiler
@pytest.mark.native_loader
def test_compiled_local_transform_body_keeps_guarded_finite_checks_in_scope(tmp_path):
    import ctypes
    import re
    import subprocess
    from pops.codegen.toolchain import _default_cxx, _probe_cxx_std, loader_cxx_std
    model, _, _, program = _transform_program(boolean_domain=True)
    generated = emit_cpp_program(program, model=_canonical_emit_model(model))
    start = generated.index("pops::Real transform_failed_")
    end = generated.index("statusA(index, 0) = transform_failed_;", start)
    body = generated[start:end] + "statusA(index, 0) = transform_failed_;"
    sources = set(re.findall(r"([A-Za-z0-9_]+A)\(index, 0\)", body)) - {"outA", "statusA"}
    source = '#include <cmath>\nnamespace pops { using Real = double; }\nnamespace Kokkos { using std::isfinite; }\n'
    source += 'extern "C" int evaluate(double q, double* output) { int index=0; double status=0;\n'
    source += "\n".join("auto %s=[&](int,int){return q;};" % name for name in sources)
    source += "\nauto outA=[&](int,int)->double&{return *output;}; auto statusA=[&](int,int)->double&{return status;};\n"
    source += body + "\nreturn static_cast<int>(status);\n}\n"
    path, binary = tmp_path / "transform.cpp", tmp_path / "transform.so"
    path.write_text(source)
    # This compiles the emitted scalar body only; it links no PoPS/Kokkos runtime ABI.
    compiler = _default_cxx()
    standard = _probe_cxx_std(compiler, loader_cxx_std())
    compiled = subprocess.run([compiler, "-std=" + standard, "-shared", "-fPIC", "-O2",
        str(path), "-o", str(binary)], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    native = ctypes.CDLL(str(binary))
    native.evaluate.argtypes = (ctypes.c_double, ctypes.POINTER(ctypes.c_double))
    native.evaluate.restype = ctypes.c_int
    for value, expected_status in ((1., 0), (-1., 1), (5., 1)):
        output = ctypes.c_double(99)
        assert native.evaluate(value, ctypes.byref(output)) == expected_status
        assert output.value == value + 1
