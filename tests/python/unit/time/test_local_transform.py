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
from pops.time import AccumulateDt, Hold, Program, Zero

from typed_program_support import state_refs

_ISSUES_REPRESENTATIONS = (
    "pops.codegen.program_emit_spatial_representation.program_issues_field_representations")


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
    input_view = re.search(
        r"(\w+A) = std::as_const\((\w+)\)\.fab\(li\)\.view\(\);", step_body)
    assert input_view is not None
    view_name, field_name = input_view.groups()
    assert view_name == field_name + "A"
    inactive_branch = step_body.split("if (transform_has_active_mask_", 1)[1].split("return;", 1)[0]
    assert "outA(index, 0) = %s(index, 0);" % view_name in inactive_branch
    assert "statusA(index, 0) = pops::Real(0);" in inactive_branch
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


@pytest.mark.parametrize("target,reduction", (
    ("system", "pointwise_status_max"), ("amr_system", "pointwise_level_status_max"),
))
def test_discharging_transform_reduces_its_status_once_on_each_path(
        target, reduction, monkeypatch) -> None:
    from tests.python.unit.codegen.test_scheduler_codegen import _block_end

    monkeypatch.setattr(_ISSUES_REPRESENTATIONS, lambda program: True)
    model, _, _, program = _transform_program()
    (transform,) = (value for value in program._values if value.op == "local_transform")
    source = emit_cpp_program(program, model=_canonical_emit_model(model), target=target)
    reductions = [match.start() for match in re.finditer(
        r"ctx\.%s\(0, transform_status_field_%d, " % (reduction, transform.id), source)]
    discharge = source.index("if (transform_%d_pending) {" % transform.id)
    discharge_end = _block_end(source, source.index("{", discharge))
    assert source.startswith("} else {", discharge_end)
    plain_end = _block_end(source, discharge_end + len("} else "))
    published = source.index("ctx.own_discharged_field_representation(", discharge)
    # The status reduction is collective. Each path converges once, the discharge before it
    # publishes its owner and the plain kernel before any consumer reads its scratch.
    assert len(reductions) == 2
    assert discharge < reductions[0] < published < discharge_end < reductions[1] < plain_end
    assert plain_end < source.index("auto& transform_observed_%d = " % transform.id)


def test_generated_operator_preflight_keeps_only_unqualified_cartesian_kernels() -> None:
    # The public emitter records source provenance around the actual dispatcher.
    assert "_emit_op_impl(" in inspect.getsource(program_emit_ops._emit_op)
    source = inspect.getsource(program_emit_ops._emit_op_impl)

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
            'json.dumps("named_flux")',
            "ctx.rhs_scratch",
        ),
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

    source_branch = branch("source", "apply")
    assert "ctx.require_cartesian_generated_operator" not in source_branch
    assert "_emit_source_kernel(" in source_branch
    rhs_branch = branch("rhs", "source")
    assert 'json.dumps("named_source")' not in rhs_branch
    assert "for source_subslot, s in enumerate(named," in rhs_branch
    assert "_emit_source_kernel(" in rhs_branch


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
def test_compiled_local_transform_body_keeps_guarded_finite_checks_in_scope(tmp_path, monkeypatch):
    import ctypes
    import re
    import subprocess
    from pops.codegen.toolchain import _default_cxx, _probe_cxx_std, loader_cxx_std
    model, _, _, program = _transform_program(boolean_domain=True)
    emit_model = _canonical_emit_model(model)
    first, last = "pops::Real transform_failed_ = pops::Real(0);", "statusA(index, 0) = transform_failed_;"
    # This Program issues no Field representation: it evaluates the plain kernel only.
    plain = emit_cpp_program(program, model=emit_model)
    assert len(re.findall(re.escape(first), plain)) == 1
    assert "CapturedTransformReads<" not in plain
    # Where a representation can reach the transform, the kernel discharging it and the
    # plain kernel each own a scalar body.
    monkeypatch.setattr(_ISSUES_REPRESENTATIONS, lambda program: True)
    generated = emit_cpp_program(program, model=emit_model)
    starts = [match.start() for match in re.finditer(re.escape(first), generated)]
    assert len(starts) == 2
    source = '#include <cmath>\nnamespace pops { using Real = double; }\nnamespace Kokkos { using std::isfinite; }\n'
    for kernel, start in enumerate(starts):
        body = generated[start:generated.index(last, start)] + last
        sources = set(re.findall(r"([A-Za-z0-9_]+A)\(index, 0\)", body)) - {"outA", "statusA"}
        # Read journals are runtime capture state; the scalar body only consults their verdict.
        captures = set(re.findall(r"\b(\w+)\.valid\(\)", body))
        source += 'extern "C" int evaluate_%d(double q, double* output) { int index=0; double status=0;\n' % kernel
        source += "".join("struct { bool valid() const { return true; } } %s;\n" % name
                          for name in sorted(captures))
        source += "\n".join("auto %s=[&](int,int){return q;};" % name for name in sorted(sources))
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
    for kernel in range(len(starts)):
        evaluate = getattr(native, "evaluate_%d" % kernel)
        evaluate.argtypes = (ctypes.c_double, ctypes.POINTER(ctypes.c_double))
        evaluate.restype = ctypes.c_int
        for value, expected_status in ((1., 0), (-1., 1), (5., 1)):
            output = ctypes.c_double(99)
            assert evaluate(value, ctypes.byref(output)) == expected_status
            assert output.value == value + 1


def _scheduled_transform_program(off):
    from pops.time import AcceptedStep, Every, Schedule

    model = Model("scheduled_transform_model", frame=Cartesian2D())
    state = model.state("U", components=("q",))
    transform = model.local_transform("held_shift", (state[0] + 1.0,), valid_if=state[0] > 0.0)
    model.module.operator_capabilities("held_shift", cacheable=True)
    program = Program("scheduled_transform_program")
    block, _ = state_refs(program, "fluid", model=model, state=state)
    temporal = program.state(block[state])
    held = transform(temporal.n, name="held_value",
                     schedule=Schedule(Every(AcceptedStep(program.clock), 2), off=off))
    program.commit(temporal.next, program.value(
        "candidate", temporal.n + program.dt * held, at=temporal.next.point))
    return model, program, held


@pytest.mark.compiler
@pytest.mark.parametrize("policy,issues", (
    (Hold, False), (AccumulateDt, False), (Zero, False), (Zero, True),
), ids=("Hold", "AccumulateDt", "Zero", "Zero-issuing"))
def test_scheduled_local_transform_program_compiles_with_its_off_cadence_value(
        policy, issues, tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    from tests.python.support.native_syntax_toolchain import configured_kokkos_toolchain

    if issues:
        monkeypatch.setattr(_ISSUES_REPRESENTATIONS, lambda program: True)
    model, program, held = _scheduled_transform_program(policy())
    generated = emit_cpp_program(program, model=_canonical_emit_model(model))
    guard = generated.index("if (ctx.schedule_decision(%d," % held.id)
    owner = "transform_closed_owner_%d" % held.id
    if issues:
        # Consumers bind after both cadence branches, so an off-cadence step reads the zeroed scratch.
        assert generated.index("std::shared_ptr<pops::MultiFab<pops::kNativeDimension>> %s;" % owner) < guard
        assert (generated.index("u%d.set_val(" % held.id, guard)
                < generated.index("auto& transform_observed_%d = " % held.id))
    else:
        # The schedule cache restores numbers only, and a Program without an issuer has no
        # Field representation to discharge: the due evaluation refuses one before evaluating.
        assert owner not in generated
        assert "transform_observed_%d" % held.id not in generated
        refusal = ("its Program issues none" if policy is Zero
                   else "schedule cache restores numerical values")
        assert (guard < generated.index(refusal)
                < generated.index("pops::Real transform_failed_ = pops::Real(0);", guard))
        if policy is Zero:
            # The consumer reads the scratch itself, which an off-cadence step zeroes.
            assert (generated.index("u%d.set_val(" % held.id, guard)
                    < generated.index("static_cast<pops::Real>(dt), u%d, dt" % held.id, guard))
    root = Path(__file__).resolve().parents[4]
    cpp = tmp_path / "program.cpp"
    cpp.write_text(generated)
    compiler, _, flags, _ = configured_kokkos_toolchain()
    compiled = subprocess.run(
        [compiler, "-std=c++20", "-DPOPS_NATIVE_DIM=2", "-DPOPS_HAS_KOKKOS=1",
         "-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI=1", "-I" + str(root / "include"), *flags,
         "-fsyntax-only", str(cpp)],
        capture_output=True, text=True, timeout=300)
    assert compiled.returncode == 0, compiled.stderr
