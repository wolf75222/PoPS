"""Shared source implementation, separate authenticated evaluations and owners."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import time

import pytest

import pops
from pops.codegen._orchestration_compile import build_program_model_graph
from pops.codegen.lowering_coverage import LoweringRejection
from pops.codegen.program_codegen import emit_cpp_program
from pops.layouts import Uniform
from pops.physics._facade import Model
from pops.time import FixedDt
from tests.python.support.layout_plan import cartesian_grid


def _resolved(*, foreign=False, changed=False, runtime=False, guarded=False, provider=False):
    def model(name):
        physical = Model(name)
        u, = physical.conservative_vars("u")
        physical.flux(x=[0 * u], y=[0 * u])
        coefficient = -2
        if runtime:
            from pops.params import RuntimeParam
            coefficient = physical.value(physical.param(RuntimeParam("rate", default=-2)))
        expression = 1 / u if guarded else coefficient * u
        if provider:
            expression = expression + physical.aux("forcing")
        source = physical.source_term("decay", [expression])
        alternate = physical.source_term("alternate", [0 * u if guarded else -3 * u])
        return physical, source, alternate

    first = model("shared_decay")
    second = model("foreign_decay") if foreign else first
    case = pops.Case("shared_source_implementations")
    program = pops.Program("shared_source_step")
    for index, (name, declarations) in enumerate((("left", first), ("right", second))):
        physical, source, alternate = declarations
        module = physical.module
        program._bind_operators(module)
        block = case.block(name, physical)
        state = program.state(block[module.state_handle(module.state_spaces()["U"])])
        if guarded:
            rhs = program.branch(
                program.norm2(state.n) > 0,
                lambda T, state=state, source=source: T.source(source, state=state.n),
                lambda T, state=state, alternate=alternate: T.source(alternate, state=state.n),
            )
        else:
            rhs = program.source(alternate if changed and index else source, state=state.n)
        program.commit(state.next, program.value(
            name + "_next", state.n + program.dt * rhs, at=state.next.point))
    program.step_strategy(FixedDt(0.01))
    case.program(program)
    return pops.resolve(pops.validate(case), layout=Uniform(cartesian_grid(n=8, periodic=True)))


def _source(resolved):
    return emit_cpp_program(resolved.time, model_graph=build_program_model_graph(resolved))


def _definitions(source):
    return re.findall(r"inline void (pops_shared_source_[0-9a-f]{64})\(", source)


def test_same_declared_model_emits_one_source_implementation_and_two_owned_calls():
    resolved = _resolved()
    source = _source(resolved)
    helper, = _definitions(source)
    assert source.count(helper + "(ctx,") == 2
    assert source.count("pops::Real(-2) * u") == 1
    assert source.count("ctx.require_cartesian_generated_operator(") == 2
    assert source.count("ctx.rhs_scratch(") == 2
    assert "provider_values_view<0>(consumer_qid, block_index, li)" in source
    left, right = resolved.blocks
    assert left.instance_owner_qid != right.instance_owner_qid
    assert left.resolved_operations.identity != right.resolved_operations.identity
    for block in resolved.blocks:
        owned_nodes = {value.id for value in resolved.time._values if value.op == "source"
                       and str(value.block.instance_owner_path.canonical()) == block.instance_owner_qid}
        selected = [operation for operation in block.resolved_operations.operations
                    if operation.guarantees.get("program_evaluation", {}).get("node_id") in owned_nodes]
        assert len(selected) == 1
        assert block.instance_owner_qid + "/program/" in source
        assert not selected[0].guards


@pytest.mark.parametrize("change", ("foreign", "changed"))
def test_changed_body_or_foreign_declaration_does_not_share_implementation(change):
    assert len(_definitions(_source(_resolved(**{change: True})))) == 2


def test_runtime_parameter_source_keeps_existing_per_block_binding_route():
    source = _source(_resolved(runtime=True))
    assert not _definitions(source)
    assert "ctx.program_params(0)" in source
    assert "ctx.program_params(1)" in source


def test_source_with_a_provider_keeps_its_exact_per_node_access_plan():
    source = _source(_resolved(provider=True))
    assert not _definitions(source)
    assert source.count("provider_values_view<1>(") == 2
    assert "providers(index, 0)" in source


def test_shared_fallible_implementation_is_called_only_inside_its_guard():
    resolved = _resolved(guarded=True)
    source = _source(resolved)
    assert len(_definitions(source)) == 2
    install = source[source.index('extern "C" void pops_install_program('):]
    first_if = install.index("if (")
    first_else = install.index("} else {", first_if)
    assert "pops_shared_source_" not in install[:first_if]
    assert "pops_shared_source_" in install[first_if:first_else]
    for block in resolved.blocks:
        selected = [operation for operation in block.resolved_operations.operations
                    if "program_evaluation" in operation.guarantees]
        assert selected and all(operation.guards for operation in selected)
        assert any("fallible" in operation.effects for operation in selected)


def test_reuse_does_not_bypass_each_instances_resolved_plan_authentication():
    resolved = _resolved()
    block = resolved.blocks[1]
    plan = block.resolved_operations
    selected = next(operation for operation in plan.operations
                    if "program_evaluation" in operation.guarantees)
    forged = replace(plan, operations=tuple(
        replace(operation, outputs=({"kind": "field", "name": "forged"},))
        if operation is selected else operation for operation in plan.operations))
    poisoned = replace(resolved, blocks=(resolved.blocks[0], replace(block, resolved_operations=forged)))
    with pytest.raises(LoweringRejection):
        _source(poisoned)


@pytest.mark.parametrize("boundary", ("program", "install", "block_compile"))
def test_shared_definition_cannot_transplant_one_blocks_resolved_plan_to_another(boundary, monkeypatch):
    resolved = _resolved()
    poisoned = replace(resolved, blocks=(resolved.blocks[0], replace(
        resolved.blocks[1], resolved_operations=resolved.blocks[0].resolved_operations)))
    from pops.codegen import _orchestration_compile as compiler

    original_compile = compiler.compile_install_model
    calls = []
    def unexpected_compile(*args, **kwargs):
        calls.append(args)
        raise AssertionError("owner preflight must precede any native compilation")
    monkeypatch.setattr(compiler, "compile_install_model", unexpected_compile)
    with pytest.raises(LoweringRejection) as caught:
        if boundary == "program":
            _source(poisoned)
        elif boundary == "install":
            compiler.compile_install_models(poisoned, {})
        else:
            block = poisoned.blocks[1]
            original_compile(block.name, block.model, block.backend, poisoned.target, {},
                             state_spaces=block.state_spaces,
                             consumer_owner_qid=block.instance_owner_qid,
                             resolved_operations=block.resolved_operations)
    assert caught.value.gate == "resolved_operation_block_owner_mismatch"
    assert calls == []


@pytest.mark.parametrize("boundary,missing_plan", (
    ("program", False), ("program", True), ("install", False),
    ("install", True), ("block_compile", False),
))
def test_missing_owner_is_rejected_before_any_block_lowering_or_compilation(
        boundary, missing_plan, monkeypatch):
    resolved = _resolved()
    block = resolved.blocks[1]
    missing = replace(block, instance_owner_qid="", resolved_operations=(
        None if missing_plan else block.resolved_operations))
    # Normal construction already rejects this. Independently exercise the
    # execution boundary against a plan corrupted after construction.
    poisoned = replace(resolved)
    object.__setattr__(poisoned, "blocks", (resolved.blocks[0], missing))
    from pops.codegen import _orchestration_compile as compiler

    original_compile = compiler.compile_install_model
    calls = []

    def unexpected_work(*args, **kwargs):
        calls.append(args)
        raise AssertionError("all-block owner preflight must precede lowering and compilation")

    monkeypatch.setattr(compiler, "compile_install_model", unexpected_work)
    monkeypatch.setattr("pops.codegen.module_lowering.lower_and_validate", unexpected_work)
    with pytest.raises(LoweringRejection) as caught:
        if boundary == "program":
            _source(poisoned)
        elif boundary == "install":
            compiler.compile_install_models(poisoned, {})
        else:
            original_compile(missing.name, missing.model, missing.backend, poisoned.target, {},
                             state_spaces=missing.state_spaces, consumer_owner_qid="",
                             resolved_operations=missing.resolved_operations)
    assert caught.value.gate == "resolved_operation_block_owner_mismatch"
    assert calls == []


@pytest.mark.parametrize("owner", (None, "", " ", 0, False))
def test_required_block_authority_rejects_missing_and_invalid_owners(owner):
    from pops.codegen._resolved_operation_ownership import require_block_plan_owner

    with pytest.raises(LoweringRejection, match="non-empty Case-block instance owner"):
        require_block_plan_owner(None, owner, where="required block", required=True)


def test_unscoped_module_plan_and_explicit_legacy_none_remain_supported():
    from pops.codegen._resolved_operation_ownership import require_block_plan_owner
    from pops.codegen.resolved_operations import build_resolved_operations

    resolved = _resolved()
    module_plan = build_resolved_operations(resolved.blocks[0].model.module)
    assert all("block_instance" not in operation.guarantees for operation in module_plan.operations)
    require_block_plan_owner(module_plan, None, where="standalone Module")
    require_block_plan_owner(None, None, where="explicit legacy adapter")


@pytest.mark.compiler
@pytest.mark.integration
@pytest.mark.parametrize("guarded", (False, True))
def test_native_shared_source_helper_preserves_two_block_updates(record_property, guarded):
    import numpy as np
    from pops._native_selector import select_native_dimension
    from tests.python.support.native_execution_context import artifact_execution_context
    from tests.python.support.requirements import missing_compiler_requirement, require_native_or_skip

    select_native_dimension(2)
    include = Path(__file__).resolve().parents[4] / "include"
    missing = missing_compiler_requirement(include)
    if missing:
        require_native_or_skip(missing, optional_skip=pytest.skip)
    resolved = _resolved(guarded=guarded)
    implementation_count = 2 if guarded else 1
    assert len(_definitions(_source(resolved))) == implementation_count
    start = time.perf_counter()
    artifact = pops.compile(resolved)
    record_property("compile_elapsed_seconds", time.perf_counter() - start)
    record_property("program_binary_bytes", Path(artifact.program.so_path).stat().st_size)
    record_property("source_implementation_count", implementation_count)
    record_property("source_evaluations_per_step", 2)
    values = {"left": 0.0, "right": 2.0} if guarded else {"left": 1.0, "right": 3.0}
    initial = {name: np.full((1, 8, 8), value) for name, value in values.items()}
    simulation = pops.bind(artifact, initial_state=initial,
                           resources={"execution_context": artifact_execution_context(artifact)})
    report = pops.run(simulation, t_end=0.02, max_steps=2)
    assert report.accepted_steps == 2
    assert report.rejected_steps == 0
    for name, value in values.items():
        expected = value
        for _ in range(2):
            expected += 0.01 * (1 / expected if guarded and expected else
                                0 if guarded else -2 * expected)
        actual = np.asarray(simulation.state_global(name))
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-14)
        if guarded and name == "left":
            np.testing.assert_array_equal(actual, initial[name])
