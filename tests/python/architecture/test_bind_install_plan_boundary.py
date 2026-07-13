"""Architecture gate: public bind consumes InstallPlan with no live-authoring fallback."""
from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

orchestration = pytest.importorskip("pops.codegen.orchestration")
plans = pytest.importorskip("pops.codegen._plans")
adapters = pytest.importorskip("pops.runtime._bind_adapters")
runtime_executor = pytest.importorskip("pops.runtime._runtime_executor")
runtime_instance = pytest.importorskip("pops.runtime.runtime_instance")
mesh_lowering = pytest.importorskip("pops.runtime._runtime_mesh_lowering")
bind_validation = pytest.importorskip("pops.runtime._bind_validation")
install_params = pytest.importorskip("pops.runtime._install_param_routing")
system_install = pytest.importorskip("pops.runtime._system_unified_install")
amr_install = pytest.importorskip("pops.runtime._amr_system_install")


_RETIRED_AUTHORING_ATTRS = {
    "_problem", "_blocks", "_block_specs", "_block_models", "_block_compiled_models",
    "_field_solvers", "_outputs", "_target", "_elliptic_fields", "dsl",
    "operator_registry", "declaration_index",
}
_RETIRED_COMPILED_ATTRS = _RETIRED_AUTHORING_ATTRS | {"_layout", "model"}
_RECONSTRUCTION_CALLS = {
    "build_problem_snapshot", "compile", "compile_install_model", "compile_model",
    "program_param_entries", "to_dsl",
}


def _tree(value):
    return ast.parse(textwrap.dedent(inspect.getsource(value)))


def _getattr_literal(node):
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        return None
    return node.args[0], node.args[1].value


def _assert_no_authoring_reconstruction(label, value):
    tree = _tree(value)
    bad_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in _RETIRED_AUTHORING_ATTRS
    }
    bad_getattrs = set()
    compiled_model_reads = []
    for node in ast.walk(tree):
        literal = _getattr_literal(node)
        if literal is None:
            continue
        owner, name = literal
        if name in _RETIRED_AUTHORING_ATTRS:
            bad_getattrs.add(name)
        if isinstance(owner, ast.Name) and owner.id == "compiled" \
                and name in _RETIRED_COMPILED_ATTRS:
            compiled_model_reads.append(name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "compiled" \
                and node.attr in _RETIRED_COMPILED_ATTRS:
            compiled_model_reads.append(node.attr)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not bad_attributes, "%s reads retired authoring attributes %s" % (
        label, sorted(bad_attributes))
    assert not bad_getattrs, "%s dynamically reads retired authoring attributes %s" % (
        label, sorted(bad_getattrs))
    assert not compiled_model_reads, "%s falls back from the artifact to compiled.%s" % (
        label, sorted(set(compiled_model_reads)))
    assert calls.isdisjoint(_RECONSTRUCTION_CALLS), \
        "%s reconstructs compile/authoring state through %s" % (
            label, sorted(calls & _RECONSTRUCTION_CALLS))


def test_bind_reads_the_install_plan_and_no_retired_authoring_mirror():
    source = textwrap.dedent(inspect.getsource(orchestration.bind))
    tree = ast.parse(source)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "InstallPlan" in calls
    assert "install" in calls
    retired = {
        "_problem", "_block_specs", "_block_models", "_block_compiled_models",
        "_layout", "_target", "_field_solvers", "_outputs",
    }
    assert attributes.isdisjoint(retired)
    assert strings.isdisjoint(retired)


def test_complete_bind_install_path_has_no_live_authoring_fallback():
    """Every public bind/install hop is plan/data-only, including helpers below bind()."""
    path = {
        "bind": orchestration.bind,
        "install": orchestration.install,
        "require_install_plan": plans.require_install_plan,
        "runtime install_plan": adapters.install_plan,
        "RuntimeInstance.__init__": runtime_instance.RuntimeInstance.__init__,
        "RuntimeInstance.run": runtime_instance.RuntimeInstance.run,
        "install_runtime_executor": runtime_executor.install_runtime_executor,
        "Uniform provider install": runtime_executor._UniformNativeProvider.install,
        "Adaptive provider install": runtime_executor._AdaptiveNativeProvider.install,
        "flow_amr_layout": mesh_lowering.flow_amr_layout,
        "_apply_refine_criterion": mesh_lowering._apply_refine_criterion,
        "_refine_threshold_value": mesh_lowering._refine_threshold_value,
        "_refine_subject_name": mesh_lowering._refine_subject_name,
        "run_bind_gates": bind_validation.run_bind_gates,
        "validate_install_arguments": bind_validation.validate_install_arguments,
        "System._install_compiled": system_install._SystemUnifiedInstall._install_compiled,
        "System._resolve_instance_model": (
            system_install._SystemUnifiedInstall._resolve_instance_model),
        "System._declared_elliptic_fields": (
            system_install._SystemUnifiedInstall._declared_elliptic_fields),
        "System._install_params": system_install._SystemUnifiedInstall._install_params,
        "System._install_program_params": (
            system_install._SystemUnifiedInstall._install_program_params),
        "AmrSystem._install_compiled": amr_install._AmrSystemInstall._install_compiled,
        "AmrSystem._declared_elliptic_fields": (
            amr_install._AmrSystemInstall._declared_elliptic_fields),
        "_require_schema": install_params._require_schema,
        "_slot_for_block": install_params._slot_for_block,
        "_resolved_value": install_params._resolved_value,
        "route_block_params": install_params.route_block_params,
        "route_program_params": install_params.route_program_params,
    }
    for label, value in path.items():
        _assert_no_authoring_reconstruction(label, value)


def test_program_param_routing_requires_captured_metadata_not_codegen_reentry():
    tree = _tree(install_params.route_program_params)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "program_param_entries" not in calls
    assert "program_param_routes" in {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
