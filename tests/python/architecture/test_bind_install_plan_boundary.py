"""Source-only gate: public bind consumes InstallPlan with no live-authoring fallback."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
POPS = ROOT / "python" / "pops"


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


def _definition(relative, qualified_name):
    """Return one top-level function or class method without importing its module."""
    path = POPS / relative
    source = path.read_text(encoding="utf-8")
    body = ast.parse(source, filename=str(path)).body
    parts = qualified_name.split(".")
    for part in parts:
        node = next((item for item in body
                     if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                     and item.name == part), None)
        assert node is not None, "%s no longer defines %s" % (relative, qualified_name)
        body = node.body
    return node


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


def _assert_no_authoring_reconstruction(label, relative, qualified_name):
    tree = _definition(relative, qualified_name)
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
    tree = _definition("codegen/_phases.py", "bind")
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
        "bind": ("codegen/_phases.py", "bind"),
        "install": ("codegen/_phases.py", "install"),
        "require_install_plan": ("codegen/_plans.py", "require_install_plan"),
        "runtime install_plan": ("runtime/_bind_adapters.py", "install_plan"),
        "RuntimeInstance.__init__": ("runtime/_runtime_instance.py", "RuntimeInstance.__init__"),
        "RuntimeInstance._run": ("runtime/_runtime_instance.py", "RuntimeInstance._run"),
        "install_runtime_executor": ("runtime/_runtime_executor.py", "install_runtime_executor"),
        "Uniform provider install": (
            "runtime/_runtime_executor.py", "_UniformNativeProvider.install"),
        "Adaptive provider install": (
            "runtime/_runtime_executor.py", "_AdaptiveNativeProvider.install"),
        "flow_amr_layout": ("runtime/_runtime_mesh_lowering.py", "flow_amr_layout"),
        "_apply_refine_criterion": (
            "runtime/_runtime_mesh_lowering.py", "_apply_refine_criterion"),
        "_refine_threshold_value": (
            "runtime/_runtime_mesh_lowering.py", "_refine_threshold_value"),
        "_refine_subject_name": (
            "runtime/_runtime_mesh_lowering.py", "_refine_subject_name"),
        "run_bind_gates": ("runtime/_bind_validation.py", "run_bind_gates"),
        "validate_install_arguments": (
            "runtime/_bind_validation.py", "validate_install_arguments"),
        "System._install_compiled": (
            "runtime/_system_unified_install.py", "_SystemUnifiedInstall._install_compiled"),
        "System._resolve_instance_model": (
            "runtime/_system_unified_install.py", "_SystemUnifiedInstall._resolve_instance_model"),
        "System._declared_elliptic_fields": (
            "runtime/_system_unified_install.py", "_SystemUnifiedInstall._declared_elliptic_fields"),
        "System._install_program_params": (
            "runtime/_system_unified_install.py", "_SystemUnifiedInstall._install_program_params"),
        "AmrSystem._install_compiled": (
            "runtime/_amr_system_install.py", "_AmrSystemInstall._install_compiled"),
        "AmrSystem._declared_elliptic_fields": (
            "runtime/_amr_system_install.py", "_AmrSystemInstall._declared_elliptic_fields"),
        "_require_schema": ("runtime/_install_param_routing.py", "_require_schema"),
        "_slot_for_block": ("runtime/_install_param_routing.py", "_slot_for_block"),
        "_resolved_value": ("runtime/_install_param_routing.py", "_resolved_value"),
        "route_block_params": ("runtime/_install_param_routing.py", "route_block_params"),
        "route_program_params": ("runtime/_install_param_routing.py", "route_program_params"),
    }
    for label, (relative, qualified_name) in path.items():
        _assert_no_authoring_reconstruction(label, relative, qualified_name)


def test_program_param_routing_requires_captured_metadata_not_codegen_reentry():
    tree = _definition("runtime/_install_param_routing.py", "route_program_params")
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
