"""Canonical validate -> resolve -> compile -> bind/install phase pipeline."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def validate(problem: Any) -> Any:
    """Validate and freeze one exact Problem without compiling or loading native code."""
    from pops.problem import Problem

    if type(problem) is not Problem:
        raise TypeError("pops.validate requires an exact pops.Problem authoring value")
    problem.validate()
    if not problem.frozen:
        raise RuntimeError("pops.validate completed without freezing the Problem")
    return problem


def resolve(
    problem: Any,
    *,
    layout: Any,
    layout_providers: Mapping[Any, Any] | None = None,
    backend: Any = None,
    time: Any = None,
    libraries: Any = (),
    compile_options: Mapping[str, Any] | None = None,
    resolved_hierarchy: Any = None,
    amr_transfer: Any = None,
    initial_condition_plan: Any = None,
    bootstrap_plan: Any = None,
) -> Any:
    """Resolve a frozen Problem into the only value accepted by :func:`compile`."""
    from pops.problem import Problem

    if type(problem) is not Problem or not problem.frozen:
        raise TypeError("pops.resolve requires the frozen Problem returned by pops.validate")
    from pops.codegen.backends import Production, _Backend, lower_backend

    selected_backend = Production() if backend is None else backend
    if not isinstance(selected_backend, _Backend):
        raise TypeError("pops.resolve backend must be a typed pops.codegen backend descriptor")
    backend_token = lower_backend(selected_backend)
    from pops.codegen._layout_resolution import (
        layout_lowering_coverage, resolve_layout, validate_layout_consumers)

    layout_authority = resolve_layout(problem, layout, providers=layout_providers)
    layout_plan = layout_authority.plan
    target = "amr_system" if layout_plan.layouts[0].adaptive else "system"
    authorities = (
        resolved_hierarchy,
        amr_transfer,
        initial_condition_plan,
        bootstrap_plan,
    )
    if any(value is not None for value in authorities):
        if target != "amr_system" or any(value is None for value in authorities):
            raise ValueError(
                "AMR hierarchy, transfer, initial-condition, and bootstrap authorities "
                "must be passed together for an AMR layout"
            )
    if time is not None and problem._time is not None and time is not problem._time:
        raise ValueError("pops.resolve received two competing time-program authorities")
    resolved_time = time if time is not None else problem._time
    from pops.time import Program
    if resolved_time is not None and type(resolved_time) is not Program:
        raise TypeError("pops.resolve time must be an exact pops.Program")
    if target == "system" and resolved_time is None:
        raise ValueError("pops.resolve requires a whole-system Program for Uniform layout")
    if resolved_time is not None and backend_token != "production":
        raise ValueError("a resolved whole-system Program requires backend=Production()")
    validate_layout_consumers(
        problem, layout_plan, time=resolved_time,
        outputs=problem._outputs, diagnostics=problem._diagnostics)
    resolved_layout = layout_authority.require_runtime()

    options = dict(compile_options or {})
    allowed_options = {"so_path", "force", "cxx", "include", "std", "debug"}
    unknown_options = sorted(set(options) - allowed_options)
    if unknown_options:
        raise TypeError("pops.resolve received unsupported compile option(s) %s" % unknown_options)
    if "libraries" in options or "backend" in options:
        raise TypeError("libraries/backend have dedicated resolve-time authorities")

    from pops.codegen._orchestration_compile import (
        capture_field_plans,
        capture_runtime_declarations,
        prepare_problem_snapshot,
        resolve_compile_libraries,
    )
    from pops.problem._detached import detached_frozen
    from pops.model.bind_schema import BindSchema
    from pops.codegen._plans import ResolvedBlock, ResolvedSimulationPlan

    detached_layout = resolved_layout
    bind_schema = BindSchema.from_problem(problem)
    compile_values = bind_schema.resolve_compile()
    def block_backend(name: str) -> str:
        dynamic = []
        for slot in (*bind_schema.runtime_slots, *bind_schema.derived_slots):
            block_ref = slot.handle.block_ref
            if block_ref is None or block_ref.local_id != name:
                continue
            if slot.kind == "runtime" or slot.declaration["phase"] == "bind":
                dynamic.append(slot)
        return "aot" if target == "system" and dynamic else backend_token

    blocks = tuple(
        ResolvedBlock(
            name=name,
            model=_resolve_problem_model(spec["model"]),
            spatial=detached_frozen(spec["spatial"]),
            backend=block_backend(name),
        )
        for name, spec in problem._blocks.items()
    )
    field_plans = capture_field_plans(
        problem, detached_frozen, target=target, layout=detached_layout)
    outputs, diagnostics = capture_runtime_declarations(problem, detached_frozen)
    resolved_libraries, snapshot_libraries = resolve_compile_libraries(tuple(libraries or ()))
    snapshot = prepare_problem_snapshot(
        problem, resolved_time, layout=layout_plan, libraries=snapshot_libraries)
    from pops._bootstrap import abi_key
    from pops.codegen._resolution import resolve_capability_evidence

    evidence = resolve_capability_evidence(
        problem, layout=layout_plan, libraries=resolved_libraries, time=resolved_time,
        module_abi_key=abi_key())
    amr_requirements = None
    amr_capabilities = None
    if bootstrap_plan is not None:
        amr_requirements = {
            "hierarchy": resolved_hierarchy.identity.to_data(),
            "transfer": amr_transfer.identity.to_data(),
            "initial_conditions": initial_condition_plan.identity.to_data(),
            "bootstrap": bootstrap_plan.identity.to_data(),
        }
        amr_capabilities = bootstrap_plan.inspect()
    return ResolvedSimulationPlan(
        snapshot=snapshot, target=target, backend=backend_token, layout=detached_layout,
        layout_plan=layout_plan,
        time=resolved_time, blocks=blocks, bind_schema=bind_schema,
        compile_values=compile_values, field_plans=field_plans, outputs=outputs,
        diagnostics=diagnostics, libraries=resolved_libraries,
        requirements={"tokens": tuple(evidence["requirements"]),
                      "layout_resources": layout_plan.resource_requirements(),
                      "amr_resources": amr_requirements},
        capabilities={"resolution": evidence,
                      "layout_plan": layout_plan.capability_evidence(),
                      "amr_bootstrap": amr_capabilities},
        lowering_coverage=layout_lowering_coverage(layout_plan), compile_options=options,
        resolved_hierarchy=resolved_hierarchy, amr_transfer=amr_transfer,
        initial_condition_plan=initial_condition_plan, bootstrap_plan=bootstrap_plan)


def compile(plan: Any) -> Any:
    """Perform total lowering of one exact resolved plan, with no support recomputation."""
    from pops.codegen._plans import ResolvedSimulationPlan

    if type(plan) is not ResolvedSimulationPlan:
        raise TypeError("pops.compile requires the ResolvedSimulationPlan returned by pops.resolve")
    plan.verify()
    from pops.codegen._orchestration_compile import (
        build_program_model_graph,
        compile_install_models,
    )

    models = compile_install_models(plan, plan.compile_options)
    program = None
    if plan.time is not None:
        from pops.codegen.compile_drivers import compile_problem

        options = dict(plan.compile_options)
        options["libraries"] = plan.libraries
        model_graph = build_program_model_graph(plan)
        program = compile_problem(
            time=plan.time, model_graph=model_graph, backend=plan.backend, target=plan.target,
            problem_snapshot=plan.snapshot, field_plans=plan.field_plans, **options)
        program._discard_authoring()
    from pops.codegen.compiled_artifact import CompiledBlockArtifact, CompiledSimulationArtifact

    blocks = tuple(
        CompiledBlockArtifact(block.name, models[block.name], block.spatial)
        for block in plan.blocks)
    artifact = CompiledSimulationArtifact(plan=plan, program=program, blocks=blocks)
    artifact.verify()
    return artifact


def bind(artifact: Any, inputs: Any) -> Any:
    """Authenticate concrete BindInputs, create one InstallPlan, and install it."""
    from pops.codegen.compiled_artifact import CompiledSimulationArtifact
    from pops.codegen._plans import BindInputs, InstallPlan

    if type(artifact) is not CompiledSimulationArtifact:
        raise TypeError("pops.bind requires an exact CompiledSimulationArtifact")
    if type(inputs) is not BindInputs:
        raise TypeError("pops.bind requires an exact pops.BindInputs value")
    artifact.verify()
    inputs.verify()
    plan = artifact.plan
    if plan.initial_condition_plan is None:
        if inputs.initial_values:
            raise ValueError("BindInputs.initial_values requires a resolved InitialConditionPlan")
    else:
        if inputs.initial_state:
            raise ValueError(
                "AMR InitialConditionPlan is the single authority; initial_state cannot duplicate it"
            )
        expected_initial = {
            row.subject.qualified_id: row.subject
            for row in plan.initial_condition_plan.bindings
        }
        from pops.mesh.amr import AnalyticReprojection
        analytic = {
            row.subject.qualified_id
            for row in plan.bootstrap_plan.selections
            if type(row.method) is AnalyticReprojection
        }
        expected_initial = {
            key: value for key, value in expected_initial.items() if key not in analytic
        }
        supplied_initial = {row.qualified_id: row for row in inputs.initial_values}
        missing = sorted(set(expected_initial) - set(supplied_initial))
        extra = sorted(set(supplied_initial) - set(expected_initial))
        if missing or extra:
            raise ValueError(
                "BindInputs.initial_values must exactly cover InitialConditionPlan; "
                "missing=%s extra=%s" % (missing, extra)
            )
    params = plan.bind_schema.resolve_bind(
        inputs.params, compile_values=plan.compile_values)
    declared = {block.name for block in artifact.blocks}
    unknown = sorted(set(inputs.initial_state) - declared)
    if unknown:
        raise ValueError("BindInputs contains state for unknown block(s) %s" % unknown)
    instances = {}
    for block in artifact.blocks:
        entry = {"model": block.model, "spatial": block.spatial}
        if block.name in inputs.initial_state:
            entry["initial"] = inputs.initial_state[block.name]
        instances[block.name] = entry
    install_plan = InstallPlan(
        artifact=artifact, bind_inputs=inputs, instances=instances, params=params,
        aux=inputs.aux, resources=inputs.resources,
        execution_context=_execution_context(artifact, inputs.resources))
    return install(install_plan)


def _execution_context(artifact: Any, resources: Any) -> Any:
    from pops.runtime.platform_manifest import execution_context_for_bind
    return execution_context_for_bind(artifact.platform_manifest, resources)


def install(plan: Any) -> Any:
    """Install one authenticated final plan; no authoring/resolve/compile inputs are accepted."""
    from pops.codegen._plans import require_install_plan

    plan = require_install_plan(plan)
    from pops.runtime._bind_adapters import install_plan as runtime_install

    return runtime_install(plan)


def _resolve_problem_model(model: Any) -> Any:
    from pops.model import Module
    from pops.physics import Model as PhysicsModel

    if isinstance(model, (Module, PhysicsModel)):
        return model
    raise TypeError(
        "Problem block physics must be a pops.physics.Model or pops.model.Module, got %s"
        % type(model).__name__)


__all__ = ["bind", "compile", "install", "resolve", "validate"]
