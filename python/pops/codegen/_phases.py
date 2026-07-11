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
    backend: Any = None,
    time: Any = None,
    libraries: Any = (),
    compile_options: Mapping[str, Any] | None = None,
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
    from pops.codegen._layout_resolution import resolve_layout, validate_layout

    resolved_layout = resolve_layout(problem, layout)
    validate_layout(problem, resolved_layout)
    from pops.mesh.layouts import AMR
    target = "amr_system" if isinstance(resolved_layout, AMR) else "system"
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

    options = dict(compile_options or {})
    allowed_options = {"so_path", "force", "cxx", "include", "std", "debug"}
    unknown_options = sorted(set(options) - allowed_options)
    if unknown_options:
        raise TypeError("pops.resolve received unsupported compile option(s) %s" % unknown_options)
    if "libraries" in options or "backend" in options:
        raise TypeError("libraries/backend have dedicated resolve-time authorities")

    from pops.codegen._orchestration_compile import (
        capture_field_solvers,
        capture_runtime_declarations,
        prepare_problem_snapshot,
        resolve_compile_libraries,
    )
    from pops.problem._detached import detached_frozen
    from pops.model.bind_schema import BindSchema
    from pops.codegen._plans import ResolvedBlock, ResolvedSimulationPlan

    detached_layout = detached_frozen(resolved_layout)
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
    field_solvers = capture_field_solvers(problem, detached_frozen)
    outputs, diagnostics = capture_runtime_declarations(problem, detached_frozen)
    resolved_libraries, snapshot_libraries = resolve_compile_libraries(tuple(libraries or ()))
    snapshot = prepare_problem_snapshot(
        problem, resolved_time, layout=detached_layout, libraries=snapshot_libraries)
    from pops._bootstrap import abi_key
    from pops.codegen._resolution import resolve_capability_evidence

    evidence = resolve_capability_evidence(
        problem, layout=detached_layout, libraries=resolved_libraries, time=resolved_time,
        module_abi_key=abi_key())
    return ResolvedSimulationPlan(
        snapshot=snapshot, target=target, backend=backend_token, layout=detached_layout,
        time=resolved_time, blocks=blocks, bind_schema=bind_schema,
        compile_values=compile_values, field_solvers=field_solvers, outputs=outputs,
        diagnostics=diagnostics, libraries=resolved_libraries,
        requirements={"tokens": tuple(evidence["requirements"])},
        capabilities={"resolution": evidence}, compile_options=options)


def compile(plan: Any) -> Any:
    """Perform total lowering of one exact resolved plan, with no support recomputation."""
    from pops.codegen._plans import ResolvedSimulationPlan

    if type(plan) is not ResolvedSimulationPlan:
        raise TypeError("pops.compile requires the ResolvedSimulationPlan returned by pops.resolve")
    plan.verify()
    from pops.codegen._orchestration_compile import compile_install_models

    models = compile_install_models(plan, plan.compile_options)
    program = None
    if plan.time is not None:
        from pops.codegen.compile_drivers import compile_problem

        options = dict(plan.compile_options)
        options["libraries"] = plan.libraries
        program = compile_problem(
            time=plan.time, model=plan.first_model, backend=plan.backend, target=plan.target,
            problem_snapshot=plan.snapshot, **options)
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
        aux=inputs.aux, resources=inputs.resources)
    return install(install_plan)


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
