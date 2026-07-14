"""Canonical validate -> resolve -> compile -> bind/install phase pipeline."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def validate(problem: Any) -> Any:
    """Validate and freeze one exact Case without compiling or loading native code."""
    from pops.problem import Case

    if type(problem) is not Case:
        raise TypeError("pops.validate requires an exact pops.Case authoring value")
    problem.validate()
    if not problem.frozen:
        raise RuntimeError("pops.validate completed without freezing the Case")
    return problem


def resolve(
    problem: Any,
    *,
    layout: Any,
    layout_providers: Mapping[Any, Any] | None = None,
    backend: Any = None,
    time: Any = None,
    components: Any = (),
    compile_options: Mapping[str, Any] | None = None,
) -> Any:
    """Resolve a frozen Case into the only value accepted by :func:`compile`."""
    from pops.problem import Case

    if type(problem) is not Case or not problem.frozen:
        raise TypeError("pops.resolve requires the frozen Case returned by pops.validate")
    if isinstance(components, (str, bytes, Mapping)):
        raise TypeError("pops.resolve components must be a finite iterable of typed components")
    try:
        components = tuple(components)
    except TypeError as exc:
        raise TypeError(
            "pops.resolve components must be a finite iterable of typed components") from exc
    from pops.codegen._backends import Production, _Backend, lower_backend

    selected_backend = Production() if backend is None else backend
    if not isinstance(selected_backend, _Backend):
        raise TypeError("pops.resolve backend must be a typed pops.codegen backend descriptor")
    backend_token = lower_backend(selected_backend)
    from pops.codegen._layout_resolution import (
        _refuse_runtime, layout_lowering_coverage, resolve_layout,
        validate_layout_mapping_components, validate_program_layout_reads)

    layout_authority = resolve_layout(problem, layout, providers=layout_providers)
    layout_plan = layout_authority.plan
    adaptive_families = {row.adaptive for row in layout_plan.layouts}
    if len(adaptive_families) != 1:
        _refuse_runtime(
            layout_plan,
            gate="mixed_uniform_amr_runtime_unavailable",
            message="one RuntimeInstance cannot mix Uniform and AMR layout families",
        )
    adaptive = next(iter(adaptive_families))
    if adaptive and len(layout_plan.layouts) != 1:
        _refuse_runtime(
            layout_plan,
            gate="heterogeneous_amr_runtime_unavailable",
            message="heterogeneous AMR layouts have no proved common regrid/transfer runtime",
        )
    target = "amr_system" if adaptive else "system"
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
    validate_program_layout_reads(problem, layout_plan, time=resolved_time)
    if len(layout_plan.layouts) > 1:
        co_layout_ops = {
            "coupled_rate", "solve_coupled_implicit", "solve_fields_from_blocks",
            "field_solve_from_blocks",
        }
        present = sorted({
            node["op"] for node in resolved_time.ir_nodes()
            if node["op"] in co_layout_ops
        })
        if present:
            _refuse_runtime(
                layout_plan,
                gate="multi_layout_colocated_kernel_unavailable",
                message=("Program operation(s) %s require one co-located grid and have no "
                         "qualified mapping lowering" % present),
            )
    resolved_layouts = layout_authority.require_runtime()
    validate_layout_mapping_components(layout_plan, components)
    if len(layout_plan.layouts) > 1 and tuple(problem.layout_subjects().fields):
        _refuse_runtime(
            layout_plan,
            gate="multi_layout_field_operator_unavailable",
            message=("FieldOperator storage/solve has no per-layout compiled partition yet; "
                     "refusing before artifact creation"),
        )
    resolved_layout = resolved_layouts.single() if len(resolved_layouts.rows) == 1 \
        else resolved_layouts

    options = dict(compile_options or {})
    allowed_options = {"so_path", "force", "cxx", "include", "std", "debug"}
    unknown_options = sorted(set(options) - allowed_options)
    if unknown_options:
        raise TypeError("pops.resolve received unsupported compile option(s) %s" % unknown_options)
    if "libraries" in options or "backend" in options:
        raise TypeError("libraries are retired and backend has a dedicated resolve-time authority")

    from pops.codegen._orchestration_compile import (
        capture_field_plans,
        prepare_problem_snapshot,
    )
    from pops.problem._detached import detached_frozen
    from pops.model.bind_schema import BindSchema
    from pops.codegen._plans import ResolvedBlock, ResolvedSimulationPlan

    detached_layout = resolved_layout
    bind_schema = BindSchema.from_problem(problem)
    compile_values = bind_schema.resolve_compile()

    resolved_blocks = []
    for name, spec in problem._blocks.items():
        state_spaces = tuple(state.local_id for state in spec["states"])
        if len(state_spaces) != 1:
            raise ValueError(
                "block %r selects %d state spaces; the installed native block contract requires "
                "exactly one. Split independently evolved states into qualified Case blocks."
                % (name, len(state_spaces))
            )
        numerics = problem._resolved_numerics_for(name)
        spatial = spec["spatial"]
        if numerics is not None:
            if spatial is not None:
                raise ValueError("block %r has competing spatial and DiscretizationPlan authorities" % name)
            spatial = numerics.primary_spatial()
        resolved_blocks.append(ResolvedBlock(
            name=name,
            model=_resolve_problem_model(spec["model"]),
            spatial=detached_frozen(spatial),
            # Bind-time values cross the native install ABI and are injected before the block
            # closures are built. A RuntimeParam never selects a second host-marshalled backend.
            backend=backend_token,
            state_spaces=state_spaces,
            state_identities=tuple(state.qualified_id for state in spec["states"]),
            numerics=numerics,
        ))
    blocks = tuple(resolved_blocks)
    resolved_hierarchy = None
    amr_transfer = None
    initial_condition_plan = None
    bootstrap_plan = None
    amr_execution = None
    amr_providers = {}
    if target == "amr_system":
        from pops.amr._resolution import (
            AMRLayoutResolver,
            AMRResolutionContext,
            ResolvedAMRAuthorities,
        )

        adaptive_layout = resolved_layouts.single()
        if not isinstance(adaptive_layout, AMRLayoutResolver):
            raise TypeError(
                "adaptive layout providers must implement "
                "resolve_amr_authorities(AMRResolutionContext)"
            )

        from pops.model import Handle

        def resolve_amr_handle(value: Any) -> Any:
            if isinstance(value, Handle) and value.is_resolved:
                return value
            return problem.resolve(value)

        context = AMRResolutionContext(
            owner=problem.owner_path.canonical(),
            layout_plan=layout_plan,
            numerics=tuple(
                block.numerics for block in blocks if block.numerics is not None),
            initials=problem.initials,
            program=resolved_time,
            resolve=resolve_amr_handle,
            components=tuple(components),
        )
        resolved_amr = adaptive_layout.resolve_amr_authorities(context)
        if type(resolved_amr) is not ResolvedAMRAuthorities:
            raise TypeError(
                "layout resolve_amr_authorities() must return exact ResolvedAMRAuthorities"
            )
        resolved_hierarchy = resolved_amr.hierarchy
        amr_transfer = resolved_amr.transfer
        initial_condition_plan = resolved_amr.initial_conditions
        bootstrap_plan = resolved_amr.bootstrap
        amr_execution = resolved_amr.execution
        amr_providers = resolved_amr.providers
    # Boundary producers depend on the resolved layout and, for adaptive states, on the exact AMR
    # transfer authority.  Compose them only after both authorities exist, then carry the single
    # executable GhostProducerPlan through compile/install.
    from dataclasses import replace
    from pops.mesh.boundaries.composition import compose_boundary_plans

    blocks = tuple(
        block if block.numerics is None else replace(
            block,
            numerics=compose_boundary_plans(
                block.numerics,
                layout_plan=layout_plan,
                amr_transfer=amr_transfer,
            ),
        )
        for block in blocks
    )
    from pops.mesh.boundaries.composition import compose_shared_interfaces
    blocks = compose_shared_interfaces(blocks, layout_plan=layout_plan)
    from pops.codegen._interface_validation import (
        validate_prepared_boundary_jacvec,
        validate_shared_interface_program,
    )

    validate_prepared_boundary_jacvec(blocks, resolved_time)
    has_shared_interfaces = validate_shared_interface_program(
        blocks, layout_plan, resolved_time, target=target,
        resolved_hierarchy=resolved_hierarchy,
    )
    field_plans = capture_field_plans(
        problem, detached_frozen, target=target, layout=detached_layout)
    from pops.codegen.program_emit_field_routes import validate_program_field_routes
    validate_program_field_routes(resolved_time, field_plans)
    snapshot = prepare_problem_snapshot(
        problem, resolved_time, layout=layout_plan, libraries=())
    from pops.codegen._resolution import resolve_capability_evidence

    amr_program_context = None
    if target == "amr_system":
        from pops.mesh._amr import FrozenHierarchy
        from pops.runtime.amr_program_support import AMRProgramSupportContext

        if resolved_hierarchy is None:
            raise TypeError("resolved AMR hierarchy evidence is missing")
        hierarchy = resolved_hierarchy.plan
        amr_program_context = AMRProgramSupportContext(
            refined_hierarchy=(
                hierarchy.level_count != 1
                or type(hierarchy.regrid) is not FrozenHierarchy
            ),
            shared_block_interfaces=has_shared_interfaces,
            field_routes_validated=True,
        )

    evidence = resolve_capability_evidence(
        problem, layout=layout_plan, libraries=(), time=resolved_time,
        module_abi_key=None, amr_program_context=amr_program_context)
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
    consumer_graph = (
        None if problem._consumers is None
        else problem._consumers.resolve(
            problem.resolve, layout_plan, owner=problem.owner_path.canonical())
    )
    return ResolvedSimulationPlan(
        snapshot=snapshot, target=target, backend=backend_token, layout=detached_layout,
        layout_plan=layout_plan,
        layout_targets={
            row.handle.qualified_id: ("amr_system" if row.adaptive else "system")
            for row in layout_plan.layouts
        },
        time=resolved_time, blocks=blocks, bind_schema=bind_schema,
        compile_values=compile_values, field_plans=field_plans, consumer_graph=consumer_graph,
        libraries=(),
        requirements={"tokens": tuple(evidence["requirements"]),
                      "layout_resources": layout_plan.resource_requirements(),
                      "amr_resources": amr_requirements},
        capabilities={"resolution": evidence,
                      "layout_plan": layout_plan.capability_evidence(),
                      "amr_bootstrap": amr_capabilities},
        lowering_coverage=layout_lowering_coverage(layout_plan), compile_options=options,
        component_inputs=tuple(components),
        resolved_hierarchy=resolved_hierarchy, amr_transfer=amr_transfer,
        initial_condition_plan=initial_condition_plan, bootstrap_plan=bootstrap_plan,
        amr_execution=amr_execution, amr_providers=amr_providers)


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
    layout_programs = ()
    if plan.time is not None:
        from pops.codegen._compile_drivers import compile_problem
        from pops.codegen._compiled_artifact import CompiledLayoutProgram
        from pops.codegen.program_models import ProgramModelGraph

        options = dict(plan.compile_options)
        options["libraries"] = plan.libraries
        if len(plan.layout_plan.layouts) == 1:
            model_graph = build_program_model_graph(plan)
            program = compile_problem(
                time=plan.time, model_graph=model_graph, backend=plan.backend, target=plan.target,
                problem_snapshot=plan.snapshot, field_plans=plan.field_plans, **options)
            program._discard_authoring()
            row = plan.layout_plan.layouts[0]
            layout_programs = (CompiledLayoutProgram(
                row.handle.qualified_id, plan.layout_targets[row.handle.qualified_id],
                tuple(block.name for block in plan.blocks), program),)
        else:
            from pathlib import Path
            from pops.codegen.program_slicing import slice_program

            block_layouts = {
                assignment.subject.local_id: assignment.layout.qualified_id
                for assignment in plan.layout_plan.assignments
                if assignment.subject_kind == "block"
            }
            compiled_rows = []
            for row in plan.layout_plan.layouts:
                layout_id = row.handle.qualified_id
                selected = tuple(
                    block for block in plan.blocks if block_layouts[block.name] == layout_id)
                selected_names = tuple(block.name for block in selected)
                sliced = slice_program(plan.time, selected_names)
                slice_options = dict(options)
                if slice_options.get("so_path") is not None:
                    path = Path(slice_options["so_path"])
                    slice_options["so_path"] = str(
                        path.with_name("%s.%s%s" % (
                            path.stem, row.handle.local_id, path.suffix or ".so")))
                compiled_program = compile_problem(
                    time=sliced,
                    model_graph=ProgramModelGraph.from_resolved_blocks(selected),
                    backend=plan.backend,
                    target=plan.layout_targets[layout_id],
                    problem_snapshot=plan.snapshot,
                    field_plans={},
                    **slice_options,
                )
                compiled_program._discard_authoring()
                compiled_rows.append(CompiledLayoutProgram(
                    layout_id, plan.layout_targets[layout_id], selected_names,
                    compiled_program))
            layout_programs = tuple(compiled_rows)
    from pops.codegen._compiled_artifact import CompiledBlockArtifact, CompiledSimulationArtifact

    blocks = tuple(
        CompiledBlockArtifact(
            block.name, models[block.name], block.spatial, block.state_spaces)
        for block in plan.blocks)
    from pops.external import CompiledComponentArtifact, ExternalComponent, compile_component

    component_artifacts = tuple(
        compile_component(item, cxx=plan.compile_options.get("cxx"),
                          include=plan.compile_options.get("include"))
        if type(item) is ExternalComponent else item
        for item in plan.component_inputs
    )
    if any(type(item) is not CompiledComponentArtifact for item in component_artifacts):
        raise TypeError("component compilation did not produce exact CompiledComponentArtifact values")
    artifact = CompiledSimulationArtifact(
        plan=plan, program=program, blocks=blocks, layout_programs=layout_programs,
        component_artifacts=component_artifacts)
    artifact.verify()
    return artifact


def bind(artifact: Any, inputs: Any) -> Any:
    """Authenticate concrete BindInputs, create one InstallPlan, and install it."""
    from pops.codegen._compiled_artifact import CompiledSimulationArtifact
    from pops.codegen._plans import BindInputs, InstallPlan

    if type(artifact) is not CompiledSimulationArtifact:
        raise TypeError("pops.bind requires an exact CompiledSimulationArtifact")
    if type(inputs) is not BindInputs:
        raise TypeError("internal bind phase requires an exact authenticated BindInputs record")
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
        from pops.mesh._amr import AnalyticReprojection
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
    components = {}
    if artifact.component_artifacts:
        from pops.codegen.cache import component_store_dir

        install_root = component_store_dir()
        for component in artifact.component_artifacts:
            installed = component.install(install_root).load()
            components[installed.component_id] = installed
    install_plan = InstallPlan(
        artifact=artifact, bind_inputs=inputs, instances=instances, params=params,
        aux=inputs.aux, resources=inputs.resources, components=components,
        execution_context=_execution_context(artifact, inputs.resources))
    return install(install_plan)


def _execution_context(artifact: Any, resources: Any) -> Any:
    from pops.runtime._platform_manifest import execution_context_for_bind
    return execution_context_for_bind(artifact.platform_manifest, resources)


def install(plan: Any) -> Any:
    """Install one authenticated final plan; no authoring/resolve/compile inputs are accepted."""
    from pops.codegen._plans import require_install_plan

    plan = require_install_plan(plan)
    from pops.runtime._bind_adapters import install_plan as runtime_install

    return runtime_install(plan)


def _resolve_problem_model(model: Any) -> Any:
    from pops.codegen._compiler_lowering import require_compiler_lowering

    require_compiler_lowering(model)
    return model


__all__ = ["bind", "compile", "install", "resolve", "validate"]
