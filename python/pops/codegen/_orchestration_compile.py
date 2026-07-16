"""Pure helpers used by resolve and total compile; no artifact mutation or install path."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def compile_install_models(plan: Any, options: Any) -> dict[str, Any]:
    compile_options = {key: value for key, value in options.items()
                       if key in ("include", "cxx", "std")}
    return {block.name: compile_install_model(
        block.name, block.model, block.backend, plan.target, compile_options,
        state_spaces=block.state_spaces)
            for block in plan.blocks}


def build_program_model_graph(plan: Any) -> Any:
    """Build the exact owner-qualified model graph for whole-Program lowering."""
    from pops.codegen._plans import ResolvedSimulationPlan

    if type(plan) is not ResolvedSimulationPlan:
        raise TypeError("program model graph requires an exact ResolvedSimulationPlan")
    from pops.codegen.program_models import ProgramModelGraph

    return ProgramModelGraph.from_resolved_blocks(plan.blocks)


def compile_install_model(name: str, model: Any, backend: str, target: str,
                          compile_options: Any, *, state_spaces: Any = ("U",)) -> Any:
    from pops.codegen.loader import CompiledModel
    from pops.codegen._compiled_model_boundary import validate_compiled_model_result
    from pops.codegen._compiled_model_identity import authenticate_compiled_model

    state_spaces = tuple(state_spaces)
    if len(state_spaces) != 1 or not isinstance(state_spaces[0], str) or not state_spaces[0]:
        raise TypeError("compiled block %r requires exactly one named state space" % name)
    if isinstance(model, CompiledModel):
        validate_compiled_model_result(model)
        if tuple(model.state_spaces) != state_spaces:
            raise ValueError("resolved compiled model state-space route disagrees with its plan")
        if model.target != target or model.backend != backend:
            raise ValueError("resolved compiled model route disagrees with its plan")
        return model
    from pops.codegen.module_lowering import lower_and_validate

    facade = model
    model, source_module = lower_and_validate(
        model, facade=facade, state_space=state_spaces[0])
    if source_module is None:
        raise TypeError(
            "resolved block %r compiler lowering has no operator-first Module authority" % name)
    source_module_hash = source_module.module_hash()
    from pops.model.manifest import build_module_manifest

    module_manifest = build_module_manifest(source_module)
    if source_module.module_hash() != source_module_hash:
        raise ValueError(
            "resolved block %r Module changed while its compile-frozen trace was captured" % name)
    compile_model = getattr(model, "compile", None)
    if not callable(compile_model):
        raise TypeError("resolved block %r has no total compile lowering" % name)
    compiled = compile_model(backend=backend, target=target, **compile_options)
    if type(compiled) is not CompiledModel:
        raise TypeError("resolved block compiler must return exact CompiledModel")
    if compiled.module_manifest is not None:
        raise TypeError(
            "model.compile() returned a CompiledModel with a pre-attached ModuleManifest; "
            "only pops.compile may attach compile-frozen trace authority")
    compiled.state_spaces = state_spaces
    validate_compiled_model_result(compiled)
    authenticate_compiled_model(model, compiled, module_hash=source_module_hash)
    object.__setattr__(
        compiled, "module_manifest", module_manifest.with_abi_key(compiled.abi_key))
    validate_compiled_model_result(compiled)
    if compiled.target != target or compiled.backend != backend:
        raise ValueError("compiled block route differs from ResolvedSimulationPlan")
    return compiled


def capture_field_plans(
    problem: Any, detach: Any, *, target: str, layout: Any,
) -> dict[str, Any]:
    """Resolve every complete field registration or refuse before artifact creation."""
    from pops.codegen.field_install import resolve_field_install_plan

    result = {}
    runtime_routes = {}
    # Authenticate the complete provider graph before canonicalizing any install identity. This is
    # one snapshot boundary: identity/hash work for field A must not influence live authoring lookup
    # for field B in the same Problem.
    prepared = []
    for name, field in problem._field_registry.resolved_items(problem.resolve):
        providers, provider_route = _field_rhs_providers(problem, field)
        prepared.append((name, field, providers, provider_route))
    for name, field, providers, provider_route in prepared:
        unknown = field.operator.unknown
        block_ref = unknown.block_ref
        declaration_ref = unknown.declaration_ref
        if block_ref is None or declaration_ref is None:
            raise TypeError(
                "FieldOperator unknown must be an owner-qualified FieldSpace instance"
            )
        block_spec = problem._block_registry.spec(block_ref.local_id)
        model = None if block_spec is None else block_spec.get("model")
        field_spaces = getattr(model, "field_spaces", None)
        if not callable(field_spaces):
            field_spaces = getattr(getattr(model, "module", None), "field_spaces", None)
        declared_spaces = field_spaces() if callable(field_spaces) else {}
        if not isinstance(declared_spaces, Mapping):
            raise TypeError("compiled model field_spaces() must return a mapping")
        output_space = declared_spaces.get(declaration_ref.local_id)
        output_components = tuple(getattr(output_space, "components", ()))
        if not output_components:
            raise ValueError(
                "FieldOperator %r output declaration %r is absent from block %r"
                % (name, declaration_ref.local_id, block_ref.local_id)
            )
        resolved = resolve_field_install_plan(
            name, detach(field), target=target, rhs_providers=providers,
            provider_route=provider_route, output_components=output_components,
            layout=layout)
        route = resolved.native_options["provider_slot"]
        provider_identity = resolved.native_options["provider_identity"]
        if route in runtime_routes:
            prior_name, prior_identity = runtime_routes[route]
            if prior_identity != provider_identity:
                raise ValueError(
                    "qualified field-provider identity digest collision between %r and %r"
                    % (prior_name, name))
            raise ValueError(
                "fields %r and %r resolve to the same native field provider %r; "
                "each FieldOperator must name a distinct qualified provider"
                % (prior_name, name, route))
        runtime_routes[route] = (name, provider_identity)
        result[name] = resolved
    return result


def _field_rhs_providers(problem: Any, registration: Any) -> tuple[tuple[Any, ...], tuple[dict[str, Any], ...]]:
    """Authenticate the ordered owner-qualified provider pack materializing one RHS graph."""
    operator = registration.operator
    authenticated: list[Any] = []
    routes: list[dict[str, Any]] = []
    composed_body = None
    for contribution in operator.providers:
        provider = contribution.provider
        if not provider.is_instance or provider.block_ref is None:
            raise ValueError(
                "field %r provider was not resolved to a Problem block instance"
                % operator.name)
        provider_declaration = provider.declaration_ref
        if provider_declaration is None:
            raise ValueError(
                "field %r provider lost its model declaration provenance" % operator.name)
        matched = False
        for block_name, spec in problem._block_registry.items():
            block_handle = problem._block_registry.handle(block_name)
            block = problem.resolve(block_handle)
            if block.canonical_identity() != provider.block_ref.canonical_identity():
                continue
            model = spec["model"]
            module = getattr(model, "module", model)
            registry = module.operator_registry()
            declared = module.operator_handle(provider_declaration.local_id)
            qualified = problem.resolve(declared, block=block_handle)
            if qualified.canonical_identity() != provider.canonical_identity():
                raise ValueError("field %r provider is not issued by its model" % operator.name)
            field_op = registry.get(registry.target_for_handle(provider_declaration.local_id))
            if field_op.kind != "field_operator":
                raise TypeError("field %r provider is not a field_operator" % operator.name)
            route = field_op.lowering.get("field_provider")
            key = route.get("key") if isinstance(route, Mapping) else None
            if not isinstance(key, str) or not key:
                raise ValueError(
                    "field %r provider has no explicit native provider key" % operator.name)
            if field_op.body is None:
                raise ValueError(
                    "field %r RHS provider has no executable expression graph" % operator.name)
            term = (field_op.body if contribution.coefficient == 1.0
                    else field_op.body * contribution.coefficient)
            composed_body = term if composed_body is None else composed_body + term
            authenticated.append(qualified)
            routes.append({
                "provider_identity": qualified.canonical_identity(),
                "owner_block": block_name,
                "key": key,
                "coefficient": contribution.coefficient,
                "measure": contribution.measure.to_data(),
                "native_measure": contribution.measure.lower_native_provider(),
            })
            matched = True
            break
        if not matched:
            raise ValueError(
                "field %r provider block is not registered by this Problem" % operator.name)
    from pops.fields._identity import strict_field_data
    expected = operator.equation.rhs
    from pops.math import Laplacian, elliptic_terms
    laplacians = [
        term for term in elliptic_terms(operator.equation.lhs)
        if isinstance(term, Laplacian)
    ]
    if len(laplacians) == 1:
        normalization = -float(laplacians[0].scale)
        if normalization != 1.0:
            expected = expected / normalization
    if strict_field_data(composed_body) != strict_field_data(expected):
        raise ValueError(
            "field %r descriptor RHS differs from its ordered provider composition"
            % operator.name)
    return tuple(authenticated), tuple(routes)


def prepare_problem_snapshot(problem: Any, time: Any, *, layout: Any, libraries: Any) -> Any:
    from pops.problem._snapshot import prepare_compile_snapshot
    return prepare_compile_snapshot(problem, time, layout=layout, libraries=libraries)


__all__ = [
    "build_program_model_graph", "capture_field_plans",
    "compile_install_model", "compile_install_models", "prepare_problem_snapshot",
]
