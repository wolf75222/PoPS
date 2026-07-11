"""Compiler-side helpers for immutable resolved and install plans."""
from __future__ import annotations

from typing import Any


def resolve_compile_libraries(
    values: tuple[Any, ...],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Normalize library inputs once and derive detached snapshot projections."""
    if not values:
        return (), ()
    from pops.codegen.library import read_library_manifest
    from pops.external.bricks import CompiledBrickRef

    compiler_values = []
    snapshot_values = []
    for value in values:
        if isinstance(value, CompiledBrickRef):
            value.validate()
            compiler_values.append(value)
            record = value.manifest_record()
            snapshot_values.append(record if record is not None else value.options())
        else:
            manifest = read_library_manifest(value)
            compiler_values.append(manifest)
            snapshot_values.append(manifest)
    return tuple(compiler_values), tuple(snapshot_values)


def compile_install_models(plan: Any, backend: str, kwargs: Any) -> dict[str, Any]:
    """Compile every block before bind and return one native loader per instance."""
    compile_kwargs = {
        key: value for key, value in kwargs.items()
        if key in ("include", "cxx", "std", "require_metadata", "hoist_reciprocals")
    }
    return {
        block.name: compile_install_model(
            block.name, block.model, backend, plan.target, compile_kwargs)
        for block in plan.blocks
    }


def compile_install_model(
    name: str,
    model: Any,
    backend: str,
    target: str,
    compile_kwargs: Any,
) -> Any:
    """Lower one model through the final compiled-value or structural protocol."""
    from pops.codegen.loader import CompiledModel
    from pops.codegen._compiled_model_boundary import validate_compiled_model_result
    from pops.codegen._compiled_model_identity import authenticate_compiled_model

    if isinstance(model, CompiledModel):
        validate_compiled_model_result(model)
        return model
    source_module_hash = None
    compile_model = getattr(model, "compile", None)
    if not callable(compile_model) and callable(getattr(model, "operator_registry", None)):
        from pops.codegen.module_lowering import _module_to_model

        module_hash = getattr(model, "module_hash", None)
        source_module_hash = module_hash() if callable(module_hash) else None
        model = _module_to_model(model)
        compile_model = getattr(model, "compile", None)
    if not callable(compile_model):
        raise NotImplementedError(
            "pops.compile: block %r model %s does not implement the install-model protocol "
            "(expected CompiledModel or compile(backend=, target=))"
            % (name, type(model).__name__))
    selected_backend = backend
    if target == "system" and model_has_runtime_params(model):
        selected_backend = "aot"
    compiled = compile_model(backend=selected_backend, target=target, **compile_kwargs)
    if not isinstance(compiled, CompiledModel):
        raise TypeError(
            "pops.compile: block %r compile() must return pops.codegen.CompiledModel, not %s; "
            "mutable/opaque loader records cannot enter InstallPlan"
            % (name, type(compiled).__name__)
        )
    validate_compiled_model_result(compiled)
    authenticate_compiled_model(model, compiled, module_hash=source_module_hash)
    if compiled.target != target:
        raise ValueError(
            "pops.compile: block %r compile() returned target=%r, expected %r"
            % (name, compiled.target, target)
        )
    return compiled


def model_has_runtime_params(model: Any) -> bool:
    params = getattr(model, "params", {})
    params = params() if callable(params) else params
    values = params.values() if hasattr(params, "values") else ()
    for declaration in values:
        kind = getattr(declaration, "kind", None)
        kind = getattr(kind, "value", kind)
        phase = getattr(declaration, "phase", None)
        phase = getattr(phase, "value", phase)
        if kind == "runtime" or (kind == "derived" and phase == "bind"):
            return True
    return False


def attach_install_plan(
    compiled: Any,
    resolved: Any,
    models: Any,
    *,
    has_program: bool,
) -> None:
    """Attach the only bind authority and remove authoring-model retention."""
    from pops.codegen._plans import InstallBlock, InstallPlan
    from pops.codegen.loader import CompiledModel

    blocks = tuple(
        InstallBlock(block.name, models[block.name], block.spatial)
        for block in resolved.blocks)
    install_plan = InstallPlan(
        snapshot_hash=resolved.snapshot.hash,
        target=resolved.target,
        layout=resolved.layout,
        blocks=blocks,
        bind_schema=resolved.bind_schema,
        field_solvers=resolved.field_solvers,
        outputs=resolved.outputs,
        diagnostics=resolved.diagnostics,
        has_program=has_program,
    )
    if isinstance(compiled, CompiledModel):
        object.__setattr__(compiled, "install_plan", install_plan)
        object.__setattr__(compiled, "bind_schema", resolved.bind_schema)
    else:
        compiled.install_plan = install_plan
        compiled.bind_schema = resolved.bind_schema
    if hasattr(compiled, "model"):
        compiled.model = blocks[0].model
    for model in models.values():
        if model is not compiled and isinstance(model, CompiledModel):
            # The artifact boundary is final: a subclass cannot replace _seal with a no-op.
            CompiledModel._seal(model)


def capture_runtime_declarations(
    problem: Any,
    detach: Any = None,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Detach and canonicalize output/diagnostic declarations."""
    if detach is None:
        from pops.problem._detached import detached_frozen

        detach = detached_frozen

    def resolved(value: Any, family: str) -> Any:
        resolve_references = getattr(value, "resolve_references", None)
        if not callable(resolve_references):
            raise TypeError(
                "%s declaration %r must implement resolve_references(resolver)"
                % (family, type(value).__name__))
        return detach(resolve_references(problem.resolve))

    outputs = tuple(resolved(value, "runtime output")
                    for value in (problem._outputs or []))
    diagnostics = tuple(resolved(value, "runtime diagnostic")
                        for value in (problem._diagnostics or []))
    return outputs, diagnostics


def capture_field_solvers(problem: Any, detach: Any) -> dict[str, Any]:
    """Resolve field references once and retain only detached solver values."""
    result = {}
    for name, field in problem._field_registry.resolved_items(problem.resolve):
        if field.solver is not None:
            result[name] = detach(field.solver)
    return result


def prepare_problem_snapshot(
    problem: Any,
    time: Any,
    *,
    layout: Any,
    libraries: Any,
) -> Any:
    """Freeze authoring before the driver computes an artifact identity."""
    from pops.problem._snapshot import prepare_compile_snapshot

    return prepare_compile_snapshot(problem, time, layout=layout, libraries=libraries)


def attach_problem_snapshot(compiled: Any, snapshot: Any) -> None:
    """Attach the snapshot already used by the driver, then seal the artifact."""
    from pops.problem._snapshot import attach_problem_snapshot

    attach_problem_snapshot(compiled, snapshot)


__all__ = [
    "attach_install_plan", "attach_problem_snapshot", "capture_field_solvers",
    "capture_runtime_declarations", "compile_install_model", "compile_install_models",
    "model_has_runtime_params", "prepare_problem_snapshot", "resolve_compile_libraries",
]
