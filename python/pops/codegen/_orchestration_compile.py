"""Pure helpers used by resolve and total compile; no artifact mutation or install path."""
from __future__ import annotations

from typing import Any


def resolve_compile_libraries(values: tuple[Any, ...]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    from pops.codegen.library import read_library_manifest
    from pops.external.bricks import CompiledBrickRef

    compiler_values, snapshot_values = [], []
    for value in values:
        if isinstance(value, CompiledBrickRef):
            compiler_values.append(value)
            record = value.manifest_record()
            snapshot_values.append(record if record is not None else value.options())
        else:
            manifest = read_library_manifest(value)
            compiler_values.append(manifest)
            snapshot_values.append(manifest)
    return tuple(compiler_values), tuple(snapshot_values)


def compile_install_models(plan: Any, options: Any) -> dict[str, Any]:
    compile_options = {key: value for key, value in options.items()
                       if key in ("include", "cxx", "std")}
    return {block.name: compile_install_model(
        block.name, block.model, block.backend, plan.target, compile_options)
            for block in plan.blocks}


def compile_install_model(name: str, model: Any, backend: str, target: str,
                          compile_options: Any) -> Any:
    from pops.codegen.loader import CompiledModel
    from pops.codegen._compiled_model_boundary import validate_compiled_model_result
    from pops.codegen._compiled_model_identity import authenticate_compiled_model

    if isinstance(model, CompiledModel):
        validate_compiled_model_result(model)
        if model.target != target or model.backend != backend:
            raise ValueError("resolved compiled model route disagrees with its plan")
        return model
    source_module_hash = None
    compile_model = getattr(model, "compile", None)
    if not callable(compile_model) and callable(getattr(model, "operator_registry", None)):
        from pops.codegen.module_lowering import _module_to_model
        source_module_hash = model.module_hash()
        model = _module_to_model(model)
        compile_model = model.compile
    if not callable(compile_model):
        raise TypeError("resolved block %r has no total compile lowering" % name)
    compiled = compile_model(backend=backend, target=target, **compile_options)
    if type(compiled) is not CompiledModel:
        raise TypeError("resolved block compiler must return exact CompiledModel")
    validate_compiled_model_result(compiled)
    authenticate_compiled_model(model, compiled, module_hash=source_module_hash)
    if compiled.target != target or compiled.backend != backend:
        raise ValueError("compiled block route differs from ResolvedSimulationPlan")
    return compiled


def capture_runtime_declarations(problem: Any, detach: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    def resolved(value: Any, family: str) -> Any:
        protocol = getattr(value, "resolve_references", None)
        if not callable(protocol):
            raise TypeError("%s must implement resolve_references" % family)
        return detach(protocol(problem.resolve))

    return (
        tuple(resolved(value, "runtime output") for value in (problem._outputs or [])),
        tuple(resolved(value, "runtime diagnostic")
              for value in (problem._diagnostics or [])),
    )


def capture_field_solvers(problem: Any, detach: Any) -> dict[str, Any]:
    return {name: detach(field.solver)
            for name, field in problem._field_registry.resolved_items(problem.resolve)
            if field.solver is not None}


def prepare_problem_snapshot(problem: Any, time: Any, *, layout: Any, libraries: Any) -> Any:
    from pops.problem._snapshot import prepare_compile_snapshot
    return prepare_compile_snapshot(problem, time, layout=layout, libraries=libraries)


__all__ = [
    "capture_field_solvers", "capture_runtime_declarations", "compile_install_model",
    "compile_install_models", "prepare_problem_snapshot", "resolve_compile_libraries",
]
