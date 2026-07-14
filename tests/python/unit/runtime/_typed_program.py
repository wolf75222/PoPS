"""Small typed-reference helpers shared by runtime unit tests.

The final Program API accepts only the case-qualified ``block[state]`` handle. Tests which exercise
Program IR in isolation
still need those real identities; these helpers build the smallest genuine
Module/Case graph without reviving free-name compatibility.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pops.model import Module
from pops.problem import Case
from pops.time import Program
from tests.python.support.layout_plan import resolved_layout_contract


_PROGRAM_CONTEXTS: dict[int, tuple[Case, Module, Any, dict[str, Any]]] = {}


def module_of(model: Any) -> Module:
    """Return the operator-first Module which owns a model's declarations."""
    module = getattr(model, "module", model)
    if not isinstance(module, Module):
        raise TypeError("typed runtime-test model must expose a pops.model.Module")
    return module


def state_handle(module: Module, state: Any = None) -> Any:
    """Return one declared state Handle, requiring an explicit choice if ambiguous."""
    if state is not None:
        if isinstance(state, str):
            state = module.state_spaces()[state]
        return module.state_handle(state)
    spaces = module.state_spaces()
    if len(spaces) != 1:
        raise ValueError("state must be selected explicitly for a multi-state Module")
    return module.state_handle(next(iter(spaces.values())))


def add_typed_block(case: Case, model: Any, block_name: str, state: Any = None):
    """Add one real block and return ``(block_handle, state_handle)``."""
    module = module_of(model)
    declaration = state_handle(module, state)
    block = case.block(block_name, module, states=(declaration,))
    return block, declaration


def typed_program_state(
    name: str,
    *,
    block_name: str = "plasma",
    model: Any = None,
    state: Any = None,
    components: tuple[str, ...] = ("u",),
):
    """Build a minimal real Program and return ``(P, module, case, block, state, temporal)``."""
    if model is None:
        module = Module("%s_model" % name)
        state = module.state_space("U", components)
    else:
        module = module_of(model)
    case = Case(name="%s_case" % name)
    block, declaration = add_typed_block(case, module, block_name, state)
    program = Program(name)
    temporal = program.state(block[declaration])
    _PROGRAM_CONTEXTS[id(program)] = (case, module, declaration, {})
    return program, module, case, block, declaration, temporal


def typed_program_states(
    name: str,
    model: Any,
    declarations: tuple[tuple[str, Any], ...],
):
    """Build one Program with several case-qualified state endpoints.

    ``declarations`` contains ``(block_name, state_space_or_handle)`` pairs.  The returned endpoint
    mapping is keyed by block name and contains genuine ``TimeState`` values; no free block/state
    spelling enters the Program IR.
    """
    module = module_of(model)
    case = Case(name="%s_case" % name)
    program = Program(name)
    endpoints = {}
    for block_name, state in declarations:
        block, declaration = add_typed_block(case, module, block_name, state)
        endpoints[block_name] = program.state(block[declaration])
    return program, module, case, endpoints


def typed_field(program: Program, name: str = "potential") -> Any:
    """Build one genuine Case-owned callable FieldHandle for a runtime IR fixture."""
    try:
        case, module, declaration, fields = _PROGRAM_CONTEXTS[id(program)]
    except KeyError as exc:
        raise ValueError("typed runtime field fixture requires typed_program_state") from exc
    existing = fields.get(name)
    if existing is not None:
        return existing

    from pops.descriptors import Descriptor
    from pops.fields import FieldDiscretization, FieldOperator
    from pops.math import ValueExpr
    from pops.math import laplacian
    from pops.model import Signature

    field_module = Module("typed_runtime_field_%s" % name)
    field_space = field_module.field_space(name, (name,))
    provider = field_module.operator(
        name=name + "_provider",
        signature=Signature((declaration.space,), field_space),
        kind="field_operator",
        expr=name + "_provider",
    )
    field_block = case.block("typed_field_%s" % name, field_module)
    unknown = field_block[field_module.field_handle(field_space)]
    state_block = next(
        block for block in case.blocks().values()
        if block.model_owner_path == module.owner_path
    )
    state_ref = state_block[declaration]

    class _Method(Descriptor):
        category = "field_method"

        def to_data(self):
            return {"type": "unit-second-order"}

    class _Solver(Descriptor):
        category = "elliptic_solver"

        def to_data(self):
            return {"type": "unit-krylov"}

    operator = FieldOperator(
        name,
        unknown=unknown,
        equation=-laplacian(ValueExpr(unknown)) == ValueExpr(state_ref),
        providers=provider,
    )
    field = case.field(
        operator,
        FieldDiscretization(method=_Method(), boundaries=(), solver=_Solver()),
    )
    fields[name] = field
    return field


def solve_field(
    program: Program,
    state: Any,
    *,
    field: Any = None,
    name: str | None = None,
    action: Any = None,
) -> Any:
    """Call the final field handle and explicitly consume its fallible outcome."""
    from pops.time import FailRun

    selected = typed_field(program) if field is None else field
    return selected(state, name=name).consume(
        action=FailRun() if action is None else action)


def typed_compiled_artifact(
    compiled: Any,
    models: Any,
    *,
    target: str = "system",
    layout: Any = None,
    has_program: bool = True,
):
    """Wrap inert compiled components in the exact compile-phase artifact.

    This is for metadata tests that intentionally do not invoke a native compiler.  Block names are
    derived from the Program's authenticated committed state handles, never supplied as a parallel
    free-name table.  ``models`` is either one real ``CompiledModel`` shared by all blocks or an
    exact block-name mapping of ``CompiledModel`` values.  The helper returns a
    ``CompiledSimulationArtifact``; it never mutates a component with bind/install phase state.
    """
    from pops.codegen._plans import ResolvedBlock, ResolvedSimulationPlan
    from pops.codegen._compiled_model_identity import model_compile_identity
    from pops.codegen._compiled_artifact import (
        CompiledBlockArtifact,
        CompiledSimulationArtifact,
    )
    from pops.codegen.loader import CompiledModel
    from pops.model.bind_schema import BindSchema
    from pops.problem._snapshot import AuthoringSnapshot
    from pops.time.references import block_name

    program = getattr(compiled, "program", None)
    commits = program.commits() if program is not None else {}
    names = tuple(dict.fromkeys(block_name(ref.block_ref) for ref in commits))
    if not names:
        raise ValueError("typed artifact fixture requires at least one committed Program state")
    by_name = dict(models) if isinstance(models, Mapping) else {name: models for name in names}
    if set(by_name) != set(names):
        raise ValueError("typed artifact models must match the Program's committed blocks")
    if any(not isinstance(model, CompiledModel) for model in by_name.values()):
        raise TypeError("typed artifact fixtures require exact CompiledModel values")

    # Recreate the compiler-side parameter schema before sealing the inert artifact.  The Program's
    # canonical block handles provide the exact case/model owner names, while CompiledModel.params
    # still contains declarations at this pre-seal point.  This mirrors public pops.compile without
    # letting inspection fall back to CompiledModel.params after the artifact boundary.
    state_refs = tuple(commits)
    block_refs = {block_name(ref.block_ref): ref.block_ref for ref in state_refs}
    state_identities = {
        block_name(ref.block_ref): ref.qualified_id for ref in state_refs
    }
    case_owner = next(iter(block_refs.values())).owner_path.nodes[0].name
    schema_problem = Case(name=case_owner)
    schema_modules = {}
    for name in names:
        model = by_name[name]
        model_owner = block_refs[name].model_owner_path.nodes[-1].name
        schema_module = Module(model_owner)
        for declaration in model.params.values():
            schema_module.param(declaration)
        schema_problem.block(name, schema_module)
        schema_modules[name] = schema_module
    schema = BindSchema.from_problem(schema_problem)

    snapshot = AuthoringSnapshot({
        "kind": "typed-runtime-test-artifact",
        "program_hash": program._ir_hash(),
        "target": target,
        "blocks": names,
    })
    backend = next(iter(by_name.values())).backend
    layout_value = layout if layout is not None else {"kind": target}
    layout_plan, layout_coverage = resolved_layout_contract(
        layout_value, target=target, block_names=names)
    plan = ResolvedSimulationPlan(
        snapshot=snapshot,
        target=target,
        backend=backend,
        layout=layout_value,
        layout_plan=layout_plan,
        layout_targets={
            row.handle.qualified_id: target for row in layout_plan.layouts
        },
        time=program if has_program else None,
        blocks=tuple(
            ResolvedBlock(
                name, schema_modules[name], None, backend, ("U",),
                (state_identities[name],))
            for name in names
        ),
        bind_schema=schema,
        compile_values=schema.resolve_compile(),
        field_plans={},
        libraries=(),
        requirements={},
        capabilities={"cpu": True, "amr": target == "amr_system"},
        lowering_coverage=layout_coverage,
    )
    compiled.bind_schema = schema
    for name, model in by_name.items():
        model.bind_schema = schema
        if getattr(model, "definition_identity", None) is None:
            model.definition_identity = model_compile_identity(schema_modules[name])
    return CompiledSimulationArtifact(
        plan=plan,
        program=compiled if has_program else None,
        blocks=tuple(
            CompiledBlockArtifact(name, by_name[name], None, ("U",))
            for name in names
        ),
    )


__all__ = [
    "add_typed_block", "module_of", "state_handle", "typed_program_state",
    "typed_program_states", "typed_compiled_artifact",
]
