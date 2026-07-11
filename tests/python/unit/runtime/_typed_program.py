"""Small typed-reference helpers shared by runtime unit tests.

The final Program API accepts only a case-owned ``BlockHandle`` plus the
model-owned state ``Handle``.  Tests which exercise Program IR in isolation
still need those real identities; these helpers build the smallest genuine
Module/Problem graph without reviving free-name compatibility.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pops.model import Module
from pops.problem import Problem
from pops.time import Program


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


def add_typed_block(case: Problem, model: Any, block_name: str, state: Any = None):
    """Add one real block and return ``(block_handle, state_handle)``."""
    module = module_of(model)
    block = case.add_block(block_name, module)
    return block, state_handle(module, state)


def typed_program_state(
    name: str,
    *,
    block_name: str = "plasma",
    model: Any = None,
    state: Any = None,
    components: tuple[str, ...] = ("u",),
    bind_operators: bool = True,
):
    """Build a minimal real Program and return ``(P, module, case, block, state, temporal)``."""
    if model is None:
        module = Module("%s_model" % name)
        state = module.state_space("U", components)
    else:
        module = module_of(model)
    case = Problem(name="%s_case" % name)
    block, declaration = add_typed_block(case, module, block_name, state)
    program = Program(name)
    if bind_operators:
        program.bind_operators(module)
    temporal = program.state(block, declaration)
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
    case = Problem(name="%s_case" % name)
    program = Program(name).bind_operators(module)
    endpoints = {}
    for block_name, state in declarations:
        block, declaration = add_typed_block(case, module, block_name, state)
        endpoints[block_name] = program.state(block, declaration)
    return program, module, case, endpoints


def attach_typed_install_plan(
    compiled: Any,
    models: Any,
    *,
    target: str = "system",
    layout: Any = None,
    has_program: bool = True,
):
    """Attach a minimal immutable InstallPlan to an inert compiled-artifact fixture.

    This is for metadata tests that intentionally do not invoke a native compiler.  Block names are
    derived from the Program's authenticated committed state handles, never supplied as a parallel
    free-name table.  ``models`` is either one real ``CompiledModel`` shared by all blocks or an
    exact block-name mapping of ``CompiledModel`` values.
    """
    from pops.codegen._plans import InstallBlock, InstallPlan
    from pops.codegen.loader import CompiledModel
    from pops.model.bind_schema import BindSchema
    from pops.problem._snapshot import AuthoringSnapshot
    from pops.time.references import block_name

    program = getattr(compiled, "program", None)
    commits = program.commits() if program is not None else {}
    names = tuple(dict.fromkeys(block_name(ref.block_ref) for ref in commits))
    if not names:
        raise ValueError("typed InstallPlan fixture requires at least one committed Program state")
    by_name = dict(models) if isinstance(models, Mapping) else {name: models for name in names}
    if set(by_name) != set(names):
        raise ValueError("typed InstallPlan models must match the Program's committed blocks")
    if any(not isinstance(model, CompiledModel) for model in by_name.values()):
        raise TypeError("typed InstallPlan fixtures require exact CompiledModel values")

    # Recreate the compiler-side parameter schema before sealing the inert artifact.  The Program's
    # canonical block handles provide the exact case/model owner names, while CompiledModel.params
    # still contains declarations at this pre-seal point.  This mirrors public pops.compile without
    # letting inspection fall back to CompiledModel.params after the artifact boundary.
    state_refs = tuple(commits)
    block_refs = {block_name(ref.block_ref): ref.block_ref for ref in state_refs}
    case_owner = next(iter(block_refs.values())).owner_path.nodes[0].name
    schema_problem = Problem(name=case_owner)
    for name in names:
        model = by_name[name]
        model_owner = block_refs[name].model_owner_path.nodes[-1].name
        schema_module = Module(model_owner)
        for declaration in model.params.values():
            schema_module.param(declaration)
        schema_problem.add_block(name, schema_module)
    schema = BindSchema.from_problem(schema_problem)

    snapshot = AuthoringSnapshot({
        "kind": "typed-runtime-test-artifact",
        "program_hash": program._ir_hash(),
        "target": target,
        "blocks": names,
    })
    compiled.install_plan = InstallPlan(
        snapshot_hash=snapshot.hash,
        target=target,
        layout=layout,
        blocks=tuple(InstallBlock(name, by_name[name], None) for name in names),
        bind_schema=schema,
        field_solvers={},
        outputs=(),
        diagnostics=(),
        has_program=has_program,
    )
    compiled.bind_schema = schema
    compiled._problem_snapshot = snapshot
    for model in dict.fromkeys(by_name.values()):
        model.bind_schema = schema
        model._seal()
    compiled._seal()
    return compiled


__all__ = [
    "add_typed_block", "module_of", "state_handle", "typed_program_state",
    "typed_program_states", "attach_typed_install_plan",
]
