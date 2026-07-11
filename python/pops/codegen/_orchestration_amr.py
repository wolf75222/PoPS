"""AMR compiler consuming only an immutable :class:`ResolvedPlan`.

Each block is compiled before bind.  If a whole-system Program exists, its AMR entrypoint is built
as a ``CompiledProblem``; otherwise the first block loader is the public artifact.  Both shapes carry
the same immutable ``InstallPlan`` and neither retains ``Problem`` or any registry.
"""
from __future__ import annotations

from typing import Any

from pops.codegen.orchestration import (
    _attach_install_plan,
    _attach_problem_snapshot,
    _compile_install_models,
)


def _compile_amr(plan: Any, backend: str, **kwargs: Any) -> Any:
    """Compile an AMR ``ResolvedPlan`` into one sealed public artifact."""
    block_models = _compile_install_models(plan, backend, kwargs)

    if plan.time is not None:
        compiled = _compile_program(plan, backend, kwargs)
        _attach_install_plan(compiled, plan, block_models, has_program=True)
    else:
        compiled = block_models[plan.blocks[0].name]
        _attach_install_plan(compiled, plan, block_models, has_program=False)

    _attach_problem_snapshot(compiled, plan.snapshot)
    return compiled


def _compile_program(plan: Any, backend: str, kwargs: Any) -> Any:
    """Build the optional whole-system Program with the same snapshot/cache authority."""
    from pops.codegen.compile_drivers import compile_problem

    program_kwargs = {
        key: value for key, value in kwargs.items()
        if key in ("so_path", "force", "cxx", "include", "std", "debug", "libraries")
    }
    return compile_problem(
        model=plan.first_model,
        time=plan.time,
        backend=backend,
        target="amr_system",
        problem_snapshot=plan.snapshot,
        **program_kwargs,
    )


__all__ = ["_compile_amr"]
