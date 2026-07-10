"""Compile-snapshot instance assembly helpers for codegen orchestration."""
from __future__ import annotations

from typing import Any


def assemble_instances(
    problem: Any,
    initial: Any,
    block_specs: Any = None,
    models: Any = None,
) -> dict:
    """Build the install mapping from compile-time block specs and initial state."""
    if problem is None and block_specs is None:
        raise TypeError(
            "pops.bind: the compiled handle carries no problem assembly "
            "(was it produced by pops.compile?)"
        )
    declared = set(block_specs) if block_specs is not None else set(problem._blocks)
    unknown = sorted(set(initial) - declared)
    if unknown:
        raise ValueError(
            "pops.bind: initial state for unknown block(s) %s; declared blocks: %s"
            % (unknown, sorted(declared))
        )
    instances = {}
    if block_specs is not None:
        for name, snapshot in block_specs.items():
            entry = {"model": snapshot["model"], "spatial": snapshot["spatial"]}
            if name in initial:
                entry["initial"] = initial[name]
            instances[name] = entry
        return instances
    from pops.codegen.orchestration import _resolve_problem_model

    for name, spec in problem._blocks.items():
        if models is not None:
            if name not in models:
                raise ValueError(
                    "pops.bind: block %r has no compiled model on the handle; the AMR handle "
                    "must carry one CompiledModel per block (was it produced by pops.compile?)"
                    % name
                )
            model = models[name]
        else:
            model = _resolve_problem_model(spec["model"])
        entry = {"model": model, "spatial": spec["spatial"]}
        if name in initial:
            entry["initial"] = initial[name]
        instances[name] = entry
    return instances


def check_problem_not_mutated(problem: Any, block_specs: Any) -> None:
    """Reject a live Problem whose block set differs from its compiled snapshot."""
    if block_specs is None or problem is None:
        return
    live = set(getattr(problem, "_blocks", {}) or {})
    frozen = set(block_specs)
    if live != frozen:
        added = sorted(live - frozen)
        removed = sorted(frozen - live)
        raise ValueError(
            "pops.bind: the Problem was mutated after pops.compile (blocks changed: added=%s "
            "removed=%s); a compiled artifact is frozen at compile time and is not affected by "
            "a later Problem mutation -- recompile the Problem (pops.compile(...)) before "
            "pops.bind(...)." % (added, removed)
        )


def problem_field_solvers(problem: Any) -> dict:
    """Return the field-name to solver mapping captured by compile."""
    if problem is None:
        return {}
    return {
        name: field.solver
        for name, field in problem._fields.items()
        if field.solver is not None
    }


__all__ = ["assemble_instances", "check_problem_not_mutated", "problem_field_solvers"]
