"""Owner-dispatch and schedule guards shared by Program code generation."""
from __future__ import annotations

from typing import Any

from pops.codegen.program_emit_kernels import _AUX_OUTPUT_OPS, ProgramValue


_MODEL_OWNER_SENSITIVE_OPS = frozenset(
    {
        "rhs",
        "source",
        "apply",
        "solve_local_linear",
        "solve_local_nonlinear",
        "coupled_rate",
        "condensed_coeffs",
        "condensed_rhs",
        "condensed_reconstruct",
        "condensed_energy",
        "max_wave_speed",
    }
)


def all_ops(program: Any) -> Any:
    """Iterate top-level nodes and the flat sub-blocks carried in their attributes."""
    for value in program._values:
        yield value
        for key in ("cond_block", "body_block", "apply_block", "residual_block"):
            block = value.attrs.get(key)
            if isinstance(block, (list, tuple)):
                yield from block


def check_model_owner_dispatch(program: Any, model: Any) -> None:
    """Refuse lowering through one physical model when Program blocks have another owner."""
    block_owners = {block.model_owner_path for block in program._block_indices()}
    if len(block_owners) > 1:
        raise NotImplementedError(
            "multi-model Program lowering needs owner->model dispatch for module metadata, state "
            "shape and kernels; Program blocks cover owners %s. Refusing a first-model fallback."
            % sorted(str(owner) for owner in block_owners)
        )
    sensitive = [
        value
        for value in all_ops(program)
        if value.op in _MODEL_OWNER_SENSITIVE_OPS and value.block is not None
    ]
    if not sensitive:
        return
    model_owner = getattr(model, "owner_path", None) if model is not None else None
    if model_owner is None and model is not None:
        model_owner = getattr(getattr(model, "_m", None), "owner_path", None)
    if model_owner is None:
        return
    for value in sensitive:
        value_owner = value.block.model_owner_path
        operator = value.attrs.get("operator_handle")
        operator_owner = getattr(operator, "owner_path", value_owner)
        if value_owner != operator_owner:
            raise ValueError(
                "Program node %r has block model owner %s but operator owner %s"
                % (value.name, value_owner, operator_owner)
            )
        if value_owner != model_owner:
            raise NotImplementedError(
                "multi-model Program lowering needs owner->model dispatch: node %r belongs to %s, "
                "but compile supplied only model %s. Refusing to lower it with the wrong physics."
                % (value.name, value_owner, model_owner)
            )


def check_schedules_lowerable(program: Any) -> None:
    """Reject schedule policies without a semantically valid native lowering."""
    for value in all_ops(program):
        schedule = value.attrs.get("schedule")
        if schedule is None or schedule.is_always():
            continue
        if not schedule.so_lowerable():
            raise NotImplementedError(
                "schedule on_end() on node %r (op '%s') is not lowerable: a compiled sim.step(dt) "
                "loop never sees an end-of-run signal, so the .so cannot know the last step. Use "
                "on_start()/every()/when()/subcycle(), or an on_end host hook (ADC-458)."
                % (value.name, value.op)
            )
        if schedule.kind == "when":
            condition = schedule.params.get("cond")
            if not isinstance(condition, ProgramValue) or condition.vtype != "bool":
                raise NotImplementedError(
                    "schedule when(cond) on node %r lowers only a Program Bool predicate (e.g. "
                    "P.norm2(r) < tol), not a Python callable (ADC-458)." % value.name
                )
        if schedule.kind == "subcycle" and value.op not in _AUX_OUTPUT_OPS:
            raise NotImplementedError(
                "schedule subcycle on node %r (op '%s') is lowerable only for a field solve (its "
                "output is the persistent System aux); a scratch-output op sub-cycled has no single "
                "result a downstream node can read (ADC-458). Sub-cycle the field solve, or express "
                "the inner steps explicitly." % (value.name, value.op)
            )


__all__ = ["all_ops", "check_model_owner_dispatch", "check_schedules_lowerable"]
