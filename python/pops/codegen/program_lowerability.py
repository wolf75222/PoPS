"""Owner-dispatch and schedule guards shared by Program code generation."""
from __future__ import annotations

from typing import Any

from pops.codegen.program_emit_kernels import ProgramValue


_MODEL_OWNER_SENSITIVE_OPS = frozenset(
    {
        "rhs",
        "source",
        "apply",
        "solve_local_linear",
        "solve_local_nonlinear",
        "coupled_rate",
        "solve_coupled_implicit",
        "condensed_coeffs",
        "condensed_rhs",
        "condensed_reconstruct",
        "condensed_energy",
        "max_wave_speed",
    }
)


def all_ops(program: Any) -> Any:
    """Iterate every node recursively, including lazy branch and solver sub-regions."""
    def walk(value: Any) -> Any:
        yield value
        for key in ("cond_block", "body_block", "apply_block", "residual_block",
                    "true_block", "false_block"):
            block = value.attrs.get(key)
            if isinstance(block, (list, tuple)):
                for nested in block:
                    yield from walk(nested)

    for value in program._values:
        yield from walk(value)


def check_model_owner_dispatch(program: Any, model: Any) -> None:
    """Refuse lowering through one physical model when Program blocks have another owner."""
    block_owners = {block.model_owner_path for block in program._block_indices()}
    from pops.codegen.program_models import ProgramModelGraph

    if type(model) is ProgramModelGraph:
        for block in program._block_indices():
            model.owner_for_block(block)
    elif len(block_owners) > 1:
        raise NotImplementedError(
            "multi-model Program lowering requires ProgramModelGraph; Program blocks cover owners "
            "%s. Refusing a representative-model fallback."
            % sorted(str(owner) for owner in block_owners))
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
    from pops.model import OwnerPath
    model_owner = OwnerPath.coerce(model_owner).canonical()
    for value in sensitive:
        value_owner = value.block.model_owner_path
        operator = value.attrs.get("operator_handle")
        operator_owner = getattr(operator, "owner_path", value_owner)
        if value_owner.canonical() != OwnerPath.coerce(operator_owner).canonical():
            raise ValueError(
                "Program node %r has block model owner %s but operator owner %s"
                % (value.name, value_owner, operator_owner)
            )
        if type(model) is ProgramModelGraph:
            model.model_for_owner(value_owner)
            continue
        if value_owner.canonical() != model_owner:
            raise NotImplementedError(
                "multi-model Program lowering needs owner->model dispatch: node %r belongs to %s, "
                "but compile supplied only model %s. Refusing to lower it with the wrong physics."
                % (value.name, value_owner, model_owner)
            )


def check_schedules_lowerable(program: Any) -> None:
    """Reject schedule policies without a semantically valid native lowering."""
    from pops.time.schedule import (
        AcceptedStep, Always, AtEnd, AtStart, Every, When,
    )
    scheduled = {
        value.id: value for value in all_ops(program)
        if value.attrs.get("schedule") is not None
    }
    for consumer in all_ops(program):
        sources = list(consumer.inputs)
        for key in ("true_result", "false_result", "body", "residual", "apply_result"):
            source = consumer.attrs.get(key)
            if isinstance(source, ProgramValue):
                sources.append(source)
        for source in sources:
            scheduled_source = scheduled.get(source.id)
            if scheduled_source is None:
                continue
            source_schedule = scheduled_source.attrs["schedule"]
            if not source_schedule.is_always() and source_schedule.off is None:
                raise ValueError(
                    "scheduled value %r is read by %r but has no explicit OffPolicy; use "
                    "Schedule(trigger, off=Hold()/Skip()/Zero()/AccumulateDt()/Error())"
                    % (scheduled_source.name, consumer.name))
    for endpoint, source in program._commits.items():
        scheduled_source = scheduled.get(source.id)
        if scheduled_source is None:
            continue
        source_schedule = scheduled_source.attrs["schedule"]
        if not source_schedule.is_always() and source_schedule.off is None:
            raise ValueError(
                "scheduled value %r is committed to %r but has no explicit OffPolicy"
                % (scheduled_source.name, endpoint))
    for value in all_ops(program):
        if value.clock != program.clock:
            raise NotImplementedError(
                "compiled Program runtime currently advances only clock %r; node %r belongs to "
                "clock %r. Keep the immutable ProgramGraph for a multi-clock backend or provide "
                "an explicit backend synchronization lowering."
                % (program.clock.name, value.name, value.clock.name))
        schedule = value.attrs.get("schedule")
        if schedule is None:
            continue
        if schedule.clock != program.clock:
            raise NotImplementedError(
                "schedule on node %r belongs to clock %r, but this native runtime advances %r"
                % (value.name, schedule.clock.name, program.clock.name))
        if type(schedule.domain) is not AcceptedStep:
            raise NotImplementedError(
                "schedule domain %s on node %r is typed and preserved, but this runtime only "
                "supports AcceptedStep; Attempt needs StepTransaction, Stage/ClockTick/AMRLevel "
                "need ADC-677, and Event/WallOutput need ConsumerGraph"
                % (type(schedule.domain).__name__, value.name))
        schedule.validate_site(clock=value.clock, point=value.point,
                               where="schedule on node %r" % value.name)
        if type(schedule.trigger) is Always:
            continue
        if type(schedule.trigger) is AtEnd:
            raise NotImplementedError(
                "schedule AtEnd on node %r (op '%s') is not lowerable: a compiled sim.step(dt) "
                "loop never sees an end-of-run signal, so the .so cannot know the last step. Use "
                "AtStart/Every/When on AcceptedStep, or an AtEnd ConsumerGraph hook."
                % (value.name, value.op)
            )
        if type(schedule.trigger) is When:
            condition = schedule.trigger.condition
            if not isinstance(condition, ProgramValue) or condition.vtype != "bool":
                raise NotImplementedError(
                    "schedule when(cond) on node %r lowers only a Program Bool predicate (e.g. "
                    "P.norm2(r) < tol), not a Python callable (ADC-458)." % value.name
                )
        if type(schedule.trigger) not in (Every, AtStart, When):
            raise NotImplementedError("schedule trigger %s is not supported by the native protocol"
                                      % type(schedule.trigger).__name__)


__all__ = ["all_ops", "check_model_owner_dispatch", "check_schedules_lowerable"]
