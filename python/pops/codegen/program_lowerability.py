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


def _check_amr_flux_weights(program: Any) -> None:
    """Prove every conservative contribution reaches a commit as exact ``weight * dt * flux``."""
    from pops.codegen.program_emit_kernels import _coeff_metadata_terms

    values = list(all_ops(program))
    stored_histories: dict[str, list[Any]] = {}
    for value in values:
        if value.op == "store_history":
            stored_histories.setdefault(value.attrs["history"], []).append(value.inputs[0])

    unknown = object()
    powers: dict[int, object | frozenset[int]] = {}
    alias_first_input = frozenset({
        "synchronize", "solve_fields", "solve_fields_from_blocks", "store_history",
        "fill_boundary", "project", "solve_outcome", "acceptance_guard",
    })

    def shifted(source: object | frozenset[int], coefficient: Any) -> object | frozenset[int]:
        terms = _coeff_metadata_terms(coefficient)
        if not terms:
            return frozenset()
        if source is unknown:
            return unknown
        assert isinstance(source, frozenset)
        return frozenset(left + right for left in source for right, _, _ in terms)

    def merged(items: list[object | frozenset[int]]) -> object | frozenset[int]:
        if any(item is unknown for item in items):
            return unknown
        result: set[int] = set()
        for item in items:
            assert isinstance(item, frozenset)
            result.update(item)
        return frozenset(result)

    # Histories create cross-step edges, so solve the finite power-set equations to a fixed point.
    for _ in range(len(values) + 1):
        changed = False
        for value in values:
            prior = powers.get(value.id, frozenset())
            if value.op == "rhs":
                current: object | frozenset[int] = (
                    frozenset({0}) if value.attrs.get("flux", True) else frozenset())
            elif value.op == "history":
                current = merged([
                    powers.get(source.id, frozenset())
                    for source in stored_histories.get(value.attrs["history"], ())
                ])
            elif value.op == "linear_combine":
                current = merged([
                    shifted(powers.get(source.id, frozenset()), coefficient)
                    for source, coefficient in zip(
                        value.inputs, value.attrs["coeffs"], strict=True)
                ])
            elif value.op in alias_first_input and value.inputs:
                current = powers.get(value.inputs[0].id, frozenset())
            else:
                dependencies = list(value.inputs)
                for key in ("true_result", "false_result", "body", "residual", "apply_result"):
                    dependency = value.attrs.get(key)
                    if isinstance(dependency, ProgramValue):
                        dependencies.append(dependency)
                inherited = merged([
                    powers.get(source.id, frozenset()) for source in dependencies
                ])
                current = unknown if inherited is unknown or inherited else frozenset()
            if current != prior:
                powers[value.id] = current
                changed = True
        if not changed:
            break
    else:
        raise ValueError("AMR conservative flux-weight proof did not converge")

    for endpoint, source in program._commits.items():
        proof = powers.get(source.id, frozenset())
        if proof is unknown:
            raise ValueError(
                "AMR conservative commit %r crosses an operation without exact flux-weight "
                "propagation; refusing artifact creation" % endpoint)
        assert isinstance(proof, frozenset)
        if proof and proof != frozenset({1}):
            raise ValueError(
                "AMR conservative commit %r has dt powers %s; every physical flux must reach "
                "the accepted ledger as an exactly proved weight * dt * flux"
                % (endpoint, sorted(proof)))


def check_schedules_lowerable(program: Any, *, target: str | None = None) -> None:
    """Reject schedule policies without a semantically valid native lowering."""
    from pops.codegen.program_emit_schedule import _lower_schedule_ir
    from pops.time.schedule import Schedule
    scheduled = {
        value.id: value for value in all_ops(program)
        if value.attrs.get("schedule") is not None
    }
    for value in scheduled.values():
        schedule = value.attrs["schedule"]
        if not isinstance(schedule, Schedule):
            raise TypeError(
                "schedule on node %r must implement the Schedule interface; got %s"
                % (value.name, type(schedule).__name__))
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
    if target == "amr_system":
        _check_amr_flux_weights(program)
    temporal_clocks = {row["id"] for row in program.temporal_manifest()["clocks"]}
    for value in all_ops(program):
        if value.clock.qualified_id not in temporal_clocks:
            raise ValueError(
                "node %r belongs to a clock absent from the temporal execution schedule"
                % value.name)
        schedule = value.attrs.get("schedule")
        if schedule is None:
            continue
        if schedule.clock != value.clock:
            raise ValueError(
                "schedule on node %r belongs to clock %r, not the node clock %r"
                % (value.name, schedule.clock.name, value.clock.name))
        schedule.validate_site(clock=value.clock, point=value.point,
                               where="schedule on node %r" % value.name)
        lowering = _lower_schedule_ir(value, schedule)
        from pops.time.schedule import ScheduleTimeline
        if target == "system" and lowering.domain.timeline is ScheduleTimeline.AMR_LEVEL:
            raise NotImplementedError(
                "AMRLevel schedule on node %r requires target='amr_system'" % value.name)
        if target == "amr_system" and schedule.needs_cache():
            raise NotImplementedError(
                "scheduled AMR node %r requires a persistent hierarchy value cache; Hold and "
                "AccumulateDt remain refused before artifact creation, while Skip/Zero/Error and "
                "domain-only schedules execute on the AMR clock provider" % value.name)


__all__ = ["all_ops", "check_model_owner_dispatch", "check_schedules_lowerable"]
