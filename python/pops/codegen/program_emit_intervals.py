"""Native interval authority using existing ProgramContext lifetime services."""
import json


def emit_authenticated_history_read(program, value, var, lines, *, target):
    """Synchronize failures before any HistoryInterval collective is entered."""
    from .program_lowerability import all_ops
    if not any(node.op == "history_interval_begin" and node.inputs[0].id == value.id
               for node in all_ops(program)):
        return False
    if target not in ("system", "amr_system") or "ncomp" in value.attrs:
        raise NotImplementedError("HistoryInterval requires a strict native full-state history read")
    key, name, lag = value.id, json.dumps(value.attrs["history"]), int(value.attrs["lag"])
    lines += [
        f"pops::MultiFab<pops::kNativeDimension>* history_interval_carrier_{key} = nullptr;",
        f"std::exception_ptr history_interval_read_error_{key};",
        f"try {{ history_interval_carrier_{key} = &ctx.history({name}, {lag}); }}",
        f"catch (...) {{ history_interval_read_error_{key} = std::current_exception(); }}",
        f"if (pops::all_reduce_max(history_interval_read_error_{key} ? 1L : 0L, ctx.prepared_execution_lane()) != 0) {{",
        f"  if (history_interval_read_error_{key}) std::rethrow_exception(history_interval_read_error_{key});",
        '  throw std::runtime_error("HistoryInterval read failed on another rank");',
        "}",
        f"auto& {var[key]} = *history_interval_carrier_{key};",
    ]
    return True


def emit_interval(program, value, var, lines, *, target, block_idx=None):
    if target not in ("system", "amr_system"):
        raise NotImplementedError("Interval requires an authenticated native invocation")
    from pops.intervals import validate_intervals
    validate_intervals(program)
    key = value.id
    if value.op.startswith("history_interval_"):
        return _emit_history_interval(program, value, var, lines, block_idx=block_idx, target=target)
    if value.op == "interval_begin":
        clock = json.dumps(value.attrs["clock"])
        lines += [
            f"[[maybe_unused]] const auto* interval_owner_{key} = &ctx;",
            f"pops::runtime::program::PreparedResourceAttempt interval_attempt_{key};",
            f"pops::runtime::multiblock::BoundaryEvaluationPoint interval_point_{key};",
            f"std::exception_ptr interval_error_{key};",
            "try {",
            "  const auto interval_scope = ctx.stage_evaluation_scope(0, 1);",
            f"  interval_point_{key} = ctx.boundary_evaluation_point(0);",
            f"  interval_attempt_{key} = ctx.resource_attempt();",
            f"  if (!interval_attempt_{key}.visible() || interval_point_{key}.clock != {clock}",
            f"      || !std::isfinite(interval_point_{key}.physical_time)",
            f"      || !std::isfinite(interval_point_{key}.dt) || !(interval_point_{key}.dt > 0))",
            '    throw std::logic_error("Interval has no matching live clock/attempt");',
            f"}} catch (...) {{ interval_error_{key} = std::current_exception(); }}",
            f"if (pops::all_reduce_max(interval_error_{key} ? 1L : 0L, ctx.prepared_execution_lane()) != 0) {{",
            f"  if (interval_error_{key}) std::rethrow_exception(interval_error_{key});",
            '  throw std::runtime_error("Interval issuance failed on another rank");',
            "}",
            f"const pops::Real interval_begin_{key} = interval_point_{key}.physical_time;",
        ]
        var[key] = f"interval_begin_{key}"
    elif value.op == "interval_end":
        begin, duration = value.inputs
        var[key] = f"interval_end_{key}"
        lines.append(f"const pops::Real {var[key]} = {var[begin.id]} + {var[duration.id]};")
    else:
        begin, duration, end = value.inputs
        from pops.time._step.strategy import ComputedDt
        if type(program._step_strategy) is ComputedDt:
            frontier = [v for v in program._values if v.op == "reached_duration"]
            if len(frontier) != 1 or tuple(v.id for v in frontier[0].inputs) != (duration.id,):
                raise ValueError("reached Interval must use the exact ComputedDt frontier SSA")
            expected = var[duration.id]
        else:
            expected = "dt"
        root = begin.id
        d, b, e = var[duration.id], var[begin.id], var[end.id]
        lines += [f"bool interval_valid_{key} = false;", "try {",
            "  const auto interval_scope = ctx.stage_evaluation_scope(0, 1);",
            f"  interval_valid_{key} = interval_owner_{root} == &ctx && interval_attempt_{root}.visible()",
            f"    && interval_attempt_{root}.same_attempt(ctx.resource_attempt())",
            f"    && interval_point_{root} == ctx.boundary_evaluation_point(0)",
            f"    && std::isfinite({b}) && std::isfinite({d}) && std::isfinite({e})",
            f"    && {d} > 0 && {e} > {b} && {e} == {b} + {d} && {d} == {expected};",
            "} catch (...) { interval_valid_%d = false; }" % key,
            f"const bool interval_checked_{key} = pops::all_reduce_max(interval_valid_{key} ? 0L : 1L, ctx.prepared_execution_lane()) == 0;",
        ]
        var[key] = f"interval_checked_{key}"


def _emit_history_interval(program, value, var, lines, *, block_idx, target):
    key = value.id
    if value.op == "history_interval_begin":
        source = value.inputs[0]
        name, lag = source.attrs["history"], int(source.attrs["lag"])
        owner = program._history_blocks.get(name)
        owner_index = (block_idx or {}).get(owner)
        state_ref = program._history_state_refs.get(name)
        if owner_index is None or state_ref is None:
            raise ValueError("HistoryInterval requires exact state and owner provenance")
        from pops.codegen.program_history_identity import history_space_identity
        row = next(row for row in program.temporal_manifest()["histories"] if row["name"] == name)
        interpolation = json.dumps(row["interpolation"], sort_keys=True, separators=(",", ":"))
        quoted = [json.dumps(x) for x in (name, state_ref.qualified_id,
                  history_space_identity(program, name), row["clock"], interpolation)]
        n, state, space, clock, interp = quoted
        storage_key = n if target == "system" else f"history_interval_storage_{key}"
        lines += [f"std::string {storage_key};"] if target == "amr_system" else []
        lines += [
            f"pops::runtime::program::PreparedResourceAttempt history_interval_attempt_{key};",
            f"pops::runtime::multiblock::BoundaryEvaluationPoint history_interval_point_{key};",
            f"pops::runtime::program::HistorySampleIdentity history_interval_sample_{key};",
            f"std::string history_interval_snapshot_{key};",
            f"pops::Real history_interval_begin_{key} = 0;",
            f"pops::Real history_interval_duration_{key} = 0;",
            f"auto history_interval_proof_{key} = [&]() {{",
            "  const auto& history = ctx.runtime_state().hist_;",
            "  pops::ExactContractBuilder exact;",
            f"  const bool cold = pops::runtime::program::interaction_history_selected(history, {storage_key}, {lag},",
            f"      ctx.sys_block({owner_index}), {state}, {space}, {clock}, {interp}, exact);",
            f"  if (cold || &{var[source.id]} != &history.histories.at({storage_key}).at({lag})",
            f"      || (history.validated_samples({storage_key}).at({lag}).kind != pops::runtime::program::HistorySampleKind::Publication",
            f"          && history.validated_samples({storage_key}).at({lag}).kind != pops::runtime::program::HistorySampleKind::Reconstruction))",
            '    throw std::logic_error("HistoryInterval requires an exact retained publication");',
            "  return std::move(exact).release();",
            "};",
            f"std::exception_ptr history_interval_error_{key};",
            "try {",
            "  const auto interval_scope = ctx.stage_evaluation_scope(0, 1);",
            f"  history_interval_point_{key} = ctx.boundary_evaluation_point(0);",
            f"  history_interval_attempt_{key} = ctx.resource_attempt();",
            f"  if (!history_interval_attempt_{key}.visible() || history_interval_point_{key}.clock != {clock})",
            '    throw std::logic_error("HistoryInterval has no matching live clock/attempt");',
            *([f"  {storage_key} = ctx.history_storage_key({n}, {owner_index});"] if target == "amr_system" else []),
            f"  history_interval_snapshot_{key} = history_interval_proof_{key}();",
            f"  history_interval_sample_{key} = ctx.runtime_state().hist_.validated_samples({storage_key}).at({lag});",
            # HistorySampleIdentity encodes binary64 independently of the selected
            # Real precision. Reject lossy conversion before any consumer or peer
            # leaves the collective issuance phase.
            f"  const double history_interval_begin64_{key} = std::bit_cast<double>(history_interval_sample_{key}.start_bits);",
            f"  const double history_interval_duration64_{key} = std::bit_cast<double>(history_interval_sample_{key}.interval_bits);",
            f"  if (!std::isfinite(history_interval_begin64_{key}) || !std::isfinite(history_interval_duration64_{key})",
            f"      || history_interval_begin64_{key} < static_cast<double>(std::numeric_limits<pops::Real>::lowest())",
            f"      || history_interval_begin64_{key} > static_cast<double>(std::numeric_limits<pops::Real>::max())",
            f"      || history_interval_duration64_{key} > static_cast<double>(std::numeric_limits<pops::Real>::max()))",
            '    throw std::logic_error("HistoryInterval publication exceeds Real range");',
            f"  history_interval_begin_{key} = static_cast<pops::Real>(history_interval_begin64_{key});",
            f"  history_interval_duration_{key} = static_cast<pops::Real>(history_interval_duration64_{key});",
            f"  if (!std::isfinite(history_interval_begin_{key}) || !std::isfinite(history_interval_duration_{key})",
            f"      || std::bit_cast<std::uint64_t>(static_cast<double>(history_interval_begin_{key})) != history_interval_sample_{key}.start_bits",
            f"      || std::bit_cast<std::uint64_t>(static_cast<double>(history_interval_duration_{key})) != history_interval_sample_{key}.interval_bits)",
            '    throw std::logic_error("HistoryInterval publication is not exactly representable in Real");',
            f"}} catch (...) {{ history_interval_error_{key} = std::current_exception(); }}",
            f"if (pops::all_reduce_max(history_interval_error_{key} ? 1L : 0L, ctx.prepared_execution_lane()) != 0) {{",
            f"  if (history_interval_error_{key}) std::rethrow_exception(history_interval_error_{key});",
            '  throw std::runtime_error("HistoryInterval issuance failed on another rank");',
            "}",
            f"if (!pops::all_ranks_agree_exact_ordered_byte_pairs({{{{\"history-interval\", history_interval_snapshot_{key}}}}}, ctx.prepared_execution_lane()))",
            '  throw std::logic_error("HistoryInterval publication differs across ranks");',
        ]
        var[key] = f"history_interval_begin_{key}"
    elif value.op == "history_interval_duration":
        var[key] = f"history_interval_duration_{value.inputs[0].id}"
    elif value.op == "history_interval_check":
        begin, duration, end, _source = value.inputs
        root = begin.id
        b, d, e = var[begin.id], var[duration.id], var[end.id]
        lines += [f"bool history_interval_valid_{key} = false;", "try {",
            "  const auto interval_scope = ctx.stage_evaluation_scope(0, 1);",
            f"  history_interval_valid_{key} = history_interval_attempt_{root}.visible()",
            f"    && history_interval_attempt_{root}.same_attempt(ctx.resource_attempt())",
            f"    && history_interval_point_{root} == ctx.boundary_evaluation_point(0)",
            f"    && history_interval_snapshot_{root} == history_interval_proof_{root}()",
            f"    && std::isfinite({b}) && std::isfinite({d}) && std::isfinite({e})",
            f"    && {d} > 0 && {e} > {b} && {e} == {b} + {d};",
            f"}} catch (...) {{ history_interval_valid_{key} = false; }}",
            f"const bool history_interval_checked_{key} = pops::all_reduce_max(history_interval_valid_{key} ? 0L : 1L, ctx.prepared_execution_lane()) == 0;",
        ]
        var[key] = f"history_interval_checked_{key}"
    else:
        raise ValueError("unknown historical interval operation")
