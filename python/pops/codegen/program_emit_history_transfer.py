"""Compile temporal transfer equations into native callbacks with exact source arity."""
import json


def emit_history_transfer_callback(data):
    """Return a closed C++ lambda; its caller owns source and parameter authentication."""
    from pops.time._history.transfer import validate_transfer_data, transfer_dimensions
    from pops.codegen.exact_transfer_expression import emit_exact_transfer_dag
    from pops.codegen.program_emit_expressions import checked_expression_dag
    nodes = validate_transfer_data(data)
    sources, targets, parameters = transfer_dimensions(data)
    inputs = tuple(("source.times[%d]" % i,) if j == 0 else ("source.intervals[%d]" % i,)
                   for i in range(sources) for j in range(2))
    inputs += (("source.accepted_time",), ("source.temporal_ratio",))
    inputs += tuple(("source.parameters[%d]" % i,) for i in range(len(parameters)))
    body, names = emit_exact_transfer_dag(nodes, inputs)
    valid_body, valid_roots, valid_bad = checked_expression_dag((data["roots"][0],), nodes, inputs)
    roots = tuple(names[index]+"()" for index in data["roots"][1:])
    lines = ["[](const pops::runtime::program::HistoryTransferInput& source) {",
        "  if (source.times.size() != %d || source.intervals.size() != %d || source.parameters.size() != %d)" % (sources, sources, len(parameters)),
        '    throw std::invalid_argument("HistoryTransfer source/parameter width differs from its graph");',
        "  for (const auto parameter : source.parameters)",
        "    if (!std::isfinite(parameter))",
        '      throw std::invalid_argument("HistoryTransfer runtime parameter is not finite");',
        "  {", *("    "+line for line in valid_body),
        "    if (%s || !(%s))" % (valid_bad, valid_roots[0]),
        '    throw std::invalid_argument("HistoryTransfer rejected its temporal validity domain");',
        "  }", *("  "+line for line in body),
        "  pops::runtime::program::HistoryTransferResult result;"]
    for i in range(targets):
        selected = roots[i*(sources+2):(i+1)*(sources+2)]
        lines.extend((
            "  {",
            "    pops::runtime::program::HistoryTransferRow row;",
            "    row.time = %s.value();" % selected[0],
            "    row.interval = %s.value();" % selected[1],
            "    row.exact_weights = {%s};" % ", ".join(selected[2:]),
            "    for (const auto& weight : row.exact_weights) row.weights.push_back(weight.value());",
            "    if (!std::isfinite(row.time) || !std::isfinite(row.interval))",
            '      throw std::invalid_argument("HistoryTransfer produced a nonfinite interval");',
            "    pops::runtime::program::validate_history_transfer_weights(row);",
            "    result.rows.push_back(std::move(row));",
            "  }"))
    lines.extend(("  return result;", "}"))
    return lines


def emit_history_transfer(name, row, *, target):
    data = row["interpolation"]
    if data.get("kind") != "history_transfer":
        return []
    if target != "amr_system":
        raise ValueError("HistoryTransfer remapping requires an AMR layout")
    from pops.time._history.transfer import require_ring_transfer
    require_ring_transfer(data, row["depth"]+1)
    identity = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return ["ctx.register_history_transfer(%s, %s," % (json.dumps(name), json.dumps(identity)),
            *emit_history_transfer_callback(data), ");"]
