"""Closed Python scalar algebra evaluated on immutable native AMR towers."""
from pops.codegen.program_emit_expressions import checked_expression_dag
from pops.time.expressions import component_names


def _contract(value):
    if value.attrs.get("schema_version") not in (1, 2) or len(value.attrs.get("expressions", ())) != 1:
        raise ValueError("composite_integral requires one closed scalar expression")
    index = value.attrs.get("support_input")
    if type(index) is not int or not 0 <= index < len(value.inputs):
        raise ValueError("composite_integral lost its exact support")
    support = value.inputs[index]
    if support.op != "state" or support.vtype != "state" or support.block != value.block:
        raise ValueError("composite_integral requires declared current State support")
    level_index = value.attrs.get("level_input")
    if value.attrs["schema_version"] == 1:
        if "level_input" in value.attrs:
            raise ValueError("composite_integral level requires its versioned domain contract")
    elif (type(level_index) is not int or not 0 <= level_index < len(value.inputs)
          or value.inputs[level_index].vtype != "int64"
          or value.inputs[level_index].block not in (None, support.block)):
        raise ValueError("composite_integral requires an owned Int64 level")
    for input_index, item in enumerate(value.inputs):
        if input_index == level_index:
            continue
        if item.vtype == "scalar":
            if item.block is not None and item.block != support.block:
                raise ValueError("composite_integral Scalar belongs to another block")
        elif (item.vtype != "state" or item.op not in {"state", "attempt_start"}
              or item.space != support.space or item.state_ref != support.state_ref
              or item.block != support.block):
            raise ValueError("composite_integral requires immutable qualified State towers")
    if level_index is not None and any(
            node[0] == "input" and node[1] == level_index for node in value.attrs["expression_nodes"]):
        raise ValueError("composite_integral level is a domain selector, not an integrand capture")
    if any(node[0] == "coefficient" for node in value.attrs["expression_nodes"]):
        raise ValueError("composite_integral cannot capture ambient coefficients")


def _helper(value):
    _contract(value)
    rows = [[] if index == value.attrs.get("level_input") else
            ["capture_%d" % index] if item.vtype == "scalar" else
            ["%s(index, %d)" % ("entry" if item.op == "attempt_start" else "current", c)
             for c in range(len(component_names(item)))]
            for index, item in enumerate(value.inputs)]
    temporaries, rendered, invalid = checked_expression_dag(
        value.attrs["expressions"], value.attrs["expression_nodes"], rows)
    return "\n".join([
        "struct ProgramCompositeIntegrand_%d {" % value.id,
        "  pops::FieldView<const pops::Real, pops::kNativeDimension> entry{}, current{};",
        *("  pops::Real capture_%d{};" % i for i, item in enumerate(value.inputs)
          if item.vtype == "scalar"),
        "  POPS_HD pops::Real operator()(const pops::Index<pops::kNativeDimension>& index) const {",
        *("    " + line for line in temporaries),
        "    return (%s) ? std::numeric_limits<pops::Real>::infinity() : %s;" % (invalid, rendered[0]),
        "  }", "};", ""])


def composite_integral_helpers(program):
    from pops.codegen.program_lowerability import all_ops
    return "\n".join(_helper(value) for value in all_ops(program) if value.op == "composite_integral")


def emit_composite_integral(value, variables, lines, *, target, block_index):
    if target != "amr_system":
        raise ValueError("composite_integral requires the native AMR hierarchy")
    _contract(value)
    kernel = "composite_integrand_%d" % value.id
    lines.append("ProgramCompositeIntegrand_%d %s{};" % (value.id, kernel))
    for index, item in enumerate(value.inputs):
        if item.vtype == "scalar":
            lines.append("%s.capture_%d = %s;" % (kernel, index, variables[item.id]))
    variables[value.id] = "composite_integral_%d" % value.id
    level_index = value.attrs.get("level_input")
    domain = "" if level_index is None else ", " + variables[value.inputs[level_index].id]
    lines.append("const pops::Real %s = ctx.composite_integral(%d, %s%s);" %
                 (variables[value.id], block_index, kernel, domain))
