"""Authenticate and lower an original residual into the common coupled provider."""
from pops.time.expressions import component_names
from .program_emit_expressions import checked_expression_dag


def product_components(value):
    attrs = value.attrs
    count = attrs["output_count"]
    if attrs.get("product_version") != 1 or type(count) is not int or count < 1:
        raise ValueError("unsupported LocalResidual product contract")
    names = attrs["unknown_names"]
    if tuple(sorted(names)) != tuple(names) or len(set(names)) != count:
        raise ValueError("LocalResidual product unknown packing changed")
    widths = tuple(len(component_names(item)) for item in value.inputs)
    if widths != tuple(attrs["product_widths"]) or any(width < 1 for width in widths):
        raise ValueError("LocalResidual product component Spaces changed")
    seeds = value.inputs[:count]
    blocks = tuple(item.block for item in seeds)
    if tuple(attrs["blocks"]) != blocks or len(set(blocks)) != count:
        raise ValueError("LocalResidual product exact block packing changed")
    if len(attrs["expressions"]) != sum(widths[:count]):
        raise ValueError("LocalResidual product original residual cardinality changed")
    if len(value.inputs) != count + len(attrs["capture_names"]):
        raise ValueError("LocalResidual product capture arity changed")
    return {item.block: tuple(range(width)) for item, width in zip(seeds, widths, strict=False)}


def product_residual_lines(value, variables):
    product_components(value)
    count = value.attrs["output_count"]
    rows, offset = [], 0
    for index, item in enumerate(value.inputs):
        width = len(component_names(item))
        if index < count:
            rows.append(["Ueval[%d]" % (offset + c) for c in range(width)])
            offset += width
        else:
            rows.append(["%sA(index, %d)" % (variables[item.id], c) for c in range(width)])
    reads = value.attrs["product_reads"]
    if any(type(i) is not int or not 0 <= i < len(rows) for i in reads):
        raise ValueError("LocalResidual product read lies outside exact arguments")
    lines, rendered, invalid = checked_expression_dag(
        value.attrs["expressions"], value.attrs["expression_nodes"], [rows[i] for i in reads])
    body = ["      " + line for line in lines]
    body.append("      if (%s) {" % invalid)
    body.extend("        rout[%d] = std::numeric_limits<pops::Real>::quiet_NaN();" % c
                for c in range(offset))
    body.extend(["        return;", "      }"])
    body.extend("      rout[%d] = %s;" % (c, expression)
                for c, expression in enumerate(rendered))
    return body
