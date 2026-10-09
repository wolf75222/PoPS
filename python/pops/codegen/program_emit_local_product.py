"""Authenticate and lower an original residual into the common coupled provider."""
import json
from pops.time.expressions import component_names
from .program_emit_expressions import checked_expression_dag


def product_operator_environments(value, model, provider_plans, block_indices, variables):
    """Keep each physical call's model, parameter table and frozen provider port distinct."""
    from .program_models import model_for_node
    from .program_emit_model_kernels import _residual_term_exprs, _provider_binding
    from .program_emit_kernels import _model_impl, _cell_locals, _has_runtime_param
    from .program_emit_kernels import program_provider_consumer_qid
    nodes = value.attrs.get("residual_block", ())
    origins = {nodes[position].id: item for position, item in
               zip(value.attrs.get("product_argument_positions", ()), value.inputs, strict=False)}
    frozen_states = {nodes[position].id for position, item in
                     zip(value.attrs.get("product_argument_positions", ())[value.attrs["output_count"]:],
                         value.inputs[value.attrs["output_count"]:], strict=True)
                     if item.vtype == "state"}
    environments, fab_setup, preparations = {}, [], []
    provider_sources = {}
    for node in nodes:
        if node.op not in ("source", "apply"):
            if node.id not in origins and node.inputs:
                origins[node.id] = origins.get(node.inputs[0].id)
            continue
        authority = model_for_node(model, node)
        impl = _model_impl(authority)
        roots = _residual_term_exprs(impl, node)
        impl.assign_runtime_indices()
        block = block_indices[node.block]
        qid = program_provider_consumer_qid(authority, node.id, node.block)
        binding = _provider_binding(impl, roots, provider_plans, qid)
        origin = origins.get(node.inputs[0].id)
        if origin is None or origin.vtype != "state":
            raise ValueError("local product operator lacks its exact State storage origin")
        origins[node.id] = origin
        if binding["count"] and node.inputs[0].id not in frozen_states:
            raise NotImplementedError(
                "LocalResidual product provider evaluation at a Newton candidate or derived State "
                "is not implemented; use an exact frozen State capture for frozen provider reads")
        # Native provider storage is not a cache indexed by arbitrary Newton
        # iterates or multiple temporal states. Do not alias two publications.
        for key, _ in provider_plans._plans[qid]:
            prior = provider_sources.setdefault((node.block, key), (origin.id, node.point))
            if prior != (origin.id, node.point):
                raise NotImplementedError(
                    "local product needs distinct frozen provider snapshots for differing State/point reads")
        if binding["count"]:
            preparations.append((node, qid, block, variables[origin.id]))
        provider_name = "product_providers_%d" % node.id
        fab_setup.append("  const auto %s = ctx.template provider_values_view<%d>(%s, %d, li);"
                         % (provider_name, binding["count"], json.dumps(qid), block))
        setup = ["  const auto providers = %s;" % provider_name]
        if _has_runtime_param(roots):
            name = "product_params_%d" % node.id
            fab_setup.append("  const pops::RuntimeParams %s = ctx.program_params(%d);" % (name, block))
            setup.append("  const auto params = %s;" % name)
        setup += _cell_locals(impl, roots, variables[origin.id], with_cons=False,
                              with_prim=False, provider_binding=binding)
        environments[node.id] = (impl, setup)
    return lambda node: environments[node.id], fab_setup, preparations


def product_components(value):
    attrs = value.attrs
    count = attrs["output_count"]
    if attrs.get("product_version") not in (1, 2) or type(count) is not int or count < 1:
        raise ValueError("unsupported LocalResidual product contract")
    names = attrs["unknown_names"]
    if tuple(sorted(names)) != tuple(names) or len(set(names)) != count:
        raise ValueError("LocalResidual product unknown packing changed")
    widths = tuple(len(component_names(item)) if item.vtype == "state" else 0
                   for item in value.inputs)
    if widths != tuple(attrs["product_widths"]) or any(width < 1 for width in widths[:count]):
        raise ValueError("LocalResidual product component Spaces changed")
    if any(item.vtype not in ("state", "fields") for item in value.inputs):
        raise ValueError("LocalResidual product capture kind changed")
    seeds = value.inputs[:count]
    blocks = tuple(item.block for item in seeds)
    if tuple(attrs["blocks"]) != blocks or len(set(blocks)) != count:
        raise ValueError("LocalResidual product exact block packing changed")
    if len(attrs["expressions"]) != sum(widths[:count]):
        raise ValueError("LocalResidual product original residual cardinality changed")
    if len(value.inputs) != count + len(attrs["capture_names"]):
        raise ValueError("LocalResidual product capture arity changed")
    if attrs["product_version"] == 2:
        nodes = attrs["residual_block"]
        positions = attrs["product_argument_positions"]
        if len(positions) != len(value.inputs):
            raise ValueError("LocalResidual product argument packing changed")
        if any(type(p) is not int or not 0 <= p < len(nodes) for p in positions):
            raise ValueError("LocalResidual product argument position changed")
        frozen_ids = {source.id: nodes[p].id for p, source in
                      zip(positions[count:], value.inputs[count:], strict=True)
                      if source.vtype == "state"}
        seen = {}
        for index, (position, source) in enumerate(zip(positions, value.inputs, strict=True)):
            if type(position) is not int or not 0 <= position < len(nodes):
                raise ValueError("LocalResidual product argument position changed")
            argument = nodes[position]
            if position in seen:
                previous = seen[position]
                if (index < count or previous < count or source is not value.inputs[previous]):
                    raise ValueError("LocalResidual product aliases distinct equation arguments")
            seen[position] = index
            if (argument.op != ("input_fields" if source.vtype == "fields" else "state")
                    or argument.inputs or argument.vtype != source.vtype or argument.block != source.block
                    or argument.space != source.space or argument.point != source.point):
                raise ValueError("LocalResidual product exact argument authority changed")
            if source.vtype == "fields":
                from pops.time.field_context import remap_field_provenance
                expected = remap_field_provenance(
                    source.field_context, lambda source: frozen_ids.get(source, source))
                if argument.field_context != expected:
                    raise ValueError("LocalResidual product frozen FieldContext authority changed")
            elif argument.state_ref != source.state_ref:
                raise ValueError("LocalResidual product exact State reference changed")
    return {item.block: tuple(range(width)) for item, width in zip(seeds, widths, strict=False)}


def product_residual_lines(value, variables, physical_environment=None):
    product_components(value)
    count = value.attrs["output_count"]
    rows, offset = [], 0
    for index, item in enumerate(value.inputs):
        width = len(component_names(item)) if item.vtype == "state" else 0
        if index < count:
            rows.append(["Ueval[%d]" % (offset + c) for c in range(width)])
            offset += width
        else:
            rows.append(["%sA(index, %d)" % (variables[item.id], c) for c in range(width)])
    prefix = []
    if value.attrs["product_version"] == 2:
        from .program_emit_model_kernels import _emit_local_residual_nodes
        nodes = value.attrs["residual_block"]
        arguments = {nodes[position].id: row for position, row in
                     zip(value.attrs["product_argument_positions"], rows, strict=True)}
        def environment(node):
            if physical_environment is None:
                raise ValueError("local product physics requires exact model/provider authority")
            return physical_environment(node)
        prefix, components = _emit_local_residual_nodes(nodes, arguments, offset, environment)
        rows = [components.get(node.id, []) for node in nodes]
    reads = value.attrs["product_reads"]
    if any(type(i) is not int or not 0 <= i < len(rows) for i in reads):
        raise ValueError("LocalResidual product read lies outside exact arguments")
    lines, rendered, invalid = checked_expression_dag(
        value.attrs["expressions"], value.attrs["expression_nodes"], [rows[i] for i in reads])
    body = ["      " + line for line in [*prefix, *lines]]
    body.append("      if (%s) {" % invalid)
    body.extend("        rout[%d] = std::numeric_limits<pops::Real>::quiet_NaN();" % c
                for c in range(offset))
    body.extend(["        return;", "      }"])
    body.extend("      rout[%d] = %s;" % (c, expression)
                for c, expression in enumerate(rendered))
    return body
