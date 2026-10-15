"""Lower closed face expressions once, retaining the exact native face product."""

from fractions import Fraction
from typing import Any


def emit_stencil_flux(value, var, lines, *, block, target):
    from pops.time._program.stencil_flux import validate_stencil_flux
    from pops.codegen.cpp_writer import _cse_emit
    from pops.time._evaluation_point import evaluation_stage_fraction
    from pops.codegen.program_value_authority import DelegatedOutputPublication

    declaration = validate_stencil_flux(value)
    state = var[value.inputs[0].id]
    prepared = "stencil_prepared_%d" % value.id
    out = "stencil_rhs_%d" % value.id
    var[value.id] = out
    start = len(lines)
    lines.append("auto& %s = ctx.rhs_scratch(%d, 0, %s);" % (out, value.id, state))
    setup_end = len(lines)
    stage = evaluation_stage_fraction(value)
    lines.append("ctx.set_stage_time(%d, %d);" % (stage.numerator, stage.denominator))
    lines.append('ctx.require_cartesian_generated_operator(%d, "stencil_flux");' % block)
    if target == "amr_system":
        from pops.codegen.program_rhs_input_trace import emit_rhs_input_trace

        trace = emit_rhs_input_trace(value, block, state, lines, target)
        lines.append(
            "ctx.prepare_generated_state(%d, %s, %d%s);"
            % (block, state, value.id, ", " + trace if trace else "")
        )
    else:
        lines.append("ctx.prepare_generated_state(%d, %s, %d);" % (block, state, value.id))
    lines.append(
        'static_assert(pops::kNativeDimension == %d, "stencil frame rank differs from native");'
        % declaration.dimension
    )
    extents = "std::array<int, pops::kNativeDimension>{%s}" % ", ".join(
        map(str, declaration.ghost_extent)
    )
    kind = (
        "pops::runtime::program::PreparedStencilFlux<pops::kNativeDimension, %d>"
        % declaration.components
    )
    lines += [
        "const auto %s_geometry = ctx.geometry();" % prepared,
        "const auto& %s_lane = ctx.prepared_execution_lane();" % prepared,
        "auto& %s = ctx.prepared_resource<%s>(%d, %d," % (prepared, kind, value.id, block),
        "  [&](const auto& resource) { return resource.matches_preparation(%s_geometry, %s_lane, %s, %s); }, ctx, %s, %s);"
        % (prepared, prepared, state, extents, state, extents),
    ]
    field_buffers = []
    field_end = 1 + len(declaration.field_components)
    for index, field in enumerate(value.inputs[1:field_end]):
        source = var[field.id]
        buffer = "stencil_field_%d_%d" % (value.id, index)
        resource = buffer + "_prepared"
        extent = "std::array<int, pops::kNativeDimension>{%s}" % ", ".join(
            map(str, declaration.field_ghost_extents[index])
        )
        resource_type = (
            "pops::runtime::program::PreparedStencilInput<pops::kNativeDimension, %d>" % index
        )
        lines += [
            "auto& %s = ctx.prepared_resource<%s>(%d, %d,"
            % (resource, resource_type, value.id, block),
            "  [&](const auto& r) { return r.matches_preparation(%s_geometry, %s_lane, %s, %s, %s); }, ctx, %s, %s, %s);"
            % (prepared, prepared, state, source, extent, state, source, extent),
        ]
        from pops.codegen.program_emit_stencil_inputs import emit_stencil_input_sample

        emit_stencil_input_sample(
            value, field, index, resource, buffer, var, lines, target=target, extent=extent
        )
        field_buffers.append(buffer)
    scalar_buffers = []
    for index, parameter in enumerate(value.inputs[field_end:]):
        buffer = "stencil_parameter_%d_%d" % (value.id, index)
        lines.append("const pops::Real %s = %s;" % (buffer, var[parameter.id]))
        scalar_buffers.append(buffer)
    before_write = len(lines)
    lines += [
        "%s.apply(%s, %s, [&](std::size_t li) {" % (prepared, state, out),
        "  const auto q = std::as_const(%s).fab(li).view();" % state,
        "  const auto geometry = %s_geometry;" % prepared,
        "  return [=] POPS_HD(const pops::Index<pops::kNativeDimension>& face, int axis) {",
        "    pops::runtime::program::StencilFaceResult<%d> result;" % declaration.components,
    ]
    for index, buffer in enumerate(field_buffers):
        # Insert captures into the host factory, before its device closure.
        location = lines.index("  const auto geometry = %s_geometry;" % prepared, before_write)
        lines.insert(location, "  const auto f%d = %s.fab(li).view();" % (index, buffer))
    bindings = {v.name: "geometry.spacing(%d)" % d for d, v in enumerate(declaration.spacing)}
    bindings.update(
        (v.name, buffer)
        for v, buffer in zip(declaration.parameter_symbols, scalar_buffers, strict=True)
    )
    for offset, component, symbol in declaration.samples:
        coordinates = ", ".join("face[%d] + (%d)" % (d, x) for d, x in enumerate(offset))
        bindings[symbol.name] = "q(pops::Index<pops::kNativeDimension>(%s), %d)" % (
            coordinates,
            component,
        )
    for field, offset, component, symbol in declaration.field_samples:
        coordinates = ", ".join("face[%d] + (%d)" % (d, x) for d, x in enumerate(offset))
        bindings[symbol.name] = "f%d(pops::Index<pops::kNativeDimension>(%s), %d)" % (
            field,
            coordinates,
            component,
        )
    for axis, expressions in enumerate(declaration.rows):
        lines.append("    %s (axis == %d) {" % ("if" if axis == 0 else "else if", axis))
        body, values, names = _cse_emit(
            expressions,
            "pops::Real",
            "      ",
            materialize_all=True,
            return_names=True,
            scalar_bindings=bindings,
        )
        lines.extend(body)
        checks = " && ".join("Kokkos::isfinite(%s)" % x for x in (*names, *values))
        lines.append("      result.valid = %s;" % (checks or "true"))
        lines.append("      result.flux = {%s};" % ", ".join(values[:-1]))
        lines.append("      result.frequency = %s;" % values[-1])
        lines.append("    }")
    lines += [
        "    return result;",
        "  };",
        "}, true);",
        "const pops::Real stencil_frequency_%d = %s.explicit_frequency();" % (value.id, prepared),
        "const auto& stencil_faces_%d = %s.faces();" % (value.id, prepared),
    ]
    if target == "amr_system":
        import json

        lines.append(
            "ctx.attach_generated_face_basis(%d, %s, %d, stencil_faces_%d, %s);"
            % (block, out, value.id, value.id, json.dumps(stencil_temporal_family(value)))
        )
    var[("partition_frequency", value.id)] = "stencil_frequency_%d" % value.id
    var[("user_face_frequency", value.id)] = "stencil_frequency_%d" % value.id
    return DelegatedOutputPublication(
        value.id, block, "RhsScratch", value.id, 0, out, start, setup_end, before_write, len(lines)
    )


def stencil_temporal_family(value):
    from pops.identity import make_identity

    return make_identity("stencil-flux-v1", value.attrs["stencil_flux"].to_data()).token


def emit_stencil_exchange_capture(value, var, *, block, target):
    """Snapshot the evaluated faces only after the guarded SSA write completed.

    The immutable native expression outlives this C++ lexical scope. Publication
    follows the field selected by control flow, including repeated evaluations
    of the same authored node; no branch-local reference escapes the region.
    """
    if target == "amr_system":
        return []
    from pops.codegen.program_emit_transport_exchanges import emit_transport_exchanges

    operation = stencil_temporal_family(value)
    return emit_transport_exchanges(
        "stencil_faces_%d" % value.id,
        operation,
        operation + "/shared-face",
        "stage:" + str(value.point) + "/evaluation:" + str(value.id),
        "pops::Real(1)",
        program_block=block,
        active_field=var[value.inputs[0].id],
        capture_to=var[value.id],
    )


def stencil_expression_write(value, var, *, has_output_boundary):
    """Independent receipt-write guards, including graphs without a Field solve.

    Fresh residuals evaluate new equations; their sampled predictor's incoming
    exchanges are not another delivery. Other writes start unproved until a
    native algebra operation installs their exact affine expression.
    """
    if (not var.get(("stencil_flux_expressions",), False)
            or value.vtype not in ("state", "rhs", "scalar_field")
            or value.op in ("state", "history")
            or not has_output_boundary and value.op != "project"):
        return [], []
    fresh = value.op in {
        "rhs", "source", "implicit_source", "apply", "diffusive_rhs",
        "principal_rate", "coupled_rate", "stencil_flux_divergence",
    } or (value.op == "pointwise_expression"
          and value.attrs.get("numerical_pointwise_rate") == "pops.pointwise-rate@1")
    inputs = [] if fresh else [
        "&(%s)" % var[source.id] for source in value.inputs
        if source.vtype in ("state", "rhs", "scalar_field", "vector_field", "mask")
    ]
    output = var[value.id]
    return (["ctx.prepare_stencil_expression_write(%s, {%s});" % (output, ", ".join(inputs))],
            ["ctx.finish_stencil_expression_write(%s);" % output])


def accepted_stencil_quadrature(program):
    """Derive receipt weights from the actual affine accepted state graph."""
    from pops.codegen.program_emit_kernels import _coeff_metadata_terms

    selected = {}
    from pops.time._program.affine_expressions import (
        affine_field_terms,
        coefficient,
        coefficient_polynomial,
    )

    def tree(weight):
        return coefficient_polynomial(weight) if isinstance(weight, dict) else weight

    def add(left, right):
        if isinstance(left, dict) and isinstance(right, dict):
            result = dict(left)
            for power, amount in right.items():
                result[power] = result.get(power, Fraction()) + amount
            return result
        return coefficient("add", tree(left), tree(right))

    from collections.abc import Mapping
    from pops.time.values import ProgramValue

    # These operations evaluate a fresh residual; sampled predictor ancestry is
    # not a second accepted face transfer. A type alone is insufficient: branch,
    # Result projection and loop aliases can also return RHS values unchanged.
    fresh_rhs = frozenset(
        (
            "rhs",
            "source",
            "implicit_source",
            "apply",
            "diffusive_rhs",
            "principal_rate",
            "coupled_rate",
        )
    )
    contained = {}

    def leaves(item):
        pending = [item]
        seen = set()
        while pending:
            item = pending.pop()
            if isinstance(item, ProgramValue):
                yield item
            elif isinstance(item, (Mapping, tuple, list)):
                if id(item) in seen:
                    continue
                seen.add(id(item))
                children = tuple(item.values()) if isinstance(item, Mapping) else item
                pending.extend(reversed(children))

    def contains(value):
        pending: list[tuple[Any, Any]] = [(value, None)]
        while pending:
            current, dependencies = pending.pop()
            if dependencies is not None:
                contained[current.id] = any(contained[child.id] for child in dependencies)
                continue
            if current.id in contained:
                continue
            contained[current.id] = False
            if current.op == "stencil_flux_divergence":
                contained[current.id] = True
                continue
            if current.vtype == "rhs" and (
                current.op in fresh_rhs
                or (
                    current.op == "pointwise_expression"
                    and current.attrs.get("numerical_pointwise_rate") == "pops.pointwise-rate@1"
                )
            ):
                continue
            # Actual outputs live in typed attributes. Dead block members are
            # not consumption; preserve the same dependency boundary.
            dependencies = list(current.inputs)
            for name, attribute in current.attrs.items():
                if not name.endswith("_block"):
                    dependencies.extend(leaves(attribute))
            pending.append((current, dependencies))
            pending.extend((child, None) for child in reversed(dependencies))
        return contained[value.id]

    def walk(value, weight):
        pending = [(value, weight)]
        while pending:
            value, weight = pending.pop()
            if value.op == "stencil_flux_divergence":
                previous = selected.get(value.id, (value, {}))[1]
                selected[value.id] = (value, add(previous, weight))
                continue
            if value.op == "acceptance_guard":
                pending.append((value.inputs[0], weight))
                continue
            affine = affine_field_terms(value)
            if affine is not None:
                pending.extend(
                    (child, coefficient("mul", tree(weight), factor))
                    for child, factor in reversed(tuple(affine))
                )
                continue
            if value.op != "linear_combine":
                if contains(value):
                    raise ValueError(
                        "accepted transform hides stencil faces outside affine quadrature"
                    )
                continue
            children = []
            for child, coefficients in zip(value.inputs, value.attrs["coeffs"], strict=True):
                if not isinstance(weight, dict):
                    children.append(
                        (child, coefficient("mul", weight, coefficient_polynomial(coefficients)))
                    )
                    continue
                product = {}
                for power, numerator, denominator in _coeff_metadata_terms(coefficients):
                    for outer, amount in weight.items():
                        product[power + outer] = product.get(
                            power + outer, Fraction()
                        ) + amount * Fraction(numerator, denominator)
                children.append((child, product))
            pending.extend(reversed(children))

    for value in program._commits.values():
        walk(value, {0: Fraction(1)})
    rows = tuple(
        (value, {p: c for p, c in weights.items() if c} if isinstance(weights, dict) else weights)
        for value, weights in selected.values()
        if not isinstance(weights, dict) or any(weights.values())
    )
    if any(isinstance(weight, dict) and set(weight) != {1} for _, weight in rows):
        raise ValueError("stencil accepted quadrature requires exact duration weights")
    return rows


def emit_accepted_stencil_exchanges(program, var, block_indices, *, target):
    # AMR attaches each evaluated face basis to its exact RHS. The native
    # FluxExpression follows aliases/linear combinations and publishes only the
    # accepted state's expression into the existing transactional reflux ledger.
    if target == "amr_system":
        return []
    from pops.codegen.program_lowerability import all_ops

    if not any(value.op == "stencil_flux_divergence" for value in all_ops(program)):
        return []
    return [
        "ctx.publish_stencil_exchanges(%d, %s);" % (block_indices[value.block], var[value.id])
        for value in program._commits.values()
    ]
