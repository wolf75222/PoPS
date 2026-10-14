"""Lower closed face expressions once, retaining the exact native face product."""
from fractions import Fraction


def emit_stencil_flux(value, var, lines, *, block, target):
    from pops.time._program.stencil_flux import validate_stencil_flux
    from pops.codegen.cpp_writer import _cse_emit
    from pops.time._evaluation_point import evaluation_stage_fraction
    from pops.codegen.program_value_authority import DelegatedOutputPublication
    if target != "system":
        raise NotImplementedError("stencil flux v1 requires a Uniform Cartesian periodic layout")
    if value.region != 0:
        raise NotImplementedError("stencil flux acceptance inside a region needs scoped face receipts")
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
    lines.append('static_assert(pops::kNativeDimension == %d, "stencil frame rank differs from native");'
                 % declaration.dimension)
    extents = "std::array<int, pops::kNativeDimension>{%s}" % ", ".join(map(str, declaration.ghost_extent))
    kind = "pops::runtime::program::PreparedStencilFlux<pops::kNativeDimension, %d>" % declaration.components
    lines += ["const auto %s_geometry = ctx.geometry();" % prepared,
              "const auto& %s_lane = ctx.prepared_execution_lane();" % prepared,
              "auto& %s = ctx.prepared_resource<%s>(%d, %d," % (prepared, kind, value.id, block),
              "  [&](const auto& resource) { return resource.matches_preparation(%s_geometry, %s_lane, %s, %s); }, ctx, %s, %s);"
              % (prepared, prepared, state, extents, state, extents)]
    field_buffers = []
    field_end = 1 + len(declaration.field_components)
    for index, field in enumerate(value.inputs[1:field_end]):
        source = var[field.id]
        buffer = "stencil_field_%d_%d" % (value.id, index)
        resource = buffer + "_prepared"
        extent = "std::array<int, pops::kNativeDimension>{%s}" % ", ".join(map(str, declaration.field_ghost_extents[index]))
        resource_type = "pops::runtime::program::PreparedStencilInput<pops::kNativeDimension, %d>" % index
        lines += ["auto& %s = ctx.prepared_resource<%s>(%d, %d," % (resource, resource_type, value.id, block),
                  "  [&](const auto& r) { return r.matches_preparation(%s_geometry, %s_lane, %s, %s, %s); }, ctx, %s, %s, %s);"
                  % (prepared, prepared, state, source, extent, state, source, extent),
                  "const auto& %s = %s.sample(%s);" % (buffer, resource, source)]
        field_buffers.append(buffer)
    scalar_buffers = []
    for index, parameter in enumerate(value.inputs[field_end:]):
        buffer = "stencil_parameter_%d_%d" % (value.id, index)
        lines.append("const pops::Real %s = %s;" % (buffer, var[parameter.id]))
        scalar_buffers.append(buffer)
    before_write = len(lines)
    lines += ["%s.apply(%s, %s, [&](std::size_t li) {" % (prepared, state, out),
              "  const auto q = std::as_const(%s).fab(li).view();" % state,
              "  const auto geometry = %s_geometry;" % prepared,
              "  return [=] POPS_HD(const pops::Index<pops::kNativeDimension>& face, int axis) {",
              "    pops::runtime::program::StencilFaceResult<%d> result;" % declaration.components]
    for index, buffer in enumerate(field_buffers):
        # Insert captures into the host factory, before its device closure.
        location = lines.index("  const auto geometry = %s_geometry;" % prepared, before_write)
        lines.insert(location, "  const auto f%d = %s.fab(li).view();" % (index, buffer))
    bindings = {v.name: "geometry.spacing(%d)" % d for d, v in enumerate(declaration.spacing)}
    bindings.update((v.name, buffer) for v, buffer in zip(declaration.parameter_symbols, scalar_buffers, strict=True))
    for offset, component, symbol in declaration.samples:
        coordinates = ", ".join("face[%d] + (%d)" % (d, x) for d, x in enumerate(offset))
        bindings[symbol.name] = "q(pops::Index<pops::kNativeDimension>(%s), %d)" % (coordinates, component)
    for field, offset, component, symbol in declaration.field_samples:
        coordinates = ", ".join("face[%d] + (%d)" % (d, x) for d, x in enumerate(offset))
        bindings[symbol.name] = "f%d(pops::Index<pops::kNativeDimension>(%s), %d)" % (field, coordinates, component)
    for axis, expressions in enumerate(declaration.rows):
        lines.append("    %s (axis == %d) {" % ("if" if axis == 0 else "else if", axis))
        body, values, names = _cse_emit(expressions, "pops::Real", "      ",
            materialize_all=True, return_names=True, scalar_bindings=bindings)
        lines.extend(body)
        checks = " && ".join("Kokkos::isfinite(%s)" % x for x in (*names, *values))
        lines.append("      result.valid = %s;" % (checks or "true"))
        lines.append("      result.flux = {%s};" % ", ".join(values[:-1]))
        lines.append("      result.frequency = %s;" % values[-1])
        lines.append("    }")
    lines += ["    return result;", "  };", "});",
              "const pops::Real stencil_frequency_%d = %s.explicit_frequency();" % (value.id, prepared),
              "const auto& stencil_faces_%d = %s.faces();" % (value.id, prepared)]
    var[("partition_frequency", value.id)] = "stencil_frequency_%d" % value.id
    var[("user_face_frequency", value.id)] = "stencil_frequency_%d" % value.id
    return DelegatedOutputPublication(value.id, block, "RhsScratch", value.id, 0, out,
                                      start, setup_end, before_write, len(lines))


def accepted_stencil_quadrature(program):
    """Derive receipt weights from the actual affine accepted state graph."""
    from pops.codegen.program_emit_kernels import _coeff_metadata_terms
    selected = {}

    from collections.abc import Mapping
    from pops.time.values import ProgramValue

    # These operations evaluate a fresh residual; sampled predictor ancestry is
    # not a second accepted face transfer. A type alone is insufficient: branch,
    # Result projection and loop aliases can also return RHS values unchanged.
    fresh_rhs = frozenset(("rhs", "source", "implicit_source", "apply",
                           "diffusive_rhs", "principal_rate", "coupled_rate"))
    contained = {}

    def leaves(item):
        if isinstance(item, ProgramValue):
            yield item
        elif isinstance(item, Mapping):
            for child in item.values():
                yield from leaves(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                yield from leaves(child)

    def contains(value):
        if value.id in contained:
            return contained[value.id]
        contained[value.id] = False
        if value.op == "stencil_flux_divergence":
            contained[value.id] = True
            return True
        if value.vtype == "rhs" and (value.op in fresh_rhs or (
                value.op == "pointwise_expression"
                and value.attrs.get("numerical_pointwise_rate") == "pops.pointwise-rate@1")):
            return False
        # Actual region outputs live in typed attributes, not necessarily inputs.
        # Merely belonging to a body block is not consumption of every dead node.
        dependencies = list(value.inputs)
        for name, attribute in value.attrs.items():
            if not name.endswith("_block"):
                dependencies.extend(leaves(attribute))
        contained[value.id] = any(contains(child) for child in dependencies)
        return contained[value.id]

    def walk(value, weight):
        if value.op == "stencil_flux_divergence":
            entry = selected.setdefault(value.id, (value, {}))[1]
            for power, coefficient in weight.items():
                entry[power] = entry.get(power, Fraction()) + coefficient
            return
        if value.op == "acceptance_guard":
            walk(value.inputs[0], weight)
            return
        if value.op != "linear_combine":
            if contains(value):
                raise ValueError("accepted transform hides stencil faces outside affine quadrature")
            return
        for child, coefficients in zip(value.inputs, value.attrs["coeffs"], strict=True):
            product = {}
            for power, numerator, denominator in _coeff_metadata_terms(coefficients):
                for outer, coefficient in weight.items():
                    product[power+outer] = product.get(power+outer, Fraction()) + coefficient*Fraction(numerator, denominator)
            walk(child, product)

    for value in program._commits.values():
        walk(value, {0: Fraction(1)})
    rows = tuple((value, {p: c for p, c in weights.items() if c})
                 for value, weights in selected.values())
    if any(set(weight) != {1} or weight[1] < 0 for _, weight in rows):
        raise ValueError("stencil accepted quadrature requires nonnegative exact dt weights")
    return rows


def emit_accepted_stencil_exchanges(program, var, block_indices, *, target):
    from pops.codegen.program_emit_transport_exchanges import emit_transport_exchanges
    from pops.codegen.program_emit_kernels import _coeff_cpp
    from pops.identity import make_identity
    rows = accepted_stencil_quadrature(program)
    if rows and target != "system":
        raise NotImplementedError("stencil receipts require Uniform native storage")
    lines = []
    for value, weight in rows:
        operation = make_identity("stencil-flux-v1", value.attrs["stencil_flux"].to_data()).token
        lines += emit_transport_exchanges("stencil_faces_%d" % value.id, operation,
            operation + "/shared-face", "stage:" + str(value.point) + "/evaluation:" + str(value.id),
            _coeff_cpp(weight), program_block=block_indices[value.block],
            active_field=var[value.inputs[0].id])
    return lines
