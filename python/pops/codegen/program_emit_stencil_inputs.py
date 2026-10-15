"""Source-owned state/field extensions for generated neighbouring-cell reads."""
import json


def cell_sample_ghost_depth(value):
    from pops.codegen.program_lowerability import all_ops
    depth=1
    for operation in all_ops(value.prog):
        if operation.op != "stencil_flux_divergence":
            continue
        declaration=operation.attrs["stencil_flux"]
        for index,field in enumerate(operation.inputs[1:1+len(declaration.field_components)]):
            if field.id==value.id:
                depth=max(depth,*declaration.field_ghost_extents[index])
    return depth


def emit_cell_sample_extension(value,var,lines,*,block,target):
    from pops.time._program.spatial_interaction import validate_cell_field
    from pops.time._evaluation_point import evaluation_stage_fraction
    from pops.codegen.program_rhs_input_trace import emit_rhs_input_trace
    source=validate_cell_field(value)
    stage=evaluation_stage_fraction(value)
    lines += ["{", "const auto cell_sample_scope_%d = ctx.stage_evaluation_scope(%d, %d);"
              % (value.id,stage.numerator,stage.denominator)]
    trace=emit_rhs_input_trace(value,block,var[source.id],lines,target)
    lines.append("ctx.prepare_generated_state(%d, %s, %d%s);" % (
        block,var[source.id],value.id,", "+trace if trace else ""))
    lines += ["long cell_sample_invalid_%d = 0;" % value.id,
              "for (int axis = 0; axis < pops::kNativeDimension; ++axis)",
              "  cell_sample_invalid_%d |= %s.ghosts()[axis] < %s.ghosts()[axis];"
              % (value.id,var[source.id],var[value.id]),
              "if (pops::all_reduce_max(cell_sample_invalid_%d, ctx.prepared_execution_lane()))" % value.id,
              '  throw std::invalid_argument("cell reconstruction exceeds its prepared state extension");',
              "std::exception_ptr cell_sample_error_%d;" % value.id, "try {",
              "for (std::size_t li = 0; li < %s.local_size(); ++li) {" % var[value.id],
              "  const auto input = std::as_const(%s).fab(li).view();" % var[source.id],
              "  const auto output = %s.fab(li).view();" % var[value.id],
              "  pops::for_each_cell(%s.fab(li).grown_box(), [=] POPS_HD(const pops::Index<pops::kNativeDimension>& cell) {" % var[value.id],
              "    for (int component = 0; component < %d; ++component)" % len(value.space.components),
              "      output(cell, component) = input(cell, component);", "  });", "}",
              "} catch (...) { cell_sample_error_%d = std::current_exception(); }" % value.id,
              "try { Kokkos::fence(); } catch (...) { if (!cell_sample_error_%d) cell_sample_error_%d = std::current_exception(); }" % (value.id, value.id),
              'pops::collectively_rethrow_exception(cell_sample_error_%d, ctx.prepared_execution_lane(), "computed cell sample extension");' % value.id,
              "}"]


def emit_stencil_input_sample(operation,field,index,resource,buffer,var,lines,*,target,extent):
    if field.attrs.get("cell_field_projection") == "pops.cell-midpoint-sampling@1":
        # The projection copied its own producer's authenticated full extension
        # at production time. It is independent of the current flux state's stage.
        lines.append("const auto& %s = %s.sample_extension(%s);" % (buffer,resource,var[field.id]))
        return
    from pops.fields._observation_contract import validate_field_observation
    _width,component,solve=validate_field_observation(field)
    if target == "amr_system":
        source=buffer+"_solution_extension"
        lines.append("const auto& %s = ctx.prepare_stencil_field_solution(%d, %s, %s, %d, %d, %d, %s);" % (
            source,solve.id,json.dumps(field.attrs["field_problem_identity"]),var[field.id],
            component,operation.id,index,extent))
        lines.append("const auto& %s = %s.sample_extension(%s, %d);" % (buffer,resource,source,component))
    else:
        physical=solve.inputs[1].attrs["physical_boundary"]
        boundary=buffer+"_physical"
        lines += ["std::array<pops::elliptic::nd::PhysicalFieldBoundary, 2 * pops::kNativeDimension> %s{};" % boundary,
                  "%s.fill(pops::elliptic::nd::PhysicalFieldBoundary::%s);" % (boundary,physical),
                  "const auto& %s = %s.sample_uniform(%s, %s);" % (buffer,resource,var[field.id],boundary)]
