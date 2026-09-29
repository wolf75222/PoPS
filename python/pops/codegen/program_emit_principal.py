"""Emit a shared native face evaluation and exact principal rate projections."""


def _checked(roots, count, indent="      ", *, primitive=False):
    from pops._ir import _wrap
    from pops.codegen.cpp_writer import _cse_emit
    bindings = {("pops_principal_p%d" if primitive else "pops_principal_%d") % i:
                ("p[%d]" if primitive else "u[%d]") % i for i in range(count)}
    lines, values, observed = _cse_emit(tuple(_wrap(root) for root in roots), "pops::Real", indent,
        materialize_all=True, return_names=True, scalar_bindings=bindings)
    invalid = " || ".join("!Kokkos::isfinite(%s)" % item for item in observed) or "false"
    return lines, values, invalid


def emit_principal_model(entry):
    from pops._ir.values import set_runtime_param_indices
    set_runtime_param_indices(dict(entry["parameter_indices"]))
    group, name = entry["group"], entry["cpp_name"]
    count, rows = group.component_count, len(group.states)
    lines = ["struct %s {" % name,
        "  static constexpr int dimension = %d, n_vars = %d;" % (group.dimension, count),
        "  static constexpr std::array<int,%d> component_counts{%s};" % (rows, ",".join(map(str,group.component_counts))),
        "  struct Schema {};", "  using State = pops::StateVec<n_vars>;",
        "  using Primitive = State;", "  std::array<pops::RuntimeParams, %d> parameter_sets{};" % rows,
        "  POPS_HD static State invalid_state() { State result{};",
        "    for (int i=0; i<n_vars; ++i) result[i]=std::numeric_limits<pops::Real>::quiet_NaN();",
        "    return result; }",
        *_emit_principal_conversions(entry),
        "  template<int Axis> POPS_HD State flux(const State& u) const {",
        "    static_assert(Axis>=0 && Axis<dimension); State result{};"]
    for axis_index, axis in enumerate(entry["axes"]):
        lines.append("    if constexpr (Axis==%d) {" % axis_index)
        offset = 0
        for row, (body, size) in enumerate(zip(entry["fluxes"], group.component_counts, strict=True)):
            lines += ["      {", "      const auto params = parameter_sets[%d];" % row]
            declarations, values, invalid = _checked(body[axis], count)
            lines.extend(declarations)
            lines.append("      if (%s) return invalid_state();" % invalid)
            lines.extend("      result[%d] = %s;" % (offset + j, value) for j, value in enumerate(values))
            lines.append("      }")
            offset += size
        lines.append("    }")
    lines += ["    return result; }",
        "  template<int Axis> POPS_HD pops::Real max_wave_speed(const State& u) const {",
        "    static_assert(Axis>=0 && Axis<dimension); pops::Real result=0;"]
    for axis_index, axis in enumerate(entry["axes"]):
        lines.append("    if constexpr (Axis==%d) {" % axis_index)
        for row in range(rows):
            lines += ["      {", "      const auto params = parameter_sets[%d];" % row]
            declarations, values, invalid = _checked(entry["waves"][axis], count)
            lines.extend(declarations)
            lines.append("      if (%s) return std::numeric_limits<pops::Real>::quiet_NaN();" % invalid)
            lines.extend("      result=Kokkos::max(result,Kokkos::abs(%s));" % value for value in values)
            lines.append("      }")
        lines.append("    }")
    lines += ["    return result; }",
        "  template<int Axis> POPS_HD void wave_speeds(const State& u,pops::Real& lower,pops::Real& upper) const {",
        "    upper=max_wave_speed<Axis>(u); lower=-upper; }", "};"]
    return "\n".join(lines) + "\n"


def emit_principal_models(program, authority):
    from pops.codegen.program_lowerability import all_ops
    from pops.codegen.program_models import model_for_node
    from .principal_lowering import principal_for_value
    entries = {}
    for value in all_ops(program):
        if value.op == "principal_rate":
            entry = principal_for_value(model_for_node(authority, value), value)
            entries.setdefault(entry["group"].identity.token, entry)
    return "".join(emit_principal_model(entry) + emit_principal_helper(entry)
                   for entry in entries.values())


def emit_principal_helper(entry):
    from .program_emit_principal_numerics import (
        emit_principal_numerical_flux, emit_principal_reconstruction)
    group, name = entry["group"], entry["cpp_name"]
    reconstruction_definition, reconstruction = emit_principal_reconstruction(entry)
    numerical_definition, numerical = emit_principal_numerical_flux(entry)
    size = len(group.states)
    lines = ["template<class Context>", "auto& %s_evaluate(Context& ctx,std::int64_t node," % name,
        "    const std::array<pops::MultiFab<pops::kNativeDimension>*,%d>& inputs," % size,
        "    const std::array<int,%d>& blocks," % size,
        "    const std::array<std::int64_t,%d>& sources) {" % size,
        "  for (int block : blocks) ctx.require_cartesian_generated_operator(block,\"principal_finite_volume\");",
        "  auto& packed=ctx.scalar_scratch(node,0,*inputs[0],%d,%d);" % (
            group.component_count,max(method.ghost_depth for method in group.methods)),
        "  const auto geometry=ctx.geometry(); const auto& lane=ctx.prepared_execution_lane();",
        "  auto& resource=ctx.template prepared_resource<pops::runtime::program::PreparedPrincipalFlux<pops::kNativeDimension,%d%s>>(" % (group.component_count,
            ",pops::nd::ReconstructionVariables::Primitive" if group.methods[0].variables.scheme == "primitive" else ""),
        "    node,blocks[0],[&](const auto& previous){return previous.matches_preparation(geometry,lane,packed);},ctx,packed);",
        "  %s model;" % name,
        "  for (int i=0;i<%d;++i) model.parameter_sets[i]=ctx.program_params(blocks[i]);" % size,
        "  if constexpr (requires { ctx.prepare_principal_inputs(inputs,blocks,node,sources); }) {",
        "    ctx.prepare_principal_inputs(inputs,blocks,node,sources);",
        "    resource.pack_prepared(inputs,packed,%s::component_counts);" % name,
        "    resource.evaluate_prepared(packed,model,%s,%s);" % (reconstruction,numerical),
        "  } else {",
        "    resource.pack(inputs,packed,%s::component_counts);" % name,
        "    resource.evaluate(packed,model,%s,%s);" % (reconstruction,numerical),
        "  }",
        "  return resource;", "}"]
    return reconstruction_definition + numerical_definition + "\n".join(lines) + "\n"


def principal_dt_bounds(program, authority):
    """Bound only unconditional evaluations of the actual beginning-of-step state.

    A prescribed step needs no speculative evaluation. Conditional bodies and
    calculated stages retain their face-frequency check at the explicit consumer;
    they must never be replayed with unrelated current-state inputs here.
    """
    from .program_lowerability import all_ops_with_ancestry
    from .program_models import model_for_node
    from .principal_lowering import principal_for_value
    from pops.time import ExternalTimeGrid, FixedDt

    if isinstance(program._step_strategy, (FixedDt, ExternalTimeGrid)):
        return []
    lines, seen = [], set()
    required = set()
    block_indices = program._block_indices()
    for value, ancestry in all_ops_with_ancestry(program):
        if value.op != "principal_rate":
            continue
        entry = principal_for_value(model_for_node(authority, value), value)
        group = entry["group"]
        required.add(group.identity.token)
        if ancestry or any(item.op != "state" for item in value.inputs):
            continue
        if group.identity.token in seen:
            continue
        seen.add(group.identity.token)
        indices = []
        for state in group.states:
            index, = [index for block, index in block_indices.items()
                      if block._resolved() == state.block_ref]
            indices.append(index)
        prefix = "principal_bound_%d" % value.id
        lines.extend(["std::array<pops::MultiFab<pops::kNativeDimension>*,%d> %s_inputs{%s};" %
                      (len(indices),prefix,",".join("&ctx.state(%d)" % index for index in indices)),
                      "auto& %s_resource=%s_evaluate(ctx,%d,%s_inputs,std::array<int,%d>{%s},std::array<std::int64_t,%d>{%s});" %
                      (prefix,entry["cpp_name"],value.id,prefix,len(indices),",".join(map(str,indices)),
                       len(indices),",".join(str(next(item.id for item in value.inputs
                           if item.block._resolved() == state.block_ref)) for state in group.states)),
                      "const pops::Real %s_frequency=%s_resource.explicit_frequency();" % (prefix,prefix),
                      "pops_program_dt_bound_value=std::min(pops_program_dt_bound_value,%s_frequency>0 ? cfl/%s_frequency : std::numeric_limits<pops::Real>::infinity());" % (prefix,prefix)])
    if required - seen and program._dt_bound is None:
        raise NotImplementedError(
            "adaptive principal groups evaluated only under a guard or on a calculated stage "
            "require an authored Program.dt_bound; their face stability is checked at explicit consumers")
    return lines


def emit_principal_rate(value, var, lines, model, block_indices, target):
    from .principal_lowering import principal_for_value
    from pops.time._evaluation_point import evaluation_stage_fraction
    if target not in {"system", "amr_system"}:
        raise NotImplementedError("principal transport requires a System or AMR Program context")
    entry = principal_for_value(model, value)
    group = entry["group"]
    inputs = []
    for state in group.states:
        matches = [item for item in value.inputs
                   if item.block._resolved() == state.block_ref and
                   item.space == (state.declaration_ref or state).space]
        if len(matches) != 1:
            raise ValueError("principal evaluation omits or duplicates an exact sampled state instance")
        inputs.append(matches[0])
    output_index, = [i for i, state in enumerate(group.states)
                     if state.block_ref == value.block._resolved()]
    key = ("principal_evaluation", group.identity.token, tuple(item.id for item in inputs),
           value.attrs.get("principal_region"))
    stage = evaluation_stage_fraction(value, ark_partition="explicit")
    lines.append("ctx.set_stage_time(%d,%d);" % (stage.numerator, stage.denominator))
    cached = var.get(key)
    if cached is not None and output_index not in cached[1]:
        cached[1].add(output_index)
        var[value.id] = cached[0][output_index]
        var[("partition_frequency", value.id)] = cached[2]
        var[("principal_frequency", value.id)] = cached[2]
        _attach_principal_faces(value, lines, group, output_index, cached[3],
                                var[value.id], block_indices, target)
        return
    prefix = "principal_%d" % value.id
    indices = [block_indices[item.block] for item in inputs]
    lines += ["std::array<pops::MultiFab<pops::kNativeDimension>*,%d> %s_inputs{%s};" %
              (len(inputs), prefix, ",".join("&" + var[item.id] for item in inputs)),
              "auto& %s_resource=%s_evaluate(ctx,%d,%s_inputs,std::array<int,%d>{%s},std::array<std::int64_t,%d>{%s});" %
              (prefix,entry["cpp_name"],value.id,prefix,len(indices),",".join(map(str,indices)),
               len(inputs),",".join(str(item.id) for item in inputs))]
    frequency = "principal_frequency_%d" % value.id
    lines.append("const pops::Real %s=%s_resource.explicit_frequency();" % (frequency, prefix))
    var[("partition_frequency", value.id)] = frequency
    var[("principal_frequency", value.id)] = frequency
    outputs = ["%s_rate_%d" % (prefix, i) for i in range(len(inputs))]
    for i, (output, item) in enumerate(zip(outputs, inputs, strict=True)):
        lines.append("auto& %s=ctx.rhs_scratch(%d,%d,%s);" % (output, value.id, i, var[item.id]))
    lines += ["std::array<pops::MultiFab<pops::kNativeDimension>*,%d> %s_outputs{%s};" %
              (len(outputs), prefix, ",".join("&" + output for output in outputs)),
              "%s_resource.publish(%s_outputs,%s::component_counts);" % (prefix, prefix, entry["cpp_name"])]
    resource = "%s_resource" % prefix
    var[key] = (outputs, {output_index}, frequency, resource)
    var[value.id] = outputs[output_index]
    _attach_principal_faces(value, lines, group, output_index, resource,
                            var[value.id], block_indices, target)


def _attach_principal_faces(value, lines, group, row, resource, output, block_indices, target):
    if target != "amr_system":
        return
    import json
    from .program_emit_ops import _rhs_flux_temporal_family
    first = sum(group.component_counts[:row])
    lines.append("ctx.attach_principal_flux(%d,%s,%d,%s.component_faces(%d,%d),%s);" % (
        block_indices[value.block], output, value.id, resource, first,
        group.component_counts[row], json.dumps(_rhs_flux_temporal_family(value))))


def _emit_principal_conversions(entry):
    count = entry["group"].component_count
    conversion = entry.get("conversion")
    if conversion is None:
        return [
            "  POPS_HD pops::nd::StateConversionStatus admissibility(const State& u) const {",
            "    for(int i=0;i<n_vars;++i) if(!Kokkos::isfinite(u[i])) return pops::nd::StateConversionStatus::NonFiniteState;",
            "    return pops::nd::StateConversionStatus::Success; }",
            "  POPS_HD pops::nd::StateConversion<Primitive> recover(const State& u) const { return {u,admissibility(u)}; }",
            "  POPS_HD pops::nd::StateConversion<State> make_conservative(const Primitive& p) const { return {p,admissibility(p)}; }"]
    lines = ["  POPS_HD bool primitive_domain(const Primitive& p) const {"]
    for row, predicates in enumerate(conversion["constraints"]):
        lines += ["    {", "      const auto params=parameter_sets[%d];" % row]
        declarations, values, invalid = _checked(predicates,count,primitive=True)
        lines.extend(declarations)
        condition = " || ".join("!(%s)" % value for value in values)
        lines.append("      if (%s%s) return false;" % (invalid," || "+condition if condition else ""))
        lines.append("    }")
    lines += ["    return true;", "  }"]
    for direction, target, function, source in (
            ("forward","Primitive","recover","State& u"),
            ("inverse","State","make_conservative","Primitive& p")):
        primitive = direction == "inverse"
        letter = "p" if primitive else "u"
        lines += ["  POPS_HD pops::nd::StateConversion<%s> %s(const %s) const {" % (target,function,source),
                  "    pops::nd::StateConversion<%s> result{};" % target,
                  "    for(int i=0;i<n_vars;++i) if(!Kokkos::isfinite(%s[i])) return result;" % letter]
        if primitive:
            lines.append("    if(!primitive_domain(p)) { result.status=pops::nd::StateConversionStatus::InvalidEquationOfState; return result; }")
        offset = 0
        for row, roots in enumerate(conversion[direction]):
            lines += ["    {", "      const auto params=parameter_sets[%d];" % row]
            declarations, values, invalid = _checked(roots,count,primitive=primitive)
            lines.extend(declarations)
            lines.append("      if(%s) return result;" % invalid)
            lines.extend("      result.value[%d]=%s;" % (offset+i,value) for i,value in enumerate(values))
            offset += len(roots)
            lines.append("    }")
        if not primitive:
            lines.append("    if(!primitive_domain(result.value)) { result.status=pops::nd::StateConversionStatus::InvalidEquationOfState; return result; }")
        lines += ["    result.status=pops::nd::StateConversionStatus::Success; return result;", "  }"]
    lines += ["  POPS_HD pops::nd::StateConversionStatus admissibility(const State& u) const { return recover(u).status; }"]
    return lines
