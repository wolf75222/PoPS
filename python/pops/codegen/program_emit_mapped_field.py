"""Pack one authenticated consumed component into private mapping storage."""
from pops.identity.scalar import scalar_cpp
from pops.fields._mapped_publication import validate_pack


def scalar_candidate_rows(token, value_id, subslot, prototype):
    return [
        "pops::MultiFab<pops::kNativeDimension>* %s_pointer = nullptr;" % token,
        "std::exception_ptr %s_allocation_error;" % token,
        "try { %s_pointer = &ctx.scalar_scratch(%d, %d, %s, 1, 0); }" % (token, value_id, subslot, prototype),
        "catch (...) { %s_allocation_error = std::current_exception(); }" % token,
        'pops::collectively_rethrow_exception(%s_allocation_error, ctx.prepared_execution_lane(), "mapped Field candidate allocation");' % token,
        "auto& %s = *%s_pointer;" % (token, token),
    ]


def emit_mapped_field_pack(value, var, lines, *, target):
    if target not in ("system", "amr_system"):
        raise NotImplementedError("mapped consumed Field output requires a native System hierarchy")
    _component, exact_factor, _solve = validate_pack(value)
    source = value.inputs[0]
    if var.get(("field_observation", source.id)) != source.attrs["field_problem_identity"]:
        raise ValueError("mapped Field source lacks emitted consumed-solve authority")
    token = "mapped_field_%d" % value.id
    source_cpp = var[source.id]
    component = value.attrs["source_component"]
    factor = scalar_cpp(exact_factor)
    lines += scalar_candidate_rows(token, value.id, 0, source_cpp)
    lines += scalar_candidate_rows(token + "_status", value.id, 1, source_cpp)
    lines += [
        "std::exception_ptr %s_error;" % token,
        "pops::Real %s_invalid = 0;" % token,
        "try {",
        "  for (std::size_t li = 0; li < %s.local_size(); ++li) {" % token,
        "    auto output = %s.fab(li).view();" % token,
        "    auto status = %s_status.fab(li).view();" % token,
        "    const auto input = (%s).fab(li).view();" % source_cpp,
        "    pops::for_each_cell(%s.box(li), [=] POPS_HD(const pops::CellIndex<pops::kNativeDimension>& index) {" % token,
        "      const pops::Real selected = input(index, %d) * (%s);" % (component, factor),
        "      output(index, 0) = selected;",
        "      status(index, 0) = std::isfinite(selected) ? pops::Real(0) : pops::Real(1);",
        "    });", "  }", "  Kokkos::fence();",
        "  %s_invalid = pops::reduce_max_local(%s_status);" % (token, token),
        "} catch (...) { %s_error = std::current_exception(); }" % token,
        'pops::collectively_rethrow_exception(%s_error, ctx.prepared_execution_lane(), "mapped Field pack");' % token,
        "if (pops::all_reduce_max(%s_invalid, ctx.prepared_execution_lane()) > 0)" % token,
        '  throw std::invalid_argument("mapped consumed Field output is nonfinite");',
    ]
    var[value.id] = token
    var[("field_observation", value.id)] = source.attrs["field_problem_identity"]
