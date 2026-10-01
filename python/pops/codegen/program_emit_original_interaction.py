"""Collective candidate interaction producer for the native original F(q)."""
import json

from pops.identity import make_identity
from pops.identity.scalar import scalar_cpp
from pops.fields._program_expression import decode_field_literal
from pops.fields.spatial_interaction import kernel_cpp
from pops.fields._original_field_interaction import interaction_identity_data, interaction_budget


def interaction_terms(value):
    data = value.attrs["source_contract"].get("interactions")
    return () if data is None else tuple(data["terms"])


def declarations(value, stem, lines, *, amr):
    terms = interaction_terms(value)
    for i, term in enumerate(terms):
        dimension = term["kernel"]["dimension"]
        lines.append('static_assert(pops::kNativeDimension == %d, "original interaction kernel dimension differs from native geometry");' % dimension)
        if amr:
            lines.append("%s_Core::hierarchy_type %s_interaction_%d;" % (stem, stem, i))
        else:
            lines.append("std::optional<pops::MultiFab<pops::kNativeDimension>> %s_interaction_%d;" % (stem, i))
    return terms


def emit_producer(value, stem, terms, owner, lines, *, amr):
    if not terms:
        return
    data = value.attrs["source_contract"]["interactions"]
    budget = interaction_budget(data["realization"])
    if amr:
        lines.append("auto %s_interaction_producer = [&](const auto& q, std::uint64_t evaluation) {" % stem)
    for i, term in enumerate(terms):
        kernel = kernel_cpp(term["kernel"]["tree"], term["kernel"]["dimension"])
        identity = make_identity("original-field-interaction", interaction_identity_data(data)).token + ":term:" + str(i)
        call = ('ctx.original_candidate_interaction(%d, %d, %s_core, %s_authority, q, std::array<int, 1>{%d}, %dULL, %s, evaluation, '
                if amr else 'ctx.spatial_interaction(%d, q, std::array<int, 1>{%d}, %dULL, %s, %s, false, ')
        args = ((value.id, owner, stem, stem, term["column"], budget, json.dumps(identity)) if amr else
                (owner, term["column"], budget, json.dumps(identity), json.dumps(value.clock.qualified_id)))
        lines.append("  auto %s_detached_%d = %s" % (stem, i, call % args) +
                     "[=] POPS_HD(const auto& x, const auto& y) { return %s; });" % kernel)
        if amr:
            lines.append("  %s_interaction_%d.swap(%s_detached_%d);" % (stem, i, stem, i))
        else:
            # The move itself is allocation-free; converge before consuming any output.
            lines.append("  pops::runtime::program::interaction_phase(ctx.prepared_execution_lane(), [&] { %s_interaction_%d.emplace(std::move(%s_detached_%d)); });" % (stem, i, stem, i))
    if amr:
        lines.append("};")


def views(stem, terms, lines, *, amr):
    for i, _ in enumerate(terms):
        field = "%s_interaction_%d[level]" % (stem, i) if amr else "*%s_interaction_%d" % (stem, i)
        lines.append("      const auto interaction_%d = std::as_const(%s).fab(patch).view();" % (i, field))


def expression(component, terms):
    return "".join(" + (%s) * interaction_%d(index, 0)" % (scalar_cpp(decode_field_literal(term["scale"]).to_python()), i)
                   for i, term in enumerate(terms) if term["row"] == component)
