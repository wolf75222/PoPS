"""C++ of a refinement criterion written as regions of ``cell`` (spec 0.3, DM13b).

The compiled Program of an adaptive layout exports the criterion as one entry that evaluates it on
the valid cells of one patch. Each input of the criterion becomes a local of the cell kernel: a cell
read is the field view of its state at the shifted index, a width is the cell size of the level
along its axis, and a runtime parameter is its bound value. The regions go through the common CSE
emitter with every operation materialized, so the kernel observes each value that it computes: a
non-finite one marks the cell, and the tagging plan then rejects the level as it does for the
bytecode of the tag graph.

The regions of one action are evaluated in full and joined after, as the bytecode joins its rules,
so a non-finite value read by any rule marks the cell. Inside one region, ``&``, ``|`` and
``where`` evaluate their right operand only when the left one does not decide.
"""
from __future__ import annotations

from typing import Any

CRITERION_ENTRY_SYMBOL = "pops_program_refinement_criterion_amr"
CRITERION_IDENTITY_SYMBOL = "pops_program_refinement_criterion_identity"
_INPUT_KIND = "refinement_criterion_input"
_KERNEL = "pops_program_refinement_criterion_kernel"
_DIM = "pops::kNativeDimension"
_MARK = "pops::runtime::amr::kGeneratedCriterion"


def _cpp_string(text: str) -> str:
    if not text.isascii() or not text.isprintable() or '"' in text or "\\" in text:
        raise ValueError("a refinement criterion identity must be printable ASCII")
    return '"%s"' % text


def _inputs(criterion: Any) -> tuple[dict[int, Any], dict[tuple[str, str], str], list[str]]:
    """The replacement of each input leaf, the C++ of each input local and the shifted indices."""
    from pops._ir.expr import Var
    from pops._ir.handle_expr import ValueExpr
    from pops._ir.visitors import _key
    from pops.mesh._amr.refinement_criterion import CellRead, CellWidth, criterion_inputs

    fields = {row.state.qualified_id: index for index, row in enumerate(criterion.fields)}
    components = {row.state.qualified_id: dict(zip(row.variables, row.components, strict=True))
                  for row in criterion.fields}
    parameters = {handle.qualified_id: index for index, handle in enumerate(criterion.parameters)}
    offsets: dict[tuple[tuple[int, int], ...], str] = {}
    names: dict[Any, str] = {}
    bindings: dict[tuple[str, str], str] = {}
    replacements: dict[int, Any] = {}
    for leaf in criterion_inputs(criterion.refine + criterion.coarsen):
        key = _key(leaf)
        name = names.get(key)
        if name is None:
            name = names[key] = "input%d_" % len(names)
            if isinstance(leaf, CellRead):
                state = leaf.handle.qualified_id
                at = "index"
                if leaf.offset:
                    at = offsets.setdefault(leaf.offset, "at%d_" % len(offsets))
                rendered = "fields[%d](%s, %d)" % (fields[state], at,
                                                   components[state][leaf.component])
            elif isinstance(leaf, CellWidth):
                rendered = "spacing[%d]" % leaf.axis
            elif isinstance(leaf, ValueExpr):
                rendered = "parameters[%d]" % parameters[leaf.handle.qualified_id]
            else:
                raise TypeError("a refinement criterion input is a cell read, a width or a "
                                "runtime parameter")
            bindings[(_INPUT_KIND, name)] = rendered
        replacements[id(leaf)] = Var(name, _INPUT_KIND)
    shifted = []
    for offset, name in offsets.items():
        shifted.append("    pops::Index<%s> %s = index;" % (_DIM, name))
        shifted.extend("    %s[%d] += %d;" % (name, axis, steps) for axis, steps in offset)
    return replacements, bindings, shifted


def emit_refinement_criterion(criterion: Any, *, native_dimension: int) -> str:
    """The kernel, the identity and the patch entry of one resolved refinement criterion."""
    from pops._ir.application import substitute_quantities
    from pops.mesh._amr import ResolvedRefinementCriterion

    from .cpp_writer import _cse_emit

    if type(criterion) is not ResolvedRefinementCriterion:
        raise TypeError("refinement criterion emission requires a ResolvedRefinementCriterion")
    if criterion.dimension != native_dimension:
        raise ValueError(
            "a refinement criterion of dimension %d cannot be compiled for native dimension %d"
            % (criterion.dimension, native_dimension))
    replacements, bindings, shifted = _inputs(criterion)
    roots = substitute_quantities(
        criterion.refine + criterion.coarsen, {}, expression_bindings=replacements)
    lines, rendered, observed = _cse_emit(
        list(roots), "pops::Real", "    ", materialize_all=True, return_names=True,
        scalar_bindings=bindings)
    count = len(criterion.refine)

    def any_of(values: list[str]) -> str:
        return " || ".join("(%s != pops::Real(0))" % value for value in values) or "false"

    field_count = len(criterion.fields)
    parameter_count = len(criterion.parameters)
    members = [
        "  pops::FieldView<const pops::Real, %s> fields[%d];" % (_DIM, field_count),
        "  pops::runtime::amr::GeneratedCriterionMarks<%s> marks;" % _DIM,
        "  std::array<pops::Real, %s> spacing;" % _DIM,
    ]
    if parameter_count:
        members.append("  pops::Real parameters[%d];" % parameter_count)
    finite = " &&\n                        ".join(
        "Kokkos::isfinite(%s)" % name for name in observed)
    body = shifted + lines + [
        "    const bool finite = %s;" % finite,
        "    const bool refine = %s;" % any_of(rendered[:count]),
        "    const bool coarsen = %s;" % any_of(rendered[count:]),
        "    marks(index) = !finite ? std::uint8_t{%sNonFinite}" % _MARK,
        "                   : refine ? std::uint8_t{%sRefine}" % _MARK,
        "                   : coarsen ? std::uint8_t{%sCoarsen}" % _MARK,
        "                   : std::uint8_t{0};",
    ]
    checks = ["patch == nullptr", "patch->fields == nullptr", "patch->marks == nullptr",
              "patch->field_count != %d" % field_count,
              "patch->parameter_count != %d" % parameter_count]
    copies = [
        "  for (std::size_t field = 0; field < %d; ++field)" % field_count,
        "    kernel.fields[field] = patch->fields[field];",
        "  kernel.marks.values = patch->marks;",
        "  kernel.marks.box = patch->box;",
        "  kernel.spacing = patch->spacing;",
    ]
    if parameter_count:
        checks.append("patch->parameters == nullptr")
        copies += [
            "  for (std::size_t parameter = 0; parameter < %d; ++parameter)" % parameter_count,
            "    kernel.parameters[parameter] = patch->parameters[parameter];",
        ]
    return "\n".join([
        "",
        "// The refinement criterion of this Program (spec 0.3, DM13b): %s." % criterion.qualified_id,
        "#include <pops/mesh/execution/for_each.hpp>",
        "#include <pops/runtime/amr/generated_refinement_criterion.hpp>",
        "#include <Kokkos_Core.hpp>",
        "#include <array>",
        "#include <cmath>",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace {",
        "",
        "struct %s {" % _KERNEL,
        *members,
        "",
        "  POPS_HD void operator()(const pops::Index<%s>& index) const {" % _DIM,
        *body,
        "  }",
        "};",
        "",
        "}  // namespace",
        "",
        'extern "C" const char* %s() {' % CRITERION_IDENTITY_SYMBOL,
        "  return %s;" % _cpp_string(criterion.qualified_id),
        "}",
        "",
        'extern "C" int %s(' % CRITERION_ENTRY_SYMBOL,
        "    const pops::runtime::amr::GeneratedCriterionPatch<%s>* patch) {" % _DIM,
        "  if (%s)" % " ||\n      ".join(checks),
        "    return 1;",
        "  %s kernel{};" % _KERNEL,
        *copies,
        "  pops::for_each_cell(patch->box, kernel);",
        "  return 0;",
        "}",
        "",
    ])


__all__ = ["CRITERION_ENTRY_SYMBOL", "CRITERION_IDENTITY_SYMBOL", "emit_refinement_criterion"]
