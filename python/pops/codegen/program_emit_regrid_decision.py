"""C++ of the regrid decision written in the Program, ``Regrid(criterion, when)`` (spec 0.3, DM13b).

The compiled Program of an adaptive layout exports two entries next to its refinement criterion.
``pops_program_regrid_indicator_amr`` evaluates one indicator of the ``when`` body over the valid
cells of one patch: a cell read is the field view of its state at the cell, a width is the cell
size of the level along its axis, and a runtime parameter is its bound value. Every operation is
materialized and observed, as for the criterion, so a non-finite value fails the decision. The
largest value is reduced over the patch; a ``smallest_cells`` indicator is generated as the largest
of the opposite value, so the core reduces every indicator with the same maximum over the patches
and the ranks. ``pops_program_regrid_decision_amr`` evaluates the body, two-valued as the ``cond``
of IR4, from the accepted step count, the accepted time and the reduced indicators.
"""
from __future__ import annotations

from typing import Any

INDICATOR_ENTRY_SYMBOL = "pops_program_regrid_indicator_amr"
DECISION_ENTRY_SYMBOL = "pops_program_regrid_decision_amr"
DECISION_IDENTITY_SYMBOL = "pops_program_regrid_decision_identity"
_INPUT_KIND = "regrid_decision_input"
_DIM = "pops::kNativeDimension"


def _cpp_string(text: str) -> str:
    if not text.isascii() or not text.isprintable() or '"' in text or "\\" in text:
        raise ValueError("a regrid decision identity must be printable ASCII")
    return '"%s"' % text


def _indicator_kernel(decision: Any, index: int, indicator: Any) -> list[str]:
    """One functor that returns the value of the indicator at a cell, or its finiteness flag."""
    from pops._ir.application import substitute_quantities
    from pops._ir.expr import Var
    from pops._ir.handle_expr import ValueExpr
    from pops._ir.visitors import _key
    from pops.mesh._amr.refinement_criterion import CellRead, CellWidth, criterion_inputs

    from .cpp_writer import _cse_emit

    fields = {row.state.qualified_id: number for number, row in enumerate(decision.fields)}
    components = {row.state.qualified_id: dict(zip(row.variables, row.components, strict=True))
                  for row in decision.fields}
    parameters = {handle.qualified_id: number
                  for number, handle in enumerate(decision.parameters)}
    names: dict[Any, str] = {}
    bindings: dict[tuple[str, str], str] = {}
    replacements: dict[int, Any] = {}
    for leaf in criterion_inputs((indicator.expression,)):
        key = _key(leaf)
        name = names.get(key)
        if name is None:
            name = names[key] = "input%d_" % len(names)
            if isinstance(leaf, CellRead):
                if leaf.offset:
                    raise NotImplementedError("an indicator reads the cell itself in this lot")
                state = leaf.handle.qualified_id
                rendered = "fields[%d](index, %d)" % (fields[state],
                                                      components[state][leaf.component])
            elif isinstance(leaf, CellWidth):
                rendered = "spacing[%d]" % leaf.axis
            elif isinstance(leaf, ValueExpr):
                rendered = "parameters[%d]" % parameters[leaf.handle.qualified_id]
            else:
                raise TypeError("an indicator input is a cell read, a width or a runtime "
                                "parameter")
            bindings[(_INPUT_KIND, name)] = rendered
        replacements[id(leaf)] = Var(name, _INPUT_KIND)
    (root,) = substitute_quantities((indicator.expression,), {},
                                    expression_bindings=replacements)
    lines, rendered, observed = _cse_emit(
        [root], "pops::Real", "    ", materialize_all=True, return_names=True,
        scalar_bindings=bindings)
    value = rendered[0] if indicator.kind == "largest" else "-(%s)" % rendered[0]
    finite = " &&\n                        ".join(
        ["Kokkos::isfinite(pops::Real(%s))" % value]
        + ["Kokkos::isfinite(%s)" % name for name in observed])
    members = [
        "  pops::FieldView<const pops::Real, %s> fields[%d];" % (_DIM, max(len(decision.fields), 1)),
        "  std::array<pops::Real, %s> spacing;" % _DIM,
        "  pops::Real parameters[%d];" % max(len(decision.parameters), 1),
        "  bool flag;",
    ]
    return [
        "struct pops_program_regrid_indicator_%d {" % index,
        *members,
        "",
        "  POPS_HD pops::Real operator()(const pops::Index<%s>& index) const {" % _DIM,
        *lines,
        "    const bool finite = %s;" % finite,
        "    if (flag)",
        "      return finite ? pops::Real(0) : pops::Real(1);",
        "    return pops::Real(%s);" % value,
        "  }",
        "};",
        "",
    ]


def _decision_body(decision: Any) -> tuple[list[str], str]:
    from pops._ir.application import substitute_quantities
    from pops._ir.expr import Var
    from pops._ir.handle_expr import ValueExpr
    from pops._ir.visitors import _key
    from pops.mesh._amr.regrid_decision import (
        AcceptedTime, CellIndicator, EveryAcceptedSteps, _leaves,
    )
    from pops._ir.expr import Const

    from .cpp_writer import _cse_emit

    indicators = {_key(indicator): number for number, indicator in enumerate(decision.indicators)}
    parameters = {handle.qualified_id: number
                  for number, handle in enumerate(decision.parameters)}
    names: dict[Any, str] = {}
    bindings: dict[tuple[str, str], str] = {}
    replacements: dict[int, Any] = {}
    for leaf in _leaves((decision.expression,)):
        if isinstance(leaf, Const):
            continue
        key = _key(leaf)
        name = names.get(key)
        if name is None:
            name = names[key] = "input%d_" % len(names)
            if isinstance(leaf, CellIndicator):
                number = indicators[key]
                rendered = ("request->indicators[%d]" % number if leaf.kind == "largest"
                            else "(-request->indicators[%d])" % number)
            elif isinstance(leaf, AcceptedTime):
                rendered = "request->time"
            elif isinstance(leaf, EveryAcceptedSteps):
                rendered = ("((request->steps > 0 && request->steps %% %d == 0) ? pops::Real(1) "
                            ": pops::Real(0))" % leaf.k)
            elif isinstance(leaf, ValueExpr):
                rendered = "request->parameters[%d]" % parameters[leaf.handle.qualified_id]
            else:
                raise TypeError("a when body reads B.steps, B.time, indicators and runtime "
                                "parameters")
            bindings[(_INPUT_KIND, name)] = rendered
        replacements[id(leaf)] = Var(name, _INPUT_KIND)
    (root,) = substitute_quantities((decision.expression,), {}, expression_bindings=replacements)
    lines, rendered = _cse_emit([root], "pops::Real", "  ", scalar_bindings=bindings)
    return lines, rendered[0]


def emit_regrid_decision(decision: Any, *, native_dimension: int) -> str:
    """The indicator kernels, the identity and the two entries of one resolved regrid decision."""
    from pops.mesh._amr import ResolvedRegridDecision

    if type(decision) is not ResolvedRegridDecision:
        raise TypeError("regrid decision emission requires a ResolvedRegridDecision")
    if decision.dimension != native_dimension:
        raise ValueError("a regrid decision of dimension %d cannot be compiled for native "
                         "dimension %d" % (decision.dimension, native_dimension))
    kernels: list[str] = []
    cases: list[str] = []
    field_count = len(decision.fields)
    parameter_count = len(decision.parameters)
    for index, indicator in enumerate(decision.indicators):
        kernels += _indicator_kernel(decision, index, indicator)
        cases += [
            "    case %d: {" % index,
            "      pops_program_regrid_indicator_%d kernel{};" % index,
            "      for (std::size_t field = 0; field < %d; ++field)" % field_count,
            "        kernel.fields[field] = patch->fields[field];",
            "      kernel.spacing = patch->spacing;",
            "      for (std::size_t parameter = 0; parameter < %d; ++parameter)" % parameter_count,
            "        kernel.parameters[parameter] = patch->parameters[parameter];",
            "      kernel.flag = true;",
            "      *patch->non_finite =",
            "          pops::for_each_cell_reduce_max(patch->box, kernel) != pops::Real(0) ? 1 : 0;",
            "      kernel.flag = false;",
            "      *patch->largest = pops::for_each_cell_reduce_max(patch->box, kernel);",
            "      return 0;",
            "    }",
        ]
    checks = ["patch == nullptr", "patch->largest == nullptr", "patch->non_finite == nullptr",
              "patch->field_count != %d" % field_count,
              "patch->parameter_count != %d" % parameter_count,
              "patch->indicator >= %d" % len(decision.indicators)]
    if field_count:
        checks.append("patch->fields == nullptr")
    if parameter_count:
        checks.append("patch->parameters == nullptr")
    lines, root = _decision_body(decision)
    decision_checks = ["request == nullptr", "request->regrid == nullptr",
                       "request->indicator_count != %d" % len(decision.indicators),
                       "request->parameter_count != %d" % parameter_count]
    if decision.indicators:
        decision_checks.append("request->indicators == nullptr")
    if parameter_count:
        decision_checks.append("request->parameters == nullptr")
    switch = ["  switch (patch->indicator) {", *cases, "    default:", "      return 1;", "  }"] \
        if cases else ["  return 1;"]
    return "\n".join([
        "",
        "// The regrid decision of this Program (spec 0.3, DM13b): %s." % decision.qualified_id,
        "#include <pops/mesh/execution/for_each.hpp>",
        "#include <pops/runtime/amr/generated_regrid_decision.hpp>",
        "#include <Kokkos_Core.hpp>",
        "#include <array>",
        "#include <cmath>",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace {",
        "",
        *kernels,
        "}  // namespace",
        "",
        'extern "C" const char* %s() {' % DECISION_IDENTITY_SYMBOL,
        "  return %s;" % _cpp_string(decision.qualified_id),
        "}",
        "",
        'extern "C" int %s(' % INDICATOR_ENTRY_SYMBOL,
        "    const pops::runtime::amr::GeneratedIndicatorPatch<%s>* patch) {" % _DIM,
        "  if (%s)" % " ||\n      ".join(checks),
        "    return 1;",
        *switch,
        "}",
        "",
        'extern "C" int %s(' % DECISION_ENTRY_SYMBOL,
        "    const pops::runtime::amr::GeneratedRegridDecisionRequest* request) {",
        "  if (%s)" % " ||\n      ".join(decision_checks),
        "    return 1;",
        *lines,
        "  *request->regrid = (%s) != pops::Real(0) ? 1 : 0;" % root,
        "  return 0;",
        "}",
        "",
    ])


__all__ = ["DECISION_ENTRY_SYMBOL", "DECISION_IDENTITY_SYMBOL", "INDICATOR_ENTRY_SYMBOL",
           "emit_regrid_decision"]
