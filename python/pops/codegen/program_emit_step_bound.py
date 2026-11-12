"""C++ of the step bound of a cells rule (spec 0.3, 2.6.2, PF8; N8 « dt= as a function of cell »,
design note ``n8/cell_step_rule/design_note.md``, D2, C4, P3).

``pops.Advance(method, dt=f)``, with ``f`` a function of ``cell``, bounds each step by the smallest
value over the valid cells of the evolving block, at the accepted state, of the expression E that
``f`` returns. The compiled Program exports ``pops_program_advance_step_bound``, which reduces one
mode of E over the cells of this rank (``include/pops/runtime/program/generated_step_bound.hpp``),
and the identity of the bound; ``System::step_bound`` reduces each mode over the ranks.

E is emitted as N12's indicators are (``program_emit_regrid_decision.py``): a cell read is the view
of the state at the cell, a width the spacing of the layout along its axis, a runtime parameter its
bound value. The operations are emitted in the order E writes them, in double precision, without
rewriting a quotient into a product by a reciprocal (P3), so ``(cfl * w) / s`` is b6's
``cfl * h / s``. Every operation is materialized and observed (C4), except:

* the branch of a ``where`` that the cell does not take: the conditional operator evaluates only the
  branch taken, and the names of the other stay at zero;
* a ``where`` itself, whose value is the branch it takes, an observed operation or a stated
  constant such as the +infinity of ``cfl_limit`` without ``max_dt`` (2.6.9);
* E itself, which may be +infinity, a step to the frontier.

E is NaN or -infinity, or an observed operation is not finite: the cell's code is
``kNonfinite`` (``step_bound_nonfinite``); E is zero or negative: ``kNonpositive``
(``step_bound_nonpositive``).
"""
from __future__ import annotations

from typing import Any

ENTRY_SYMBOL = "pops_program_advance_step_bound"
IDENTITY_SYMBOL = "pops_program_advance_step_bound_identity"
_INPUT_KIND = "step_bound_input"
_DIM = "pops::kNativeDimension"


def _cpp_string(text: str) -> str:
    if not text.isascii() or not text.isprintable() or '"' in text or "\\" in text:
        raise ValueError("a step bound identity is printable ASCII")
    return '"%s"' % text


def _parameter_indices(program: Any, model: Any, block: int, bound: Any) -> dict[str, int]:
    """The within-block index of each runtime parameter that E reads, from the table that the
    Program's kernels already route (ADC-510); a parameter that no kernel of the method reads has no
    installed value, and is refused."""
    if not bound.parameters:
        return {}
    from pops.codegen.program_emit_params import program_param_entries
    from pops.time._program.traced_method import NotYetElaborated

    entries = {name: index for blk, name, index, _ in program_param_entries(program, model)
               if blk == block}
    indices = {}
    for handle in bound.parameters:
        name = handle.local_id
        if name not in entries:
            raise NotYetElaborated(
                "dt= as a function of cell reads the runtime parameter %s, which no kernel of the "
                "method reads: its value is installed with the parameters of a kernel (a later "
                "lot of N8 « dt= as a function of cell »)" % name)
        indices[name] = entries[name]
    return indices


def _kernel(bound: Any, components: dict[str, int], parameters: dict[str, int]) -> list[str]:
    """The functor that returns one mode of E at a cell."""
    from pops._ir.application import substitute_quantities
    from pops._ir.control_expr import Where
    from pops._ir.expr import Var
    from pops._ir.values import RuntimeParamRef
    from pops._ir.visitors import _key
    from pops.mesh._amr.refinement_criterion import CellRead, CellWidth
    from pops.time._step.bound import _leaves

    from .cpp_writer import _cse_emit

    names: dict[Any, str] = {}
    bindings: dict[tuple[str, str], str] = {}
    replacements: dict[int, Any] = {}
    for leaf in _leaves(bound.expression):
        if not isinstance(leaf, (CellRead, CellWidth, RuntimeParamRef)):
            continue
        key = _key(leaf)
        name = names.get(key)
        if name is None:
            name = names[key] = "input%d_" % len(names)
            if isinstance(leaf, CellRead):
                rendered = "field(index, %d)" % components[leaf.component]
            elif isinstance(leaf, CellWidth):
                rendered = "spacing[%d]" % leaf.axis
            else:
                rendered = "params.get(%d)" % parameters[leaf.name]
            bindings[(_INPUT_KIND, name)] = rendered
        replacements[id(leaf)] = Var(name, _INPUT_KIND)
    (root,) = substitute_quantities((bound.expression,), {}, expression_bindings=replacements)
    root_key = _key(root)

    def observe(expression: Any) -> bool:
        return not isinstance(expression, Where) and _key(expression) != root_key

    lines, rendered, observed = _cse_emit(
        [root], "pops::Real", "    ", materialize_all=True, return_names=True,
        scalar_bindings=bindings, observe=observe)
    finite = " &&\n        ".join(["true"] + ["Kokkos::isfinite(%s)" % name for name in observed])
    return [
        "struct pops_program_advance_step_bound_kernel {",
        "  pops::FieldView<const pops::Real, %s> field;" % _DIM,
        "  std::array<pops::Real, %s> spacing{};" % _DIM,
        "  pops::RuntimeParams params{};",
        "  std::array<std::int64_t, %s> lower{};" % _DIM,
        "  std::array<std::int64_t, %s> length{};" % _DIM,
        "  int mode = 0;",
        "  pops::Real target = 0;",
        "",
        "  POPS_HD pops::Real operator()(const pops::Index<%s>& index) const {" % _DIM,
        *lines,
        "    const pops::Real bound = pops::Real(%s);" % rendered[0],
        "    const bool observed = %s;" % finite,
        "    const pops::Real infinity = Kokkos::Experimental::infinity<pops::Real>::value;",
        "    const pops::Real lowest = Kokkos::Experimental::finite_min<pops::Real>::value;",
        "    pops::Real code = pops::Real(1);",
        "    if (!observed || Kokkos::isnan(bound) || bound == -infinity)",
        "      code = pops::Real(3);",
        "    else if (!(bound > pops::Real(0)))",
        "      code = pops::Real(2);",
        "    else if (bound == infinity)",
        "      code = pops::Real(0);",
        "    if (mode == 0)",
        "      return code;",
        "    if (mode == 1)",
        "      return code == pops::Real(1) ? -bound : lowest;",
        "    if (code != target)",
        "      return lowest;",
        "    std::int64_t linear = 0;",
        "    for (int axis = %s - 1; axis >= 0; --axis)" % _DIM,
        "      linear = linear * length[axis] + (static_cast<std::int64_t>(index[axis]) - "
        "lower[axis]);",
        "    return -static_cast<pops::Real>(linear);",
        "  }",
        "};",
        "",
    ]


def emit_step_bound(program: Any, model: Any, target: str) -> str:
    """The kernel, the identity and the entry of the step bound of ``program``'s cells rule, or
    nothing for a Program without one."""
    bound = getattr(program, "_advance_step_bound", None)
    if bound is None:
        return ""
    from pops.time._program.traced_method import NotYetElaborated
    from pops.time._step.bound import ResolvedStepBound

    if type(bound) is not ResolvedStepBound:
        raise TypeError("Program step bound lost its resolved record")
    if target != "system":
        raise NotYetElaborated(
            "dt= as a function of cell under an adaptive layout: the bound of each level is a "
            "later lot of N8 « dt= as a function of cell » with N12 and N10")
    blocks = program._block_indices()
    if len(blocks) != 1:
        raise NotYetElaborated(
            "dt= as a function of cell on a Program of %d blocks: this lot bounds the step of one "
            "evolving block (a later lot of N8 « dt= as a function of cell »)" % len(blocks))
    (block,) = blocks.values()
    space = tuple(bound.state.space.components)
    components = {name: space.index(name) for name in bound.components}
    parameters = _parameter_indices(program, model, block, bound)
    kernel = _kernel(bound, components, parameters)
    params = ["  kernel.params = ctx.program_params(%d);" % block] if parameters else []
    return "\n".join([
        "",
        "// The step bound of the cells rule of the elaborated Advance (N8 « dt= as a function of",
        "// cell », D2): the smallest value over the valid cells of the evolving block at the accepted",
        "// state, reduced here over this rank's cells and by System::step_bound over the ranks.",
        "namespace {",
        "",
        *kernel,
        "}  // namespace",
        "",
        'extern "C" const char* %s() {' % IDENTITY_SYMBOL,
        "  return %s;" % _cpp_string(bound.qualified_id),
        "}",
        "",
        'extern "C" int %s(pops::System<%s>* sys, int mode, double target, double* result) {'
        % (ENTRY_SYMBOL, _DIM),
        "  if (result == nullptr || mode < 0 || mode > 2)",
        "    return 1;",
        "  auto ctx = pops::runtime::program::make_program_execution_view(sys);",
        "  const pops::MultiFab<%s>& state = ctx.state(%d);" % (_DIM, block),
        "  pops_program_advance_step_bound_kernel kernel{};",
        "  const auto& geometry = ctx.geometry();",
        "  for (int axis = 0; axis < %s; ++axis) {" % _DIM,
        "    kernel.spacing[axis] = geometry.spacing(axis);",
        "    kernel.lower[axis] = static_cast<std::int64_t>(geometry.domain().lo[axis]);",
        "    kernel.length[axis] = static_cast<std::int64_t>(geometry.domain().length(axis));",
        "  }",
        *params,
        "  kernel.mode = mode;",
        "  kernel.target = static_cast<pops::Real>(target);",
        "  pops::Real largest = std::numeric_limits<pops::Real>::lowest();",
        "  for (int local = 0; local < static_cast<int>(state.local_size()); ++local) {",
        "    const auto& box = state.box(local);",
        "    if (box.empty())",
        "      continue;",
        "    kernel.field = state.fab(local).view();",
        "    largest = std::max(largest, pops::for_each_cell_reduce_max(box, kernel));",
        "  }",
        "  *result = static_cast<double>(largest);",
        "  return 0;",
        "}",
        "",
    ])


__all__ = ["ENTRY_SYMBOL", "IDENTITY_SYMBOL", "emit_step_bound"]
