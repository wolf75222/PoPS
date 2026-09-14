"""Reuse transparent numerical source implementations inside one Program translation unit.

Owner-qualified calls and provider plans remain separate. This collector only
shares an identical function body; it never shares evaluations, outputs or launches.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def _transparent_roots(impl: Any, roots: Any) -> bool:
    from pops._ir.expr import (Abs, Add, Compare, Const, Div, Maximum, Minimum,
                              Mul, Neg, Pow, Sqrt, Sub, Var)
    from pops._ir.visitors import _children

    allowed = {Abs, Add, Compare, Const, Div, Maximum, Minimum, Mul, Neg, Pow, Sqrt, Sub, Var}
    seen: set[int] = set()

    def visit(value: Any) -> bool:
        if id(value) in seen:
            return True
        seen.add(id(value))
        if type(value) not in allowed:
            return False  # Includes native effects and runtime parameter reads.
        if type(value) is Var:
            if value.kind == "prim":
                recipe = impl.prim_defs.get(value.name)
                return recipe is not None and visit(recipe)
            return value.kind == "cons" and value.name in impl.cons_names
        return all(visit(child) for child in _children(value))

    return all(visit(root) for root in roots)


class ProgramSourceKernelHelpers:
    """Private emission collector scoped to one exact Program compilation."""

    def __init__(self) -> None:
        self._definitions: dict[str, str] = {}

    def call(self, impl: Any, roots: Any, *, binding: Any, state_var: str,
             out_var: str, block_index: int) -> list[str] | None:
        if binding["count"] != 0 or not _transparent_roots(impl, roots):
            return None
        from pops.codegen.program_emit_kernels import _cell_locals

        # These fixed input/output names are private C++ argument names, never
        # scientific identities. Authentication has already bound every leaf.
        numerical = _cell_locals(impl, roots, "kernel_input", with_cons=True,
                                 with_prim=True, provider_binding=binding)
        numerical += ["outA(index, %d) = %s;" % (i, root.to_cpp())
                      for i, root in enumerate(roots)]
        owner = str(impl.owner_path.canonical())
        key = json.dumps({"owner": owner, "components": tuple(impl.cons_names),
                          "numerical": numerical}, sort_keys=True, separators=(",", ":"))
        name = "pops_shared_source_" + hashlib.sha256(key.encode()).hexdigest()
        definition = "\n".join([
            "// Shared numerical implementation; each call retains its own evaluation and owner.",
            "template <class Context>",
            "inline void %s(Context& ctx, const char* consumer_qid, int block_index," % name,
            "    const pops::MultiFab<pops::kNativeDimension>& kernel_input,",
            "    pops::MultiFab<pops::kNativeDimension>& kernel_output) {",
            "  for (int li = 0; li < kernel_output.local_size(); ++li) {",
            "    const pops::FieldView<pops::Real, pops::kNativeDimension> outA = "
            "kernel_output.fab(li).view();",
            "    const pops::FieldView<const pops::Real, pops::kNativeDimension> "
            "kernel_inputA = kernel_input.fab(li).view();",
            # Keep the exact zero-provider binding check at the original call site
            # in execution order; no numerical work occurs at helper definition.
            "    const auto providers = ctx.template provider_values_view<0>("
            "consumer_qid, block_index, li);",
            "    pops::for_each_cell(kernel_output.box(li), [=] POPS_HD("
            "const pops::CellIndex<pops::kNativeDimension>& index) {",
            *("      " + line for line in numerical),
            "    });",
            "  }",
            "}",
        ])
        previous = self._definitions.setdefault(name, definition)
        if previous != definition:
            raise ValueError("shared source implementation identity collision")
        return ["%s(ctx, %s, %d, %s, %s);" % (
            name, json.dumps(binding["qid"]), block_index, state_var, out_var)]

    def cpp(self) -> str:
        return "\n\n".join(self._definitions.values()) + ("\n" if self._definitions else "")
