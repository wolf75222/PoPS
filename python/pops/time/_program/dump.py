"""Readable, non-codegen projections of an authored time Program."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops.time.references import block_name, state_name

if TYPE_CHECKING:
    from pops.time._program.contract import _ProgramBase
else:
    _ProgramBase = object


class _ProgramDump(_ProgramBase):
    """Operator-first, board and C++-plan inspection views."""

    def _render_node(self, value: Any) -> str:
        """Render one IR value as an operator-first line (introspection, not codegen)."""
        inputs = ", ".join(self._canonical_value(item).name for item in value.inputs)
        extra = ""
        keys = {key: item for key, item in value.attrs.items() if key != "coeffs"}
        if "coeffs" in value.attrs:
            extra = "  # coeffs=%s" % (value.attrs["coeffs"],)
        elif keys:
            extra = "  # %s" % (keys,)
        return "%-16s = P.%s(%s)%s" % (value.name, value.op, inputs, extra)

    def dump_operator_ir(self) -> str:
        """Return the operator-first view shared by primitive and board authoring."""
        lines = ["# operator-first Program IR: %s" % self.name]
        for value in self._values:
            lines.append("  " + self._render_node(value))
        for state_ref, state in self._commits.items():
            lines.append(
                "  T.commit(T.state(%s[%s]).next, %s)"
                % (
                    block_name(state_ref.block_ref),
                    state_name(state_ref),
                    state.name,
                )
            )
        return "\n".join(lines)

    def dump_board(self) -> str:
        """Return the board-level view, followed by its operator-first lowering."""
        return (
            "# board program %s lowers to the operator-first IR (board == operator-first):\n%s"
            % (self.name, self.dump_operator_ir())
        )

    def dump_cpp_plan(self) -> str:
        """Return a readable ProgramContext call plan, not generated C++."""
        lines = ["// C++ plan for GeneratedProgram step of %s" % self.name]
        for value in self._values:
            inputs = ", ".join(self._canonical_value(item).name for item in value.inputs)
            if value.op == "coupled_rate":
                blocks = ", ".join(
                    block_name(block) for block in value.attrs.get("blocks", [])
                )
                lines.append(
                    "  // %s: multi-state for_each_cell rate kernel over (%s) for blocks "
                    "[%s];  // ADC-457" % (value.name, inputs, blocks)
                )
            elif value.op == "coupled_rate_out":
                lines.append(
                    "  // %s = %s.rate[%r];  // ADC-457 (block projection, no ctx call)"
                    % (value.name, inputs, value.attrs.get("out_block"))
                )
            else:
                lines.append("  ctx.%s(%s);  // -> %s" % (value.op, inputs, value.name))
        for state_ref, state in self._commits.items():
            lines.append(
                "  ctx.commit(%r, %s);"
                % (block_name(state_ref.block_ref), state.name)
            )
        return "\n".join(lines)


__all__ = ["_ProgramDump"]
