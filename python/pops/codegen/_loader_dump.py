"""IR, generated-C++ and schedule dumps shared by compiled problem handles."""
from __future__ import annotations

from typing import Any


class CompiledProblemDumpMixin:
    """Inert debug exports kept separate from compiled-artifact metadata."""

    def dump_ir(self, path: Any = None) -> Any:
        """Return or write the exact serialized Program IR."""
        import json

        program = self._require_program("dump_ir")
        blob = json.dumps(program._serialize(), indent=2, sort_keys=True)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(blob)
            return path
        return blob

    def dump_cpp(self, target: Any) -> Any:
        """Write the exact compiler-owned C++ translation unit."""
        import os

        program = self._require_program("dump_cpp")
        src = self._generated_cpp
        if src is None:
            # Advanced handles built outside compile_problem have no persisted translation unit.
            src = program.emit_cpp_program(model=self.model)
        name = self.program_name or "problem"
        if str(target).endswith(".cpp"):
            out_path = str(target)
            parent = os.path.dirname(out_path) or "."
        else:
            parent = str(target)
            out_path = os.path.join(parent, "%s.cpp" % name)
        if not os.path.isdir(parent):
            raise NotADirectoryError(
                "dump_cpp: the target directory %r does not exist; create it first "
                "(dump_cpp does not allocate or create directories)." % (parent,))
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(src)
        return out_path

    def dump_schedule(self, path: Any = None) -> Any:
        """Return or write the Program's deterministic block commit order."""
        program = self._require_program("dump_schedule")
        commits = program.commits()
        order = program._block_indices() if hasattr(program, "_block_indices") else {}
        ordered = sorted(
            commits, key=lambda state_ref: order.get(state_ref.block_ref, len(order)))
        lines = ["schedule for Program %r (block commit order):"
                 % (self.program_name or "problem")]
        from pops.time.references import block_name

        for state_ref in ordered:
            state = commits[state_ref]
            block = state_ref.block_ref
            lines.append("  %2d  commit %-14s <- %s"
                         % (order.get(block, -1), block_name(block),
                            getattr(state, "name", "?")))
        if not ordered:
            lines.append("  (no committed block)")
        text = "\n".join(lines)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    def _require_program(self, who: Any) -> Any:
        """Return the carried Program or fail without fabricating debug data."""
        program = self.program
        if program is None:
            raise ValueError(
                "%s: this CompiledProblem carries no Program (the lowered pops.time.Program is "
                "unavailable on this handle), so the IR / C++ / schedule cannot be dumped." % who)
        return program


__all__ = ["CompiledProblemDumpMixin"]
