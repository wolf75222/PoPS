"""A test-only C++ patch: the first evaluation of the default assessment ``finite`` of an Advance
rejects its candidate once, whatever its value (N8 lot « reuse across stages », note section 5).

The patch rewrites the emitted source of a System Program: a flag owned by the installed step
callback, false at install, is set by the first evaluation that passes the guard, which then
throws the rejection that the guard itself throws. Every later evaluation passes as emitted. The
patched source has its own content address (``program_artifact_spec(source=...)``), so its
artifact never shares a cache entry with the unpatched one. Nothing under ``python/pops``,
``src`` or ``include`` names this patch (``test_one_shot_rejection_is_test_only``).
"""
from __future__ import annotations

import contextlib

import pops.codegen.program_codegen as program_codegen

INSTALL = "  ctx.install([=](double dt) {\n"
GUARD = "Advance assessment 'finite' rejected the candidate"
FLAG = "  auto one_shot_rejection = std::make_shared<bool>(false);\n"
REJECTION = ("if (!*one_shot_rejection) { *one_shot_rejection = true; "
             "throw pops::runtime::program::StepAttemptRejected("
             "pops::SolveStatus::kInvalidEvaluation, \"assessment\", "
             "\"test-only one-shot rejection\"); }")


def patch_source(source: str) -> str:
    """``source`` with the one-shot rejection after the guard of ``finite``; refuses a source
    that has not exactly one install and one such guard."""
    if source.count(INSTALL) != 1:
        raise AssertionError("one-shot rejection: the source has %d System installs"
                             % source.count(INSTALL))
    lines = source.split("\n")
    guards = [index for index, line in enumerate(lines) if GUARD in line]
    if len(guards) != 1:
        raise AssertionError("one-shot rejection: the source has %d guards of 'finite'"
                             % len(guards))
    (index,) = guards
    if lines[index + 1].strip() != "}":
        raise AssertionError("one-shot rejection: the guard of 'finite' is not one throw")
    lines.insert(index + 2, REJECTION)
    return "\n".join(lines).replace(INSTALL, FLAG + INSTALL)


SUFFIX = " one-shot rejection"


@contextlib.contextmanager
def one_shot_rejection(monkeypatch, route):
    """Within the block, every emitted System Program carries the patch. ``route``, the route name
    of the compile that the block wraps, must end with the suffix ``one-shot rejection``: what
    separates the patched artifact is its content address, and the suffix keeps a production
    route name from ever naming a patched compile."""
    if not isinstance(route, str) or not route.endswith(SUFFIX):
        raise ValueError("one-shot rejection: the route %r does not end with %r" % (route, SUFFIX))
    with monkeypatch.context() as patch:
        for name in ("emit_cpp_program", "_emit_resolved_cpp_program"):
            emit = getattr(program_codegen, name)

            def patched(*args, _emit=emit, **kwargs):
                return patch_source(_emit(*args, **kwargs))
            patch.setattr(program_codegen, name, patched)
        yield


__all__ = ["SUFFIX", "one_shot_rejection", "patch_source"]
