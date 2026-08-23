"""Bounded compiler-diagnostic retention contract for native code generation."""
from __future__ import annotations

import re

import pytest


def test_long_compiler_diagnostic_keeps_warning_head_and_terminal_error() -> None:
    from pops.codegen.toolchain import _COMPILER_DIAGNOSTIC_BUDGET, _format_compiler_output

    warning_head = "warning: generated CUDA declaration\n" * 400
    terminal_error = "nvcc fatal: distinctive terminal diagnostic\n"
    formatted = _format_compiler_output(warning_head + terminal_error)

    assert formatted.startswith("warning: generated CUDA declaration\n")
    assert formatted.endswith(terminal_error)
    omitted = re.search(r"\[(\d+) compiler-output characters omitted\]", formatted)
    assert omitted is not None
    assert int(omitted.group(1)) > 0
    assert len(formatted) <= _COMPILER_DIAGNOSTIC_BUDGET


def test_compiler_diagnostic_formatter_preserves_short_and_empty_text() -> None:
    from pops.codegen.toolchain import _format_compiler_output

    assert _format_compiler_output("short compiler error") == "short compiler error"
    assert _format_compiler_output(b"decoded compiler bytes") == "decoded compiler bytes"
    assert _format_compiler_output("") == ""


def test_run_compile_reports_the_terminal_diagnostic_after_warning_flood(monkeypatch) -> None:
    from pops.codegen import toolchain

    warning_head = b"warning: generated CUDA declaration\n" * 400
    terminal_error = b"nvcc fatal: distinctive terminal diagnostic\n"
    result = type("CompileResult", (), {"returncode": 2, "stderr": warning_head + terminal_error,
                                         "stdout": b""})()
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: result)

    with pytest.raises(RuntimeError, match="distinctive terminal diagnostic") as failure:
        toolchain._run_compile(["nvcc_wrapper", "model.cpp"], "native loader")

    assert "Command: nvcc_wrapper model.cpp" in str(failure.value)
    assert "compiler-output characters omitted" in str(failure.value)
