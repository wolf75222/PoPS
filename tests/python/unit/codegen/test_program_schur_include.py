"""ADC-587: the generated problem.so pulls the condensed-Schur operator header ONLY on a Schur route.

The runtime-half sibling of tests/python/architecture/test_no_schur_header_leak.py (which pins the
source-parse hygiene of program_context.hpp without importing pops): these checks lower a real
Forward-Euler and a real condensed-Schur Program through the codegen, so they need the compiled
_pops extension and live in the unit/codegen tier the CI shards run WITH the module built -- in the
source-only architecture lane they would never execute.
"""
import pytest

# import pops loads _pops; skip honestly on a source-only checkout (the bootstrap wraps the raw
# ModuleNotFoundError in a plain ImportError, hence exc_type).
pops_time = pytest.importorskip("pops.time", exc_type=ImportError)
pops_lib_time = pytest.importorskip("pops.lib.time", exc_type=ImportError)


def _emit(program_name, build):
    """Lower a Program built by @p build(P) and return its emitted .so source."""
    program = pops_time.Program(program_name)
    build(program)
    return program.emit_cpp_program()


def test_forward_euler_program_excludes_schur_includes():
    """A Schur-free (Forward-Euler) Program's generated .so excludes coupling/schur/** (ADC-587)."""
    src = _emit("fe", lambda P: pops_lib_time.forward_euler(P, "gas"))
    assert "program/program_context.hpp" in src, "the .so must always include the runtime facade"
    assert "coupling/schur" not in src, (
        "a Schur-free Program's generated .so must not include coupling/schur/** -- the codegen "
        "should emit the condensed-Schur operator header only when a Schur op is in the IR"
    )


def test_condensed_schur_program_includes_operator_module():
    """A condensed-Schur Program's generated .so DOES pull the operator module (ADC-587 positive)."""
    src = _emit("schur", lambda P: pops_lib_time.condensed_schur(P, "gas", alpha=1.0, theta=1.0))
    assert "coupling/schur/program/condensed_schur_operator.hpp" in src, (
        "a condensed-Schur Program must include the native operator module header"
    )
    assert "pops::coupling::schur::program::" in src, (
        "the schur ops must lower to the pops::coupling::schur::program:: free functions"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
