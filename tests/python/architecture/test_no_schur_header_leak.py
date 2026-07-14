"""The final generic Program runtime must not leak retired Schur/Lorentz machinery.

``ProgramContext`` is the backend-neutral seam included by every generated artifact.  Global
implicit work is authored as a matrix-free ``LinearProblem`` and a hierarchy provider is selected
explicitly; there is no public Schur program, solver or source-stage route.  This source-only gate
therefore locks the generic facade and native bindings against accidentally reintroducing that
retired vocabulary or its compatibility controls.
"""
import pathlib

# tests/python/architecture/<this file> -> repo root is parents[3].
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PROGRAM_CONTEXT = (REPO_ROOT / "include" / "pops" / "runtime" / "program"
                   / "program_context.hpp")
RETIRED_NATIVE_HEADERS = (
    "include/pops/coupling/schur/core/schur_condensation.hpp",
    "include/pops/coupling/schur/core/schur_source_kernels.hpp",
    "include/pops/coupling/schur/source/condensed_schur_source_stepper.hpp",
    "include/pops/coupling/schur/source/polar_condensed_schur_source_stepper.hpp",
    "include/pops/coupling/schur/amr/amr_condensed_schur_source_stepper.hpp",
    "include/pops/numerics/linalg/lorentz_eliminator.hpp",
)

# Tokens that must never appear in the generic runtime facade (case-insensitive, whole-word where it
# matters).  Provider-specific multigrid state belongs to the elliptic provider implementation.
_FORBIDDEN_TOKENS = ("schur", "lorentz", "electrostatic", "condensedschur", "geometricmg")

# Headers the facade must no longer include (the split moved their consumers to the Schur module).
_FORBIDDEN_INCLUDES = (
    "coupling/schur/core/schur_condensation.hpp",
    "numerics/elliptic/mg/geometric_mg.hpp",
    "numerics/linalg/lorentz_eliminator.hpp",
)


def test_program_context_has_no_schur_tokens():
    """program_context.hpp carries zero Schur/Lorentz/electrostatic tokens (ADC-587)."""
    assert PROGRAM_CONTEXT.exists(), "program_context.hpp not found at %s" % PROGRAM_CONTEXT
    text = PROGRAM_CONTEXT.read_text(encoding="utf-8")
    lowered = text.lower()
    offenders = [tok for tok in _FORBIDDEN_TOKENS if tok in lowered]
    assert not offenders, (
        "program_context.hpp must remain backend-neutral, but it still mentions retired "
        "Schur/Lorentz machinery %s; move provider-specific work behind the generic elliptic "
        "provider seam" % offenders
    )


def test_program_context_drops_schur_includes():
    """program_context.hpp no longer #includes the Schur / MG / Lorentz headers (ADC-587)."""
    text = PROGRAM_CONTEXT.read_text(encoding="utf-8")
    include_lines = [ln for ln in text.splitlines() if ln.lstrip().startswith("#include")]
    joined = "\n".join(include_lines)
    leaked = [inc for inc in _FORBIDDEN_INCLUDES if inc in joined]
    assert not leaked, (
        "program_context.hpp must not include the Schur/MG/Lorentz headers after the split, but it "
        "still includes %s" % leaked
    )


def test_native_source_stage_headers_are_retired():
    """The generated Program route is the only condensed time-integration path."""
    leaked = [path for path in RETIRED_NATIVE_HEADERS if (REPO_ROOT / path).exists()]
    assert not leaked, "retired native condensed-source headers still exist: %s" % leaked


def test_native_bindings_do_not_reintroduce_source_stage_controls():
    """The System and AMR pybind surfaces expose no inert compatibility setters."""
    binding_files = (
        REPO_ROOT / "python" / "bindings" / "core" / "init" / "init_system.cpp",
        REPO_ROOT / "python" / "bindings" / "core" / "init" / "init_amr.cpp",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in binding_files)
    for retired in ('def("set_source_stage"', 'def("set_time_scheme"',
                    'def("set_gauss_policy"'):
        assert retired not in text, "retired native binding leaked: %s" % retired


if __name__ == "__main__":
    # Runnable directly (the source-only architecture gate also collects it).
    test_program_context_has_no_schur_tokens()
    test_program_context_drops_schur_includes()
    test_native_source_stage_headers_are_retired()
    test_native_bindings_do_not_reintroduce_source_stage_controls()
    print("OK test_no_schur_header_leak")
