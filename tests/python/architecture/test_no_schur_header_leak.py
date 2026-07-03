"""ADC-587: the generic runtime facade and a Schur-free Program must not leak Schur/Lorentz.

The Phase-4 refactor splits the condensed-Schur / Lorentz operator out of the ProgramContext
runtime facade into ``include/pops/coupling/schur/program/`` so that:

  1. ``include/pops/runtime/program/program_context.hpp`` -- the generic seam a generated problem.so
     always includes -- carries ZERO Schur/Lorentz/electrostatic tokens and no longer #includes the
     Schur condensation / geometric multigrid / Lorentz eliminator headers; and
  2. a Program whose IR has NO Schur op emits a .so whose #include set excludes ``coupling/schur/**``
     (only a condensed-Schur Program pulls the operator module in).

This file pins part 1 as a pure SOURCE-PARSE check (no ``pops`` / ``_pops`` import), so the
source-only architecture gate always executes it. Part 2 needs the compiled ``_pops`` extension
(``import pops`` loads it) and lives in tests/python/unit/codegen/test_program_schur_include.py,
the tier the CI shards run WITH the module built.
"""
import pathlib

# tests/python/architecture/<this file> -> repo root is parents[3].
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PROGRAM_CONTEXT = (REPO_ROOT / "include" / "pops" / "runtime" / "program"
                   / "program_context.hpp")

# Tokens that must never appear in the generic runtime facade after the split (case-insensitive,
# whole-word where it matters). ``electrostatic`` and ``CondensedSchur`` catch the operator names;
# ``GeometricMG`` catches the multigrid preconditioner state that moved out.
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
        "program_context.hpp must be Schur/Lorentz-free after the ADC-587 split, but it still "
        "mentions %s -- move the offending material into "
        "include/pops/coupling/schur/program/" % offenders
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


if __name__ == "__main__":
    # Runnable directly (the source-only architecture gate also collects it).
    test_program_context_has_no_schur_tokens()
    test_program_context_drops_schur_includes()
    print("OK test_no_schur_header_leak")
