"""The non-negotiable source contract for a final PoPS release.

This module deliberately contains only identities which are reviewed with the
release process.  Both the executable gate and the release preflight import it,
so a new example cannot be silently added to one path but omitted from the
other.
"""
from __future__ import annotations

from pathlib import Path


FINAL_SPECIFICATION = Path("docs/design/SPECIFICATION_TECHNIQUE_FINALE_POPS_ARCHITECTURE.md")
FINAL_EXAMPLES = (
    Path("examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py"),
    Path("examples/final/EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py"),
    Path("examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py"),
    Path("examples/final/EXEMPLE_SPEC_FINALE_15_MOMENTS_HYQMOM.py"),
)
REQUIRED_PROOF_MARKERS = (
    "HDF5:",
    "ParaView:",
    "checkpoint:",
    "bit-identical restart:",
)
# The published wheel matrix is CPU/Kokkos Serial without MPI or parallel HDF5. The full suite still
# runs; this supported-platform subset is repeated with a strict all-pass/no-hidden-skip policy.
PYTHON_REQUIRED_SELECTION = "not mpi and not hdf5"
REQUIRED_RELEASE_GATES = (
    "official_build",
    "doctor",
    "codesign",
    "native_conformance",
    "python_conformance",
    "examples",
    "artifact_reopen",
    "strict_restart",
    "documentation",
    "generated_products",
    "diff",
)


def source_contract_errors(root: Path) -> list[str]:
    """Return every deterministic final-source contract violation.

    This is intentionally source-only: it is used before starting a costly
    build and can be exercised in isolation by architecture tests.
    """

    errors: list[str] = []
    specification = root / FINAL_SPECIFICATION
    if not specification.is_file() or not specification.read_text(encoding="utf-8").strip():
        errors.append("missing canonical final specification: %s" % FINAL_SPECIFICATION)

    examples_dir = root / "examples" / "final"
    actual = tuple(sorted(path.relative_to(root) for path in examples_dir.glob("*.py"))) \
        if examples_dir.is_dir() else ()
    expected = tuple(sorted(FINAL_EXAMPLES))
    if actual != expected:
        errors.append("final examples must be exactly %s (found %s)" % (expected, actual))

    for relative in FINAL_EXAMPLES:
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if "--output-dir" not in text:
            errors.append("%s must accept an explicit --output-dir" % relative)
        if 'if __name__ == "__main__"' not in text:
            errors.append("%s must remain directly executable" % relative)
        missing = [marker for marker in REQUIRED_PROOF_MARKERS if marker not in text]
        if missing:
            errors.append("%s lacks final proof markers %s" % (relative, missing))
    return errors


def require_source_contract(root: Path) -> None:
    errors = source_contract_errors(root)
    if errors:
        raise ValueError("; ".join(errors))
