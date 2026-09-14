"""Generated Program DSOs can link the field publication transaction seam."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[4]


def test_generated_field_outcome_transaction_has_exported_native_instantiation():
    header = (ROOT / "include/pops/runtime/system.hpp").read_text(encoding="utf-8")
    context = (ROOT / "include/pops/runtime/program/program_context.hpp").read_text(
        encoding="utf-8"
    )
    implementation = (ROOT / "src/runtime/system/system_fields.cpp").read_text(
        encoding="utf-8"
    )

    # This call lives in a separately compiled Program DSO: private C++ access is
    # granted by friendship, while dynamic linking also requires symbol visibility.
    assert "system_->run_field_publication_outcome_(" in context
    assert re.search(
        r"POPS_EXPORT\s+SolveOutcome\s+run_field_publication_outcome_\s*\(", header
    )
    assert re.search(
        r"template\s+SolveOutcome\s+System<kNativeDimension>::"
        r"run_field_publication_outcome_\s*\(",
        implementation,
    )
