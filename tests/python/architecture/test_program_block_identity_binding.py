"""Program loaders require explicit block identities; positional binding is forbidden."""
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
LOADERS = (
    ROOT / "src" / "runtime" / "system" / "system_io.cpp",
    ROOT / "src" / "runtime" / "amr" / "amr_system.cpp",
)


@pytest.mark.parametrize("path", LOADERS, ids=("system", "amr_system"))
def test_program_loader_requires_the_complete_block_identity_table(path):
    source = path.read_text(encoding="utf-8")
    assert "if (!block_count || !block_name)" in source
    assert "does not export the required block identity table" in source
    assert "pops_program_block_count + pops_program_block_name" in source
    assert "Positional Program-to-" in source
    assert "regenerate the Program library" in source


@pytest.mark.parametrize("path", LOADERS, ids=("system", "amr_system"))
def test_program_loader_has_no_empty_map_positional_fallback(path):
    source = path.read_text(encoding="utf-8")
    assert "set_program_block_map({})" not in source
    assert "pre-Spec-3 .so" not in source
