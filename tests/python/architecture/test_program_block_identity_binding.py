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
    assert "candidate_tables.blocks" in source
    assert "program_block_map.assign" in source
    assert "Program requires block instance" in source
    if path.name == "system_io.cpp":
        # Uniform admits a genuinely state-free Program only when both sides have no block-owned
        # authority. It must still reject using an empty table as the historical positional map.
        assert "state-free Program declares block-owned authority" in source
        assert "state-free Program has an empty block identity table" in source
        assert "positional Program-to-System binding is not supported" in source
    else:
        # AMR always owns hierarchy block state, so its v5 table can never be empty.
        assert "compiled Program has no explicit v5 block identity table" in source


@pytest.mark.parametrize("path", LOADERS, ids=("system", "amr_system"))
def test_program_loader_has_no_empty_map_positional_fallback(path):
    source = path.read_text(encoding="utf-8")
    assert "set_program_block_map({})" not in source
    assert "pre-Spec-3 .so" not in source
