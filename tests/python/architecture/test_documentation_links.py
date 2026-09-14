"""Documentation checks include executable collections and respect the requested root."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("pops_check_docs", ROOT / "docs/check_docs.py")
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


@pytest.mark.parametrize("directory", ["docs/tutorials", "examples", "benchmarks"])
def test_broken_collection_link_is_rejected(tmp_path, capsys, directory):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/docmap.toml").write_text("[docs]\n")
    page = tmp_path / directory / "README.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text("[missing](missing.py)\n")
    assert checker.check(root=tmp_path) == 1
    assert str(page.relative_to(tmp_path)) in capsys.readouterr().err


def test_requested_root_uses_its_own_docmap_and_link_targets(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/docmap.toml").write_text('[docs."README.md"]\ndepends_on = []\n')
    (tmp_path / "README.md").write_text("[script](demo.py)\n")
    (tmp_path / "demo.py").write_text("pass\n")
    assert checker.check(root=tmp_path) == 0


def test_code_examples_are_not_treated_as_real_links(tmp_path):
    page = tmp_path / "README.md"
    text = "```markdown\n[placeholder](absent.md)\n```\n"
    page.write_text(text)
    violations = []
    checker.check_links(page, text, violations, tmp_path)
    assert violations == []
