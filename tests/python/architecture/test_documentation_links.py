"""Documentation checks include executable collections and respect the requested root."""
from __future__ import annotations

import importlib.util
import hashlib
import json
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


def _raw_archive(root):
    archive = root / "docs/evidence/retained-run"
    archive.mkdir(parents=True)
    member = archive / "historical.md"
    member.write_text("Captured prose\u2014unchanged [old context](missing.md)\n")
    row = {"source": "/original/location.md", "bytes": member.stat().st_size,
           "sha256": hashlib.sha256(member.read_bytes()).hexdigest()}
    manifest = archive / "manifest.json"
    manifest.write_text(json.dumps({"schema": 1, "files": {"historical.md": row}}))
    def seal():
        (root / "docs/docmap.toml").write_text(
            '[docs]\n[retained_evidence."docs/evidence/retained-run"]\n'
            'members = 1\nmetadata = {}\n'
            '[retained_evidence."docs/evidence/retained-run".catalogs]\n'
            '"manifest.json" = "' + hashlib.sha256(manifest.read_bytes()).hexdigest() + '"\n')
    seal()
    return archive, member, manifest, seal


def test_exact_archived_prose_is_checked_by_integrity_without_rewriting(tmp_path):
    _archive, member, _manifest, _seal = _raw_archive(tmp_path)
    before = member.read_bytes()
    assert checker.check(root=tmp_path) == 0
    assert member.read_bytes() == before
    assert member in checker.md_files(tmp_path)  # Corpus inventory is unchanged.


@pytest.mark.parametrize("fault", (
    "missing_member", "changed_member", "changed_catalog", "empty_catalog",
    "unlisted_member", "escaping_member", "wrong_bytes", "wrong_schema",
    "active_mapped_page", "symlink_member", "outside_evidence_scope",
))
def test_nonconforming_archives_fail_closed(tmp_path, fault):
    archive, member, manifest, seal = _raw_archive(tmp_path)
    data = json.loads(manifest.read_text())
    if fault == "missing_member":
        member.unlink()
    elif fault == "changed_member":
        member.write_text("changed\n")
    elif fault == "changed_catalog":
        manifest.write_text("{}")  # Digest change is not resealed.
    elif fault == "unlisted_member":
        (archive / "unlisted.md").write_text("new active [broken](missing.md)\n")
    elif fault == "symlink_member":
        member.unlink()
        member.symlink_to(tmp_path / "absent.md")
    elif fault == "active_mapped_page":
        with (tmp_path / "docs/docmap.toml").open("a") as stream:
            stream.write('\n[docs."docs/evidence/retained-run/historical.md"]\ndepends_on = []\n')
    elif fault == "outside_evidence_scope":
        text = (tmp_path / "docs/docmap.toml").read_text().replace(
            'docs/evidence/retained-run', 'docs')
        (tmp_path / "docs/docmap.toml").write_text(text)
    else:
        if fault == "empty_catalog":
            data["files"] = {}
        elif fault == "escaping_member":
            data["files"]["../../../README.md"] = data["files"].pop("historical.md")
        elif fault == "wrong_bytes":
            data["files"]["historical.md"]["bytes"] += 1
        elif fault == "wrong_schema":
            data = []
        manifest.write_text(json.dumps(data)); seal()
    assert checker.check(root=tmp_path) == 1


def test_new_active_markdown_broken_link_still_fails_with_valid_archive(tmp_path):
    _raw_archive(tmp_path)
    page = tmp_path / "docs/tutorials/new-active.md"
    page.parent.mkdir()
    page.write_text("[broken](missing.py)\n")
    assert checker.check(root=tmp_path) == 1


def test_proof_catalog_cannot_reclassify_root_active_page(tmp_path):
    _archive, _member, manifest, seal = _raw_archive(tmp_path)
    (tmp_path / "README.md").write_text("[broken](missing.py)\n")
    data = json.loads(manifest.read_text())
    data["files"]["../../../README.md"] = data["files"].pop("historical.md")
    manifest.write_text(json.dumps(data)); seal()
    assert checker.check(root=tmp_path) == 1
