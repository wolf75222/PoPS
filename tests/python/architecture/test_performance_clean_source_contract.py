"""Regression contracts: performance evidence must originate from a clean Git tree."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[3]
PERFORMANCE = REPOSITORY / "benchmarks/performance/advection_sine"
PROFILE = PERFORMANCE / "profiling"


def _scope(path: Path, import_root: Path) -> dict[str, object]:
    sys.path.insert(0, str(import_root))
    try:
        return runpy.run_path(str(path))
    finally:
        sys.path.remove(str(import_root))


def _dirty_manifest(scope: dict[str, object]) -> dict[str, object]:
    entries: list[object] = []
    tree_digest = scope["_tree_digest"](entries)
    return {
        "schema": "pops.performance.source-export.v1",
        "base_sha": "0" * 40,
        "source_dirty": True,
        "tree_sha256": tree_digest,
        "archive_sha256": "1" * 64,
        "archive_format": "tar-pax",
        "file_count": 0,
        "files": entries,
    }


def _clean_manifest(scope: dict[str, object]) -> dict[str, object]:
    manifest = _dirty_manifest(scope)
    manifest["source_dirty"] = False
    return manifest


def test_export_refuses_a_dirty_git_worktree_before_writing_evidence(tmp_path: Path) -> None:
    scope = _scope(PERFORMANCE / "prepare_export.py", PERFORMANCE)
    source = tmp_path / "source"
    source.mkdir()
    for arguments in (
        ("git", "init", "-q", str(source)),
        ("git", "-C", str(source), "config", "user.email", "benchmark@example.invalid"),
        ("git", "-C", str(source), "config", "user.name", "Benchmark Contract"),
    ):
        subprocess.run(arguments, check=True)
    tracked = source / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(source), "add", "tracked.txt"), check=True)
    subprocess.run(("git", "-C", str(source), "commit", "-qm", "initial"), check=True)
    tracked.write_text("modified\n", encoding="utf-8")

    archive = tmp_path / "evidence" / "source.tar"
    manifest = tmp_path / "evidence" / "source.manifest.json"
    with pytest.raises(scope["ExportError"], match="dirty Git worktree"):
        scope["create_export"](source, archive, manifest)
    assert not archive.exists()
    assert not manifest.exists()


def test_collection_and_scaling_plot_refuse_a_dirty_source_manifest(tmp_path: Path) -> None:
    export = _scope(PERFORMANCE / "prepare_export.py", PERFORMANCE)
    collector = _scope(PERFORMANCE / "collect_results.py", PERFORMANCE)
    plotter = _scope(PERFORMANCE / "plot_scaling.py", PERFORMANCE)
    root = tmp_path / "campaign"
    raw = root / "raw"
    report = root / "report"
    raw.mkdir(parents=True)
    report.mkdir()
    manifest = _dirty_manifest(export)
    (raw / "source.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (raw / "build.receipt.json").write_text("{}\n", encoding="utf-8")
    summary = {
        "schema": "pops.performance.advection-sine.summary.v2",
        "campaign": "strong-openmp",
        "route": "kokkos_openmp",
        "scaling": "strong",
        "source_manifest": manifest,
        "rows": [{"workers": 1}],
    }
    summary_path = report / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (root / "COMPLETE.json").write_text(
        json.dumps(
            {
                "schema": "pops.performance.advection-sine.complete.v2",
                "campaign": "strong-openmp",
                "source_tree_sha256": manifest["tree_sha256"],
                "raw": {"sha256": "0" * 64, "files": []},
                "report": {"sha256": "0" * 64, "files": []},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(export["ExportError"], match="source_dirty=false"):
        export["create_complete_receipt"](
            raw,
            report,
            "strong-openmp",
            "1",
            manifest["tree_sha256"],
            root / "new-COMPLETE.json",
        )
    with pytest.raises(collector["CampaignError"], match="source manifest is dirty"):
        collector["_require_receipts"](raw, {})
    with pytest.raises(export["ExportError"], match="source_dirty=false"):
        export["verify_complete_receipt"](root)
    with pytest.raises(ValueError, match="clean source manifest"):
        plotter["_authenticated_summary_root"](summary_path, summary)


def test_complete_seal_refuses_dirty_summary_source_linked_to_clean_raw_manifest(
    tmp_path: Path,
) -> None:
    export = _scope(PERFORMANCE / "prepare_export.py", PERFORMANCE)
    root = tmp_path / "campaign"
    raw = root / "raw"
    report = root / "report"
    raw.mkdir(parents=True)
    report.mkdir()
    manifest = _clean_manifest(export)
    (raw / "source.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    build = {
        "schema": "pops.performance.advection-sine.build-receipt.v3",
        "source": {"tree_sha256": manifest["tree_sha256"]},
        "campaign": {"id": "strong-openmp"},
    }
    (raw / "build.receipt.json").write_text(json.dumps(build), encoding="utf-8")
    dirty_summary_manifest = dict(manifest)
    dirty_summary_manifest["source_dirty"] = True
    summary = {
        "campaign": "strong-openmp",
        "source_manifest": dirty_summary_manifest,
        "build_receipt": build,
    }
    (report / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (report / "measurements.csv").write_text("point\nfull\n", encoding="utf-8")

    output = root / "COMPLETE.json"
    with pytest.raises(export["ExportError"], match="raw/build/summary provenance"):
        export["create_complete_receipt"](
            raw,
            report,
            "strong-openmp",
            "1",
            manifest["tree_sha256"],
            output,
        )
    assert not output.exists()


def test_profile_collection_and_complete_verifier_refuse_dirty_source_manifests(
    tmp_path: Path,
) -> None:
    export = _scope(PERFORMANCE / "prepare_export.py", PERFORMANCE)
    profile = _scope(PROFILE / "profile_contract.py", PROFILE)
    root = tmp_path / "profile"
    source_root = root / "source-tree"
    source_root.mkdir(parents=True)
    manifest = _dirty_manifest(export)
    (root / "source.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(profile["ProfileContractError"], match="source_dirty=false"):
        profile["source_manifest_receipt"](
            manifest_path=root / "source.manifest.json", source_root=source_root
        )

    (root / "COMPLETE.json").write_text(
        json.dumps(
            {
                "schema": "pops.performance.advection-sine.macos-profile.v1.complete.v1",
                "runs": {"sample": 5, "xctrace_time_profiler": 5},
                "source": {},
                "evidence": {"sha256": "0" * 64, "files": []},
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "schema": "pops.performance.advection-sine.macos-profile.v1.summary",
        "tools": {"sample": 5, "xctrace_time_profiler": 5},
        "leaves": {"sample": [{}] * 5, "xctrace": [{}] * 5},
        "provenance": {
            "source": {"source_dirty": True, "tree_sha256": "a" * 64},
            "native": {"sha256": "b" * 64, "build_fingerprint": "c" * 64},
            "build": {
                "source_tree_sha256": "a" * 64,
                "native_sha256": "b" * 64,
                "native_build_fingerprint": "c" * 64,
            },
        },
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(profile["ProfileContractError"], match="linked source/native"):
        profile["create_profile_complete_receipt"](root)
    with pytest.raises(profile["ProfileContractError"], match="source_dirty=false"):
        profile["verify_profile_complete_receipt"](root)
