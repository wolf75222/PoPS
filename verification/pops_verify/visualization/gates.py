"""Phase 8 visual completeness and quality gates."""
from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from jsonschema import Draft202012Validator

from verification.pops_verify.visualization.catalog import iter_catalog
from verification.pops_verify.visualization.plots import PUBLICATION_FORMATS, file_sha256

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_visuals.v1.json"


class VisualsGateError(RuntimeError):
    pass


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "analysis" / "visual_manifest.json"
    if not path.is_file():
        raise VisualsGateError(f"missing visual_manifest.json: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    errors = list(_validator().iter_errors(manifest))
    if errors:
        raise VisualsGateError(f"invalid visual_manifest.json: {errors[0].message}")
    return manifest


def check_visual_manifest(run_dir: str | Path, *, suite: str = "pr") -> dict[str, Any]:
    root = Path(run_dir)
    manifest = _load_manifest(root)
    figures = list(manifest.get("figures") or [])
    for item in figures:
        if item.get("role") == "hero" and not item.get("quantitative_companion"):
            raise VisualsGateError("hero figure is missing a quantitative companion")
    if suite == "pr":
        figures = [item for item in figures if item.get("pr")]
    ids = {item["figure_id"] for item in figures}
    for item in figures:
        if not item.get("units"):
            raise VisualsGateError(f"{item['figure_id']} is missing units")
        if not item.get("source_files"):
            raise VisualsGateError(f"{item['figure_id']} is missing source data")
        if item.get("role") == "publication":
            have_outputs = set(item.get("outputs") or {})
            have_hashes = set(item.get("output_hashes") or {})
            required = set(PUBLICATION_FORMATS)
            if have_outputs != required:
                raise VisualsGateError(
                    f"{item['figure_id']} publication figure is missing required formats"
                )
            if have_hashes != required:
                raise VisualsGateError(
                    f"{item['figure_id']} publication figure is missing required hashes"
                )
        for source in item["source_files"]:
            if not (root / source).is_file():
                raise VisualsGateError(f"missing visual_data: {source}")
        for fmt, rel in item.get("outputs", {}).items():
            path = root / rel
            if not path.is_file() or path.stat().st_size <= 0:
                raise VisualsGateError(f"missing or empty figure: {rel}")
            expected = item.get("output_hashes", {}).get(fmt)
            if expected and file_sha256(path) != expected:
                raise VisualsGateError(f"hash mismatch for {rel}")
        if item.get("kind") == "hero_figure" and item.get("quantitative_companion") not in ids:
            if item.get("quantitative_companion") is None:
                raise VisualsGateError("hero figure is missing a quantitative companion")
    if any(item.get("kind") == "animation" for item in figures):
        if not any(item.get("kind") == "storyboard" for item in manifest["figures"]):
            if suite == "release":
                raise VisualsGateError("animation requires a storyboard")
    return {"case_id": manifest["case_id"], "figures": len(figures)}


def check_release_completeness(
    campaign_dir: str | Path,
    *,
    suite: str = "release",
    executed_only: bool = False,
) -> dict[str, Any]:
    root = Path(campaign_dir)
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for manifest_path in sorted(root.glob("*/*/analysis/visual_manifest.json")):
        run_dir = manifest_path.parents[1]
        manifest = _load_manifest(run_dir)
        check_visual_manifest(run_dir, suite=suite)
        provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
        found[(manifest["case_id"], int(provenance["dimension"]))] = manifest
    missing: list[str] = []
    checked = 0
    for entry in iter_catalog():
        for axis, code, dim in zip(("1d", "2d", "3d"), entry.dimension_codes, (1, 2, 3), strict=True):
            if code != "R":
                continue
            key = (entry.case_id, dim)
            if key not in found:
                if executed_only:
                    continue
                missing.append(f"{entry.case_id}/{axis}")
                continue
            checked += 1
            manifest = found[key]
            if manifest["verdict"] in {"not-run", "not-supported"}:
                continue
            have = {item["figure_id"] for item in manifest["figures"]}
            required = set(entry.artifacts[axis])
            if suite == "pr":
                from verification.pops_verify.visualization.render import PR_KINDS

                required &= PR_KINDS
            absent = sorted(required - have)
            if absent:
                missing.append(f"{entry.case_id}/{axis}: {', '.join(absent)}")
    if missing:
        raise VisualsGateError(
            "release visual matrix is incomplete: " + "; ".join(missing[:12])
        )
    return {"cells_checked": checked, "missing": missing}
