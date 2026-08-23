#!/usr/bin/env python3
"""Render top-frame and acquisition-composition figures from collected profiles only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

from plot_scaling import (  # noqa: E402
    _create_staging_directory,
    _publish_staging_directory,
    _sha256,
    _write_publication_manifest,
)
from profile_contract import (  # noqa: E402
    PROFILE_SCHEMA,
    ProfileContractError,
    external_profile_publication_path,
    verify_profile_complete_receipt,
)


PROFILE_PLOT_SCHEMA = "pops.performance.advection-sine.macos-profile-plot-publication.v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _icicle(plt, rows: list[object], output: Path) -> None:
    """Render exact sample Call-graph leaf stacks, not invented parent counts."""
    tree: dict[str, object] = {"weight": 0, "children": {}}
    for row in rows:
        if type(row) is not dict or type(row.get("frames")) is not list:
            raise SystemExit("profile plotting refused: invalid collected sample stack")
        frames = row["frames"]
        weight = row.get("weight")
        if not frames or any(type(frame) is not str or not frame for frame in frames):
            raise SystemExit("profile plotting refused: empty collected sample stack")
        if type(weight) is not int or weight < 1:
            raise SystemExit("profile plotting refused: invalid sample stack weight")
        node = tree
        node["weight"] = int(node["weight"]) + weight
        for frame in frames:
            children = node["children"]
            if not isinstance(children, dict):
                raise SystemExit("profile plotting refused: malformed sample stack tree")
            child = children.setdefault(frame, {"weight": 0, "children": {}})
            if not isinstance(child, dict):
                raise SystemExit("profile plotting refused: malformed sample stack child")
            child["weight"] = int(child["weight"]) + weight
            node = child
    total = int(tree["weight"])
    if total < 1:
        raise SystemExit("profile plotting refused: no sample stack weight")
    max_depth = max(len(row["frames"]) for row in rows if isinstance(row, dict))
    figure, axis = plt.subplots(figsize=(13, max(4, 1.15 * max_depth)))

    def color(label: str) -> str:
        return "#" + hashlib.sha256(label.encode("utf-8")).hexdigest()[:6]

    def draw(node: dict[str, object], x: float, width: float, depth: int, label: str) -> None:
        if depth:
            axis.barh(
                max_depth - depth,
                width,
                left=x,
                height=0.82,
                color=color(label),
                edgecolor="white",
                linewidth=0.5,
            )
            if width >= 0.07:
                axis.text(
                    x + width / 2,
                    max_depth - depth,
                    label[:44],
                    ha="center",
                    va="center",
                    fontsize=7,
                    clip_on=True,
                )
        children = node["children"]
        if not isinstance(children, dict):
            raise SystemExit("profile plotting refused: malformed sample stack tree")
        cursor = x
        ordered = sorted(children.items(), key=lambda item: (-int(item[1]["weight"]), item[0]))
        for child_label, child in ordered:
            if not isinstance(child, dict):
                raise SystemExit("profile plotting refused: malformed sample stack child")
            child_width = width * int(child["weight"]) / int(node["weight"])
            draw(child, cursor, child_width, depth + 1, child_label)
            cursor += child_width

    draw(tree, 0.0, 1.0, 0, "")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.8, max_depth - 0.1)
    axis.set_xlabel("part du poids exclusif des feuilles Call graph")
    axis.set_ylabel("profondeur de pile")
    axis.set_title("macOS /usr/bin/sample — icicle des piles de feuilles exactes")
    axis.set_yticks(range(max_depth))
    axis.set_yticklabels([str(max_depth - level) for level in range(max_depth)])
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(output / ("sample_icicle." + suffix), dpi=180)
    plt.close(figure)


def _write_analysis(staging: Path, summary: dict, complete: dict, summary_path: Path) -> None:
    """Publish an evidence-derived profile reading beside the immutable media."""
    top = summary["sample_top15"]
    composition = summary["sample_image_composition"]
    total_weight = sum(int(row["weight"]) for row in composition)
    hotspot_lines = [
        f"- `{row['frame']}` : {int(row['weight'])} ({int(row['weight']) / total_weight:.1%})."
        for row in top[:5]
    ]
    image_lines = [f"- `{row['image']}` : {int(row['weight'])}." for row in composition]
    source = complete["source"]
    native = complete["native"]
    text = "\n".join(
        [
            "# Analyse générée — profil macOS",
            "",
            "Cette analyse est produite uniquement à partir du summary et du `COMPLETE.json` "
            "scellés; elle ne remplace pas `MACOS_PROFILE_ANALYSIS.md` du dépôt.",
            "",
            "## Identité de la preuve",
            "",
            f"- Summary SHA-256 : `{_sha256(summary_path)}`.",
            f"- COMPLETE SHA-256 : `{_sha256(summary_path.parent / 'COMPLETE.json')}`.",
            f"- Source tree : `{source['tree_sha256']}` ; build fingerprint natif : `{native['build_fingerprint']}`.",
            f"- Extension : `{native['path']}`, SHA-256 `{native['sha256']}`.",
            "",
            "## Ce qui a été acquis",
            "",
            "Cinq processus complets `sample` et cinq processus complets `xctrace Time Profiler` "
            "ont été scellés. Les poids ci-dessous proviennent exclusivement des feuilles de piles "
            "`sample`, pas de compteurs inclusifs reconstruits.",
            "",
            "## Figures et interprétation",
            "",
            "![Feuilles les plus lourdes](sample_top15.png)",
            "",
            "Hotspots observés :",
            *hotspot_lines,
            "",
            "![Composition par image](sample_image_composition.png)",
            "",
            "Composition observée :",
            *image_lines,
            "",
            "![Icicle des piles de feuilles](sample_icicle.png)",
            "",
            "L'icicle conserve les chemins issus de l'indentation réelle du Call graph; il aide à "
            "relier les feuilles dominantes à leur contexte d'appel, sans inventer de coûts parents.",
            "",
            "## Limites",
            "",
            "Ce profil décrit ce point macOS OpenMP t8 et le cycle de vie Python public; il ne fournit "
            "ni scaling, ni mesure GPU/MPI, ni attribution causale exacte, ni résultat AMR. Les dix "
            "processus réduisent le risque d'un échantillon isolé mais ne remplacent pas les campagnes "
            "ROMEO scellées.",
            "",
        ]
    )
    (staging / "ANALYSIS.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = _arguments()
    try:
        summary_path = args.summary.resolve(strict=True)
        if summary_path.name != "summary.json":
            raise ProfileContractError("profile plot input must be the sealed summary.json")
        complete = verify_profile_complete_receipt(summary_path.parent)
        publication_root = external_profile_publication_path(
            profile_root=summary_path.parent, publication_root=args.output
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit("profile plotting refused: %s" % error) from error
    if type(summary) is not dict or summary.get("schema") != PROFILE_SCHEMA + ".summary":
        raise SystemExit("profile plotting refused: uncollected or incompatible summary")
    rows = summary.get("sample_top15")
    if type(rows) is not list or not rows:
        raise SystemExit("profile plotting refused: no collected sample frames")
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("profile plotting requires matplotlib") from error
    staging = _create_staging_directory(publication_root)
    publication = {
        "schema": PROFILE_PLOT_SCHEMA,
        "input": {
            "summary_filename": "summary.json",
            "summary_sha256": _sha256(summary_path),
            "complete_filename": "COMPLETE.json",
            "complete_sha256": _sha256(summary_path.parent / "COMPLETE.json"),
            "complete_schema": complete["schema"],
            "profile_schema": summary["schema"],
        },
    }
    try:
        output = staging
        labels = [row["frame"] for row in rows][::-1]
        weights = [row["weight"] for row in rows][::-1]
        figure, axis = plt.subplots(figsize=(11, 7))
        axis.barh(labels, weights, color="#306998")
        axis.set_xlabel("poids agrégé des échantillons / 5 processus")
        axis.set_title("macOS /usr/bin/sample — 15 feuilles les plus lourdes")
        figure.tight_layout()
        for suffix in ("png", "svg"):
            figure.savefig(output / ("sample_top15." + suffix), dpi=180)
        plt.close(figure)
        composition = summary.get("sample_image_composition")
        if type(composition) is not list or not composition:
            raise SystemExit("profile plotting refused: no collected image composition")
        image_labels = [row["image"] for row in composition][::-1]
        image_weights = [row["weight"] for row in composition][::-1]
        figure, axis = plt.subplots(figsize=(11, 6))
        axis.barh(image_labels, image_weights, color="#f58518")
        axis.set_xlabel("poids agrégé des feuilles sample / 5 processus")
        axis.set_title("macOS /usr/bin/sample — composition par image")
        figure.tight_layout()
        for suffix in ("png", "svg"):
            figure.savefig(output / ("sample_image_composition." + suffix), dpi=180)
        plt.close(figure)
        stacks = summary.get("sample_leaf_stacks")
        if type(stacks) is not list or not stacks:
            raise SystemExit("profile plotting refused: no collected sample leaf stacks")
        _icicle(plt, stacks, output)
        _write_analysis(staging, summary, complete, summary_path)
        _write_publication_manifest(staging, publication)
        _publish_staging_directory(staging, publication_root)
    except BaseException:
        # A staging directory is non-evidence and is safe to remove on local failure.
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
