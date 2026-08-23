#!/usr/bin/env python3
"""Render static scaling figures from one authenticated summary JSON."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

from common import SUMMARY_SCHEMA
from prepare_export import ExportError, verify_complete_receipt


BLUE = "#1769aa"
ORANGE = "#d97706"
INK = "#20242a"
GREY = "#7a828c"
GRID = "#d9dee5"
PLOT_PUBLICATION_SCHEMA = "pops.performance.advection-sine.plot-publication.v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict or payload.get("schema") != SUMMARY_SCHEMA:
        raise ValueError("unexpected summary schema")
    rows = payload.get("rows")
    if type(rows) is not list or not rows:
        raise ValueError("summary contains no plotted rows")
    workers = [row.get("workers") for row in rows]
    if any(type(value) is not int or value < 1 for value in workers):
        raise ValueError("workers must be positive integers")
    if len(set(workers)) != len(workers):
        raise ValueError("workers must be unique within one scaling series")
    for row in rows:
        for field in (
            "median_seconds",
            "mad_seconds",
            "throughput_cell_updates_per_second",
            "mass_drift",
            "l2_error",
        ):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"row {row.get('point')}: {field} is not numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"row {row.get('point')}: {field} is not finite/nonnegative")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _require_fresh_path(path: Path) -> None:
    if path.is_symlink() or _path_exists(path):
        raise FileExistsError("refusing to replace an existing plot artifact: %s" % path)


def _authenticated_summary_root(summary_path: Path, summary: dict) -> dict:
    """Refuse orphan summaries: every figure must start from sealed evidence."""
    if summary_path.name != "summary.json" or summary_path.parent.name != "report":
        raise ValueError("summary must be the report/summary.json sealed by COMPLETE.json")
    root = summary_path.parent.parent
    try:
        complete = verify_complete_receipt(root)
    except ExportError as error:
        raise ValueError("summary is not covered by an authentic COMPLETE receipt: %s" % error) from error
    if complete.get("campaign") != summary.get("campaign"):
        raise ValueError("COMPLETE campaign differs from the summary campaign")
    return complete


def _publication_manifest(
    summary_path: Path, summary: dict, complete: dict | None = None
) -> dict:
    payload: dict[str, object] = {
        "schema": PLOT_PUBLICATION_SCHEMA,
        "renderer": {
            "filename": Path(__file__).name,
            "sha256": _sha256(Path(__file__)),
        },
        "input": {
            "summary_filename": summary_path.name,
            "summary_sha256": _sha256(summary_path),
            "summary_schema": summary["schema"],
            "campaign": summary["campaign"],
            "route": summary["route"],
            "scaling": summary["scaling"],
        },
    }
    if complete is not None:
        payload["input"].update(
            {
                "complete_filename": "COMPLETE.json",
                "complete_sha256": _sha256(summary_path.parent.parent / "COMPLETE.json"),
                "complete_schema": complete["schema"],
            }
        )
    payload["publication_identity"] = _canonical_sha256(payload)
    return payload


def _canonical_publication_target(output: Path) -> Path:
    """Refuse a lexical target first; only its parent may be canonicalised."""
    lexical_output = output.absolute()
    _require_fresh_path(lexical_output)
    parent = lexical_output.parent
    if _path_exists(parent) and not parent.is_dir():
        raise NotADirectoryError(parent)
    parent.mkdir(parents=True, exist_ok=True)
    canonical_output = parent.resolve(strict=True) / lexical_output.name
    _require_fresh_path(canonical_output)
    return canonical_output


def _create_staging_directory(output: Path) -> Path:
    output = _canonical_publication_target(output)
    return Path(tempfile.mkdtemp(prefix=".%s.staging-" % output.name, dir=output.parent))


def _write_publication_manifest(staging: Path, manifest: dict) -> Path:
    target = staging / "plot_manifest.json"
    _require_fresh_path(target)
    media = [
        {
            "path": path.relative_to(staging).as_posix(),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path != target
    ]
    if not media:
        raise ValueError("plot publication produced no media artifacts")
    payload = dict(manifest)
    payload["media"] = media
    target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return target


def _atomic_rename_noreplace(staging: Path, output: Path) -> None:
    """Use the platform no-replace rename primitive; never emulate it with os.rename."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(staging)
    output_bytes = os.fsencode(output)
    if sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace publication requires libc renameat2 on Linux",
            ) from error
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, output_bytes, 1)  # AT_FDCWD, RENAME_NOREPLACE
    elif sys.platform == "darwin":
        try:
            rename = libc.renamex_np
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace publication requires renamex_np on macOS",
            ) from error
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(source_bytes, output_bytes, 0x0004)  # RENAME_EXCL
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace plot publication is unavailable on this platform",
        )
    if result == 0:
        return
    failure = ctypes.get_errno()
    if failure in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            failure, "refusing to replace an existing plot publication", str(output)
        )
    raise OSError(failure, os.strerror(failure), str(output))


def _publish_staging_directory(staging: Path, output: Path) -> Path:
    output = _canonical_publication_target(output)
    _atomic_rename_noreplace(staging, output)
    return output


def _style_axis(axis, *, log_x: bool = True) -> None:
    axis.grid(True, color=GRID, linewidth=0.8, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GREY)
    axis.tick_params(colors=INK)
    if log_x:
        axis.set_xscale("log", base=2)


def _write_analysis(
    staging: Path, summary: dict, complete: dict, rows: list[dict], summary_path: Path
) -> None:
    """Write an evidence-only companion report beside, never over, repository docs."""
    source = summary["source_manifest"]
    table = [
        "| Point | Workers | Nodes | Median (s) | MAD (s) | Throughput |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table.append(
            "| {point} | {workers} | {nodes} | {median_seconds:.6g} | {mad_seconds:.3g} | "
            "{throughput_cell_updates_per_second:.6g} |".format(**row)
        )
    figure_interpretation = (
        "La figure compare les médianes du cycle de vie Python public, avec les MAD comme "
        "barres d'incertitude. Elle montre les mesures observées; elle n'extrapole aucune "
        "performance entre points."
    )
    if summary["scaling"] == "strong":
        tail = rows[-1]
        scaling_interpretation = (
            f"Au dernier point ({tail['workers']} workers), le speedup mesuré est "
            f"{float(tail['speedup']):.4g} et l'efficacité parallèle "
            f"{float(tail['parallel_efficiency']):.4g}."
        )
    elif summary["scaling"] == "weak_spatial":
        tail = rows[-1]
        scaling_interpretation = (
            f"Au dernier point ({tail['workers']} workers), l'efficacité weak observée est "
            f"{float(tail['parallel_efficiency']):.4g}; le nombre de pas reste fixe."
        )
    else:
        scaling_interpretation = "Campagne de référence : aucun speedup ni efficacité n'est inféré."
    text = "\n".join(
        [
            "# Analyse générée — performance advection sinusoïdale",
            "",
            "Cette analyse est générée exclusivement depuis le summary scellé et accompagne cette "
            "publication; elle ne remplace pas les modèles `ANALYSIS.md` du dépôt.",
            "",
            "## Identité de la preuve",
            "",
            f"- Campagne : `{summary['campaign']}` ; route : `{summary['route']}` ; scaling : `{summary['scaling']}`.",
            f"- Summary : `{summary_path.name}`, SHA-256 `{_sha256(summary_path)}`.",
            f"- COMPLETE : SHA-256 `{_sha256(summary_path.parent.parent / 'COMPLETE.json')}` ; source tree `{complete['source_tree_sha256']}`.",
            f"- Source export : base `{source['base_sha']}`, dirty `{source['source_dirty']}`, tree `{source['tree_sha256']}`.",
            "",
            "## Mesures",
            "",
            *table,
            "",
            "## Figure et interprétation",
            "",
            "![Vue d'ensemble scaling](scaling_overview.png)",
            "",
            figure_interpretation,
            "",
            scaling_interpretation,
            "",
            "## Limites",
            "",
            "Les allocations/nœuds et workers ci-dessus sont les limites matérielles déclarées par "
            "chaque point; le reçu Slurm scellé est l'autorité pour la topologie observée. Ce workload "
            "est uniforme et synchrone : ces résultats ne qualifient ni n'extrapolent AMR, regridding, "
            "refluxing, patch mobile ou subcycling.",
            "",
        ]
    )
    (staging / "ANALYSIS.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = _arguments()
    try:
        summary_path = args.summary.resolve()
        summary = _load(summary_path)
        complete = _authenticated_summary_root(summary_path, summary)
        manifest = _publication_manifest(summary_path, summary, complete)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (OSError, ValueError, json.JSONDecodeError, ImportError) as error:
        print(f"plot refused: {error}", file=sys.stderr)
        return 2

    rows = sorted(summary["rows"], key=lambda row: int(row["workers"]))
    workers = [int(row["workers"]) for row in rows]
    seconds = [float(row["median_seconds"]) for row in rows]
    mad = [float(row["mad_seconds"]) for row in rows]
    throughput = [float(row["throughput_cell_updates_per_second"]) for row in rows]
    efficiency = [row.get("parallel_efficiency") for row in rows]
    speedup = [row.get("speedup") for row in rows]
    ideal = [row.get("ideal_speedup") for row in rows]

    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.2))
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.84, hspace=0.42, wspace=0.24)
    figure.patch.set_facecolor("white")
    figure.suptitle(
        f"Periodic sine advection — {summary['campaign']}",
        color=INK,
        fontsize=15,
        fontweight="semibold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.925,
        f"{summary['route']} · {summary['scaling']} · median of fenced max-rank timings",
        ha="center",
        va="top",
        color=GREY,
        fontsize=9,
    )

    runtime_axis = axes[0, 0]
    runtime_axis.errorbar(
        workers,
        seconds,
        yerr=mad,
        color=BLUE,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2,
        capsize=3,
    )
    runtime_axis.set_yscale("log")
    runtime_axis.set_title("Time to solution", loc="left", color=INK)
    runtime_axis.set_ylabel("seconds (median ± MAD)")
    runtime_axis.set_xlabel("CPU workers or GPUs")
    _style_axis(runtime_axis)

    scaling_axis = axes[0, 1]
    if summary["scaling"] == "strong":
        scaling_axis.plot(
            workers, speedup, color=BLUE, marker="o", linewidth=2, label="measured speedup"
        )
        scaling_axis.plot(workers, ideal, color=INK, linestyle="--", linewidth=1.4, label="ideal")
        scaling_axis.set_ylabel("speedup")
        scaling_axis.set_title("Strong-scaling speedup", loc="left", color=INK)
        scaling_axis.legend(frameon=False, loc="upper left")
    elif summary["scaling"] == "weak_spatial":
        scaling_axis.plot(workers, efficiency, color=BLUE, marker="o", linewidth=2)
        scaling_axis.axhline(1.0, color=INK, linestyle="--", linewidth=1.4)
        scaling_axis.set_ylabel("weak efficiency")
        scaling_axis.set_title("Fixed-step weak efficiency", loc="left", color=INK)
    else:
        scaling_axis.scatter(workers, seconds, color=BLUE, edgecolor=INK, zorder=3)
        scaling_axis.set_ylabel("seconds")
        scaling_axis.set_title("Reference measurement", loc="left", color=INK)
    scaling_axis.set_xlabel("CPU workers or GPUs")
    _style_axis(scaling_axis)

    throughput_axis = axes[1, 0]
    throughput_axis.plot(
        workers, throughput, color=ORANGE, marker="s", markerfacecolor="white", linewidth=2
    )
    throughput_axis.set_title("Update throughput", loc="left", color=INK)
    throughput_axis.set_ylabel("SSPRK stage-cell updates / s")
    throughput_axis.set_xlabel("CPU workers or GPUs")
    _style_axis(throughput_axis)

    quality_axis = axes[1, 1]
    l2_error = [float(row["l2_error"]) for row in rows]
    mass_drift = [max(float(row["mass_drift"]), 1e-18) for row in rows]
    quality_axis.plot(workers, l2_error, color=BLUE, marker="o", linewidth=2, label="L2 error")
    quality_axis.plot(
        workers,
        mass_drift,
        color=ORANGE,
        marker="s",
        markerfacecolor="white",
        linewidth=2,
        label="mass drift",
    )
    quality_axis.set_yscale("log")
    quality_axis.set_title("Scientific guardrails", loc="left", color=INK)
    quality_axis.set_ylabel("error")
    quality_axis.set_xlabel("CPU workers or GPUs")
    quality_axis.legend(frameon=False, loc="best")
    _style_axis(quality_axis)

    output = args.output
    try:
        staging = _create_staging_directory(output)
        stem = staging / "scaling_overview"
        figure.savefig(stem.with_suffix(".png"), dpi=180, facecolor="white")
        figure.savefig(stem.with_suffix(".svg"), facecolor="white")
        _write_analysis(staging, summary, complete, rows, summary_path)
        _write_publication_manifest(staging, manifest)
        published = _publish_staging_directory(staging, output)
    except (OSError, ValueError) as error:
        print(f"plot refused: {error}", file=sys.stderr)
        return 2
    finally:
        plt.close(figure)
    print(f"figure: {published / 'scaling_overview.png'}")
    print(f"figure: {published / 'scaling_overview.svg'}")
    print(f"manifest: {published / 'plot_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
