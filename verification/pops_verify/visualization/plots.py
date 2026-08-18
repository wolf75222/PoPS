"""Deterministic figure transforms and headless rendering."""
from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Any, Iterable

from verification.pops_verify.visualization.data import VisualsError
from verification.pops_verify.visualization.style import (
    RENDERER_VERSION,
    configure_matplotlib,
)

EXACT_STYLE = {"color": "#222222", "linestyle": "-", "linewidth": 2.0}
NUMERICAL_MARKERS = ("o", "s", "D", "^", "v")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return f"sha256:{digest.hexdigest()}"


def _require_series(payload: dict[str, Any]) -> list[dict[str, Any]]:
    series = payload.get("series")
    if not isinstance(series, list) or not series:
        raise VisualsError("empty visual_data series")
    cleaned: list[dict[str, Any]] = []
    for item in series:
        if not isinstance(item, dict):
            raise VisualsError("empty visual_data series")
        x_values = item.get("x")
        y_values = item.get("y")
        if not isinstance(x_values, list) or not isinstance(y_values, list):
            raise VisualsError("empty visual_data series")
        if not x_values or not y_values or len(x_values) != len(y_values):
            raise VisualsError("empty visual_data series")
        cleaned.append(dict(item))
    return cleaned


def prepare_convergence(payload: dict[str, Any]) -> dict[str, Any]:
    units = payload.get("units") or {}
    if not units.get("x") or not units.get("y"):
        raise VisualsError("convergence figure is missing units")
    return {
        "kind": "spatial_convergence",
        "figure_id": payload.get("figure_id", "spatial_convergence"),
        "scale": "loglog",
        "xlabel": units["x"],
        "ylabel": units["y"],
        "series": _require_series(payload),
        "reference_slopes": list(payload.get("reference_slopes") or []),
        "title": payload.get("title") or "Spatial convergence",
    }


def prepare_profile(payload: dict[str, Any]) -> dict[str, Any]:
    units = payload.get("units") or {}
    series = _require_series(payload)
    styled = []
    for item in series:
        name = str(item.get("name") or "series")
        style = "exact" if name.lower() == "exact" else "numerical"
        styled.append({**item, "name": name, "style": style})
    return {
        "kind": "reference_profile",
        "figure_id": payload.get("figure_id", "reference_profile"),
        "scale": "linear",
        "xlabel": units.get("x") or "x",
        "ylabel": units.get("y") or "value",
        "series": styled,
        "title": payload.get("title") or "Exact / numerical profile",
    }


def prepare_field(payload: dict[str, Any]) -> dict[str, Any]:
    field = payload.get("field")
    if not isinstance(field, list) or not field:
        raise VisualsError("empty visual_data field")
    values = [float(value) for row in field for value in row]
    peak = max(abs(value) for value in values) if values else 0.0
    if peak == 0.0:
        peak = 1.0
    signed = payload.get("kind") == "signed_error_field"
    return {
        "kind": payload.get("kind", "field_snapshot"),
        "figure_id": payload.get("figure_id", "field"),
        "x": list(payload.get("x") or []),
        "y": list(payload.get("y") or []),
        "field": field,
        "cmap": "RdBu_r" if signed else "viridis",
        "vmin": -peak if signed else min(values),
        "vmax": peak if signed else max(values),
        "center": 0.0 if signed else None,
        "xlabel": (payload.get("units") or {}).get("x") or "x",
        "ylabel": (payload.get("units") or {}).get("y") or "y",
        "clabel": (payload.get("units") or {}).get("field") or "value",
        "title": payload.get("title") or payload.get("kind") or "field",
    }


def _caption_text(caption: str | None, provenance_sha: str | None) -> str:
    parts = [part for part in (caption, provenance_sha) if part]
    return " | ".join(parts)


def _draw(prepared: dict[str, Any], caption: str | None, provenance_sha: str | None):
    plt = configure_matplotlib()
    figure, axes = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    kind = prepared["kind"]
    if kind in {"spatial_convergence", "temporal_convergence"}:
        for index, item in enumerate(prepared["series"]):
            marker = NUMERICAL_MARKERS[index % len(NUMERICAL_MARKERS)]
            axes.loglog(
                item["x"],
                item["y"],
                marker=marker,
                linestyle="-",
                label=item.get("name") or f"series-{index}",
            )
        for slope in prepared.get("reference_slopes") or []:
            order = float(slope["order"])
            anchor = slope["anchor"]
            x0, y0 = float(anchor[0]), float(anchor[1])
            xs = list(prepared["series"][0]["x"])
            ys = [y0 * (x0 / x) ** order for x in xs]
            axes.loglog(xs, ys, color="#222222", linestyle="--", label=f"order {order:g}")
        axes.legend()
    elif kind in {"reference_profile", "signed_error_profile", "invariants_vs_time"}:
        for item in prepared["series"]:
            if item.get("style") == "exact":
                axes.plot(item["x"], item["y"], label=item["name"], **EXACT_STYLE)
            else:
                axes.plot(item["x"], item["y"], marker="o", label=item["name"])
        axes.legend()
    elif "field" in prepared:
        mesh = axes.pcolormesh(
            prepared["x"],
            prepared["y"],
            prepared["field"],
            cmap=prepared["cmap"],
            vmin=prepared["vmin"],
            vmax=prepared["vmax"],
            shading="nearest",
        )
        colorbar = figure.colorbar(mesh, ax=axes)
        colorbar.set_label(prepared.get("clabel") or "")
        axes.set_aspect("equal", adjustable="box")
    else:
        raise VisualsError(f"unsupported figure kind: {kind}")
    axes.set_xlabel(prepared.get("xlabel") or "")
    axes.set_ylabel(prepared.get("ylabel") or "")
    axes.set_title(prepared.get("title") or "")
    note = _caption_text(caption, provenance_sha)
    if note:
        figure.text(0.01, 0.01, note, fontsize=8)
    return figure


def render_prepared(
    prepared: dict[str, Any],
    output_dir: str | Path,
    *,
    formats: Iterable[str] = ("svg", "png", "pdf"),
    caption: str | None = None,
    provenance_sha: str | None = None,
) -> dict[str, str]:
    figure = _draw(prepared, caption, provenance_sha)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = prepared.get("figure_id") or "figure"
    outputs: dict[str, str] = {}
    metadata = {"Creator": RENDERER_VERSION, "Date": None}
    try:
        for fmt in formats:
            path = root / f"{stem}.{fmt}"
            figure.savefig(path, format=fmt, metadata=metadata)
            if path.stat().st_size <= 0:
                raise VisualsError(f"empty figure output: {path}")
            outputs[fmt] = str(path)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
    return outputs
