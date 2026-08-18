"""Deterministic figure transforms and headless rendering."""
from __future__ import annotations

from pathlib import Path
import hashlib
import math
from typing import Any, Iterable

from verification.pops_verify.visualization.data import VisualsError
from verification.pops_verify.visualization.style import (
    RENDERER_VERSION,
    configure_matplotlib,
)

EXACT_STYLE = {"color": "#222222", "linestyle": "-", "linewidth": 2.0}
NUMERICAL_MARKERS = ("o", "s", "D", "^", "v")
FIGURE_ORDER_THRESHOLD = 1.8
ROUNDING_FLOOR = 1.0e-14
PUBLICATION_FORMATS = ("svg", "png", "pdf")
FIXTURE_LABEL = "DETERMINISTIC FIXTURE — not a PoPS campaign result"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return f"sha256:{digest.hexdigest()}"


def observed_order(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2 or len(y_values) < 2:
        raise VisualsError("cannot observe order from fewer than two points")
    orders: list[float] = []
    for left, right in zip(range(len(x_values) - 1), range(1, len(x_values)), strict=True):
        x0, x1 = float(x_values[left]), float(x_values[right])
        y0, y1 = float(y_values[left]), float(y_values[right])
        if x0 <= 0.0 or x1 <= 0.0 or y0 <= 0.0 or y1 <= 0.0 or x0 == x1:
            raise VisualsError("cannot observe order from non-positive values")
        orders.append(math.log(y0 / y1) / math.log(x1 / x0))
    return sum(orders) / len(orders)


def _series_spacings(x_values: list[float]) -> list[float]:
    xs = [float(value) for value in x_values]
    if len(xs) >= 2 and all(xs[index] > xs[index + 1] for index in range(len(xs) - 1)):
        return xs
    return [1.0 / value for value in xs]


def _interval_orders(errors: list[float], spacings: list[float]) -> list[float]:
    orders: list[float] = []
    for left, right in zip(range(len(errors) - 1), range(1, len(errors)), strict=True):
        e0, e1 = float(errors[left]), float(errors[right])
        h0, h1 = float(spacings[left]), float(spacings[right])
        if e0 <= 0.0 or e1 <= 0.0 or h0 <= 0.0 or h1 <= 0.0 or h0 == h1:
            raise VisualsError("cannot observe order from non-positive values")
        orders.append(math.log(e0 / e1) / math.log(h0 / h1))
    return orders


def derive_figure_verdict(payload: dict[str, Any]) -> str:
    kind = payload.get("kind")
    if kind in {"spatial_convergence", "temporal_convergence", "coarse_fine_error"}:
        series = payload.get("series") or []
        if not series:
            raise VisualsError("empty visual_data series")
        names = {"Linf", "L∞", "linf"}
        chosen = next((item for item in series if item.get("name") in names), None)
        if chosen is None:
            chosen = next((item for item in series if item.get("name") == "L2"), series[0])
        errors = [float(value) for value in chosen["y"]]
        spacings = _series_spacings(list(chosen["x"]))
        orders = _interval_orders(errors, spacings)
        usable = {index for index, error in enumerate(errors) if error > ROUNDING_FLOOR}
        interval_ids = [
            index
            for index in range(len(orders))
            if index in usable and (index + 1) in usable
        ]
        if len(interval_ids) < 2:
            return "fail"
        gated = [orders[index] for index in interval_ids[-2:]]
        if all(value >= FIGURE_ORDER_THRESHOLD for value in gated):
            return "pass"
        return "fail"
    return "pass"


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
        "kind": payload.get("kind") or "spatial_convergence",
        "figure_id": payload.get("figure_id", "spatial_convergence"),
        "scale": "loglog",
        "xlabel": units["x"],
        "ylabel": units["y"],
        "series": _require_series(payload),
        "reference_slopes": list(payload.get("reference_slopes") or []),
        "title": payload.get("title") or "Spatial convergence",
        "verdict": derive_figure_verdict(payload),
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
        "kind": payload.get("kind") or "reference_profile",
        "figure_id": payload.get("figure_id", "reference_profile"),
        "scale": "linear",
        "xlabel": units.get("x") or "x",
        "ylabel": units.get("y") or "value",
        "series": styled,
        "title": payload.get("title") or "Exact / numerical profile",
        "verdict": "pass",
    }


def prepare_signed_error_profile(payload: dict[str, Any]) -> dict[str, Any]:
    series = _require_series(payload)
    if [item.get("name") for item in series] != ["error"]:
        raise VisualsError("signed_error_profile must contain only the error series")
    values = [float(value) for value in series[0]["y"]]
    if min(values) >= 0 or max(values) <= 0:
        raise VisualsError("signed_error_profile must contain both signs")
    prepared = prepare_profile(payload)
    prepared["kind"] = "signed_error_profile"
    prepared["ylabel"] = (payload.get("units") or {}).get("y") or "signed error"
    return prepared


def prepare_field(payload: dict[str, Any]) -> dict[str, Any]:
    field = payload.get("field")
    if not isinstance(field, list) or not field:
        raise VisualsError("empty visual_data field")
    values = [float(value) for row in field for value in row]
    peak = max(abs(value) for value in values) if values else 0.0
    if peak == 0.0:
        peak = 1.0
    signed = payload.get("kind") in {"signed_error_field"}
    limits = payload.get("color_limits") or payload.get("color_range")
    if isinstance(limits, list) and len(limits) == 2:
        vmin, vmax = float(limits[0]), float(limits[1])
    else:
        vmin = -peak if signed else min(values)
        vmax = peak if signed else max(values)
    return {
        "kind": payload.get("kind", "field_snapshot"),
        "figure_id": payload.get("figure_id", "field"),
        "x": list(payload.get("x") or []),
        "y": list(payload.get("y") or []),
        "field": field,
        "cmap": "RdBu_r" if signed else "viridis",
        "vmin": vmin,
        "vmax": vmax,
        "center": 0.0 if signed else None,
        "xlabel": (payload.get("units") or {}).get("x") or "x",
        "ylabel": (payload.get("units") or {}).get("y") or "y",
        "clabel": (payload.get("units") or {}).get("field") or "value",
        "title": payload.get("title") or payload.get("kind") or "field",
        "verdict": "pass",
    }


def prepare_backend_parity(payload: dict[str, Any]) -> dict[str, Any]:
    backends = payload.get("backends")
    values = payload.get("values")
    metrics = payload.get("metrics") or ["value"]
    if not isinstance(backends, list) or not backends:
        raise VisualsError("backend_parity is missing backends")
    if not isinstance(values, list) or len(values) != len(backends):
        raise VisualsError("backend_parity values must match backends")
    return {
        "kind": "backend_parity",
        "figure_id": payload.get("figure_id", "backend_parity"),
        "backends": backends,
        "metrics": metrics,
        "values": values,
        "xlabel": (payload.get("units") or {}).get("x") or "backend",
        "ylabel": (payload.get("units") or {}).get("y") or "metric",
        "title": payload.get("title") or "Backend parity",
        "verdict": "pass",
    }


def prepare_performance_breakdown(payload: dict[str, Any]) -> dict[str, Any]:
    stages = payload.get("stages")
    seconds = payload.get("seconds")
    if not isinstance(stages, list) or not stages:
        raise VisualsError("performance_breakdown is missing stages")
    if not isinstance(seconds, list) or len(seconds) != len(stages):
        raise VisualsError("performance_breakdown seconds must match stages")
    return {
        "kind": "performance_breakdown",
        "figure_id": payload.get("figure_id", "performance_breakdown"),
        "stages": stages,
        "seconds": [float(value) for value in seconds],
        "xlabel": (payload.get("units") or {}).get("x") or "stage",
        "ylabel": (payload.get("units") or {}).get("y") or "s",
        "title": payload.get("title") or "Performance breakdown",
        "verdict": "pass",
    }


def prepare_heatmap(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows")
    columns = payload.get("columns")
    values = payload.get("values")
    if not rows or not columns or not values:
        raise VisualsError("heatmap is missing rows, columns, or values")
    return {
        "kind": payload.get("kind") or "orders_heatmap",
        "figure_id": payload.get("figure_id", "orders_heatmap"),
        "rows": list(rows),
        "columns": list(columns),
        "values": values,
        "xlabel": (payload.get("units") or {}).get("x") or "column",
        "ylabel": (payload.get("units") or {}).get("y") or "row",
        "clabel": (payload.get("units") or {}).get("field") or "value",
        "title": payload.get("title") or "Heatmap",
        "verdict": payload.get("verdict") or "pass",
    }


def prepare_amr_boxes(payload: dict[str, Any]) -> dict[str, Any]:
    boxes = payload.get("boxes")
    if not isinstance(boxes, list) or not boxes:
        raise VisualsError("amr_boxes is missing boxes")
    return {
        "kind": "amr_boxes",
        "figure_id": payload.get("figure_id", "amr_boxes"),
        "boxes": boxes,
        "xlabel": (payload.get("units") or {}).get("x") or "x",
        "ylabel": (payload.get("units") or {}).get("y") or "y",
        "zlabel": (payload.get("units") or {}).get("z") or "z",
        "title": payload.get("title") or "AMR boxes",
        "verdict": "pass",
    }


def prepare_not_run(figure_id: str, reason: str) -> dict[str, Any]:
    return {
        "kind": "not_run_panel",
        "figure_id": figure_id,
        "reason": reason,
        "title": figure_id,
        "verdict": "not-run",
    }


def _caption_text(caption: str | None, provenance_sha: str | None) -> str:
    parts = [part for part in (caption, provenance_sha) if part]
    return " | ".join(parts)


def qualify_fixture_caption(caption: str | None) -> str:
    text = caption or ""
    if "campaign dashboard" in text.lower():
        text = ""
    if "DETERMINISTIC FIXTURE" in text:
        return text
    if text:
        return f"{FIXTURE_LABEL} | {text}"
    return FIXTURE_LABEL


def _draw_boxes_3d(axes, boxes: list[dict[str, Any]]) -> None:
    for box in boxes:
        lo = box["lo"]
        hi = box["hi"]
        xs = [lo[0], hi[0]]
        ys = [lo[1], hi[1]]
        zs = [lo[2], hi[2]]
        corners = [
            (xs[i], ys[j], zs[k])
            for k in (0, 1)
            for j in (0, 1)
            for i in (0, 1)
        ]
        edges = (
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )
        for start, end in edges:
            pair = [corners[start], corners[end]]
            axes.plot(
                [pair[0][0], pair[1][0]],
                [pair[0][1], pair[1][1]],
                [pair[0][2], pair[1][2]],
                color="#0072B2" if box.get("level", 0) == 0 else "#D55E00",
            )


def _draw(prepared: dict[str, Any], caption: str | None, provenance_sha: str | None):
    plt = configure_matplotlib()
    kind = prepared["kind"]
    if kind == "amr_boxes":
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        figure = plt.figure(figsize=(7.2, 4.8))
        axes = figure.add_subplot(111, projection="3d")
        _draw_boxes_3d(axes, prepared["boxes"])
        axes.set_xlabel(prepared.get("xlabel") or "x")
        axes.set_ylabel(prepared.get("ylabel") or "y")
        axes.set_zlabel(prepared.get("zlabel") or "z")
        axes.set_title(prepared.get("title") or "")
        note = _caption_text(caption, provenance_sha)
        if note:
            figure.text(0.01, 0.01, note, fontsize=8)
        return figure
    if kind == "report_figure":
        panels = prepared.get("panels") or []
        figure, axes_list = plt.subplots(
            1, max(len(panels), 1), figsize=(10.0, 4.4), constrained_layout=True
        )
        if max(len(panels), 1) == 1:
            axes_list = [axes_list]
        for axes, panel in zip(axes_list, panels, strict=False):
            if panel.get("type") == "convergence":
                axes.loglog(panel["x"], panel["y"], marker="o", label=panel.get("name"))
                axes.set_xlabel(panel.get("xlabel") or "1/h")
                axes.set_ylabel(panel.get("ylabel") or "error")
            else:
                axes.plot(panel["x"], panel["y"], label=panel.get("name"))
                axes.set_xlabel(panel.get("xlabel") or "x")
                axes.set_ylabel(panel.get("ylabel") or "value")
            axes.set_title(panel.get("title") or "")
            axes.legend()
        figure.suptitle(prepared.get("title") or "report figure")
        note = _caption_text(caption, provenance_sha)
        if note:
            figure.text(0.01, 0.01, note, fontsize=8)
        return figure
    figure, axes = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    if kind in {"spatial_convergence", "temporal_convergence", "coarse_fine_error"}:
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
    elif kind in {
        "reference_profile",
        "signed_error_profile",
        "invariants_vs_time",
        "phase_amplitude",
        "frequency_spectrum",
        "symmetry_metric",
        "linecuts",
        "linecut",
        "reference_comparison",
    }:
        for item in prepared["series"]:
            if item.get("style") == "exact":
                axes.plot(item["x"], item["y"], label=item["name"], **EXACT_STYLE)
            else:
                axes.plot(item["x"], item["y"], marker="o", label=item["name"])
        axes.legend()
    elif kind == "backend_parity":
        backends = prepared["backends"]
        metrics = list(prepared["metrics"])
        values = prepared["values"]
        positions = list(range(len(backends)))
        width = 0.8 / max(len(metrics), 1)
        for index, metric in enumerate(metrics):
            heights = [float(row[index]) for row in values]
            axes.bar(
                [pos + index * width for pos in positions],
                heights,
                width=width,
                label=metric,
            )
        axes.set_xticks([pos + width * (len(metrics) - 1) / 2.0 for pos in positions])
        axes.set_xticklabels(backends, rotation=15)
        axes.legend()
    elif kind == "performance_breakdown":
        axes.bar(prepared["stages"], prepared["seconds"])
    elif kind in {"orders_heatmap", "component_coverage"}:
        mesh = axes.imshow(prepared["values"], cmap="viridis", aspect="auto")
        axes.set_xticks(range(len(prepared["columns"])))
        axes.set_xticklabels(prepared["columns"], rotation=15)
        axes.set_yticks(range(len(prepared["rows"])))
        axes.set_yticklabels(prepared["rows"])
        for irow, row in enumerate(prepared["values"]):
            for icol, value in enumerate(row):
                axes.text(icol, irow, f"{value:g}", ha="center", va="center", color="white")
        colorbar = figure.colorbar(mesh, ax=axes)
        colorbar.set_label(prepared.get("clabel") or "")
    elif kind == "not_run_panel":
        axes.text(0.5, 0.5, f"not-run\n{prepared.get('reason')}", ha="center", va="center")
        axes.set_axis_off()
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
    if kind != "not_run_panel":
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
    try:
        for fmt in formats:
            path = root / f"{stem}.{fmt}"
            if fmt == "pdf":
                metadata = {
                    "Creator": RENDERER_VERSION,
                    "Producer": RENDERER_VERSION,
                    "CreationDate": None,
                    "ModDate": None,
                }
            else:
                metadata = {"Creator": RENDERER_VERSION, "Date": None}
            figure.savefig(path, format=fmt, metadata=metadata)
            if path.stat().st_size <= 0:
                raise VisualsError(f"empty figure output: {path}")
            outputs[fmt] = str(path)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
    return outputs
