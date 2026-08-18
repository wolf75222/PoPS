"""Release gallery and transverse dashboards (plan §40.7, §40.8)."""
from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Iterable

from verification.pops_verify.visualization.plots import render_prepared

DASHBOARDS = (
    "orders_heatmap",
    "amr_degradation",
    "component_coverage",
    "backend_parity",
    "conservation_dashboard",
    "poisson_dashboard",
    "temporal_dashboard",
    "amr_dashboard",
    "performance_dashboard",
    "failure_map",
)


def _load_summary(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _na_text(summary: dict[str, Any], key: str) -> str:
    reasons = summary.get("not_applicable_reason") or {}
    return reasons.get(key) or reasons.get(f"{key}.*") or "not applicable"


def _placeholder(title: str, message: str) -> dict[str, Any]:
    return {
        "kind": "reference_profile",
        "figure_id": title,
        "scale": "linear",
        "xlabel": "status",
        "ylabel": "value",
        "series": [{"name": message, "x": [0.0, 1.0], "y": [0.0, 0.0], "style": "exact"}],
        "title": f"{title}: {message}",
    }


def _orders_prepared(summary: dict[str, Any]) -> dict[str, Any]:
    orders = summary.get("orders") or []
    if not orders:
        return _placeholder("orders_heatmap", _na_text(summary, "orders"))
    labels = [
        f"{item['case_id']} {item['variable']} {item['observed_order']}"
        for item in orders
        if item.get("observed_order") is not None
    ]
    return {
        "kind": "spatial_convergence",
        "figure_id": "orders_heatmap",
        "scale": "loglog",
        "xlabel": "case index",
        "ylabel": "observed order",
        "series": [
            {
                "name": f"{item['case_id']} {item['variable']}",
                "x": [index + 1, index + 2],
                "y": [
                    float(item["observed_order"]),
                    float(item["observed_order"]),
                ],
                "style": "numerical",
            }
            for index, item in enumerate(orders)
            if item.get("observed_order") is not None
        ],
        "reference_slopes": [],
        "title": "Observed orders: " + "; ".join(labels),
    }


def _failures_prepared(summary: dict[str, Any]) -> dict[str, Any]:
    failures = summary.get("failures") or []
    if not failures:
        return _placeholder("failure_map", "none")
    return {
        "kind": "reference_profile",
        "figure_id": "failure_map",
        "scale": "linear",
        "xlabel": "failure index",
        "ylabel": "count",
        "series": [
            {
                "name": item["case_id"],
                "x": [index, index + 1],
                "y": [1.0, 1.0],
                "style": "numerical",
            }
            for index, item in enumerate(failures)
        ],
        "title": "Failure map",
    }


def _topic_prepared(name: str, summary: dict[str, Any], topic: dict[str, Any] | None) -> dict[str, Any]:
    if topic is None or all(value is None for value in topic.values()):
        return _placeholder(name, _na_text(summary, name.split("_")[0] if name != "backend_parity" else "parallel_invariance"))
    xs = []
    ys = []
    labels = []
    for index, (key, value) in enumerate(topic.items(), start=1):
        if isinstance(value, bool):
            xs.append(float(index))
            ys.append(1.0 if value else 0.0)
            labels.append(key)
        elif isinstance(value, (int, float)):
            xs.append(float(index))
            ys.append(float(value))
            labels.append(key)
    if not xs:
        return _placeholder(name, _na_text(summary, name))
    return {
        "kind": "reference_profile",
        "figure_id": name,
        "scale": "linear",
        "xlabel": "metric",
        "ylabel": "value",
        "series": [
            {"name": label, "x": [x, x + 0.25], "y": [y, y], "style": "numerical"}
            for label, x, y in zip(labels, xs, ys, strict=True)
        ],
        "title": name,
    }


def render_release_gallery(
    output_dir: str | Path,
    *,
    formats: Iterable[str] = ("svg", "png", "pdf"),
    caption: str | None = "campaign dashboard",
) -> dict[str, str]:
    root = Path(output_dir)
    summary = _load_summary(root)
    sha = summary.get("repository_sha")
    gallery_dir = root / "analysis" / "figures" / "publication"
    mapping: dict[str, dict[str, Any]] = {
        "orders_heatmap": _orders_prepared(summary),
        "failure_map": _failures_prepared(summary),
        "amr_degradation": _topic_prepared("amr_degradation", summary, summary.get("amr")),
        "component_coverage": _placeholder(
            "component_coverage",
            ",".join(summary["coverage"]["components"]),
        ),
        "backend_parity": _topic_prepared(
            "backend_parity", summary, summary.get("parallel_invariance")
        ),
        "conservation_dashboard": _placeholder(
            "conservation_dashboard", _na_text(summary, "coupling")
        ),
        "poisson_dashboard": _topic_prepared("poisson_dashboard", summary, summary.get("poisson")),
        "temporal_dashboard": _orders_prepared(summary),
        "amr_dashboard": _topic_prepared("amr_dashboard", summary, summary.get("amr")),
        "performance_dashboard": _placeholder(
            "performance_dashboard", _na_text(summary, "performance")
        ),
    }
    mapping["temporal_dashboard"]["figure_id"] = "temporal_dashboard"
    outputs: dict[str, str] = {}
    for name in DASHBOARDS:
        prepared = mapping[name]
        rendered = render_prepared(
            prepared,
            gallery_dir,
            formats=formats,
            caption=caption,
            provenance_sha=sha,
        )
        outputs[name] = rendered.get("svg") or next(iter(rendered.values()))
    report_path = root / "REPORT.md"
    if report_path.is_file():
        block = ["", "## Visual gallery", ""]
        for name, path in outputs.items():
            rel = Path(path).resolve().relative_to(root.resolve())
            block.append(f"- `{name}`: `{rel}`")
        report_path.write_text(
            report_path.read_text(encoding="utf-8") + "\n".join(block) + "\n",
            encoding="utf-8",
        )
    return outputs
