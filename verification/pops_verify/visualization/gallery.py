"""Release gallery and transverse dashboards (plan §40.7, §40.8)."""
from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any, Iterable

from jsonschema import Draft202012Validator

from verification.pops_verify.visualization.plots import (
    FIXTURE_LABEL,
    file_sha256,
    prepare_heatmap,
    prepare_not_run,
    qualify_fixture_caption,
    render_prepared,
)
from verification.pops_verify.visualization.style import RENDERER_SCRIPT, RENDERER_VERSION

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_visuals.v1.json"

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
    return reasons.get(key) or reasons.get(f"{key}.*") or "not-run: topic not present"


def _topic_values(topic: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(topic, dict):
        return {}
    collected: dict[str, float] = {}
    for key, value in topic.items():
        if value is None:
            continue
        if isinstance(value, bool):
            collected[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            collected[key] = float(value)
    return collected


def _heatmap(name: str, rows: list[str], columns: list[str], values: list[list[float]], title: str) -> dict[str, Any]:
    return prepare_heatmap(
        {
            "kind": name if name in {"orders_heatmap", "component_coverage"} else "orders_heatmap",
            "figure_id": name,
            "rows": rows,
            "columns": columns,
            "values": values,
            "units": {"x": "column", "y": "row", "field": name},
            "title": title,
            "verdict": "pass",
        }
    )


def _orders(summary: dict[str, Any], *, temporal: bool) -> tuple[dict[str, Any], str]:
    name = "temporal_dashboard" if temporal else "orders_heatmap"
    orders = []
    for item in summary.get("orders") or []:
        if item.get("observed_order") is None:
            continue
        kind = str(item.get("kind") or "spatial")
        if temporal and kind != "temporal":
            continue
        if not temporal and kind == "temporal":
            continue
        orders.append(item)
    if not orders:
        reason = (
            "no temporal orders in this report"
            if temporal
            else _na_text(summary, "orders")
        )
        return prepare_not_run(name, reason), "not-run"
    cases: list[str] = []
    kinds: list[str] = []
    lookup: dict[tuple[str, str], float] = {}
    labels = []
    for item in orders:
        case_id = str(item["case_id"])
        kind = str(item.get("kind") or "spatial")
        variable = str(item.get("variable") or "scalar")
        order = float(item["observed_order"])
        row = f"{case_id} {variable}"
        if row not in cases:
            cases.append(row)
        if kind not in kinds:
            kinds.append(kind)
        lookup[(row, kind)] = order
        labels.append(f"{case_id} {variable} {order}")
    values = [[lookup.get((row, kind), float("nan")) for kind in kinds] for row in cases]
    title = ("Temporal orders: " if temporal else "Observed orders: ") + "; ".join(labels)
    prepared = _heatmap(name, cases, kinds, values, title)
    prepared["figure_id"] = name
    prepared["kind"] = "orders_heatmap"
    return prepared, "pass"


def _failures(summary: dict[str, Any]) -> tuple[dict[str, Any], str]:
    failures = summary.get("failures") or []
    if not failures:
        return prepare_not_run("failure_map", "no failures recorded"), "not-run"
    rows = [str(item["case_id"]) for item in failures]
    prepared = _heatmap(
        "failure_map",
        rows,
        ["failed"],
        [[1.0] for _ in rows],
        "Failure map: " + ", ".join(rows),
    )
    prepared["figure_id"] = "failure_map"
    prepared["kind"] = "orders_heatmap"
    return prepared, "pass"


def _topic_dashboard(
    name: str,
    summary: dict[str, Any],
    topic: dict[str, Any] | None,
    reason_key: str,
) -> tuple[dict[str, Any], str]:
    values = _topic_values(topic)
    if not values:
        return prepare_not_run(name, _na_text(summary, reason_key)), "not-run"
    keys = list(values)
    prepared = _heatmap(
        name,
        [name],
        keys,
        [[values[key] for key in keys]],
        f"{name}: " + ", ".join(keys),
    )
    prepared["figure_id"] = name
    prepared["kind"] = "orders_heatmap"
    return prepared, "pass"


def _coverage(summary: dict[str, Any]) -> tuple[dict[str, Any], str]:
    components = list((summary.get("coverage") or {}).get("components") or [])
    if not components:
        return prepare_not_run("component_coverage", "coverage.components missing"), "not-run"
    prepared = _heatmap(
        "component_coverage",
        ["coverage"],
        components,
        [[1.0] * len(components)],
        "Component coverage: " + ",".join(components),
    )
    prepared["figure_id"] = "component_coverage"
    prepared["kind"] = "component_coverage"
    return prepared, "pass"


def _replace_gallery_section(text: str, block: str) -> str:
    marker = "## Visual gallery"
    body = block.strip() + "\n"
    if marker not in text:
        return text.rstrip() + "\n\n" + body
    head, rest = text.split(marker, 1)
    parts = re.split(r"^## ", rest, maxsplit=1, flags=re.M)
    remainder = f"## {parts[1]}" if len(parts) > 1 else ""
    return head.rstrip() + "\n\n" + body + remainder


def _gallery_digests(root: Path) -> dict[str, str]:
    summary = root / "summary.json"
    if not summary.is_file():
        raise FileNotFoundError(summary)
    program = root / "program.json"
    native = root / "native_artifact.json"
    return {
        "resolved_case_digest": file_sha256(summary),
        "program_digest": (
            file_sha256(program)
            if program.is_file()
            else "not-run: program.json absent from gallery directory"
        ),
        "native_artifact_digest": (
            file_sha256(native)
            if native.is_file()
            else "not-run: native_artifact.json absent from gallery directory"
        ),
    }


def _combine_status(statuses: dict[str, str]) -> str:
    if any(value == "fail" for value in statuses.values()):
        return "fail"
    if any(value == "not-run" for value in statuses.values()):
        return "not-run"
    if statuses and all(value == "pass" for value in statuses.values()):
        return "pass"
    return "not-run"


def render_release_gallery(
    output_dir: str | Path,
    *,
    formats: Iterable[str] = ("svg", "png", "pdf"),
    caption: str | None = None,
) -> dict[str, str]:
    root = Path(output_dir)
    summary = _load_summary(root)
    sha = summary.get("repository_sha") or "fixture:unspecified"
    caption_text = qualify_fixture_caption(caption or FIXTURE_LABEL)
    gallery_dir = root / "analysis" / "figures" / "publication"
    mapping: dict[str, tuple[dict[str, Any], str]] = {
        "orders_heatmap": _orders(summary, temporal=False),
        "failure_map": _failures(summary),
        "amr_degradation": _topic_dashboard(
            "amr_degradation", summary, summary.get("amr"), "amr"
        ),
        "component_coverage": _coverage(summary),
        "backend_parity": _topic_dashboard(
            "backend_parity", summary, summary.get("parallel_invariance"), "parallel_invariance"
        ),
        "conservation_dashboard": _topic_dashboard(
            "conservation_dashboard", summary, summary.get("coupling"), "coupling"
        ),
        "poisson_dashboard": _topic_dashboard(
            "poisson_dashboard", summary, summary.get("poisson"), "poisson"
        ),
        "temporal_dashboard": _orders(summary, temporal=True),
        "amr_dashboard": _topic_dashboard(
            "amr_dashboard", summary, summary.get("amr"), "amr"
        ),
        "performance_dashboard": _topic_dashboard(
            "performance_dashboard", summary, summary.get("performance"), "performance"
        ),
    }
    outputs: dict[str, str] = {}
    status: dict[str, str] = {}
    for name in DASHBOARDS:
        prepared, dash_status = mapping[name]
        rendered = render_prepared(
            prepared,
            gallery_dir,
            formats=formats,
            caption=caption_text,
            provenance_sha=sha,
        )
        outputs[name] = rendered.get("svg") or next(iter(rendered.values()))
        status[name] = dash_status
    status["verdict"] = _combine_status({key: status[key] for key in DASHBOARDS})
    status_path = root / "analysis" / "gallery_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    data_kind = (
        "deterministic_fixture"
        if str(sha).startswith("fixture:")
        else "campaign"
    )
    digests = _gallery_digests(root)
    figures = []
    for name in DASHBOARDS:
        path = Path(outputs[name])
        rel = str(path.resolve().relative_to(root.resolve()))
        fmt = path.suffix.lstrip(".")
        figures.append(
            {
                "case_id": "campaign-gallery",
                "run_id": str(summary.get("suite") or "gallery"),
                "figure_id": name,
                "kind": name,
                "role": "publication",
                "source_files": ["summary.json"],
                "variables": [name],
                "units": {"x": "column", "y": "row"},
                "transform": None,
                "color_range": None,
                "resolutions": [],
                "amr_levels": [],
                "times": [],
                "step_numbers": [],
                "repository_sha": sha,
                "resolved_case_digest": digests["resolved_case_digest"],
                "program_digest": digests["program_digest"],
                "native_artifact_digest": digests["native_artifact_digest"],
                "renderer": {"script": RENDERER_SCRIPT, "version": RENDERER_VERSION},
                "output_hashes": {fmt: file_sha256(path)},
                "outputs": {fmt: rel},
                "proves": f"{name} reconstructed from the campaign summary.",
                "does_not_prove": (
                    "This fixture is not a live PoPS campaign result."
                    if data_kind == "deterministic_fixture"
                    else "The dashboard does not invent values missing from summary.json."
                ),
                "quantitative_companion": None,
                "pr": False,
                "verdict": status[name],
            }
        )
    manifest = {
        "schema": "pops.verification.visual_manifest.v1",
        "case_id": "campaign-gallery",
        "run_id": str(summary.get("suite") or "gallery"),
        "data_kind": data_kind,
        "suite": summary.get("suite") if summary.get("suite") in {"pr", "nightly", "weekly", "release", "two_node"} else "pr",
        "verdict": status["verdict"],
        "repository_sha": sha,
        "resolved_case_digest": digests["resolved_case_digest"],
        "program_digest": digests["program_digest"],
        "native_artifact_digest": digests["native_artifact_digest"],
        "renderer": {"script": RENDERER_SCRIPT, "version": RENDERER_VERSION},
        "figures": figures,
        "dimensions": {
            "1d": {
                "status": "not_applicable",
                "justification": "Gallery is a campaign-level dashboard, not a 1-d run.",
                "artifacts": [],
            },
            "2d": {
                "status": "not_applicable",
                "justification": "Gallery is a campaign-level dashboard, not a 2-d run.",
                "artifacts": [],
            },
            "3d": {
                "status": "not_applicable",
                "justification": "Gallery is a campaign-level dashboard, not a 3-d run.",
                "artifacts": [],
            },
        },
        "proves": "Gallery dashboards reconstruct from summary.json without invented series.",
        "does_not_prove": "Missing topics are not-run and are not live campaign science.",
    }
    Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(
        manifest
    )
    (root / "analysis" / "visual_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    report_path = root / "REPORT.md"
    if report_path.is_file():
        lines = ["## Visual gallery", ""]
        for name, path in outputs.items():
            rel = Path(path).resolve().relative_to(root.resolve())
            lines.append(f"- `{name}`: `{rel}` ({status[name]})")
        report_path.write_text(
            _replace_gallery_section(report_path.read_text(encoding="utf-8"), "\n".join(lines)),
            encoding="utf-8",
        )
    return outputs
