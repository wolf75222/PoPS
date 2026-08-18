"""Render a campaign pops.verification.report.v1 summary to Markdown/CSV/JSON.

Plan §31: validate before publication. Measured values stay in the document.
This module does not run cases.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
ARTIFACTS = {
    "report_md": "REPORT.md",
    "summary_json": "summary.json",
    "coverage_csv": "coverage.csv",
    "failures_csv": "failures.csv",
}
COVERAGE_HEADER = [
    "component",
    "cases_planned",
    "cases_run",
    "cases_passed",
    "cases_failed",
    "cases_not_supported",
]
FAILURES_HEADER = ["case_id", "reason", "metrics_ref", "provenance_ref"]


def _validator() -> Draft202012Validator:
    raw = SCHEMA_PATH.read_text(encoding="utf-8")
    schema = json.loads(raw)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _format_validation_error(error: ValidationError) -> str:
    location = ".".join(str(part) for part in error.absolute_path)
    if location:
        return f"{location}: {error.message}"
    return error.message


def _validate_summary(summary) -> None:
    if not isinstance(summary, dict):
        raise ValueError("summary must be a dict")
    try:
        validator = _validator()
    except (OSError, json.JSONDecodeError, SchemaError) as exc:
        raise ValueError(f"invalid schema {SCHEMA_PATH}: {exc}") from exc
    errors = sorted(
        validator.iter_errors(summary), key=lambda item: list(item.absolute_path)
    )
    if errors:
        details = "; ".join(_format_validation_error(error) for error in errors)
        raise ValueError(details)


def _lookup_reason(reasons: dict, path: str) -> str | None:
    if path in reasons:
        return reasons[path]
    parts = path.split(".")
    for key, value in reasons.items():
        key_parts = key.split(".")
        if len(key_parts) != len(parts):
            continue
        if all(
            token == "*" or token == part
            for token, part in zip(key_parts, parts, strict=True)
        ):
            return value
    return None


def _topic_reason(reasons: dict, prefix: str) -> str | None:
    matches: list[str] = []
    if prefix in reasons:
        matches.append(reasons[prefix])
    wildcard = f"{prefix}.*"
    if wildcard in reasons:
        matches.append(reasons[wildcard])
    for key, value in reasons.items():
        if key.startswith(prefix + ".") and value not in matches:
            matches.append(value)
    if not matches:
        return None
    return "; ".join(matches)


def _all_null(value) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return all(_all_null(item) for item in value.values())
    return False


def _format_measured(obj: dict, prefix: str, reasons: dict) -> str:
    if _all_null(obj):
        return _topic_reason(reasons, prefix) or "not applicable"
    lines: list[str] = []
    for key, value in obj.items():
        path = f"{prefix}.{key}"
        if value is None:
            reason = _lookup_reason(reasons, path) or _topic_reason(reasons, prefix)
            lines.append(f"{key}: {reason or 'not applicable'}")
        elif isinstance(value, dict):
            rate = value.get("cells_per_second")
            notes = value.get("notes", "")
            if rate is None:
                reason = _lookup_reason(reasons, path) or "not applicable"
                lines.append(f"{key}: {reason}; notes={notes}")
            else:
                lines.append(f"{key}: cells_per_second={rate} notes={notes}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _format_orders(orders: list, reasons: dict) -> str:
    if not orders:
        return reasons.get("orders") or "not applicable"
    lines = []
    for item in orders:
        observed = item["observed_order"]
        if observed is None:
            lines.append(
                f"{item['case_id']} {item['kind']} {item['variable']}: not applicable"
            )
        else:
            lines.append(
                f"{item['case_id']} {item['kind']} {item['variable']} "
                f"observed_order={observed} threshold={item['threshold']}"
            )
    return "\n".join(lines)


def _format_failures(failures: list) -> str:
    if not failures:
        return "none"
    return "\n".join(f"{item['case_id']}: {item['reason']}" for item in failures)


def _render_report(summary: dict) -> str:
    reasons = summary.get("not_applicable_reason") or {}
    coverage = summary["coverage"]
    not_tested = list(coverage.get("not_tested") or [])
    for reason in reasons.values():
        if reason not in not_tested:
            not_tested.append(reason)
    not_tested_block = "\n".join(not_tested) if not_tested else "none"
    return "\n".join(
        [
            "# Verification report",
            "",
            f"suite: {summary['suite']}",
            f"repository: {summary['repository']}",
            f"repository_sha: {summary['repository_sha']}",
            f"max_nodes: {summary['max_nodes']}",
            "",
            "## 1. Components covered",
            ", ".join(coverage["components"]),
            "",
            "## 2. Backends and dimensions tested",
            f"backends: {', '.join(summary['execution_spaces'])}",
            f"dimensions: {', '.join(str(dim) for dim in summary['native_dimensions'])}",
            "",
            "## 3. Failures",
            _format_failures(summary["failures"]),
            "",
            "## 4. Orders measured",
            _format_orders(summary["orders"], reasons),
            "",
            "## 5. AMR order and invariants",
            _format_measured(summary["amr"], "amr", reasons),
            "",
            "## 6. Poisson and the field",
            _format_measured(summary["poisson"], "poisson", reasons),
            "",
            "## 7. Coupling phase, sign, and energy",
            _format_measured(summary["coupling"], "coupling", reasons),
            "",
            "## 8. Parallel invariance (ranks, threads, GPU)",
            _format_measured(summary["parallel_invariance"], "parallel_invariance", reasons),
            "",
            "## 9. Performance on one and two nodes",
            _format_measured(summary["performance"], "performance", reasons),
            "",
            "## 10. Not tested",
            not_tested_block,
            "",
        ]
    )


def _coverage_csv(summary: dict) -> str:
    coverage = summary["coverage"]
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(COVERAGE_HEADER)
    writer.writerow(
        [
            ";".join(coverage["components"]),
            coverage["cases_planned"],
            coverage["cases_run"],
            coverage["cases_passed"],
            coverage["cases_failed"],
            coverage["cases_not_supported"],
        ]
    )
    return buf.getvalue()


def _failures_csv(summary: dict) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(FAILURES_HEADER)
    for item in summary["failures"]:
        writer.writerow(
            [
                item["case_id"],
                item["reason"],
                item["metrics_ref"],
                item["provenance_ref"],
            ]
        )
    return buf.getvalue()


def write_verification_report(summary, output_dir) -> dict:
    _validate_summary(summary)
    payload = json.dumps(summary, indent=2) + "\n"
    reloaded = json.loads(payload)
    _validate_summary(reloaded)
    contents = {
        "summary.json": payload,
        "coverage.csv": _coverage_csv(summary),
        "failures.csv": _failures_csv(summary),
        "REPORT.md": _render_report(summary),
    }
    path = Path(output_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"cannot create output_dir {path}: {exc}") from exc
    for name, text in contents.items():
        (path / name).write_text(text, encoding="utf-8")
    return dict(ARTIFACTS)
