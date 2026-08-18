"""Render a campaign pops.verification.report.v1 summary (plan §31 / §31.1)."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from test_verification_report_schema import LOCAL_PR_SUMMARY
from verification.pops_verify.report import write_verification_report

REPO_ROOT = Path(__file__).resolve().parents[3]
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
SECTION_31_1_TOPICS = (
    "components",
    "backends",
    "dimensions",
    "fail",
    "orders",
    "amr",
    "poisson",
    "coupling",
    "parallel",
    "performance",
    "not tested",
)


def _summary(**updates) -> dict:
    instance = copy.deepcopy(LOCAL_PR_SUMMARY)
    instance.update(updates)
    return instance


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _written_names(output: Path) -> set[str]:
    if not output.exists():
        return set()
    return {path.name for path in output.iterdir() if path.is_file()}


def _read_csv(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def _section(markdown: str, needle: str) -> str:
    lines = markdown.splitlines()
    start = None
    for index, line in enumerate(lines):
        if needle.lower() in line.lower() and line.lstrip().startswith("#"):
            start = index
            break
    if start is None:
        return ""
    stop = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].lstrip().startswith("#"):
            stop = index
            break
    return "\n".join(lines[start:stop])


def test_valid_summary_writes_four_artifacts(tmp_path: Path):
    output = tmp_path / "report"
    written = write_verification_report(_summary(), output)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (output / name).is_file()


def test_summary_json_validates_and_keeps_schema_id(tmp_path: Path):
    output = tmp_path / "report"
    write_verification_report(_summary(), output)
    text = (output / "summary.json").read_text(encoding="utf-8")
    assert text.endswith("\n")
    loaded = json.loads(text)
    assert loaded["schema"] == "pops.verification.report.v1"
    _validator().validate(loaded)
    assert loaded["artifacts"] == LOCAL_PR_SUMMARY["artifacts"]
    assert not any(Path(value).is_absolute() for value in loaded["artifacts"].values())


def test_coverage_csv_header_and_counts(tmp_path: Path):
    output = tmp_path / "report"
    summary = _summary()
    write_verification_report(summary, output)
    rows = _read_csv(output / "coverage.csv")
    assert rows[0] == COVERAGE_HEADER
    assert len(rows) == 2
    coverage = summary["coverage"]
    assert rows[1] == [
        ";".join(coverage["components"]),
        str(coverage["cases_planned"]),
        str(coverage["cases_run"]),
        str(coverage["cases_passed"]),
        str(coverage["cases_failed"]),
        str(coverage["cases_not_supported"]),
    ]


def test_empty_failures_csv_is_header_only(tmp_path: Path):
    output = tmp_path / "report"
    write_verification_report(_summary(), output)
    rows = _read_csv(output / "failures.csv")
    assert rows == [FAILURES_HEADER]


def test_nonempty_failures_csv_has_case_and_reason(tmp_path: Path):
    output = tmp_path / "report"
    summary = _summary()
    summary["coverage"]["cases_failed"] = 1
    summary["coverage"]["cases_passed"] = 0
    summary["failures"] = [
        {
            "case_id": "CP-02",
            "reason": "spatial order below threshold",
            "metrics_ref": "CP-02/metrics.json",
            "provenance_ref": "CP-02/provenance.json",
        }
    ]
    write_verification_report(summary, output)
    rows = _read_csv(output / "failures.csv")
    assert rows[0] == FAILURES_HEADER
    assert len(rows) == 2
    assert rows[1][0] == "CP-02"
    assert rows[1][1] == "spatial order below threshold"


def test_report_md_answers_ten_topics_and_null_amr_reason(tmp_path: Path):
    output = tmp_path / "report"
    write_verification_report(_summary(), output)
    text = (output / "REPORT.md").read_text(encoding="utf-8")
    lowered = text.lower()
    for topic in SECTION_31_1_TOPICS:
        assert topic in lowered
    amr = _section(text, "AMR")
    assert "AMR not run in local pr" in amr
    assert "1.95" not in amr
    assert "order_retained" not in amr.lower() or "null" in amr.lower()


def test_report_md_shows_measured_order(tmp_path: Path):
    output = tmp_path / "report"
    write_verification_report(_summary(), output)
    text = (output / "REPORT.md").read_text(encoding="utf-8")
    assert "1.95" in text
    orders = _section(text, "order")
    assert "1.95" in orders


def test_invalid_summary_writes_no_files(tmp_path: Path):
    output = tmp_path / "report"
    bad_nodes = _summary(max_nodes=3)
    with pytest.raises(ValueError):
        write_verification_report(bad_nodes, output)
    assert _written_names(output) == set()

    missing_coverage = _summary()
    del missing_coverage["coverage"]
    with pytest.raises(ValueError):
        write_verification_report(missing_coverage, output)
    assert _written_names(output) == set()
    assert not output.exists()


def test_non_dict_summary_raises_and_writes_nothing(tmp_path: Path):
    output = tmp_path / "report"
    with pytest.raises(ValueError):
        write_verification_report("not a summary", output)
    with pytest.raises(ValueError):
        write_verification_report(None, output)
    assert _written_names(output) == set()
    assert not output.exists()


def test_written_summary_json_revalidates(tmp_path: Path):
    output = tmp_path / "report"
    write_verification_report(_summary(), output)
    loaded = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"


def test_empty_orders_print_not_applicable_reason(tmp_path: Path):
    output = tmp_path / "report"
    summary = _summary()
    summary["orders"] = []
    summary["not_applicable_reason"]["orders"] = "no order campaign ran"
    write_verification_report(summary, output)
    text = (output / "REPORT.md").read_text(encoding="utf-8")
    orders = _section(text, "order")
    assert "no order campaign ran" in orders
    assert "1.95" not in orders
