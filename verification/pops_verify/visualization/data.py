"""Load run artefacts for Phase 8 rendering without inventing numbers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from verification.pops_verify.metrics import SCHEMA_PATH as METRICS_SCHEMA
from verification.pops_verify.provenance import SCHEMA_PATH as PROVENANCE_SCHEMA

ALLOWED_VERDICTS = ("pass", "fail", "not-supported", "not-run")
ALLOWED_DATA_KINDS = ("deterministic_fixture", "campaign")
FIXTURE_LABEL = "DETERMINISTIC FIXTURE — not a PoPS campaign result"
CAMPAIGN_LABEL = "campaign result"


class VisualsError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RunBundle:
    run_dir: Path
    case_id: str
    run_id: str
    data_kind: str
    data_kind_label: str
    verdict: str
    metrics: dict[str, Any]
    provenance: dict[str, Any]


def _load_schema(path: Path) -> Draft202012Validator:
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _read_json(path: Path, *, kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise VisualsError(f"missing {kind}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VisualsError(f"invalid JSON in {kind} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise VisualsError(f"{kind} {path} must be a JSON object")
    return payload


def _validate(document: dict[str, Any], schema_path: Path, *, kind: str) -> None:
    try:
        _load_schema(schema_path).validate(document)
    except ValidationError as exc:
        raise VisualsError(f"invalid {kind}: {exc.message}") from exc


def _status_payload(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "status.json"
    if not path.is_file():
        raise VisualsError(f"missing status.json: {path}")
    return _read_json(path, kind="status")


def load_run_bundle(run_dir: str | Path) -> RunBundle:
    root = Path(run_dir)
    metrics = _read_json(root / "metrics.json", kind="metrics")
    provenance = _read_json(root / "provenance.json", kind="provenance")
    _validate(metrics, METRICS_SCHEMA, kind="metrics")
    _validate(provenance, PROVENANCE_SCHEMA, kind="provenance")
    status = _status_payload(root)
    verdict = status.get("verdict")
    if verdict not in ALLOWED_VERDICTS:
        raise VisualsError("status.json verdict must be pass/fail/not-supported/not-run")
    data_kind = status.get("data_kind", "campaign")
    if data_kind not in ALLOWED_DATA_KINDS:
        raise VisualsError("status.json data_kind must be campaign or deterministic_fixture")
    case_id = status.get("case_id") or metrics.get("case_id") or provenance.get("case_id")
    run_id = status.get("run_id") or root.name
    if not isinstance(case_id, str) or not case_id:
        raise VisualsError("case_id missing from status/metrics")
    label = FIXTURE_LABEL if data_kind == "deterministic_fixture" else CAMPAIGN_LABEL
    return RunBundle(
        run_dir=root,
        case_id=case_id,
        run_id=str(run_id),
        data_kind=data_kind,
        data_kind_label=label,
        verdict=verdict,
        metrics=metrics,
        provenance=provenance,
    )


def _series_nonempty(series: Any) -> bool:
    if not isinstance(series, list) or not series:
        return False
    for item in series:
        if not isinstance(item, dict):
            return False
        x_values = item.get("x")
        y_values = item.get("y")
        if not isinstance(x_values, list) or not isinstance(y_values, list):
            return False
        if not x_values or not y_values or len(x_values) != len(y_values):
            return False
    return True


def _load_csv_series(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise VisualsError(f"empty visual_data CSV: {path}")
    grouped: dict[str, dict[str, list]] = {}
    for row in rows:
        name = row.get("name") or "series"
        grouped.setdefault(name, {"x": [], "y": [], "unit": row.get("unit") or "1"})
        grouped[name]["x"].append(float(row["x"]))
        grouped[name]["y"].append(float(row["y"]))
    return {
        "figure_id": path.stem,
        "kind": path.stem,
        "units": {"x": "1", "y": "1"},
        "variables": list(grouped),
        "series": [
            {"name": name, "x": values["x"], "y": values["y"], "unit": values["unit"]}
            for name, values in grouped.items()
        ],
    }


def load_visual_series(run_dir: str | Path, figure_id: str) -> dict[str, Any]:
    bundle = load_run_bundle(run_dir)
    visual_dir = bundle.run_dir / "analysis" / "visual_data"
    json_path = visual_dir / f"{figure_id}.json"
    csv_path = visual_dir / f"{figure_id}.csv"
    if json_path.is_file():
        payload = _read_json(json_path, kind="visual_data")
    elif csv_path.is_file():
        payload = _load_csv_series(csv_path)
    else:
        raise VisualsError(f"missing visual_data for {figure_id}: {json_path}")
    series_kind = payload.get("data_kind", bundle.data_kind)
    if series_kind != bundle.data_kind:
        raise VisualsError(
            f"visual_data data_kind {series_kind!r} does not match run {bundle.data_kind!r}"
        )
    units = payload.get("units")
    if not isinstance(units, dict) or not units:
        raise VisualsError(f"visual_data {figure_id} is missing units")
    if "field" in payload:
        field = payload["field"]
        if not isinstance(field, list) or not field:
            raise VisualsError(f"empty visual_data field: {figure_id}")
        return payload
    frames = payload.get("frames")
    if isinstance(frames, list) and frames:
        payload = dict(payload)
        payload["data_kind"] = bundle.data_kind
        payload["verdict"] = bundle.verdict
        return payload
    if not _series_nonempty(payload.get("series")):
        raise VisualsError(f"empty visual_data series: {figure_id}")
    payload = dict(payload)
    payload["data_kind"] = bundle.data_kind
    payload["verdict"] = bundle.verdict
    return payload
