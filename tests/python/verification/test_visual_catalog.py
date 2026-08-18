"""Exhaustive Phase 8 case × dimension catalog (plan §40.5, §40.6.11, §40.8)."""
from __future__ import annotations

from verification.pops_verify.visualization.catalog import (
    HERO_CASES,
    P0_STATIC_CASES,
    SCIENTIFIC_CASE_IDS,
    catalog_entry,
    dimension_code,
    iter_catalog,
    visual_contract_for,
)

EXPECTED_IDS = (
    "TR-01",
    "TR-02",
    "TR-03",
    "TR-04",
    "TR-05",
    "TR-06",
    "TR-07",
    "EU-01",
    "EU-02",
    "EU-03",
    "EU-04",
    "EU-05",
    "EU-06",
    "PO-01",
    "PO-02",
    "PO-03",
    "PO-04",
    "PO-05",
    "PO-06",
    "PO-07",
    "CP-01",
    "CP-02",
    "CP-03",
    "CP-04",
    "CP-05",
    "CP-06",
    "CP-07",
    "CP-08",
    "CP-09",
    "CP-10",
    "CP-11",
    "CP-12",
    "TM-01",
    "TM-02",
    "TM-03",
    "TM-04",
    "TM-05",
    "TM-06",
    "TM-07",
    "TM-08",
    "AM-01",
    "AM-02",
    "AM-03",
    "AM-04",
    "AM-05",
    "AM-06",
    "AM-07",
    "AM-08",
    "AM-09",
    "AM-10",
    "AM-11",
    "AM-12",
    "RB-01",
    "RB-02",
    "RB-03",
    "RB-04",
    "RB-05",
    "RB-06",
    "RB-07",
    "RB-08",
    "RB-09",
    "GE-01",
    "GE-02",
    "GE-03",
    "GE-04",
    "GE-05",
    "GE-06",
    "IF-01",
    "IF-02",
    "IF-03",
    "IF-04",
    "IF-05",
    "IF-06",
    "IF-07",
    "IF-08",
    "IF-09",
    "IF-10",
    "PF-01",
    "PF-02",
    "PF-03",
    "PF-04",
    "PF-05",
    "PF-06",
    "PF-07",
    "PF-08",
    "PF-09",
    "PF-10",
    "PF-11",
    "PF-12",
)

# Plan §40.6.11 dimension codes: R / E / N/A
EXPECTED_DIMS = {
    "TR-01": ("R", "R", "R"),
    "TR-02": ("R", "R", "E"),
    "TR-03": ("N/A", "R", "E"),
    "TR-04": ("N/A", "E", "R"),
    "TR-05": ("R", "R", "R"),
    "TR-06": ("E", "R", "R"),
    "TR-07": ("R", "R", "E"),
    "EU-01": ("R", "R", "R"),
    "EU-02": ("N/A", "R", "E"),
    "EU-03": ("R", "R", "R"),
    "EU-04": ("R", "R", "E"),
    "EU-05": ("N/A", "R", "E"),
    "EU-06": ("R", "R", "R"),
    "PO-01": ("R", "R", "R"),
    "PO-02": ("R", "R", "E"),
    "PO-03": ("R", "R", "E"),
    "PO-04": ("E", "R", "R"),
    "PO-05": ("R", "R", "R"),
    "PO-06": ("R", "R", "R"),
    "PO-07": ("R", "R", "R"),
    "CP-01": ("R", "R", "E"),
    "CP-02": ("R", "R", "E"),
    "CP-03": ("R", "E", "E"),
    "CP-04": ("N/A", "R", "R"),
    "CP-05": ("R", "E", "E"),
    "CP-06": ("R", "E", "N/A"),
    "CP-07": ("R", "R", "E"),
    "CP-08": ("R", "R", "R"),
    "CP-09": ("R", "R", "E"),
    "CP-10": ("R", "E", "N/A"),
    "CP-11": ("N/A", "R", "N/A"),
    "CP-12": ("R", "R", "R"),
    "TM-01": ("R", "R", "E"),
    "TM-02": ("R", "E", "N/A"),
    "TM-03": ("R", "R", "R"),
    "TM-04": ("R", "R", "R"),
    "TM-05": ("R", "E", "N/A"),
    "TM-06": ("R", "E", "E"),
    "TM-07": ("R", "R", "E"),
    "TM-08": ("R", "R", "E"),
    "AM-01": ("R", "R", "R"),
    "AM-02": ("R", "R", "E"),
    "AM-03": ("E", "R", "R"),
    "AM-04": ("R", "R", "R"),
    "AM-05": ("R", "R", "E"),
    "AM-06": ("R", "R", "R"),
    "AM-07": ("R", "R", "R"),
    "AM-08": ("R", "R", "R"),
    "AM-09": ("R", "R", "R"),
    "AM-10": ("R", "R", "R"),
    "AM-11": ("R", "R", "E"),
    "AM-12": ("E", "R", "R"),
    "RB-01": ("R", "R", "R"),
    "RB-02": ("R", "E", "E"),
    "RB-03": ("R", "E", "E"),
    "RB-04": ("R", "E", "N/A"),
    "RB-05": ("E", "R", "R"),
    "RB-06": ("R", "R", "R"),
    "RB-07": ("N/A", "R", "N/A"),
    "RB-08": ("N/A", "R", "N/A"),
    "RB-09": ("R", "N/A", "N/A"),
    "GE-01": ("E", "R", "N/A"),
    "GE-02": ("E", "R", "N/A"),
    "GE-03": ("R", "R", "R"),
    "GE-04": ("E", "R", "N/A"),
    "GE-05": ("E", "R", "N/A"),
    "GE-06": ("N/A", "R", "N/A"),
    "IF-01": ("R", "R", "R"),
    "IF-02": ("R", "R", "R"),
    "IF-03": ("R", "R", "R"),
    "IF-04": ("R", "R", "R"),
    "IF-05": ("R", "R", "R"),
    "IF-06": ("R", "R", "R"),
    "IF-07": ("R", "R", "E"),
    "IF-08": ("R", "R", "R"),
    "IF-09": ("R", "R", "R"),
    "IF-10": ("R", "R", "R"),
    "PF-01": ("R", "R", "R"),
    "PF-02": ("R", "R", "R"),
    "PF-03": ("R", "R", "R"),
    "PF-04": ("R", "R", "R"),
    "PF-05": ("R", "R", "R"),
    "PF-06": ("R", "R", "R"),
    "PF-07": ("E", "R", "R"),
    "PF-08": ("R", "R", "R"),
    "PF-09": ("E", "R", "R"),
    "PF-10": ("R", "R", "R"),
    "PF-11": ("E", "R", "R"),
    "PF-12": ("E", "R", "E"),
}

P0_EXPECTED = (
    "TR-01",
    "EU-01",
    "EU-02",
    "PO-01",
    "CP-01",
    "CP-02",
    "TM-01",
    "TM-02",
    "AM-01",
    "AM-09",
    "RB-01",
    "RB-05",
    "IF-01",
    "IF-03",
    "PF-06",
)
HERO_EXPECTED = (
    "TR-03",
    "EU-02",
    "CP-02",
    "CP-11",
    "GE-06",
    "AM-01",
    "AM-02",
    "AM-03",
    "AM-11",
    "RB-05",
    "RB-08",
    "GE-03",
    "PF-06",
    "PF-11",
)


def test_catalog_covers_all_89_scientific_cases():
    assert SCIENTIFIC_CASE_IDS == EXPECTED_IDS
    assert len(list(iter_catalog())) == 89


def test_dimension_codes_match_plan_40_6_11():
    for case_id, expected in EXPECTED_DIMS.items():
        assert dimension_code(case_id) == expected, case_id


def test_na_dimensions_have_explicit_justification():
    for entry in iter_catalog():
        for axis, code in zip(("1d", "2d", "3d"), entry.dimension_codes, strict=True):
            if code == "N/A":
                assert entry.na_reasons[axis], entry.case_id


def test_p0_and_hero_sets_match_plan():
    assert P0_STATIC_CASES == P0_EXPECTED
    assert HERO_CASES == HERO_EXPECTED


def test_required_dimension_declares_minimum_artifacts():
    for entry in iter_catalog():
        for axis, code in zip(("1d", "2d", "3d"), entry.dimension_codes, strict=True):
            if code == "R":
                assert entry.artifacts[axis], entry.case_id
                assert "report_figure" in entry.artifacts[axis], entry.case_id


def test_visual_contract_is_schema_valid():
    from jsonschema import Draft202012Validator

    schema_path = (
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "schemas"
        / "verification_visuals.v1.json"
    )
    schema = __import__("json").loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    contract = visual_contract_for("AM-01")
    validator.validate(contract)
    assert contract["schema"] == "pops.verification.visuals.v1"
    assert contract["dimensions"]["1d"]["status"] == "required"


def test_unknown_case_is_rejected():
    import pytest

    with pytest.raises(KeyError):
        catalog_entry("PH-00")


def test_every_catalog_contract_is_schema_valid():
    import json
    from pathlib import Path

    from jsonschema import Draft202012Validator

    schema_path = Path(__file__).resolve().parents[3] / "schemas" / "verification_visuals.v1.json"
    validator = Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")))
    for case_id in SCIENTIFIC_CASE_IDS:
        validator.validate(visual_contract_for(case_id))
