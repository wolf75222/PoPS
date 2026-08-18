"""JSON Schema for Phase 8 visual contracts and run manifests (plan §40.2, §40.9)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_visuals.v1.json"

# Plan §40.9 machine-readable contract, plus the §40.11 dimension block.
PLAN_SECTION_40_9_CONTRACT = {
    "schema": "pops.verification.visuals.v1",
    "required": [
        "reference_comparison",
        "spatial_convergence",
        "coarse_fine_error",
        "amr_patch_map",
        "storyboard",
        "animation",
    ],
    "optional": ["hero_figure"],
    "animation": {
        "id": "wave_coarse_fine_crossing",
        "master": "mp4",
        "preview": "gif",
        "frames": "accepted_states",
        "color_limits": "global",
        "overlays": ["amr_patches", "coarse_fine_interfaces", "time", "leaf_cells"],
        "key_events": [
            "before_entry",
            "entry",
            "inside_fine",
            "exit",
            "periodic_crossing",
            "final",
        ],
    },
    "acceptance": {
        "require_source_data": True,
        "require_manifest": True,
        "require_fixed_animation_scale": True,
        "require_storyboard_for_animation": True,
        "forbid_missing_units": True,
    },
    "dimensions": {
        "1d": {
            "status": "required",
            "required": [
                "reference_profile",
                "signed_error_profile",
                "spatial_convergence",
            ],
        },
        "2d": {
            "status": "required",
            "required": [
                "exact_field",
                "numerical_field",
                "signed_error_field",
                "linecuts",
                "report_figure",
            ],
        },
        "3d": {
            "status": "extended",
            "required_when_executed": [
                "slice_xy",
                "slice_xz",
                "slice_yz",
                "linecut",
                "amr_boxes",
                "report_figure",
            ],
        },
    },
    "report": {
        "require_quantitative_companion": True,
        "require_source_data": True,
        "require_same_run_identity": True,
    },
}

PLAN_SECTION_40_2_MANIFEST = {
    "schema": "pops.verification.visual_manifest.v1",
    "case_id": "AM-01",
    "run_id": "fixture-am01-2d",
    "data_kind": "deterministic_fixture",
    "suite": "pr",
    "verdict": "pass",
    "repository_sha": "57ed20571b2255bc6610cb09ceb9ad7925a2b173",
    "resolved_case_digest": "sha256:resolved-case",
    "program_digest": "sha256:program",
    "native_artifact_digest": "sha256:native-leaf",
    "renderer": {
        "script": "scripts/render_verification_visuals.py",
        "version": "pops.verification.visuals.v1",
    },
    "proves": "AMR coarse-fine crossing figures reconstruct from versioned visual_data.",
    "does_not_prove": "This fixture is not a live PoPS campaign result.",
    "figures": [
        {
            "case_id": "AM-01",
            "run_id": "fixture-am01-2d",
            "figure_id": "spatial_convergence",
            "kind": "spatial_convergence",
            "role": "publication",
            "source_files": ["analysis/visual_data/spatial_convergence.json"],
            "variables": ["scalar"],
            "units": {"x": "1/h", "y": "L2 error"},
            "transform": "loglog",
            "color_range": None,
            "resolutions": [16, 32, 64, 128],
            "amr_levels": [0, 1],
            "times": [1.0],
            "step_numbers": [100],
            "repository_sha": "57ed20571b2255bc6610cb09ceb9ad7925a2b173",
            "resolved_case_digest": "sha256:resolved-case",
            "program_digest": "sha256:program",
            "native_artifact_digest": "sha256:native-leaf",
            "renderer": {
                "script": "scripts/render_verification_visuals.py",
                "version": "pops.verification.visuals.v1",
            },
            "output_hashes": {
                "svg": "sha256:deadbeef",
            },
            "outputs": {
                "svg": "analysis/figures/publication/spatial_convergence.svg",
            },
            "proves": "Observed spatial order on four resolutions.",
            "does_not_prove": "Temporal order.",
            "quantitative_companion": "spatial_convergence",
            "pr": True,
        }
    ],
    "dimensions": {
        "1d": {
            "status": "not_applicable",
            "justification": "This fixture run is two-dimensional only.",
            "artifacts": [],
        },
        "2d": {
            "status": "required",
            "justification": None,
            "artifacts": ["spatial_convergence"],
        },
        "3d": {
            "status": "extended",
            "justification": "3-d not executed in this fixture run.",
            "artifacts": [],
        },
    },
}

REQUIRED_CONTRACT = (
    "schema",
    "required",
    "optional",
    "animation",
    "acceptance",
    "dimensions",
    "report",
)
REQUIRED_MANIFEST = (
    "schema",
    "case_id",
    "run_id",
    "data_kind",
    "suite",
    "verdict",
    "repository_sha",
    "figures",
    "dimensions",
    "proves",
    "does_not_prove",
)


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_schema_file_is_draft_2020_12():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator.check_schema(schema)


def test_plan_40_9_contract_is_valid():
    _validator().validate(PLAN_SECTION_40_9_CONTRACT)


def test_plan_40_2_manifest_is_valid():
    _validator().validate(PLAN_SECTION_40_2_MANIFEST)


def test_contract_missing_required_is_rejected():
    instance = copy.deepcopy(PLAN_SECTION_40_9_CONTRACT)
    del instance["acceptance"]
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_manifest_unknown_verdict_is_rejected():
    instance = copy.deepcopy(PLAN_SECTION_40_2_MANIFEST)
    instance["verdict"] = "pretty-enough"
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_manifest_unknown_data_kind_is_rejected():
    instance = copy.deepcopy(PLAN_SECTION_40_2_MANIFEST)
    instance["data_kind"] = "hand_copied_curve"
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_figure_missing_units_is_rejected():
    instance = copy.deepcopy(PLAN_SECTION_40_2_MANIFEST)
    del instance["figures"][0]["units"]
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_not_applicable_dimension_requires_justification():
    instance = copy.deepcopy(PLAN_SECTION_40_2_MANIFEST)
    instance["dimensions"]["1d"]["justification"] = None
    with pytest.raises(ValidationError):
        _validator().validate(instance)
