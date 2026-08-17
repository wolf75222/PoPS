"""JSON Schema for the parsed verification/manifest.toml object (plan §5)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 test environments
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_manifest.v1.json"
MANIFEST_PATH = REPO_ROOT / "verification" / "manifest.toml"

# Plan §5 example: same header as the current manifest plus one CP-02 case.
PLAN_SECTION_5_EXAMPLE = """\
schema = "pops.verification.manifest.v1"
repository = "wolf75222/PoPS"
max_nodes = 2

[current_capabilities]
exact_native_dimension = true
cartesian_system_runtime = true
polar_system_runtime = false
amr_total_levels_baseline = 3
amr_refinement_ratios_baseline = [2, 2]
hdf5_requires_mpi = true

[[case]]
id = "CP-02"
path = "verification/cases/euler_poisson/langmuir_cold/run.py"
name = "Cold Langmuir wave"
verification_kind = "code-verification"
evidence_status = "required"
physics = ["continuity", "momentum", "poisson", "electrostatic_source"]
oracle = "linear_eigenmode_and_closed_form"
native_dimensions = [1, 2]
execution_spaces = ["KokkosSerial", "KokkosOpenMP", "KokkosCuda"]
mpi_modes = ["off", "on"]
suites = ["pr", "nightly", "weekly", "release", "two_node"]
requires = [
  "public_case_pipeline",
  "cartesian_layout",
  "poisson",
  "field_at_program_stage",
]

[case.resources.pr]
nodes = 1
mpi_ranks = 1
omp_threads = 1
resolutions = [32, 64, 128]

[case.resources.two_node]
nodes = [1, 2]
mpi_ranks_per_node = [1, 2, 4]
gpus_per_node = [1, 2, 4]
max_wall_seconds = 3600

[case.acceptance]
spatial_order_min = 1.8
temporal_order_min = 1.8
poisson_relative_residual_max = 1.0e-10
finite = true
charge_conservation = true
"""

_CAPABILITY_KEYS = (
    "exact_native_dimension",
    "cartesian_system_runtime",
    "polar_system_runtime",
    "amr_total_levels_baseline",
    "amr_refinement_ratios_baseline",
    "hdf5_requires_mpi",
)


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _parse_toml(text: str) -> dict:
    return tomllib.loads(text)


def _current_manifest() -> dict:
    return _parse_toml(MANIFEST_PATH.read_text(encoding="utf-8"))


def _plan_example() -> dict:
    return _parse_toml(PLAN_SECTION_5_EXAMPLE)


def test_current_manifest_is_valid():
    _validator().validate(_current_manifest())


def test_empty_case_array_is_valid():
    instance = _current_manifest()
    instance["case"] = []
    _validator().validate(instance)


def test_plan_section_5_example_is_valid():
    _validator().validate(_plan_example())


def test_acceptance_allows_additional_properties():
    instance = _plan_example()
    instance["case"][0]["acceptance"]["later_case_metric"] = 0.5
    _validator().validate(instance)


def test_rejects_max_nodes_other_than_two():
    instance = _current_manifest()
    instance["max_nodes"] = 1
    with pytest.raises(ValidationError):
        _validator().validate(instance)


@pytest.mark.parametrize("key", _CAPABILITY_KEYS)
def test_rejects_missing_current_capabilities_key(key):
    instance = _current_manifest()
    del instance["current_capabilities"][key]
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_illegal_verification_kind():
    instance = _plan_example()
    instance["case"][0]["verification_kind"] = "benchmark"
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_illegal_evidence_status():
    instance = _plan_example()
    instance["case"][0]["evidence_status"] = "optional"
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_pr_nodes_greater_than_two():
    instance = _plan_example()
    instance["case"][0]["resources"]["pr"]["nodes"] = 3
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_two_node_nodes_item_greater_than_two():
    instance = _plan_example()
    instance["case"][0]["resources"]["two_node"]["nodes"] = [1, 3]
    with pytest.raises(ValidationError):
        _validator().validate(instance)


@pytest.mark.parametrize("field", ("execution_spaces", "mpi_modes", "suites"))
def test_rejects_unknown_catalog_token(field):
    instance = _plan_example()
    instance["case"][0][field] = ["typo"]
    with pytest.raises(ValidationError):
        _validator().validate(instance)


@pytest.mark.parametrize("field", ("execution_spaces", "mpi_modes", "suites"))
def test_rejects_empty_catalog_array(field):
    instance = _plan_example()
    instance["case"][0][field] = []
    with pytest.raises(ValidationError):
        _validator().validate(instance)
