"""A real qualification receipt: M01 compiled, loaded and advanced by PoPS (ROM-17, K6).

The receipt observes the public lifecycle (validate, resolve, compile, bind, run) in this process;
``scripts/qualification_receipt.py`` judges it from files and process state.  The receipt is then
admitted against the cohort of the exact installation under test: this clean checkout and the
retained wheel the installation is byte-identical to.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse

import pytest

import pops

from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import native_profile, profile_gather_value
from tests.python.support.requirements import require_native_or_skip

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import qualification_receipt as receipts  # noqa: E402
import sdk_cohort as cohort  # noqa: E402

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _receipt_directory(tmp_path: Path) -> Path:
    # CI uploads its pytest receipt directory; a local run keeps the receipt beside the test.
    ci = os.environ.get("POPS_CI_PYTEST_TIMINGS_DIR")
    return (Path(ci) if ci else tmp_path) / "qualification-receipts"


def _retained_wheel(sdk: dict) -> Path | None:
    wheel = sdk["installation"]["wheel"]
    if wheel is None or urlparse(wheel["url"]).scheme != "file":
        return None
    path = Path(unquote(urlparse(wheel["url"]).path))
    return path if path.is_file() else None


def _not_official(data: dict) -> list[str]:
    """Why this run cannot qualify a cohort (empty in the installed CI lane)."""
    reasons = list(data["overlay"]["findings"])
    if _retained_wheel(data["sdk"]) is None:
        reasons.append("the installation names no retained wheel file")
    return reasons


def _installation_cohort(data: dict) -> dict:
    """The cohort of the checkout and installation under test, built in memory.

    CI installs one retained wheel per dimension into a fresh prefix; the wheel's own provenance is
    the CI run that built it from this checkout.  The installation is re-proved byte-identical to
    that wheel here, so the cohort binds exactly the bytes this receipt ran on.
    """
    spec = importlib.util.spec_from_file_location(
        "_pops_receipt_wheel_proof", SCRIPTS / "prove_installed_wheel.py")
    prover = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prover)
    source = cohort.source_identity(ROOT)
    sdk = cohort.sdk_identity(Path(data["sdk"]["package_dir"]))
    proof = prover.installed_wheel_proof(
        _retained_wheel(sdk), expected_dimensions=[row["dimension"] for row in sdk["native"]])
    provenance = {
        "kind": "retained-wheel-proof",
        "attested": True,
        "proof": proof,
        "leaves": [{"dimension": item["dimension"], "sha256": item["sha256"],
                    "attestation": None, "attestation_sha256": None,
                    "built_from_commit": None, "problem": None}
                   for item in proof["native_variants"]],
        "limit": "the retained wheel was built from this checkout by the CI run that installed "
                 "it; no installation-local build attestation exists",
    }
    return cohort.cohort_record(source, sdk, cohort.coherence(ROOT, Path(sdk["package_dir"])),
                                provenance)


def test_m01_receipt_distinguishes_compiled_loaded_and_advanced_and_qualifies_its_cohort(
    tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property
):
    del isolated_native_cache, native_cxx, kokkos_root
    _native, world = native_profile(dimension=2, scope="qualification receipt")
    import examples.api_v02.common as common
    import examples.api_v02.models_M01_M14 as models

    gather = None if world is None else (lambda local: profile_gather_value(world, local))
    receipt = receipts.Receipt(
        "M01 compiled, loaded and advanced in System", source_root=ROOT,
        directory=_receipt_directory(tmp_path), gather=gather,
        domain="M01: 2D periodic scalar advection, 16x8 cells, Dim=2")
    with receipt:
        receipt.equations(models, common)
        spec = models.build_m01()
        plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
        receipt.plan(plan)
        artifact = pops.compile(plan)
        receipt.compiled(artifact)
        receipt.parameters(initial_values=dict(spec.initial_values), t_end=spec.t_end,
                           dt=spec.dt, max_steps=32)
        runtime = pops.bind(artifact, initial_values={
            subject: value.copy() for subject, value in spec.initial_values.items()},
            resources={"execution_context": artifact_execution_context(artifact)})
        receipt.loaded(runtime)
        report = pops.run(runtime, t_end=spec.t_end, max_steps=32, console=False)
        receipt.advanced(report, runtime)
        metrics = spec.invariants(runtime)
        receipt.property(
            "M01 mass, positivity and short-time accuracy", value=metrics,
            bound={"mass_defect": 2.e-10, "minimum": .799, "reference_l1": .012},
            passed=bool(metrics["passed"]), oracle="exact translated cosine, NumPy",
            domain=spec.profile)
    data = receipts.read_receipt(receipt.path)
    record_property("receipt", str(receipt.path))

    # Each lifecycle step is its own recorded stage, observed rather than assumed.
    assert not data["failures"], data["failures"]
    assert [row["stage"] for row in data["stages"]] == list(receipts.STAGES)
    stages = {row["stage"]: row["evidence"] for row in data["stages"]}
    assert stages["C"]["same_plan_as_G"]
    assert sorted(dso["role"] for dso in stages["C"]["dsos"]) == ["model", "program"]
    model = next(dso for dso in stages["C"]["dsos"] if dso["role"] == "model")
    assert model["generated_cpp"]["sha256"] == model["generated_cpp"]["recorded_source_sha256"]
    assert model["generated_cpp"]["recorded_binary_sha256"] == model["sha256"]
    assert stages["loaded"]["system"] == "System"
    assert stages["loaded"]["all_mapped"], stages["loaded"]["artifact_dsos"]
    assert stages["loaded"]["installed_program_matches"]
    assert stages["advanced"]["same_artifact_as_C"]
    assert stages["advanced"]["accepted_steps"] == 2
    assert stages["advanced"]["runtime_time"] == pytest.approx(spec.t_end, abs=2.e-13)
    assert stages["S"]["passed"], metrics
    assert data["backend"]["native_loaded"] and data["backend"]["dimension"] == 2
    assert data["ranks"]["size"] == (1 if world is None else int(world.size))
    assert data["ranks"]["consensus"] is True

    # Admission: only a clean checkout and the exact retained wheel qualify a cohort.
    reasons = _not_official(data)
    if reasons:
        require_native_or_skip("receipt admission needs the installed CI lane: "
                               + "; ".join(reasons), optional_skip=pytest.skip)
    assert data["levels"] == dict.fromkeys(receipts.LADDER, True), data["limits"]
    record = _installation_cohort(data)
    verdict = receipts.admit(data, record, require="R")
    assert verdict["admitted"], verdict["reasons"]
    record_property("cohort_id", record["cohort_id"])
