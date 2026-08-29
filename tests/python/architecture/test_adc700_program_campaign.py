"""Source witnesses for the ADC-700 Python MODULE campaign.

These tests intentionally do not launch MPI, compile CUDA, or claim ROMEO execution.  They protect
the fail-closed campaign contract while hardware qualification remains an authenticated batch-job
artifact.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[3]
ADC = ROOT / "benchmarks" / "adc700"
DRIVER = ADC / "program_cutover.py"
VERIFIER = ADC / "verify.py"
ORACLE = ADC / "program_cutover.cpp"
CMAKE = ADC / "CMakeLists.txt"
SBATCH = ROOT / "benchmarks" / "romeo" / "adc700_program_cutover.sbatch"
MANIFEST = ROOT / "benchmarks" / "manifest.toml"
BASELINE = "db3d390f43dfb14f12e88db31a9b3e631ff50488"


def test_adc700_candidate_is_python_module_lifecycle_and_oracle_has_no_lambda() -> None:
    driver = DRIVER.read_text(encoding="utf-8")
    oracle = ORACLE.read_text(encoding="utf-8")
    assert "pops.Program" in driver
    assert "pops.validate" in driver
    assert "pops.resolve" in driver
    assert "pops.compile" in driver
    assert "compile(MODULE)" in driver
    assert "AmrSystem" in driver and "install_program" in driver
    assert "Profile.Advanced()" in driver
    assert "_executor" in driver and "profile_context" in driver
    assert "scratch_allocs" in driver
    assert "operations * levels" in driver
    assert "never O(cells)" in driver
    assert "Uniform" in driver and "AMRRegrid.frozen" in driver
    assert "_euler_case" in driver and "adaptive=True" in driver
    assert "bind_install_route" in driver and "AmrSystem.install_program" in driver
    assert 'memory_view.get("entries")' in VERIFIER.read_text(encoding="utf-8")
    assert "_require_baseline_toolchain_attestation" in driver
    assert "_merge_cmake_target_contract" in driver and "toolchain_probe" in driver
    assert "toolchain_build_attested" in driver and "toolchain_build_attested" in ORACLE.read_text(encoding="utf-8")
    assert "native_std != 20" in driver and 'artifact_std != "c++20"' in driver
    assert "POPS_ADC700_TOOLCHAIN_RECEIPT" in oracle
    assert "POPS_ADC700_TOOLCHAIN_ATTESTED" in oracle
    assert "CMake preflight binding" in oracle
    assert '\\"toolchain\\":%s' in oracle
    assert '\\"distribute_coarse\\":true' in oracle
    assert '\\"coarse_max_grid\\":%d' in oracle
    assert "context->install" not in oracle
    assert "install_forward_euler_program" not in oracle
    assert "program_only" not in oracle


def test_adc700_box_tokens_match_cpp_for_nontrivial_fine_boxes() -> None:
    """The Python AMR tuple seam must use the C++ oracle's exact box spelling."""
    import ast

    tree = ast.parse(DRIVER.read_text(encoding="utf-8"), filename=str(DRIVER))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_box_token")
    namespace: dict[str, object] = {"Any": object, "CampaignError": ValueError}
    code = compile(
        ast.fix_missing_locations(ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
            type_ignores=[],
        )),
        str(DRIVER),
        "exec",
    )
    exec(code, namespace)
    box_token = namespace["_box_token"]
    assert callable(box_token)
    assert box_token((1, (9, 8), (16, 15))) == "1:9,8,16,15"
    assert box_token((0, (-4, 3), (7, 19))) == "0:-4,3,7,19"


def test_adc700_cmake_only_builds_native_oracle() -> None:
    cmake = CMAKE.read_text(encoding="utf-8")
    assert "POPS_ADC700_ROUTE STREQUAL \"pre_cutover_native\"" in cmake
    assert "the candidate program_only route is benchmarks/adc700/program_cutover.py" in cmake
    assert "POPS_ADC700_PRE_CUTOVER=1" in cmake
    assert "POPS_ADC700_PRE_CUTOVER=0" not in cmake
    assert "_adc700_verify_shared_toolchain" in cmake
    assert "CMAKE_CXX_STANDARD 20" in cmake
    assert "MPI_CXX_COMPILE_OPTIONS" in cmake and "Kokkos::kokkos" in cmake
    for witness in (
        "INTERFACE_COMPILE_OPTIONS", "INTERFACE_COMPILE_DEFINITIONS", "INTERFACE_LINK_OPTIONS",
        "INTERFACE_LINK_LIBRARIES", "_adc700_kokkos_abi", "_adc700_mpi_abi",
        "_adc700_command_provenance", "_effective_compile_flags", "_effective_link_flags",
        "_adc700_parse_file_rows", "_adc700_require_hex", "_adc700_require_integer", "string(LENGTH",
        "find_package(MPI REQUIRED COMPONENTS CXX)", "missing/extra fields", "CMAKE_CXX_STANDARD",
    ):
        assert witness in cmake
    assert "IN_LIST _required_flag" not in cmake
    assert "POPS_ADC700_TOOLCHAIN_RECEIPT_SHA256" in cmake
    assert "POPS_ADC700_TOOLCHAIN_ATTESTED=1" in cmake
    assert 'NOT _requested_std STREQUAL "c++20"' in cmake
    assert "POPS_ADC700_TOOLCHAIN_REVISION" in ORACLE.read_text(encoding="utf-8")


def test_adc700_cmake_refuses_non_cxx20_configuration(tmp_path: Path) -> None:
    """A caller-provided C++23 cache value must fail before any receipt is consumed."""
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake is unavailable for the source-level negative")
    process = subprocess.run(
        [
            cmake,
            "-S",
            str(ADC),
            "-B",
            str(tmp_path / "cxx23"),
            "-DPOPS_ADC700_TOOLCHAIN_PROBE_ONLY=ON",
            "-DCMAKE_CXX_STANDARD=23",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert process.returncode != 0
    assert "exact C++20" in (process.stdout + process.stderr)


def test_adc700_verifier_rejects_missing_candidate_provenance() -> None:
    import importlib.util

    loader = importlib.util.spec_from_file_location("adc700_verify_source_test", VERIFIER)
    assert loader is not None and loader.loader is not None
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)

    def row(route: str, revision: str) -> dict:
        return {
            "schema": module.SCHEMA,
            "route": route,
            "revision": revision,
            "execution_space": "Cuda",
            "mpi_ranks": 4,
            "execution_concurrency": 1,
            "real_bytes": 8,
            "parameters": {"n": 32, "warmups": 1, "measured_steps": 2, "dt": 1e-4},
            "topology": {"levels": 1, "patches": 0, "boxes": ""},
            "timing": {"per_step_seconds": 1.0},
            "signature": {
                "mass": 1.0,
                "checksum": 2.0,
                "checksum_square": 3.0,
                "maximum": 1.0,
            },
            "validation": {"passed": True},
        }

    report = {
        "schema": module.REPORT_SCHEMA,
        "schema_version": 1,
        "status": "passed",
        "provenance": {
            "baseline_revision": BASELINE,
            "candidate_revision": "candidate",
        },
        "device": {
            "mpi_ranks": 4,
            "one_distinct_device_per_rank": True,
            "execution_space": "Cuda",
            "assignments": [{"rank": i, "uuid": "GPU-%d" % i} for i in range(4)],
        },
        "performance": {"passed": True, "median": 1.0},
        "numerical_parity": {"passed": True},
    }
    rows = [row("pre_cutover_native", BASELINE), row("program_only", "candidate")]
    rows.extend([row("program_only", "candidate"), row("pre_cutover_native", BASELINE)])
    with pytest.raises(module.EvidenceError, match="installed wheel proof"):
        module.verify(
            report,
            rows,
            [{"rank": i, "uuid": "GPU-%d" % i} for i in range(4)],
            candidate_revision="candidate",
        )


def test_adc700_verifier_rejects_non_archived_candidate_source(tmp_path: Path) -> None:
    import importlib.util

    loader = importlib.util.spec_from_file_location("adc700_verify_archive_test", VERIFIER)
    assert loader is not None and loader.loader is not None
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)

    with pytest.raises(module.EvidenceError, match="immutable candidate archive"):
        module._candidate_source_root(tmp_path / "checkout")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    with pytest.raises(module.EvidenceError, match="immutable"):
        module._candidate_source_root(candidate)


def test_adc700_romeo_job_is_four_gpu_abba_and_runs_python_candidate() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --ntasks=4" in text
    assert "#SBATCH --gpus-per-node=4" in text
    assert "#SBATCH --gpus-per-task=1" in text
    assert BASELINE in text
    assert "build_python.sh" in text and "prove_installed_wheel.py" in text
    assert "program_cutover.py" in text and "--route=program_only" in text
    assert "compare.py" in text and "verify.py" in text
    assert "run_one baseline baseline" in text
    assert "run_one candidate candidate" in text
    assert "ABBA_BLOCKS" in text and "-ge 5" in text
    assert "cat-file -e" in text and "cat-file -t" in text and "git archive" in text
    assert "ls-tree" in text and "100755" in text
    assert "status --porcelain --untracked-files=all" in text
    assert "chmod -R a-w" in text
    assert "archive_receipt.py" in text and "candidate-archive-receipt.json" in text
    assert "ARCHIVE_HELPER" in text and "--helper" in text and "outside both extracted trees" in text
    assert 'python3 "${ARCHIVE_HELPER}"' in text
    assert 'python3 "${CAMPAIGN_ROOT}/benchmarks/adc700/archive_receipt.py"' not in text
    assert "toolchain-receipt.json" in text and "toolchain-only" in text
    assert "build-toolchain-probe" in text and "toolchain-probe" in text
    assert "POPS_ADC700_TOOLCHAIN_RECEIPT_SHA256" in text
    assert "GPU_WRAPPER" in text
    assert "POPS_ADC700_GPU_UUID" in (ADC / "with_gpu_identity.sh").read_text(encoding="utf-8")
    for helper in (
        "benchmarks/adc700/with_gpu_identity.sh",
        "benchmarks/adc700/archive_receipt.py",
        "scripts/build_python.sh",
        "scripts/prove_installed_wheel.py",
        "scripts/preserve_native_variants.py",
        "scripts/codesign_pops_extensions.py",
        "scripts/verify_installed_native.py",
    ):
        assert helper in text


def test_adc700_verifier_reauthenticates_archive_toolchain_wheel_and_raw_rows() -> None:
    driver = DRIVER.read_text(encoding="utf-8")
    verifier = VERIFIER.read_text(encoding="utf-8")
    for witness in (
        "_candidate_source_root",
        "_archive_contract",
        "_toolchain_receipt_contract",
        "candidate_archive_tree_sha256",
        "_command_version",
        "compiler_sha256",
        "abi_sha256",
        "proof_script_sha256",
        "_harness_contract",
        "_raw_gpu_contract",
        "_assert_report_aggregates",
        "candidate extension provenance differs between ABBA runs",
        "baseline C++ toolchain receipt differs from candidate provenance",
        "report provenance toolchain/topology differs from raw JSONL",
        "external immutable archive receipt helper",
    ):
        assert witness in verifier
    for harness in (
        "benchmarks/adc700/with_gpu_identity.sh",
        "scripts/preserve_native_variants.py",
        "scripts/write_native_variant_manifest.py",
        "scripts/codesign_pops_extensions.py",
        "scripts/verify_installed_native.py",
    ):
        assert harness in driver and harness in verifier


def test_adc700_archive_receipt_rejects_writable_or_unlinked_tree(tmp_path: Path) -> None:
    import importlib.util

    archive = ADC / "archive_receipt.py"
    loader = importlib.util.spec_from_file_location("adc700_archive_receipt_test", archive)
    assert loader is not None and loader.loader is not None
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    with pytest.raises(module.ArchiveReceiptError, match="writable"):
        module.build(candidate, role="candidate", revision=BASELINE, output=tmp_path / "receipt.json")


def test_adc700_archive_receipt_accepts_pinned_baseline_without_in_tree_helper(
    tmp_path: Path,
) -> None:
    """The external helper must verify the pinned baseline, which predates ADC-700."""
    import importlib.util

    archive = ADC / "archive_receipt.py"
    loader = importlib.util.spec_from_file_location("adc700_baseline_receipt_test", archive)
    assert loader is not None and loader.loader is not None
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)

    baseline = tmp_path / "baseline"
    baseline.mkdir()
    archive_bytes = subprocess.run(
        ["git", "-C", str(ROOT), "archive", BASELINE],
        check=True,
        capture_output=True,
    ).stdout
    subprocess.run(
        ["tar", "-xf", "-", "-C", str(baseline)],
        input=archive_bytes,
        check=True,
    )
    subprocess.run(["chmod", "-R", "a-w", str(baseline)], check=True)
    assert not (baseline / "benchmarks" / "adc700" / "archive_receipt.py").exists()

    helper = tmp_path / "archive-receipt.py"
    shutil.copy2(archive, helper)
    helper.chmod(0o444)
    receipt = tmp_path / "baseline-receipt.json"
    payload = module.build(
        baseline,
        role="baseline",
        revision=BASELINE,
        output=receipt,
        helper=helper,
    )
    assert payload["receipt_script"] == {
        "path": str(helper.resolve()),
        "sha256": module._sha256(helper),
    }
    verified = module.verify(
        baseline,
        role="baseline",
        revision=BASELINE,
        receipt=receipt,
        helper=helper,
    )
    assert verified["tree_sha256"] == payload["tree_sha256"]


def test_adc700_manifest_declares_authenticated_probe_contract() -> None:
    with MANIFEST.open("rb") as stream:
        manifest = tomllib.load(stream)
    campaign = manifest["campaigns"]["adc700_program_cutover"]
    assert campaign["baseline_revision"] == BASELINE
    assert campaign["minimum_ratio"] == 0.98
    assert campaign["minimum_abba_blocks"] == 5
    assert campaign["minimum_mpi_ranks"] == 4
    assert campaign["minimum_gpu_count"] == 4
    assert campaign["requires_mpi_communicator"] == "MPI_COMM_WORLD"
    assert campaign["requires_per_run_gpu_uuid"] is True
    assert campaign["requires_immutable_archive"] is True
    assert campaign["requires_archived_tree_receipt"] is True
    assert campaign["requires_archive_receipt_before_each_run"] is True
    assert campaign["requires_exact_baseline_candidate_toolchain"] is True
    assert campaign["requires_cmake_target_toolchain_probe"] is True
    assert campaign["requires_baseline_toolchain_build_attestation"] is True
    assert campaign["baseline_cxx_standard"] == 20
    assert campaign["patch_layout_distribute_coarse"] is True
    assert campaign["patch_layout_coarse_max_grid"] == "n/2"
    assert campaign["driver"].endswith("benchmarks/adc700/program_cutover.py")
    assert campaign["verifier"].endswith("benchmarks/adc700/verify.py")
    assert campaign["requires_zero_allocation_after_prepare"] is True
    assert campaign["requires_installed_wheel_proof"] is True
    assert campaign["wheel_proof_schema_version"] == 3
    assert campaign["dispatch_complexity"] == "O(operations*levels), never O(cells)"
    assert campaign["probes"] == ["uniform", "amr_refined_planned"]
    assert campaign["probe_resolutions"] == [16, 32]


def test_adc700_verifier_rejects_forged_aggregates_and_preparation_gaps() -> None:
    import importlib.util

    loader = importlib.util.spec_from_file_location("adc700_verify_contract_test", VERIFIER)
    assert loader is not None and loader.loader is not None
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)

    def measurement(route: str, seconds: float) -> dict:
        return {
            "route": route,
            "timing": {"per_step_seconds": seconds},
            "signature": {
                "mass": 1.0,
                "checksum": 2.0,
                "checksum_square": 3.0,
                "maximum": 4.0,
            },
        }

    rows = []
    for _ in range(module.MINIMUM_ABBA_BLOCKS):
        rows.extend([
            measurement(module.BASELINE_ROUTE, 1.0),
            measurement(module.CANDIDATE_ROUTE, 1.01),
            measurement(module.CANDIDATE_ROUTE, 1.01),
            measurement(module.BASELINE_ROUTE, 1.0),
        ])
    performance = module._derived_performance(rows)
    numerical = module._derived_numerical(rows)
    report = {"performance": dict(performance), "numerical_parity": dict(numerical)}
    module._assert_report_aggregates(report, performance=performance, numerical=numerical)

    forged = {"performance": {**performance, "median": performance["median"] + 1.0},
              "numerical_parity": dict(numerical)}
    with pytest.raises(module.EvidenceError, match="performance.median"):
        module._assert_report_aggregates(forged, performance=performance, numerical=numerical)

    probe = {
        "status": "passed",
        "preparation": {
            "bind_complete": True,
            "bind_install_route": "AmrSystem.install_program",
            "scope": "bind+warmups",
            "warmups": 1,
            "profile": {
                "profile": "advanced", "source": "snapshot", "schema_version": 1,
                "counters": {"kernels": 1},
            },
            "counters_before_reset": {"kernels": 1},
            "reset_after_preparation": True,
        },
        "allocations_after_prepare": 0,
        "dispatches": 1,
        "operations": 1,
        "levels": 2,
        "measured_steps": 1,
        "dispatch_bound": 8,
        "allocation_free": True,
        "dispatch_complexity": module.DISPATCH_COMPLEXITY,
        "cell_count": 16,
        "fixed_operations": True,
        "fixed_levels": True,
        "cell_independent_dispatches": True,
        "profile": {
            "profile": "advanced",
            "source": "snapshot",
            "schema_version": 1,
            "counters": {"kernels": 1},
            "views": {"by_memory": {"available": True, "entries": {"scratch_allocs": 0}}},
        },
    }
    module._probe_one(probe, label="amr_refined_planned", where="test probe")
    with pytest.raises(module.EvidenceError, match="allocation freedom"):
        module._probe_one(
            {**probe, "allocations_after_prepare": 1, "allocation_free": False},
            label="amr_refined_planned", where="test probe",
        )
    with pytest.raises(module.EvidenceError, match="bind/warmup"):
        module._probe_one(
            {**probe, "preparation": {**probe["preparation"], "reset_after_preparation": False}},
            label="amr_refined_planned", where="test probe",
        )
    with pytest.raises(module.EvidenceError, match="Advanced snapshot"):
        module._probe_one(
            {**probe, "profile": {**probe["profile"], "source": "legacy_text"}},
            label="amr_refined_planned", where="test probe",
        )
