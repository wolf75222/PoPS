"""Source-only profile admission and mandatory rank receipt checks; no native execution."""
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from pops._generated_release_contract import NATIVE_ABI_VERSION
from tests.python.support.initial_field_ghost_profile import native_profile
from tests.python.integration.mpi.test_initial_field_ghost_mpi import require_rank_results


@pytest.fixture
def profile_policy(monkeypatch, tmp_path):
    import pops
    import pops._native_selector as selector
    import pops._native_collectives as collectives
    import pops.codegen.abi as abi
    import pops.codegen.toolchain as toolchain
    import scripts.verify_installed_native as verifier

    package = tmp_path / "pops"
    origin = package / "_native" / "dim2" / "_pops.so"
    world = SimpleNamespace(rank=0, size=1)
    calls = []
    native = SimpleNamespace(__abi_version__=NATIVE_ABI_VERSION, __has_mpi__=False,
                             mpi_world=lambda: calls.append("MPI") or world)
    monkeypatch.delenv("POPS_TEST_INITIAL_GHOST_EXPECT_RANKS", raising=False)
    monkeypatch.setattr(pops, "__file__", str(package / "__init__.py"))
    monkeypatch.setattr(selector, "select_native_dimension", lambda dimension: native)
    monkeypatch.setattr(verifier, "verify_installed_native", lambda **expect: origin)
    monkeypatch.setattr(abi, "module_header_signature", lambda: "exact-sdk")
    monkeypatch.setattr(toolchain, "pops_include", lambda: package / "include")
    monkeypatch.setattr(toolchain, "pops_header_signature", lambda path: "exact-sdk")
    monkeypatch.setattr(collectives, "require_world", lambda resource: resource)
    return native, world, calls


def test_serial_profile_never_materializes_an_mpi_communicator(profile_policy):
    native, _world, calls = profile_policy
    assert native_profile() == (native, None)
    assert calls == []


@pytest.mark.parametrize("version", (NATIVE_ABI_VERSION - 1, NATIVE_ABI_VERSION + 1,
                                     str(NATIVE_ABI_VERSION), True))
def test_profile_rejects_every_nonexact_release_abi(profile_policy, version):
    native, _world, _calls = profile_policy
    native.__abi_version__ = version
    with pytest.raises(AssertionError, match="exact declared native ABI"):
        native_profile()


def test_profile_refuses_a_foreign_sdk_even_when_native_version_matches(profile_policy, monkeypatch):
    import pops.codegen.toolchain as toolchain

    monkeypatch.setattr(toolchain, "pops_header_signature", lambda path: "foreign-sdk")
    with pytest.raises(AssertionError, match="SDK header identities differ"):
        native_profile()


def test_expected_singleton_mpi_is_never_satisfied_by_serial(profile_policy, monkeypatch):
    monkeypatch.setenv("POPS_TEST_INITIAL_GHOST_EXPECT_RANKS", "1")
    with pytest.raises(AssertionError, match="requested world"):
        native_profile()


@pytest.mark.parametrize("ranks", (1, 2))
def test_mpi_profile_requires_the_requested_real_world_shape(profile_policy, monkeypatch, ranks):
    native, world, calls = profile_policy
    native.__has_mpi__ = True
    world.size = ranks
    monkeypatch.setenv("POPS_TEST_INITIAL_GHOST_EXPECT_RANKS", str(ranks))
    assert native_profile() == (native, world)
    assert calls == ["MPI"]
    world.size = 3
    with pytest.raises(AssertionError, match="requested world"):
        native_profile()


def test_rank_receipts_refuse_missing_rank_and_silent_skip(tmp_path):
    names = ("test_public_initial_field_fresh_before_ghost_and_positive_point",
             "test_public_initial_ghost_rank_fault_keeps_prepublication_owner")
    for rank in (0, 1):
        root = ET.Element("testsuite")
        for name in names:
            ET.SubElement(root, "testcase", name=name)
        ET.ElementTree(root).write(tmp_path / ("rank%d.xml" % rank))
    require_rank_results(tmp_path, 2)
    (tmp_path / "rank1.xml").unlink()
    with pytest.raises(AssertionError, match="omitted its pytest receipt"):
        require_rank_results(tmp_path, 2)
    root = ET.Element("testsuite")
    for name in names:
        ET.SubElement(ET.SubElement(root, "testcase", name=name), "skipped")
    ET.ElementTree(root).write(tmp_path / "rank1.xml")
    with pytest.raises(AssertionError, match="mandatory outcomes"):
        require_rank_results(tmp_path, 2)


def test_independent_amr_mpi_drivers_each_preserve_all_cases_and_cleanup_budget(tmp_path, monkeypatch):
    from tests.python.integration.mpi import (
        test_public_amr_profile_mpi1 as mpi1,
        test_public_amr_profile_mpi2 as mpi2,
    )
    from tests.python.support import mpi_fixture_driver, public_amr_profile_mpi as profile

    calls = []
    monkeypatch.setattr(profile, "run_fixture_world", lambda **arguments: calls.append(arguments))
    mpi1.test_public_amr_fixtures_on_authenticated_mpi1_world(tmp_path)
    mpi2.test_public_amr_fixtures_on_authenticated_mpi2_world(tmp_path)
    assert [call["ranks"] for call in calls] == [1, 2]
    assert sum(profile.EXPECTED_CASES.values()) == 18
    for call in calls:
        assert call["tests"] is profile.TESTS
        assert call["expected_cases"] is profile.EXPECTED_CASES
        assert call["timeout_seconds"] == 840
        assert mpi_fixture_driver.PROCESS_TIMEOUT_SECONDS - call["timeout_seconds"] == 60
    assert sum(sum(call["expected_cases"].values()) for call in calls) == 36
    assert sum(call["ranks"] * sum(call["expected_cases"].values()) for call in calls) == 54


def test_uniform_mpi_drivers_keep_full_native_groups_separate_from_source_mms(tmp_path, monkeypatch):
    from tests.python.integration.mpi import (
        test_public_evolved_original_stage_mpi1 as original1,
        test_public_evolved_original_stage_mpi2 as original2,
        test_public_captured_diffusion_mpi1 as captured1,
        test_public_captured_diffusion_mpi2 as captured2,
    )
    from tests.python.support import public_uniform_profile_mpi as profile

    calls = []
    monkeypatch.setattr(profile, "run_fixture_world", lambda **arguments: calls.append(arguments))
    original1.test_public_evolved_original_stage_on_authenticated_mpi1_world(tmp_path)
    original2.test_public_evolved_original_stage_on_authenticated_mpi2_world(tmp_path)
    captured1.test_public_captured_and_candidate_diffusion_on_authenticated_mpi1_world(tmp_path)
    captured2.test_public_captured_and_candidate_diffusion_on_authenticated_mpi2_world(tmp_path)
    assert [call["ranks"] for call in calls] == [1, 2, 1, 2]
    assert [sum(call["expected_cases"].values()) for call in calls] == [8, 8, 4, 4]
    for call in calls:
        assert {selector.split("::", 1)[1] for selector in call["tests"]} == set(call["expected_cases"])
        assert call["timeout_seconds"] == 840
    assert sum(sum(call["expected_cases"].values()) for call in calls) == 24
    assert sum(call["ranks"] * sum(call["expected_cases"].values()) for call in calls) == 36


def test_field_profile_drivers_keep_every_declared_native_group_and_world(tmp_path, monkeypatch):
    import importlib
    from tests.python.support import public_field_profile_mpi as profile

    stems = {
        "completed_original": "test_public_completed_original_interaction",
        "amr_original": "test_public_amr_original_field",
        "imex_time": "test_public_imex_nonautonomous_field",
        "nonlinear_mixed": "test_public_nonlinear_mixed_field",
        "spatial_interaction": "test_public_spatial_interaction",
    }
    expected_counts = {"completed_original": 2, "amr_original": 4,
                       "imex_time": 1, "nonlinear_mixed": 5, "spatial_interaction": 8}
    assert profile.GROUPS["spatial_interaction"][1] == {
        "test_public_spatial_interaction_saved_history_and_exact_replay": 3,
        "test_public_spatial_interaction_refuses_without_publication": 3,
        "test_public_spatial_interaction_ignores_inactive_source_pole": 1,
        "test_public_spatial_interaction_nonperiodic_amr_refuses_at_bind": 1,
    }
    calls = []
    monkeypatch.setattr(profile, "run_fixture_world", lambda **arguments: calls.append(arguments))
    for group, stem in stems.items():
        start = len(calls)
        for ranks in (1, 2):
            module = importlib.import_module("tests.python.integration.mpi.%s_mpi%d" % (stem, ranks))
            getattr(module, "test_public_%s_on_authenticated_mpi%d_world" % (group, ranks))(tmp_path)
        pair = calls[start:]
        assert [call["ranks"] for call in pair] == [1, 2]
        for call in pair:
            assert sum(call["expected_cases"].values()) == expected_counts[group]
            assert call["tests"] is profile.GROUPS[group][0]
            assert call["expected_cases"] is profile.GROUPS[group][1]
            assert {selector.split("::", 1)[1] for selector in call["tests"]} == set(call["expected_cases"])
            assert call["expected_rank_env"] == "POPS_TEST_PUBLIC_NATIVE_EXPECT_RANKS"
            assert call["timeout_seconds"] == 840
    assert sum(sum(call["expected_cases"].values()) for call in calls) == 40
    assert sum(call["ranks"] * sum(call["expected_cases"].values()) for call in calls) == 60


def test_checkpoint_profile_drivers_keep_all_seventeen_cases_in_each_world(tmp_path, monkeypatch):
    import importlib
    from tests.python.support import public_checkpoint_profile_mpi as profile

    expected_counts = {"accepted_halo": 2, "scalar_bound_capture": 4,
                       "imex_checkpoint": 2, "amr12_carrier": 1, "typed_newton": 8}
    assert set(profile.GROUPS) == set(expected_counts)
    calls = []
    monkeypatch.setattr(profile, "run_fixture_world", lambda **arguments: calls.append(arguments))
    for group, count in expected_counts.items():
        for ranks in (1, 2):
            module = importlib.import_module(
                "tests.python.integration.mpi.test_public_%s_mpi%d" % (group, ranks))
            getattr(module, "test_public_%s_on_authenticated_mpi%d_world" % (group, ranks))(tmp_path)
            call = calls[-1]
            assert call["ranks"] == ranks
            assert call["tests"] is profile.GROUPS[group][0]
            assert call["expected_cases"] is profile.GROUPS[group][1]
            assert sum(call["expected_cases"].values()) == count
            assert {selector.split("::", 1)[1] for selector in call["tests"]} == set(call["expected_cases"])
            assert call["expected_rank_env"] == "POPS_TEST_PUBLIC_NATIVE_EXPECT_RANKS"
            assert call["timeout_seconds"] == 840
    assert sum(sum(call["expected_cases"].values()) for call in calls) == 34
    assert sum(call["ranks"] * sum(call["expected_cases"].values()) for call in calls) == 51


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("family", ("accepted_halo", "scalar_bound_capture", "imex_checkpoint",
                                  "amr12_carrier", "typed_newton"))
def test_checkpoint_receipts_refuse_missing_parameter_skip_and_duplicate(tmp_path, ranks, family):
    from tests.python.support.mpi_fixture_driver import require_rank_results
    from tests.python.support.public_checkpoint_profile_mpi import GROUPS

    expected = GROUPS[family][1]
    for rank in range(ranks):
        root = ET.Element("testsuite")
        for name, count in expected.items():
            for parameter in range(count):
                ET.SubElement(root, "testcase", name="%s[%d]" % (name, parameter))
        ET.ElementTree(root).write(tmp_path / ("rank%d.xml" % rank))
    require_rank_results(tmp_path, ranks, expected, purpose=family)
    receipt = tmp_path / "rank0.xml"
    root = ET.parse(receipt).getroot()
    ET.SubElement(root.find("testcase"), "skipped")
    ET.ElementTree(root).write(receipt)
    with pytest.raises(AssertionError, match="mandatory outcomes"):
        require_rank_results(tmp_path, ranks, expected, purpose=family)
    root.find("testcase").remove(root.find("testcase/skipped"))
    root.remove(root.find("testcase"))
    ET.ElementTree(root).write(receipt)
    with pytest.raises(AssertionError, match="case inventory differs"):
        require_rank_results(tmp_path, ranks, expected, purpose=family)


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("family", ("amr", "evolved_original", "captured_diffusion", "spatial_interaction"))
def test_public_receipts_cover_every_parameter_on_every_rank(tmp_path, ranks, family):
    from tests.python.support.mpi_fixture_driver import require_rank_results as require_all
    from tests.python.support.public_amr_profile_mpi import EXPECTED_CASES as AMR_CASES
    from tests.python.support.public_uniform_profile_mpi import (
        EVOLVED_ORIGINAL_CASES, CAPTURED_DIFFUSION_CASES,
    )
    from tests.python.support.public_field_profile_mpi import GROUPS as FIELD_CASES

    expected_cases = {"amr": AMR_CASES, "evolved_original": EVOLVED_ORIGINAL_CASES,
                      "captured_diffusion": CAPTURED_DIFFUSION_CASES,
                      "spatial_interaction": FIELD_CASES["spatial_interaction"][1]}[family]

    for rank in range(ranks):
        root = ET.Element("testsuite")
        for name, count in expected_cases.items():
            for parameter in range(count):
                ET.SubElement(root, "testcase", name="%s[%d]" % (name, parameter),
                              classname="public_amr")
        ET.ElementTree(root).write(tmp_path / ("rank%d.xml" % rank))
    require_all(tmp_path, ranks, expected_cases, purpose="public " + family + " profile")
    # Same rank count must not authenticate an incomplete parameter inventory.
    receipt = tmp_path / "rank0.xml"
    root = ET.parse(receipt).getroot()
    root.remove(root.find("testcase"))
    ET.ElementTree(root).write(receipt)
    with pytest.raises(AssertionError, match="case inventory differs"):
        require_all(tmp_path, ranks, expected_cases, purpose="public " + family + " profile")


@pytest.mark.parametrize("timeout", (0, 901, True, None))
def test_mpi_child_watchdog_cannot_disable_or_exceed_existing_cap(tmp_path, timeout):
    from tests.python.support.mpi_fixture_driver import run_fixture_world

    with pytest.raises(ValueError, match="existing 900-second ceiling"):
        run_fixture_world(ranks=1, directory=tmp_path, tests=(), expected_cases={},
                          expected_rank_env="UNUSED", purpose="budget source check",
                          timeout_seconds=timeout)
