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


def test_initial_failure_component_requires_normative_local_authority_and_written_status():
    """The test DSO must respect the real Ghost ABI and never recover MPI authority."""
    import ast
    from pathlib import Path
    from tests.python.support.initial_ghost_failure_component import package_data

    _manifest, payload = package_data()
    source = payload.decode()
    prepare = source.split("int prepare(", 1)[1].split("void destroy(", 1)[0]
    assert "POPS_EXECUTION_NONCOLLECTIVE_IDENTITY_V1" in prepare
    assert 'std::strcmp(execution.communicator_datatype_identity,"none")' in prepare
    assert 'std::strcmp(identity,"serial")' in prepare
    assert 'std::strcmp(identity,"MPI_COMM_WORLD")' in prepare
    assert 'std::string("pops.amr.package-assembly/")+execution.execution_identity' in prepare
    assert 'const bool package_lane=identity==package_identity;' in prepare
    assert 'std::strcmp(declared,execution.execution_identity)!=0' in prepare
    assert 'exact_world_package_descendant(identity,execution.execution_identity)' in prepare
    assert 'suffix.find(\'/\',split+1)==std::string::npos' in source
    assert 'canonical_epoch(suffix.substr(0,split))&&canonical_epoch(suffix.substr(split+1))' in source
    assert 'std::strncmp(identity' not in prepare
    assert 'std::strcmp(execution.communicator_datatype_identity,"MPI_DOUBLE")' in prepare
    assert "execution.communicator_f_handle||execution.communicator_datatype_f_handle" in prepare
    assert "MPI_Comm" not in source and "#include <mpi.h>" not in source
    # Only an absent status pointer can prevent writing an ABI refusal. All valid
    # caller-supplied status paths must return an explicit successful/refused outcome.
    assert prepare.count("return 51;") == 1
    assert "if(!status)return 51;" in prepare
    assert "return rejected(status,51," in prepare
    assert "*status={sizeof(PopsComponentStatusV1),0,POPS_COMPONENT_CONTINUE_V1,nullptr}" in prepare

    fixture = Path(__file__).resolve().parents[2] / "integration/amr/test_public_initial_field_ghost_failure.py"
    tree = ast.parse(fixture.read_text())
    diagnostic = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "setenv" and len(node.args) == 2 \
                and isinstance(node.args[0], ast.Constant) \
                and node.args[0].value in ("POPS_TEST_INITIAL_GHOST_RANK", "POPS_TEST_INITIAL_GHOST_SIZE"):
            diagnostic[node.args[0].value] = ast.unparse(node.args[1])
    assert diagnostic == {"POPS_TEST_INITIAL_GHOST_RANK": "str(rank)",
                          "POPS_TEST_INITIAL_GHOST_SIZE": "str(size)"}


def test_independent_amr_mpi_drivers_each_preserve_all_cases_and_cleanup_budget(tmp_path, monkeypatch):
    from collections import Counter
    from tests.python.integration.mpi import (
        test_public_amr_profile_mpi1 as mpi1,
        test_public_amr_profile_mpi2 as mpi2,
    )
    from tests.python.support import mpi_fixture_driver, public_amr_profile_mpi as profile

    calls = []
    monkeypatch.setattr(profile, "run_fixture_world", lambda **arguments: calls.append(arguments))
    for module, ranks in ((mpi1, 1), (mpi2, 2)):
        for partition in profile.PARTITIONS:
            getattr(module, "test_public_amr_fixtures_on_authenticated_mpi%d_world" % ranks)(tmp_path, partition)
    assert [call["ranks"] for call in calls] == [1] * 5 + [2] * 5
    assert sum(profile.EXPECTED_CASES.values()) == 18
    for ranks in (1, 2):
        world = [call for call in calls if call["ranks"] == ranks]
        native_nodes = [selector for call in world for selector in call["tests"]]
        assert len(native_nodes) == len(set(native_nodes)) == 5
        assert {selector.split("::", 1)[0] for selector in native_nodes} == set(profile.TESTS)
        union = Counter()
        for partition, call in zip(profile.PARTITIONS, world, strict=True):
            assert call["tests"] is profile.PARTITIONS[partition][0]
            assert call["expected_cases"] is profile.PARTITIONS[partition][1]
            assert {selector.split("::", 1)[1] for selector in call["tests"]} == set(call["expected_cases"])
            assert call["directory"] == tmp_path / (partition + "-mpi%d" % ranks)
            assert call["timeout_seconds"] == 840
            assert mpi_fixture_driver.PROCESS_TIMEOUT_SECONDS - call["timeout_seconds"] == 60
            union.update(call["expected_cases"])
        assert union == profile.EXPECTED_CASES
    assert sum(sum(call["expected_cases"].values()) for call in calls) == 36
    assert sum(call["ranks"] * sum(call["expected_cases"].values()) for call in calls) == 54
    assert len({call["directory"] for call in calls}) == len(calls)


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
    from collections import Counter
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
        for ranks in (1, 2):
            module = importlib.import_module("tests.python.integration.mpi.%s_mpi%d" % (stem, ranks))
            target = getattr(module, "test_public_%s_on_authenticated_mpi%d_world" % (group, ranks))
            start = len(calls)
            if group == "amr_original":
                partitions = profile.AMR_ORIGINAL_PARTITIONS
                for partition in partitions:
                    target(tmp_path, partition)
            else:
                partitions = {None: profile.GROUPS[group]}
                target(tmp_path)
            world = calls[start:]
            assert [call["ranks"] for call in world] == [ranks] * len(partitions)
            union = Counter()
            native_nodes = []
            for partition, call in zip(partitions, world, strict=True):
                assert call["tests"] is partitions[partition][0]
                assert call["expected_cases"] is partitions[partition][1]
                assert {selector.split("::", 1)[1] for selector in call["tests"]} == set(call["expected_cases"])
                assert call["expected_rank_env"] == "POPS_TEST_PUBLIC_NATIVE_EXPECT_RANKS"
                assert call["timeout_seconds"] == 840
                if partition is not None:
                    assert call["directory"] == tmp_path / (partition + "-mpi%d" % ranks)
                union.update(call["expected_cases"])
                native_nodes.extend(call["tests"])
            assert union == profile.GROUPS[group][1]
            assert sum(union.values()) == expected_counts[group]
            assert len(native_nodes) == len(set(native_nodes))
    assert sum(sum(call["expected_cases"].values()) for call in calls) == 40
    assert sum(call["ranks"] * sum(call["expected_cases"].values()) for call in calls) == 60


def test_checkpoint_profile_drivers_keep_all_seventeen_cases_in_each_world(tmp_path, monkeypatch):
    from collections import Counter
    import importlib
    from tests.python.support import public_checkpoint_profile_mpi as profile
    from tests.python.support import public_typed_newton_mpi as newton

    expected_counts = {"accepted_halo": 2, "scalar_bound_capture": 4,
                       "imex_checkpoint": 2, "amr12_carrier": 1, "typed_newton": 8}
    assert set(profile.GROUPS) == set(expected_counts)
    calls = []

    def capture(**arguments):
        calls.append(arguments)
        if arguments["tests"][0].startswith(
                "tests/python/integration/runtime/test_public_newton_typed_original.py::"):
            arguments["directory"].mkdir()
            name = arguments["tests"][0].split("::", 1)[1]
            for rank in range(arguments["ranks"]):
                receipt = ET.Element("testsuite")
                ET.SubElement(receipt, "testcase", name=name,
                              classname="tests.python.integration.runtime.test_public_newton_typed_original")
                ET.ElementTree(receipt).write(arguments["directory"] / ("rank%d.xml" % rank))

    monkeypatch.setattr(profile, "run_fixture_world", capture)
    monkeypatch.setattr(newton, "run_fixture_world", capture)
    for group, count in expected_counts.items():
        for ranks in (1, 2):
            module = importlib.import_module(
                "tests.python.integration.mpi.test_public_%s_mpi%d" % (group, ranks))
            function = getattr(module, "test_public_%s_on_authenticated_mpi%d_world" % (group, ranks))
            partitions = tuple(newton.TYPED_NEWTON_PARTITIONS) if group == "typed_newton" else (None,)
            union, native_nodes = Counter(), []
            for partition in partitions:
                parent = tmp_path
                if partition is None:
                    function(parent)
                else:
                    parent = tmp_path / (group + "-" + partition + "-mpi%d" % ranks)
                    parent.mkdir()
                    function(parent, partition)
                call = calls[-1]
                assert call["ranks"] == ranks
                assert call["directory"] == parent / (group + "-mpi%d" % ranks)
                if partition is None:
                    assert call["tests"] is profile.GROUPS[group][0]
                    assert call["expected_cases"] is profile.GROUPS[group][1]
                    assert sum(call["expected_cases"].values()) == count
                else:
                    assert call["tests"] == (newton.TYPED_NEWTON_PARTITIONS[partition],)
                    assert call["expected_cases"] == {"test_public_typed_original_policy": 1}
                assert {selector.split("::", 1)[1].split("[", 1)[0]
                        for selector in call["tests"]} == set(call["expected_cases"])
                assert call["expected_rank_env"] == "POPS_TEST_PUBLIC_NATIVE_EXPECT_RANKS"
                assert call["timeout_seconds"] == 840
                union.update(call["expected_cases"])
                native_nodes.extend(call["tests"])
            assert union == profile.GROUPS[group][1]
            assert sum(union.values()) == count
            assert len(native_nodes) == len(set(native_nodes))
    assert sum(sum(call["expected_cases"].values()) for call in calls) == 34
    assert sum(call["ranks"] * sum(call["expected_cases"].values()) for call in calls) == 51
    assert len({call["directory"] for call in calls}) == len(calls)


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("group", ("accepted_halo", "scalar_bound_capture", "imex_checkpoint",
                                  "amr12_carrier", "typed_newton"))
def test_checkpoint_runner_keeps_existing_pytest_root_and_refuses_reused_world(
        tmp_path, monkeypatch, ranks, group):
    """Exercise real directory/receipt admission with an explicitly inert MPI process stub."""
    from types import SimpleNamespace
    from pops import _native_selector
    from tests.python.support import mpi_fixture_driver as driver
    from tests.python.support import public_checkpoint_profile_mpi as profile

    native = SimpleNamespace(__has_mpi__=True, __mpi_contract__="source-only launcher token")
    monkeypatch.setattr(_native_selector, "select_native_dimension", lambda dimension: native)
    monkeypatch.setattr(driver, "mpi_launcher_or_skip", lambda *args, **kwargs: ["source-only-mpiexec"])
    calls = []

    def inert_process(command, **kwargs):
        calls.append((command, kwargs))
        directory = tmp_path / (group + "-mpi%d" % ranks)
        assert directory.is_dir()
        assert command[command.index("-c") + 2] == str(directory)
        for rank in range(ranks):
            receipt = ET.Element("testsuite")
            for name, count in profile.GROUPS[group][1].items():
                for parameter in range(count):
                    ET.SubElement(receipt, "testcase", name="%s[%d]" % (name, parameter),
                                  classname="source_only_mpi_stub")
            ET.ElementTree(receipt).write(directory / ("rank%d.xml" % rank))
        return SimpleNamespace(returncode=0, stdout="source-only process stub", stderr="")

    monkeypatch.setattr(driver.subprocess, "run", inert_process)
    sentinel = tmp_path / "parent-owned"
    sentinel.write_text("pytest root must remain intact")
    profile.run_public_checkpoint_profile_world(group=group, ranks=ranks, directory=tmp_path)
    assert sentinel.read_text() == "pytest root must remain intact"
    assert len(calls) == 1 and calls[0][1]["timeout"] == 840
    with pytest.raises(FileExistsError):
        profile.run_public_checkpoint_profile_world(group=group, ranks=ranks, directory=tmp_path)
    assert len(calls) == 1, "a reused namespace must fail before any MPI process starts"


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


@pytest.mark.parametrize("partition", (None, "missing", True, 1, "tag_selection[0]"))
def test_amr_partition_admission_refuses_undeclared_ids_before_launch(tmp_path, monkeypatch, partition):
    from tests.python.support import public_amr_profile_mpi as profile
    calls = []
    monkeypatch.setattr(profile, "run_fixture_world", lambda **arguments: calls.append(arguments))
    with pytest.raises(ValueError, match="exact declared partition"):
        profile.run_public_amr_profile_world(ranks=1, directory=tmp_path, partition=partition)
    assert calls == []


@pytest.mark.parametrize("partition", (None, "missing", True, 1, "saved_restart[0]"))
def test_original_amr_partition_admission_refuses_undeclared_ids_before_launch(tmp_path, monkeypatch, partition):
    from tests.python.support import public_field_profile_mpi as profile
    calls = []
    monkeypatch.setattr(profile, "run_fixture_world", lambda **arguments: calls.append(arguments))
    with pytest.raises(ValueError, match="exact declared partition"):
        profile.run_public_field_profile_world(group="amr_original", ranks=1,
                                               directory=tmp_path, partition=partition)
    assert calls == []


@pytest.mark.parametrize("group", ("completed_original", "imex_time", "nonlinear_mixed", "spatial_interaction"))
def test_unpartitioned_field_profiles_refuse_foreign_partition(tmp_path, monkeypatch, group):
    from tests.python.support import public_field_profile_mpi as profile
    calls = []
    monkeypatch.setattr(profile, "run_fixture_world", lambda **arguments: calls.append(arguments))
    with pytest.raises(ValueError, match="only the original AMR"):
        profile.run_public_field_profile_world(group=group, ranks=1, directory=tmp_path,
                                               partition="saved_restart")
    assert calls == []


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("family", ("amr", "original"))
def test_partition_rank_receipts_refuse_omission_duplicate_failure_error_and_skip(tmp_path, ranks, family):
    from tests.python.support.mpi_fixture_driver import require_rank_results as require_all
    from tests.python.support.public_amr_profile_mpi import PARTITIONS
    from tests.python.support.public_field_profile_mpi import AMR_ORIGINAL_PARTITIONS
    partitions = PARTITIONS if family == "amr" else AMR_ORIGINAL_PARTITIONS
    for partition, (_, expected) in partitions.items():
        directory = tmp_path / partition
        directory.mkdir()
        for rank in range(ranks):
            root = ET.Element("testsuite")
            for name, count in expected.items():
                for parameter in range(count):
                    ET.SubElement(root, "testcase", name="%s[%d]" % (name, parameter), classname="partition_source_stub")
            ET.ElementTree(root).write(directory / ("rank%d.xml" % rank))
        require_all(directory, ranks, expected, purpose=partition)
        path = directory / "rank0.xml"
        root = ET.parse(path).getroot()
        raw = path.read_bytes()
        root.remove(root.find("testcase"))
        ET.ElementTree(root).write(path)
        with pytest.raises(AssertionError, match="case inventory differs"):
            require_all(directory, ranks, expected, purpose=partition)
        path.write_bytes(raw)
        root = ET.parse(path).getroot()
        ET.SubElement(root, "testcase", **root.find("testcase").attrib)
        ET.ElementTree(root).write(path)
        with pytest.raises(AssertionError, match="case inventory differs"):
            require_all(directory, ranks, expected, purpose=partition)
        for failure in ("failure", "error", "skipped"):
            path.write_bytes(raw)
            root = ET.parse(path).getroot()
            ET.SubElement(root.find("testcase"), failure)
            ET.ElementTree(root).write(path)
            with pytest.raises(AssertionError, match="mandatory outcomes"):
                require_all(directory, ranks, expected, purpose=partition)


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("as_text", (False, True))
def test_mpi_timeout_preserves_partial_streams_and_original_exception(tmp_path, monkeypatch, capsys, ranks, as_text):
    import subprocess
    from pops import _native_selector
    from tests.python.support import mpi_fixture_driver as driver
    native = SimpleNamespace(__has_mpi__=True, __mpi_contract__="source-only token")
    monkeypatch.setattr(_native_selector, "select_native_dimension", lambda dimension: native)
    monkeypatch.setattr(driver, "mpi_launcher_or_skip", lambda *args, **kwargs: ["source-only-mpiexec"])
    stdout = "partial case PASSED\nactive case " if as_text else b"partial case PASSED\nactive case \xff"
    stderr = "compiler stage\n" if as_text else b"compiler stage\n\xfe"
    exception = subprocess.TimeoutExpired(["source-only-mpiexec"], 840, output=stdout, stderr=stderr)
    calls = []

    def timed_out(command, **kwargs):
        calls.append((command, kwargs))
        assert "['-v','-ra','--maxfail=1'" in command[command.index("-c") + 1]
        raise exception

    monkeypatch.setattr(driver.subprocess, "run", timed_out)
    directory = tmp_path / "child"
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        driver.run_fixture_world(ranks=ranks, directory=directory, tests=("declared-source-case",),
            expected_cases={"declared-source-case": 1}, expected_rank_env="SOURCE_ONLY_WORLD",
            purpose="source-only timeout guard", timeout_seconds=840)
    assert raised.value is exception and len(calls) == 1 and calls[0][1]["timeout"] == 840
    raw_stdout = stdout.encode("utf-8") if as_text else stdout
    raw_stderr = stderr.encode("utf-8") if as_text else stderr
    assert (directory / "launcher.stdout.log").read_bytes() == raw_stdout
    assert (directory / "launcher.stderr.log").read_bytes() == raw_stderr
    assert (directory / "launcher.log").read_bytes() == raw_stdout + raw_stderr
    assert "active case" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        driver.run_fixture_world(ranks=ranks, directory=directory, tests=("declared-source-case",),
            expected_cases={"declared-source-case": 1}, expected_rank_env="SOURCE_ONLY_WORLD",
            purpose="source-only namespace guard", timeout_seconds=840)
    assert len(calls) == 1, "a reused child namespace must fail before another process starts"
