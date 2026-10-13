"""Source-only admission and receipt contracts for the exact Newton MPI partitions."""
import ast
import importlib
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from tests.python.support import public_typed_newton_mpi as harness


ROOT = Path(__file__).resolve().parents[3]
NATIVE_FILE = "tests/python/integration/runtime/test_public_newton_typed_original.py"
NATIVE_FUNCTION = "test_public_typed_original_policy"
NATIVE_CLASSNAME = "tests.python.integration.runtime.test_public_newton_typed_original"
PARTITIONS = (
    "relative_uniform", "relative_amr", "floor_uniform", "floor_amr",
    "absolute_uniform", "absolute_amr", "overflow_uniform", "overflow_amr",
)


def _write_receipt(directory, rank, *, name, classname=NATIVE_CLASSNAME):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite", tests="1", failures="0", errors="0", skipped="0")
    ET.SubElement(suite, "testcase", classname=classname, name=name)
    ET.ElementTree(root).write(directory / ("rank%d.xml" % rank))


def test_partition_table_is_the_complete_native_fixture_cartesian_product():
    tree = ast.parse((ROOT / NATIVE_FILE).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == NATIVE_FUNCTION)
    parameters = {
        ast.literal_eval(decorator.args[0]): ast.literal_eval(decorator.args[1])
        for decorator in function.decorator_list
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "parametrize"
    }
    native_ids = tuple(policy + "-" + backend for policy in parameters["policy"]
                       for backend in parameters["backend"])
    assert tuple(harness.TYPED_NEWTON_PARTITIONS) == PARTITIONS
    assert tuple(harness.TYPED_NEWTON_PARTITIONS.values()) == tuple(
        NATIVE_FILE + "::" + NATIVE_FUNCTION + "[" + case + "]" for case in native_ids)
    assert len(set(native_ids)) == 8


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("partition", PARTITIONS)
def test_each_partition_retains_singleton_selector_rank_authority_and_budget(
        monkeypatch, tmp_path, ranks, partition):
    native_name = NATIVE_FUNCTION + "[" + partition.replace("_", "-") + "]"
    calls = []

    def receive(**kwargs):
        calls.append(kwargs)
        kwargs["directory"].mkdir()
        for rank in range(ranks):
            _write_receipt(kwargs["directory"], rank, name=native_name)

    monkeypatch.setattr(harness, "run_fixture_world", receive)
    harness.run_public_typed_newton_partition(partition=partition, ranks=ranks, directory=tmp_path)
    assert len(calls) == 1
    call = calls[0]
    assert call["tests"] == (NATIVE_FILE + "::" + native_name,)
    assert call["ranks"] == ranks
    assert call["expected_cases"] == {NATIVE_FUNCTION: 1}
    assert call["expected_rank_env"] == "POPS_TEST_PUBLIC_NATIVE_EXPECT_RANKS"
    assert call["timeout_seconds"] == 840
    assert call["directory"] == tmp_path / ("typed_newton-mpi%d" % ranks)


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("wrong_name,wrong_classname", (
    (NATIVE_FUNCTION + "[absolute-uniform]", NATIVE_CLASSNAME),
    (NATIVE_FUNCTION + "[relative-amr]", NATIVE_CLASSNAME),
    (NATIVE_FUNCTION, NATIVE_CLASSNAME),
    (NATIVE_FUNCTION + "[relative-uniform]", "tests.python.integration.runtime.foreign_fixture"),
))
def test_same_base_count_cannot_substitute_policy_backend_or_fixture_identity(
        monkeypatch, tmp_path, ranks, wrong_name, wrong_classname):
    def receive(**kwargs):
        kwargs["directory"].mkdir()
        for rank in range(ranks):
            name = wrong_name if rank == ranks - 1 else NATIVE_FUNCTION + "[relative-uniform]"
            classname = wrong_classname if rank == ranks - 1 else NATIVE_CLASSNAME
            _write_receipt(kwargs["directory"], rank, name=name, classname=classname)

    monkeypatch.setattr(harness, "run_fixture_world", receive)
    with pytest.raises(AssertionError, match="exact parameter identity differs"):
        harness.run_public_typed_newton_partition(
            partition="relative_uniform", ranks=ranks, directory=tmp_path)


@pytest.mark.parametrize("partition", (None, True, 0, "", "relative-uniform", "unknown", []))
def test_unknown_partition_is_rejected_before_launch(monkeypatch, tmp_path, partition):
    monkeypatch.setattr(harness, "run_fixture_world", lambda **kwargs: pytest.fail("launched"))
    with pytest.raises(ValueError, match="exact declared partition"):
        harness.run_public_typed_newton_partition(partition=partition, ranks=1, directory=tmp_path)


@pytest.mark.parametrize("ranks", (None, True, False, 0, -1, 3, "1", 1.0))
def test_rank_authority_is_exact_and_validated_before_launch(monkeypatch, tmp_path, ranks):
    monkeypatch.setattr(harness, "run_fixture_world", lambda **kwargs: pytest.fail("launched"))
    with pytest.raises(ValueError, match="explicit MPI1 or MPI2"):
        harness.run_public_typed_newton_partition(
            partition="relative_uniform", ranks=ranks, directory=tmp_path)


@pytest.mark.parametrize("ranks", (1, 2))
def test_wrapper_collects_all_eight_exact_ids_and_preserves_world(monkeypatch, tmp_path, ranks):
    wrapper = importlib.import_module(
        "tests.python.integration.mpi.test_public_typed_newton_mpi%d" % ranks)
    function = getattr(wrapper, "test_public_typed_newton_on_authenticated_mpi%d_world" % ranks)
    mark = next(mark for mark in function.pytestmark if mark.name == "parametrize")
    assert mark.args == ("partition", PARTITIONS)
    calls = []
    monkeypatch.setattr(wrapper, "run_public_typed_newton_partition", lambda **kwargs: calls.append(kwargs))
    for partition in PARTITIONS:
        function(tmp_path, partition)
    assert calls == [dict(partition=partition, ranks=ranks, directory=tmp_path)
                     for partition in PARTITIONS]
