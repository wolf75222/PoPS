"""Pure checks for the script-to-pytest MPI receiver and its partition units; no native execution."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from tests.python.support import mpi_partition_units
from tests.python.support.mpi_pytest_entrypoint import partition_units, receipt_identities

INVENTORY = ["tests/python/r.py::test_case[1]", "tests/python/r.py::test_case[2]",
             "tests/python/r.py::test_other"]


def _receipt(tmp_path, *, cases=2, failure=None, duplicate=False):
    root = ET.Element("testsuite")
    for number in range(cases):
        node = ET.SubElement(root, "testcase", classname="receiver",
                             name="case%d" % (0 if duplicate else number))
        if failure and number == 0:
            ET.SubElement(node, failure)
    path = tmp_path / "rank.xml"
    ET.ElementTree(root).write(path)
    return path


def test_exact_rank_receipt_has_stable_case_identity(tmp_path):
    assert receipt_identities(_receipt(tmp_path), 2) == (("receiver", "case0"), ("receiver", "case1"))


@pytest.mark.parametrize("tag", ("failure", "error", "skipped"))
def test_rank_receipt_refuses_every_nonpass(tmp_path, tag):
    with pytest.raises(ValueError, match="mandatory"):
        receipt_identities(_receipt(tmp_path, failure=tag), 2)


@pytest.mark.parametrize("count", (0, 1, 3))
def test_rank_receipt_refuses_empty_missing_or_excess_cases(tmp_path, count):
    with pytest.raises(ValueError, match="mandatory"):
        receipt_identities(_receipt(tmp_path, cases=count), 2)


def test_rank_receipt_refuses_duplicate_identity(tmp_path):
    with pytest.raises(ValueError, match="mandatory"):
        receipt_identities(_receipt(tmp_path, duplicate=True), 2)


@pytest.mark.parametrize("count", (True, 0, -1, 2.0))
def test_expected_cardinality_is_exact_positive_integer(tmp_path, count):
    with pytest.raises(ValueError, match="positive"):
        receipt_identities(_receipt(tmp_path), count)


@pytest.mark.parametrize("mutation", ("type", "field", "identity", "empty", "unqualified", "padded",
                                      "duplicate_inventory", "duplicate_unit", "outside", "scalar"))
def test_partition_spec_refuses_every_inexact_shape(mutation):
    spec = {"partition": "first", "inventory": list(INVENTORY), "units": INVENTORY[:2]}
    if mutation == "type":
        spec = list(spec.items())
    elif mutation == "field":
        spec["timeout_seconds"] = 900
    elif mutation == "identity":
        spec["partition"] = "First-1"
    elif mutation == "empty":
        spec["units"] = []
    elif mutation == "unqualified":
        spec["units"] = ["test_case[1]"]
    elif mutation == "padded":
        spec["inventory"][0] += " "
    elif mutation == "duplicate_inventory":
        spec["inventory"].append(INVENTORY[0])
    elif mutation == "duplicate_unit":
        spec["units"] = [INVENTORY[0], INVENTORY[0]]
    elif mutation == "outside":
        spec["units"] = ["tests/python/r.py::test_added"]
    else:
        spec["inventory"] = INVENTORY[0]
    with pytest.raises(ValueError, match="MPI partition units"):
        mpi_partition_units.validate(spec)


def test_partition_collection_must_equal_the_inventory_before_units_run():
    spec = mpi_partition_units.validate({"partition": "first", "inventory": INVENTORY, "units": INVENTORY[1:]})
    assert mpi_partition_units.selected(reversed(INVENTORY), spec) == set(INVENTORY[1:])
    for collected, detail in (
            (INVENTORY[:2], "missing=['tests/python/r.py::test_other'] extra=[]"),
            (INVENTORY + ["tests/python/r.py::test_added"], "missing=[] extra=['tests/python/r.py::test_added']"),
            (INVENTORY[:2] + ["tests/python/r.py::test_renamed"],
             "missing=['tests/python/r.py::test_other'] extra=['tests/python/r.py::test_renamed']")):
        with pytest.raises(ValueError, match=re.escape(detail)):
            mpi_partition_units.selected(collected, spec)
    with pytest.raises(ValueError, match="duplicated"):
        mpi_partition_units.selected(INVENTORY + INVENTORY[:1], spec)


def test_partition_plugin_keeps_units_reports_the_rest_deselected_and_refuses_no_spec(monkeypatch):
    reported = []
    config = SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items: reported.extend(items)))
    items = [SimpleNamespace(nodeid=node) for node in INVENTORY]
    monkeypatch.setenv(mpi_partition_units.ENVIRONMENT,
                       mpi_partition_units.encode("first", INVENTORY, INVENTORY[2:]))
    mpi_partition_units.pytest_collection_modifyitems(config, items)
    assert [item.nodeid for item in items] == INVENTORY[2:]
    assert [item.nodeid for item in reported] == INVENTORY[:2]
    monkeypatch.delenv(mpi_partition_units.ENVIRONMENT)
    with pytest.raises(ValueError, match="without its declared units"):
        mpi_partition_units.pytest_collection_modifyitems(config, [SimpleNamespace(nodeid=node) for node in INVENTORY])


def test_partition_plugin_runs_only_declared_units_of_an_unchanged_real_collection(tmp_path):
    root = Path(__file__).resolve().parents[3]
    (tmp_path / "test_receiver.py").write_text(
        "import pytest\n\n\n@pytest.mark.parametrize('case', (1, 2, 3))\ndef test_case(case):\n    pass\n\n\n"
        "def test_other():\n    pass\n")
    nodes = ["test_receiver.py::test_case[%d]" % case for case in (1, 2, 3)] + ["test_receiver.py::test_other"]

    def run(inventory):
        environment = {**os.environ, "PYTHONPATH": str(root),
                       mpi_partition_units.ENVIRONMENT: mpi_partition_units.encode("half", inventory, nodes[1:3])}
        environment.pop("PYTEST_ADDOPTS", None)
        # As in mpi_pytest_entrypoint.run, the plugin module is imported before pytest loads it.
        launcher = ("import sys, pytest, tests.python.support.mpi_partition_units; "
                    "sys.exit(pytest.main(sys.argv[1:]))")
        return subprocess.run(
            [sys.executable, "-c", launcher, "-q", "-p", "no:cacheprovider", "--rootdir", str(tmp_path),
             "-W", "error::pytest.PytestAssertRewriteWarning",
             "-p", "tests.python.support.mpi_partition_units", "--junitxml", str(tmp_path / "rank0.xml"),
             "test_receiver.py"], cwd=tmp_path, env=environment, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=300, check=False)

    result = run(nodes)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed, 2 deselected" in result.stdout
    assert receipt_identities(tmp_path / "rank0.xml", 2) == (
        ("test_receiver", "test_case[2]"), ("test_receiver", "test_case[3]"))
    # A renamed case changes the collection: nothing runs and the launch fails.
    result = run(nodes[:3] + ["test_receiver.py::test_renamed"])
    assert result.returncode != 0
    assert "collection differs from its declared inventory" in result.stdout + result.stderr
    assert " passed" not in result.stdout


def test_entrypoint_partition_requires_the_receivers_exact_declared_cases():
    receiver = "tests/python/r.py"
    assert partition_units(receiver, 3, {}) is None
    raw = mpi_partition_units.encode("first", INVENTORY, INVENTORY[:1])
    assert partition_units(receiver, 3, {mpi_partition_units.ENVIRONMENT: raw})["units"] == INVENTORY[:1]
    for owner, count in ((receiver, 4), ("tests/python/other.py", 3)):
        with pytest.raises(ValueError, match="exact receiver cases"):
            partition_units(owner, count, {mpi_partition_units.ENVIRONMENT: raw})


def test_parameter_partitions_share_one_inventory_and_count_cases_per_function():
    partitions = mpi_partition_units.parameter_partitions({
        "first": ("m.py::test_f[1]", "m.py::test_f[2]"), "second": ("m.py::test_f[3]", "m.py::test_g")})
    for identity, (selectors, _cases, encoded) in partitions.items():
        assert selectors == ("m.py::test_f", "m.py::test_g")
        assert json.loads(encoded) == {
            "partition": identity,
            "inventory": ["m.py::test_f[1]", "m.py::test_f[2]", "m.py::test_f[3]", "m.py::test_g"],
            "units": list({"first": ("m.py::test_f[1]", "m.py::test_f[2]"),
                           "second": ("m.py::test_f[3]", "m.py::test_g")}[identity])}
    assert partitions["first"][1] == {"test_f": 2} and partitions["second"][1] == {"test_f": 1, "test_g": 1}
    for invalid in ({"first": ("m.py::test_f[1]",), "second": ("m.py::test_f[1]",)},
                    {"first": ("m.py::test_f[1]",), "second": ()}):
        with pytest.raises(ValueError, match="nonempty and disjoint"):
            mpi_partition_units.parameter_partitions(invalid)
