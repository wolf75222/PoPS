"""Pure XML checks for the script-to-pytest MPI receiver; no native execution."""
import xml.etree.ElementTree as ET

import pytest

from tests.python.support.mpi_pytest_entrypoint import receipt_identities


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
