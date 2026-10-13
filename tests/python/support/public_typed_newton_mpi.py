"""Exact singleton projections of the complete public typed Newton MPI family."""
from pathlib import Path
import xml.etree.ElementTree as ET

from tests.python.support.mpi_fixture_driver import run_fixture_world
from tests.python.support.public_checkpoint_profile_mpi import CHILD_TIMEOUT_SECONDS


_NATIVE_PARENT = (
    "tests/python/integration/runtime/test_public_newton_typed_original.py"
    "::test_public_typed_original_policy"
)
_NATIVE_CLASSNAME = "tests.python.integration.runtime.test_public_newton_typed_original"
TYPED_NEWTON_PARTITIONS = {
    "relative_uniform": _NATIVE_PARENT + "[relative-uniform]",
    "relative_amr": _NATIVE_PARENT + "[relative-amr]",
    "floor_uniform": _NATIVE_PARENT + "[floor-uniform]",
    "floor_amr": _NATIVE_PARENT + "[floor-amr]",
    "absolute_uniform": _NATIVE_PARENT + "[absolute-uniform]",
    "absolute_amr": _NATIVE_PARENT + "[absolute-amr]",
    "overflow_uniform": _NATIVE_PARENT + "[overflow-uniform]",
    "overflow_amr": _NATIVE_PARENT + "[overflow-amr]",
}


def run_public_typed_newton_partition(*, partition, ranks, directory: Path) -> None:
    if type(partition) is not str or partition not in TYPED_NEWTON_PARTITIONS:
        raise ValueError("typed Newton requires one exact declared partition")
    if type(ranks) is not int or ranks not in (1, 2):
        raise ValueError("typed Newton requires an explicit MPI1 or MPI2 world")
    native_node = TYPED_NEWTON_PARTITIONS[partition]
    child_directory = directory / ("typed_newton-mpi%d" % ranks)
    purpose = "public checkpoint typed_newton " + partition + " profile"
    run_fixture_world(
        ranks=ranks, directory=child_directory, tests=(native_node,),
        expected_cases={"test_public_typed_original_policy": 1},
        expected_rank_env="POPS_TEST_PUBLIC_NATIVE_EXPECT_RANKS", purpose=purpose,
        timeout_seconds=CHILD_TIMEOUT_SECONDS,
    )
    # The common driver authenticates rank inventory, outcomes and uniqueness.
    # A singleton must additionally retain the exact policy/backend identity.
    expected_identity = (_NATIVE_CLASSNAME, native_node.split("::", 1)[1])
    for rank in range(ranks):
        cases = ET.parse(child_directory / ("rank%d.xml" % rank)).getroot().findall(".//testcase")
        actual = [(case.attrib.get("classname"), case.attrib.get("name")) for case in cases]
        assert actual == [expected_identity], (
            purpose + " MPI rank%d exact parameter identity differs" % rank)
