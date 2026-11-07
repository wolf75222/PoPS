"""Run one declared pytest receiver under the existing script-style MPI CI contract."""
from __future__ import annotations

import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def receipt_identities(path: Path, expected_cases: int) -> tuple[tuple[str, str], ...]:
    if type(expected_cases) is not int or expected_cases < 1:
        raise ValueError("MPI pytest entrypoint needs a positive exact case count")
    cases = list(ET.parse(path).getroot().iter("testcase"))
    identities = tuple(sorted((case.attrib.get("classname", ""), case.attrib.get("name", ""))
                              for case in cases))
    if (len(cases) != expected_cases or len(set(identities)) != expected_cases
            or any(not owner or not name for owner, name in identities)
            or any(case.find(tag) is not None for case in cases
                   for tag in ("failure", "error", "skipped"))):
        raise ValueError("MPI pytest receiver did not pass every mandatory case without skips")
    return identities


def partition_units(receiver: str, expected_cases: int, environment=None) -> dict | None:
    """Return the allocation's declared partition of this receiver, or None for the whole module.

    The inventory must be exactly the receiver's declared case count, all owned by the receiver;
    the plugin then proves at collection that the module still has exactly those cases.
    """
    from tests.python.support import mpi_partition_units

    spec = mpi_partition_units.from_environment(environment)
    if spec is None:
        return None
    if (len(spec["inventory"]) != expected_cases
            or any(node.split("::", 1)[0] != receiver for node in spec["inventory"])):
        raise ValueError("MPI pytest partition inventory differs from its exact receiver cases")
    return spec


def run(receiver: str, *, expected_cases: int, ranks: int = 2) -> int:
    """Called only by a declared script entrypoint; ordinary collection imports no test module."""
    import pytest
    from pops import _pops

    world = _pops.mpi_world()
    if int(world.size) != ranks:
        raise RuntimeError("MPI pytest receiver launched with an undeclared world size")
    root = Path(__file__).resolve().parents[3]
    partition = world.collective_prepare(lambda: partition_units(receiver, expected_cases))
    selected_cases = expected_cases if partition is None else len(partition["units"])

    def prepare():
        relative = Path(receiver)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py":
            raise ValueError("invalid MPI pytest receiver path")
        if not (root / relative).is_file():
            raise ValueError("missing MPI pytest receiver")
        if type(expected_cases) is not int or expected_cases < 1:
            raise ValueError("invalid mandatory case count")
        directory = Path(os.environ.get("POPS_CI_MPI_ENTRYPOINT_RECEIPTS",
                                       str(root / "build-mpi/python-entrypoint-receipts")))
        suffix = "" if partition is None else "." + partition["partition"]
        directory = directory / (relative.stem + suffix + ".np%d" % ranks)
        directory.mkdir(parents=True, exist_ok=True)
        xml = directory / ("rank%d.xml" % int(world.rank))
        if xml.exists():
            raise ValueError("MPI pytest receiver refuses a stale per-rank receipt")
        return xml

    xml = world.collective_prepare(prepare)
    selection = [] if partition is None else ["-p", "tests.python.support.mpi_partition_units"]
    code = int(pytest.main(["-q", "-ra", "--maxfail=1", "-o", "pythonpath=", *selection,
                            "--junitxml=" + str(xml), receiver]))
    statuses = world.allgather_bytes(world.collective_prepare(lambda: str(code).encode("ascii")))
    if any(status != b"0" for status in statuses):
        return 1
    identities = world.collective_prepare(lambda: receipt_identities(xml, selected_cases))
    if partition is not None:
        units = {tuple(node.split("::", 1)) for node in partition["units"]}
        names = {(str(Path(owner).with_suffix("")).replace("/", "."), name) for owner, name in units}
        if set(identities) != names:
            raise RuntimeError("MPI pytest partition receipt differs from its declared units")
    frame = world.collective_prepare(lambda: json.dumps(identities, separators=(",", ":")).encode())
    if not world.agree_control_bytes(b"pops.mpi-pytest-identities@1", frame):
        raise RuntimeError("MPI pytest receiver cases differ between ranks")
    return 0
