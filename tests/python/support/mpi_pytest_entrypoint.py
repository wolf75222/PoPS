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


def run(receiver: str, *, expected_cases: int, ranks: int = 2) -> int:
    """Called only by a declared script entrypoint; ordinary collection imports no test module."""
    import pytest
    from pops import _pops

    world = _pops.mpi_world()
    if int(world.size) != ranks:
        raise RuntimeError("MPI pytest receiver launched with an undeclared world size")
    root = Path(__file__).resolve().parents[3]

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
        directory = directory / (relative.stem + ".np%d" % ranks)
        directory.mkdir(parents=True, exist_ok=True)
        xml = directory / ("rank%d.xml" % int(world.rank))
        if xml.exists():
            raise ValueError("MPI pytest receiver refuses a stale per-rank receipt")
        return xml

    xml = world.collective_prepare(prepare)
    code = int(pytest.main(["-q", "-ra", "--maxfail=1", "-o", "pythonpath=",
                            "--junitxml=" + str(xml), receiver]))
    statuses = world.allgather_bytes(world.collective_prepare(lambda: str(code).encode("ascii")))
    if any(status != b"0" for status in statuses):
        return 1
    identities = world.collective_prepare(lambda: receipt_identities(xml, expected_cases))
    frame = world.collective_prepare(lambda: json.dumps(identities, separators=(",", ":")).encode())
    if not world.agree_control_bytes(b"pops.mpi-pytest-identities@1", frame):
        raise RuntimeError("MPI pytest receiver cases differ between ranks")
    return 0
