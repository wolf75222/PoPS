"""Run public native fixtures in authenticated MPI worlds and require every rank receipt."""
from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from tests.python.support.mpi_launcher import mpi_launcher_or_skip
from tests.python.support.requirements import require_mpi_or_skip


ROOT = Path(__file__).resolve().parents[3]
PROCESS_TIMEOUT_SECONDS = 15 * 60


def require_rank_results(directory: Path, ranks: int, expected_cases: dict[str, int], *,
                         purpose: str) -> None:
    """Reject omitted ranks or parameter cases, failed cases and silent native skips."""
    for rank in range(ranks):
        receipt = directory / ("rank%d.xml" % rank)
        if not receipt.is_file():
            raise AssertionError(purpose + " MPI rank%d omitted its pytest receipt" % rank)
        cases = ET.parse(receipt).getroot().findall(".//testcase")
        actual = Counter(case.attrib["name"].split("[", 1)[0] for case in cases)
        assert actual == expected_cases, purpose + " MPI rank case inventory differs"
        identities = {(case.attrib.get("classname"), case.attrib["name"]) for case in cases}
        assert len(identities) == len(cases), purpose + " MPI rank duplicated a parameter case"
        assert not any(case.find(tag) is not None for case in cases
                       for tag in ("failure", "error", "skipped")), (
            purpose + " MPI rank%d did not receive all mandatory outcomes" % rank)


def run_fixture_world(*, ranks: int, directory: Path, tests: tuple[str, ...],
                      expected_cases: dict[str, int], expected_rank_env: str,
                      purpose: str, timeout_seconds: int = PROCESS_TIMEOUT_SECONDS) -> None:
    """Stay outside MPI while each child receives its requested exact native world."""
    if type(timeout_seconds) is not int or not 0 < timeout_seconds <= PROCESS_TIMEOUT_SECONDS:
        raise ValueError("MPI child watchdog must fit the existing 900-second ceiling")
    try:
        from pops._native_selector import select_native_dimension

        native = select_native_dimension(2)
    except (ImportError, RuntimeError) as exc:
        require_mpi_or_skip(purpose + " cannot import its native module: %s" % exc,
                            optional_skip=pytest.skip)
        raise AssertionError("pytest.skip unexpectedly returned") from exc
    if getattr(native, "__has_mpi__", False) is not True:
        require_mpi_or_skip(purpose + " MPI proof requires an MPI-enabled native module",
                            optional_skip=pytest.skip)
        raise AssertionError("pytest.skip unexpectedly returned")
    launcher = mpi_launcher_or_skip(getattr(native, "__mpi_contract__", None), purpose=purpose)
    directory.mkdir()
    environment = dict(os.environ)
    environment["POPS_REQUIRE_MPI_TESTS"] = "1"
    environment["POPS_REQUIRE_NATIVE_TESTS"] = "1"
    environment["POPS_NATIVE_DIM"] = "2"
    environment[expected_rank_env] = str(ranks)
    bootstrap = (
        "from pops._native_selector import select_native_dimension; "
        "native=select_native_dimension(2); world=native.mpi_world(); "
        "import pathlib,sys,pytest; "
        "receipt=pathlib.Path(sys.argv[1])/('rank%d.xml'%world.rank); "
        "raise SystemExit(pytest.main(['-v','-ra','--maxfail=1', "
        "'--junitxml='+str(receipt), *sys.argv[2:]]))"
    )
    command = [*launcher, "-n", str(ranks), sys.executable, "-u", "-c", bootstrap,
               str(directory), *tests]
    try:
        result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                                text=True, check=False, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        def partial_bytes(value):
            if value is None:
                return b""
            return value if isinstance(value, bytes) else value.encode("utf-8")

        stdout, stderr = partial_bytes(error.stdout), partial_bytes(error.stderr)
        (directory / "launcher.stdout.log").write_bytes(stdout)
        (directory / "launcher.stderr.log").write_bytes(stderr)
        (directory / "launcher.log").write_bytes(stdout + stderr)
        print(purpose + " MPI%d child timed out after %d seconds; partial output follows"
              % (ranks, timeout_seconds), flush=True)
        print((stdout + stderr).decode("utf-8", errors="replace"), end="", flush=True)
        raise
    output = result.stdout + result.stderr
    (directory / "launcher.log").write_text(output)
    assert result.returncode == 0, (
        purpose + " MPI%d process failed with status %d\n%s"
        % (ranks, result.returncode, output))
    require_rank_results(directory, ranks, expected_cases, purpose=purpose)
