"""DM8 for finite arrays, observed by Kokkos Tools on the actual Program DSO.

The Kokkos Tools library of ``tests/python/support/kokkos_residency_tool.cpp`` reports the deep
copies that each step makes. A Program whose array operations all have a device lowering moves no
array byte inside a step. A Program with one host-placed map is the negative control: the library
sees the one host copy of the Array it reads and its publication, and the residency counters of
the Program DSO count the same bytes (PF2). A fold sequenced on the host reads one element of its
carry per iteration and copies once the Array it reads without redefining it.
"""
import json
import os
import subprocess
import sys

import pytest

from tests.python.support import finite_residency_probe as probe
from tests.python.support.requirements import REPO_ROOT

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

TOOL = REPO_ROOT / "tests" / "python" / "support" / "kokkos_residency_tool.cpp"
# The memory space of the default execution space, by Kokkos name (DM7).
TARGET_SPACES = {"Serial": "Host", "OpenMP": "Host", "Threads": "Host", "Cuda": "Cuda",
                 "HIP": "HIP", "SYCL": "SYCLDeviceUSM"}
# Leaves of the probe element (exact Int64, value Float64, even Bool stored as unsigned char).
ELEMENT_BYTES = 8 + 8 + 1
LEAVES = 3


@pytest.fixture(scope="module")
def residency_tool(tmp_path_factory, native_cxx):
    library = tmp_path_factory.mktemp("kokkos-residency") / "libpops_kokkos_residency.so"
    built = subprocess.run([native_cxx, "-std=c++17", "-O1", "-fPIC", "-shared", str(TOOL), "-o",
                            str(library)], capture_output=True, text=True, check=False)
    assert built.returncode == 0, built.stdout + built.stderr
    return library


def _run(kind, tool, tmp_path):
    windows = tmp_path / (kind + "-windows.jsonl")
    report = tmp_path / (kind + "-report.json")
    environment = dict(os.environ, KOKKOS_TOOLS_LIBS=str(tool),
                       POPS_KOKKOS_RESIDENCY_OUTPUT=str(windows))
    result = subprocess.run(
        [sys.executable, "-m", "tests.python.support.finite_residency_probe", kind, str(report)],
        cwd=REPO_ROOT, env=environment, capture_output=True, text=True, timeout=1800, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [json.loads(line) for line in windows.read_text(encoding="utf-8").splitlines()]
    steps = [line for line in lines if line["region"] == "pops::step"]
    # One window per step: the library was loaded and saw every step.
    assert len(steps) == probe.STEPS, lines
    return lines, steps, json.loads(report.read_text(encoding="utf-8"))


def _allocations(steps):
    spaces = {}
    for step in steps:
        for space, count in step["array_allocations"].items():
            spaces[space] = spaces.get(space, 0) + count
    return spaces


def test_device_lowered_arrays_move_no_array_byte_inside_a_step(
        residency_tool, tmp_path, isolated_native_cache, kokkos_root, record_property):
    del isolated_native_cache, kokkos_root
    windows, steps, report = _run("device", residency_tool, tmp_path)
    assert all(window["array_data"] == {"count": 0, "bytes": 0} for window in windows), windows
    assert all(window["array_labels"] == [] for window in windows)
    assert report["pops_finite_host_map_count"] == 0
    assert report["pops_finite_kokkos_launch_count"] > 0
    assert report["pops_finite_device_to_host_bytes"] == 0
    assert report["pops_finite_host_to_device_bytes"] == 0
    assert report["pops_finite_host_transfer_count"] == 0
    # Each step allocates the columns of its two maps once, in the target memory space.
    assert _allocations(steps) == {TARGET_SPACES[report["execution_space"]]: 2 * LEAVES * probe.STEPS}
    if os.environ.get("POPS_REQUIRE_DEVICE") == "1":
        assert report["execution_space"] in ("Cuda", "HIP", "SYCL"), report["execution_space"]
    record_property("kokkos_execution_space", report["execution_space"])
    record_property("residency_windows", json.dumps(windows, sort_keys=True))


def test_host_placed_map_is_seen_crossing_and_is_counted(
        residency_tool, tmp_path, isolated_native_cache, kokkos_root, record_property):
    del isolated_native_cache, kokkos_root
    _, steps, report = _run("host", residency_tool, tmp_path)
    column_bytes = probe.COUNT * ELEMENT_BYTES
    for step in steps:
        # One host copy per column of the Array that the map reads, then one publication per
        # column, instead of one read per leaf of each element.
        assert step["array_data"] == {"count": 2 * LEAVES, "bytes": 2 * column_bytes}, step
    assert report["pops_finite_host_map_count"] == probe.STEPS
    assert report["pops_finite_device_to_host_bytes"] == probe.STEPS * column_bytes
    assert report["pops_finite_host_to_device_bytes"] == probe.STEPS * column_bytes
    assert report["pops_finite_host_transfer_count"] == probe.STEPS * 2 * LEAVES
    assert sum(step["array_data"]["bytes"] for step in steps) == (
        report["pops_finite_device_to_host_bytes"] + report["pops_finite_host_to_device_bytes"])
    # The host map publishes its columns once into the target memory space.
    assert _allocations(steps) == {TARGET_SPACES[report["execution_space"]]: 3 * LEAVES * probe.STEPS}
    record_property("residency_steps", json.dumps(steps, sort_keys=True))


def test_host_fold_reads_its_carry_and_copies_what_it_does_not_redefine_once(
        residency_tool, tmp_path, isolated_native_cache, kokkos_root, record_property):
    del isolated_native_cache, kokkos_root
    _, steps, report = _run("fold", residency_tool, tmp_path)
    carry_reads = LEAVES * probe.FOLDS
    copied = (probe.FOLDS + probe.COUNT) * ELEMENT_BYTES
    for step in steps:
        # One read per leaf of the carry element of each iteration, then one host copy per
        # column of the Array that the fold reads without redefining it.
        assert step["array_data"] == {"count": carry_reads + LEAVES, "bytes": copied}, step
    assert report["pops_finite_host_map_count"] == 0
    assert report["pops_finite_device_to_host_bytes"] == probe.STEPS * copied
    assert report["pops_finite_host_to_device_bytes"] == 0
    assert report["pops_finite_host_transfer_count"] == probe.STEPS * (carry_reads + LEAVES)
    # Each iteration maps the carry on the device, after the two maps of the probe.
    assert _allocations(steps) == {
        TARGET_SPACES[report["execution_space"]]: (2 + probe.FOLDS) * LEAVES * probe.STEPS}
    record_property("residency_steps", json.dumps(steps, sort_keys=True))
