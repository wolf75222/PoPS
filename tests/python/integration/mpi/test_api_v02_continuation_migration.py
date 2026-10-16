"""One serial orchestrator, four independently terminated real MPI worlds."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

from tests.python.support.mpi_launcher import mpi_launcher_or_skip
from tests.python.support.requirements import require_mpi_or_skip

pytestmark = [pytest.mark.mpi, pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
PROBE = Path(__file__).with_name("probe_api_v02_continuation_migration.py")


def test_runtime_instance_continuation_forest_migrates_1_2_3_1(tmp_path, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    from pops._native_selector import select_native_dimension
    native = select_native_dimension(2)
    if native.__has_mpi__ is not True:
        require_mpi_or_skip("Continuation rank migration requires native MPI", optional_skip=pytest.skip)
    launcher = mpi_launcher_or_skip(native.__mpi_contract__, purpose="Continuation forest rank migration")
    environment = dict(os.environ)
    cache = tmp_path / "native-cache"
    cache.mkdir()
    environment.update(POPS_CACHE_DIR=str(cache), POPS_NATIVE_CACHE_DIR=str(cache), POPS_NATIVE_DIM="2")
    for phase, ranks in enumerate((1, 2, 3, 1)):
        environment["POPS_TEST_PUBLIC_NATIVE_EXPECT_RANKS"] = str(ranks)
        result = subprocess.run([*launcher, "-n", str(ranks), sys.executable, "-u", str(PROBE),
                                 str(tmp_path), str(phase)], cwd=PROBE.parents[4], env=environment,
                                capture_output=True, text=True, timeout=15*60, check=False)
        (tmp_path / ("phase-%d.log" % phase)).write_text(result.stdout+result.stderr)
        assert result.returncode == 0, result.stdout+result.stderr
        assert result.stdout.count("PASS native forest phase=%d ranks=%d" % (phase, ranks)) == ranks
    receipts = [json.loads((tmp_path / ("receipt-%d.json" % phase)).read_text()) for phase in range(4)]
    assert [len(receipt["ranks"]) for receipt in receipts] == [1, 2, 3, 1]
    assert len({receipt["artifact"] for receipt in receipts}) == 1
    assert len({receipt["program"] for receipt in receipts}) == 1
    assert receipts[-1]["step"] == 4
