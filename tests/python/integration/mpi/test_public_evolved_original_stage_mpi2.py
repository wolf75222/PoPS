"""All eight evolved-original cases on every rank in one mandatory MPI2 world."""
import pytest

from tests.python.support.public_uniform_profile_mpi import run_evolved_original_world


pytestmark = pytest.mark.mpi


def test_public_evolved_original_stage_on_authenticated_mpi2_world(tmp_path):
    run_evolved_original_world(ranks=2, directory=tmp_path / "mpi2")
