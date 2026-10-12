"""All eight evolved-original fixture cases in one mandatory MPI1 world."""
import pytest

from tests.python.support.public_uniform_profile_mpi import run_evolved_original_world


pytestmark = pytest.mark.mpi


def test_public_evolved_original_stage_on_authenticated_mpi1_world(tmp_path):
    run_evolved_original_world(ranks=1, directory=tmp_path / "mpi1")
