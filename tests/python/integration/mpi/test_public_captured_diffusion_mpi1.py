"""All four captured/candidate native diffusion cases in one mandatory MPI1 world."""
import pytest

from tests.python.support.public_uniform_profile_mpi import run_captured_diffusion_world


pytestmark = pytest.mark.mpi


def test_public_captured_and_candidate_diffusion_on_authenticated_mpi1_world(tmp_path):
    run_captured_diffusion_world(ranks=1, directory=tmp_path / "mpi1")
