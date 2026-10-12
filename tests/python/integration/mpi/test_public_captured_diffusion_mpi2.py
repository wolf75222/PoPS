"""All four captured/candidate diffusion cases on every rank in one MPI2 world."""
import pytest

from tests.python.support.public_uniform_profile_mpi import run_captured_diffusion_world


pytestmark = pytest.mark.mpi


def test_public_captured_and_candidate_diffusion_on_authenticated_mpi2_world(tmp_path):
    run_captured_diffusion_world(ranks=2, directory=tmp_path / "mpi2")
