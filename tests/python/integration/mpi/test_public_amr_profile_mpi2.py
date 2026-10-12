"""All 18 public AMR fixture cases on every rank in one mandatory MPI2 world."""
import pytest

from tests.python.support.public_amr_profile_mpi import run_public_amr_profile_world


pytestmark = pytest.mark.mpi


def test_public_amr_fixtures_on_authenticated_mpi2_world(tmp_path):
    run_public_amr_profile_world(ranks=2, directory=tmp_path / "mpi2")
