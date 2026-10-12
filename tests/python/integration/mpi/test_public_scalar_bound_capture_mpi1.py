"""Mandatory complete scalar_bound_capture family on a genuine MPI1 world."""
from tests.python.support.public_checkpoint_profile_mpi import run_public_checkpoint_profile_world


def test_public_scalar_bound_capture_on_authenticated_mpi1_world(tmp_path):
    run_public_checkpoint_profile_world(group='scalar_bound_capture', ranks=1, directory=tmp_path)
