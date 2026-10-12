"""Mandatory complete typed_newton family on a genuine MPI1 world."""
from tests.python.support.public_checkpoint_profile_mpi import run_public_checkpoint_profile_world


def test_public_typed_newton_on_authenticated_mpi1_world(tmp_path):
    run_public_checkpoint_profile_world(group='typed_newton', ranks=1, directory=tmp_path)
