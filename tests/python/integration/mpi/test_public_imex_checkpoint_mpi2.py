"""Mandatory complete imex_checkpoint family on a genuine MPI2 world."""
from tests.python.support.public_checkpoint_profile_mpi import run_public_checkpoint_profile_world


def test_public_imex_checkpoint_on_authenticated_mpi2_world(tmp_path):
    run_public_checkpoint_profile_world(group='imex_checkpoint', ranks=2, directory=tmp_path)
