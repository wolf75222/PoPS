"""Every completed_original native case on every rank in one mandatory MPI1 world."""
import pytest

from tests.python.support.public_field_profile_mpi import run_public_field_profile_world


pytestmark = pytest.mark.mpi


def test_public_completed_original_on_authenticated_mpi1_world(tmp_path):
    run_public_field_profile_world(group="completed_original", ranks=1,
                                   directory=tmp_path / "mpi1")
