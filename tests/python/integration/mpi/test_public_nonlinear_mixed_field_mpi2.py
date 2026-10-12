"""Every nonlinear_mixed native case on every rank in one mandatory MPI2 world."""
import pytest

from tests.python.support.public_field_profile_mpi import run_public_field_profile_world


pytestmark = pytest.mark.mpi


def test_public_nonlinear_mixed_on_authenticated_mpi2_world(tmp_path):
    run_public_field_profile_world(group="nonlinear_mixed", ranks=2,
                                   directory=tmp_path / "mpi2")
