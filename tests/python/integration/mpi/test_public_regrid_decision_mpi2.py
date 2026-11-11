"""Regrid(criterion, when) written in the Program on a genuine MPI2 world (N12 lot 2)."""
import pytest

from tests.python.support.public_regrid_decision_mpi import run_public_regrid_decision_world


pytestmark = pytest.mark.mpi


def test_public_regrid_decision_on_authenticated_mpi2_world(tmp_path):
    run_public_regrid_decision_world(ranks=2, directory=tmp_path)
