"""Regrid(criterion, when) written in the Program on a genuine MPI1 world (N12 lot 2)."""
import pytest

from tests.python.support.public_regrid_decision_mpi import run_public_regrid_decision_world


pytestmark = pytest.mark.mpi


def test_public_regrid_decision_on_authenticated_mpi1_world(tmp_path):
    run_public_regrid_decision_world(ranks=1, directory=tmp_path)
