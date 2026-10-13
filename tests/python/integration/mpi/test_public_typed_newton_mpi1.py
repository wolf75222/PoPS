"""Mandatory exact typed_newton partitions on a genuine MPI1 world."""
import pytest

from tests.python.support.public_typed_newton_mpi import (
    TYPED_NEWTON_PARTITIONS,
    run_public_typed_newton_partition,
)


@pytest.mark.parametrize("partition", tuple(TYPED_NEWTON_PARTITIONS))
def test_public_typed_newton_on_authenticated_mpi1_world(tmp_path, partition):
    run_public_typed_newton_partition(partition=partition, ranks=1, directory=tmp_path)
