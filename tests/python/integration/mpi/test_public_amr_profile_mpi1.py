"""All 18 public AMR cases, partitioned into mandatory bounded MPI1 worlds."""
import pytest

from tests.python.support.public_amr_profile_mpi import PARTITIONS, run_public_amr_profile_world


pytestmark = pytest.mark.mpi


@pytest.mark.parametrize("partition", tuple(PARTITIONS))
def test_public_amr_fixtures_on_authenticated_mpi1_world(tmp_path, partition):
    run_public_amr_profile_world(ranks=1, directory=tmp_path / (partition + "-mpi1"),
                                 partition=partition)
