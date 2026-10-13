"""Every original AMR case in disjoint bounded mandatory MPI1 worlds."""
import pytest

from tests.python.support.public_field_profile_mpi import AMR_ORIGINAL_PARTITIONS, run_public_field_profile_world


pytestmark = pytest.mark.mpi


@pytest.mark.parametrize("partition", tuple(AMR_ORIGINAL_PARTITIONS))
def test_public_amr_original_on_authenticated_mpi1_world(tmp_path, partition):
    run_public_field_profile_world(group="amr_original", ranks=1,
                                   directory=tmp_path / (partition + "-mpi1"), partition=partition)
