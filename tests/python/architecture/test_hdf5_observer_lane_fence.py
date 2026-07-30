"""ADC-683 fences for observer-owned collective HDF5 communication."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HEADER = ROOT / "include/pops/runtime/output/hdf5_collective.hpp"
SOURCE = ROOT / "src/runtime/output/hdf5_collective.cpp"
BINDING = ROOT / "python/bindings/core/init/init_parallel_hdf5.cpp"
WRITER = ROOT / "python/pops/output/_writers/hdf5.py"


def test_native_hdf5_surface_has_no_process_world_overload_or_probe():
    header = HEADER.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")

    assert "WorldCommunicator" not in header
    assert "WorldCommunicator" not in source
    assert "world_communicator.hpp" not in source
    assert "MPI_COMM_WORLD" not in source
    assert "MPI_Initialized" not in source
    assert "const CommunicatorView& communicator" in header


def test_python_hdf5_route_requires_a_duplicated_observer_lane():
    binding = BINDING.read_text(encoding="utf-8")
    writer = WRITER.read_text(encoding="utf-8")

    assert "WorldCommunicator" not in binding
    assert "world_communicator.hpp" not in binding
    assert "py::isinstance<pops::ObserverMpiLane>" in binding
    assert "requires an exact duplicated observer MPI lane" in binding
    assert "require_communicator(communicator, allow_world=False)" in writer
