"""ADC-683 fences for run-owned native ROOT scientific-output communication."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
COLLECTIVE = ROOT / "include/pops/runtime/output_piece_collective.hpp"
SYSTEM = ROOT / "include/pops/runtime/system.hpp"
AMR = ROOT / "include/pops/runtime/amr_system.hpp"
SYSTEM_BINDING = ROOT / "python/bindings/core/init/init_system.cpp"
AMR_BINDING = ROOT / "python/bindings/core/init/init_amr.cpp"
RUNTIME = ROOT / "python/pops/runtime/_runtime_consumers.py"
STUB = ROOT / "python/pops/_pops.pyi"


def test_native_root_output_surface_requires_an_owned_consumer_lane():
    collective = COLLECTIVE.read_text(encoding="utf-8")
    system = SYSTEM.read_text(encoding="utf-8")
    amr = AMR.read_text(encoding="utf-8")

    assert "WorldCommunicator" not in collective
    assert "MPI_COMM_WORLD" not in collective
    assert "const ObserverMpiLane& lane" in collective
    assert "const ObserverMpiLane& lane" in system
    assert "const ObserverMpiLane& lane" in amr


def test_python_root_output_bridge_rejects_the_process_world_type():
    system = SYSTEM_BINDING.read_text(encoding="utf-8")
    amr = AMR_BINDING.read_text(encoding="utf-8")
    stub = STUB.read_text(encoding="utf-8")

    assert "WorldCommunicator" not in system
    assert "WorldCommunicator" not in amr
    assert "const ObserverMpiLane& lane" in system
    assert "const ObserverMpiLane& lane" in amr
    assert "lane: _NativeObserverMpiLane" in stub


def test_runtime_materializes_and_closes_one_root_output_lane_per_run():
    runtime = RUNTIME.read_text(encoding="utf-8")

    assert 'lane_identity = "scientific-output/root/%s" % run_identity.token' in runtime
    assert "self._communicator.duplicate_observer_lane(lane_identity)" in runtime
    assert "root_lane.close_collectively()" in runtime
    assert "native_communicator = lane_provider()" in runtime
