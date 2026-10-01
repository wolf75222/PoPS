"""Independent Python transport/frontier review; recorder is never a native codec."""
from pathlib import Path
from types import SimpleNamespace
import ast
import sys
import numpy as np
import pops
import pytest
from pops.runtime import _checkpoint_state_carriers as carrier

IMAGE = b"POPSCAR1opaque-recorded-source-geometry"

@pytest.fixture
def topology(monkeypatch):
    from pops.output import _checkpoint_collective as collective
    monkeypatch.setattr(collective, "checkpoint_topology", lambda owner: collective.CheckpointTopology(0, 1))
    assert Path(pops.__file__).resolve() == Path(__file__).resolve().parents[2]/"python/pops/__init__.py"
    assert "pops._pops" not in sys.modules
    assert not any(name.startswith("pops._native.dim") for name in sys.modules)

@pytest.mark.parametrize("bad", (bytearray(IMAGE), memoryview(IMAGE), b"POPSCAR1", b"POPSCAR2broken", "POPSCAR1wrong"))
def test_capture_refuses_nonexact_native_image_without_payload_write(topology, bad):
    payload = {}
    sim = SimpleNamespace(checkpoint_state_carriers=lambda: bad)
    with pytest.raises(ValueError):
        carrier.capture_checkpoint_state_carriers(None, sim, payload)
    assert payload == {}

@pytest.mark.parametrize("bad", (None, np.array([1], dtype=np.int64), np.array([[1]], dtype=np.uint8), np.frombuffer(b"POPSCAR2wrong", dtype=np.uint8)))
def test_restore_preflight_refuses_archive_frontier_before_codec(topology, bad):
    calls = []
    sim = SimpleNamespace(validate_checkpoint_state_carriers=lambda image: calls.append(image), restore_checkpoint_state_carriers=lambda image: calls.append("mutation"))
    payload = {} if bad is None else {carrier.STATE_CARRIERS_KEY: bad}
    with pytest.raises(ValueError):
        carrier.prepare_checkpoint_state_carriers(None, sim, payload)
    assert calls == []

@pytest.mark.parametrize("missing", ("checkpoint_state_carriers", "validate_checkpoint_state_carriers", "restore_checkpoint_state_carriers"))
def test_missing_native_method_refuses_at_collective_preflight(topology, missing, monkeypatch):
    from pops.output import _checkpoint_collective as collective
    calls, votes = [], []
    methods = {"checkpoint_state_carriers": lambda: calls.append("capture") or IMAGE,
               "validate_checkpoint_state_carriers": lambda image: calls.append("validate"),
               "restore_checkpoint_state_carriers": lambda image: calls.append("restore")}
    del methods[missing]
    real = collective.consensus
    def vote(topology, phase, **kwargs):
        votes.append((phase, kwargs.get("error")))
        return real(topology, phase, **kwargs)
    monkeypatch.setattr(collective, "consensus", vote)
    with pytest.raises(TypeError):
        if missing == "checkpoint_state_carriers":
            carrier.capture_checkpoint_state_carriers(None, SimpleNamespace(**methods), {})
        else:
            carrier.prepare_checkpoint_state_carriers(None, SimpleNamespace(**methods), {carrier.STATE_CARRIERS_KEY: np.frombuffer(IMAGE, dtype=np.uint8)})
    assert calls == [] and len(votes) == 1 and isinstance(votes[0][1], TypeError)


def test_capture_copies_and_prepare_only_validates_recorded_image(topology):
    calls = []
    sim = SimpleNamespace(checkpoint_state_carriers=lambda: IMAGE,
        validate_checkpoint_state_carriers=lambda image: calls.append(("validate", image)),
        restore_checkpoint_state_carriers=lambda image: calls.append(("restore", image)))
    payload = {}
    carrier.capture_checkpoint_state_carriers(None, sim, payload)
    assert payload[carrier.STATE_CARRIERS_KEY].flags.owndata
    assert payload[carrier.STATE_CARRIERS_KEY].dtype == np.uint8
    assert carrier.prepare_checkpoint_state_carriers(None, sim, payload) == IMAGE
    assert calls == [("validate", IMAGE)]


def test_carrier_publication_follows_replay_clock_warmstarts_before_contract_and_regrid():
    source = Path(pops.__file__).parent/"runtime/_amr_checkpoint_v3.py"
    tree = ast.parse(source.read_text())
    apply = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "apply_v3")
    calls = [(getattr(n.func, "attr", getattr(n.func, "id", "")), n.lineno) for n in ast.walk(apply) if isinstance(n, ast.Call)]
    def line(name):
        return min(number for called, number in calls if called == name)
    assert line("_restore_histories_v3") < line("set_clock") < line("restore_fields") < line("restore_checkpoint_state_carriers")
    regrid = min(n.lineno for n in ast.walk(apply) if isinstance(n, ast.Attribute) and n.attr == "regrid_on_restart")
    assert line("restore_checkpoint_state_carriers") < line("validate_restored_contract") < regrid


def test_obsolete_amr_payload_version_refuses_before_geometry_preparation():
    from pops._generated_release_contract import AMR_CHECKPOINT_PAYLOAD_VERSION
    from pops.runtime._checkpoint_manifest import require_exact_payload_version
    assert AMR_CHECKPOINT_PAYLOAD_VERSION == 12
    with pytest.raises(ValueError):
        require_exact_payload_version({"pops_amr_checkpoint_version": np.array(11, dtype=np.int64)}, key="pops_amr_checkpoint_version", expected=12, runtime_kind="AMR")
    source = (Path(pops.__file__).parent/"runtime/_amr_system_io.py").read_text()
    start = source.index("def _prepare_checkpoint_restart")
    end = source.index("def _begin_checkpoint_restart", start)
    section = source[start:end]
    assert section.index("require_exact_payload_version(\n") < section.index("prepare_v3(\n")



def test_native_validator_failure_propagates_without_restore(topology):
    def invalid(image):
        raise ValueError("recorded source geometry invalid")
    calls = []
    sim = SimpleNamespace(validate_checkpoint_state_carriers=invalid, restore_checkpoint_state_carriers=lambda image: calls.append(image))
    with pytest.raises(ValueError, match="recorded source geometry"):
        carrier.prepare_checkpoint_state_carriers(None, sim, {carrier.STATE_CARRIERS_KEY: np.frombuffer(IMAGE, dtype=np.uint8)})
    assert calls == []
    source = (Path(pops.__file__).parent/"output/_checkpoint_collective.py").read_text()
    # The real outer restart protocol closes the fallible preparation before mutation.
    assert source.index('error=prepare_error)') < source.index('error=begin_error)')
