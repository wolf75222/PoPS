"""Batch reporting preserves validation, revisions and the legacy protocol."""

import pytest

from pops.runtime._program_manifest_snapshot import _NAMES, program_manifest_reader
from pops.runtime._amr_checkpoint_contract import contract_for
from pops.runtime.program_report import _amr_temporal_report
from test_amr_checkpoint_contract import _Sim


class BatchSim(_Sim):
    def __init__(self):
        self.batch_calls = 0
        self.revision = "0"
        self.failed = False
        self.last_tables = None

    def program_manifest_snapshot(self):
        self.batch_calls += 1
        if self.failed:
            raise RuntimeError("stale accepted authority")
        base = super()
        self.last_tables = {name: getattr(base, name)() for name in _NAMES}
        return self.last_tables


@pytest.mark.parametrize("consumer", [contract_for, _amr_temporal_report])
def test_batch_equals_all_legacy_projections_without_calling_legacy(consumer):
    baseline = consumer(_Sim())
    sim = BatchSim()
    def forbidden():
        raise AssertionError("individual manifest must not be read")
    for name in _NAMES:
        setattr(sim, name, forbidden)
    assert consumer(sim) == baseline
    assert sim.batch_calls == 1
    assert consumer(sim) == baseline
    assert sim.batch_calls == 2


def test_invocation_snapshot_is_owned_and_next_revision_is_read_fresh():
    class Revision:
        revision = "0"
        def program_manifest_snapshot(self):
            self.rows = {name: [[self.revision]] for name in _NAMES}
            return self.rows
    sim = Revision()
    first = program_manifest_reader(sim)
    sim.rows["program_clock_manifest"][0][0] = "tampered after return"
    sim.revision = "1"
    second = program_manifest_reader(sim)
    assert first("program_clock_manifest") == (("0",),)
    assert second("program_clock_manifest") == (("1",),)
    sim.revision = "0"  # Restoring an earlier revision must not reuse the last report.
    restored = program_manifest_reader(sim)
    assert restored("program_clock_manifest") == first("program_clock_manifest")


@pytest.mark.parametrize("consumer", [contract_for, _amr_temporal_report])
def test_native_validation_error_never_falls_back_or_reuses_previous_report(consumer):
    sim = BatchSim()
    consumer(sim)
    sim.failed = True
    with pytest.raises(RuntimeError, match="stale accepted authority"):
        consumer(sim)
    assert sim.batch_calls == 2


@pytest.mark.parametrize("corrupt", ["missing", "extra", "row", "item", "noncallable"])
def test_malformed_batch_is_rejected_without_legacy_fallback(corrupt):
    class Malformed:
        def program_manifest_snapshot(self):
            rows = {name: [] for name in _NAMES}
            if corrupt == "missing":
                rows.pop("program_clock_manifest")
            elif corrupt == "extra":
                rows["unexpected"] = []
            elif corrupt == "row":
                rows["program_clock_manifest"] = ["bad"]
            else:
                rows["program_clock_manifest"] = [[1]]
            return rows
    sim = Malformed()
    if corrupt == "noncallable":
        sim.program_manifest_snapshot = 1
    def forbidden(name):
        raise AssertionError("fallback must not hide a malformed native batch")
    with pytest.raises((ValueError, TypeError), match="manifest snapshot"):
        program_manifest_reader(sim, fallback=forbidden)


def test_missing_batch_keeps_explicit_legacy_report_defaults():
    reader = program_manifest_reader(object(), fallback=lambda name: [])
    assert reader("program_clock_manifest") == []


def test_present_none_and_failing_descriptor_cannot_downgrade_to_legacy():
    class PresentNone:
        program_manifest_snapshot = None
    class FailingDescriptor:
        calls = 0
        @property
        def program_manifest_snapshot(self):
            self.calls += 1
            raise AttributeError("descriptor validation failed")
    def forbidden(name):
        raise AssertionError("fallback must not hide present batch failure")
    with pytest.raises(TypeError, match="not callable"):
        program_manifest_reader(PresentNone(), fallback=forbidden)
    sim = FailingDescriptor()
    with pytest.raises(AttributeError, match="descriptor validation failed"):
        program_manifest_reader(sim, fallback=forbidden)
    assert sim.calls == 1
