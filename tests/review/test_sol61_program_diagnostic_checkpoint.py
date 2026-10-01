"""Source/runtime archive checks; native transport/engine substitutes are explicit."""

from pathlib import Path
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest

from pops.runtime import _checkpoint_program_diagnostics as codec
from pops.runtime import _checkpoint_resource_budget as budget
from pops.output import _checkpoint_collective as collective
from pops.output._checkpoint_contract import CheckpointResourceBudget
from pops.runtime._multi_layout_executor import _MultiLayoutUniformExecutor
from pops.runtime._runtime_instance import RuntimeInstance

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("program_diagnostics_state", "program_diagnostics_offsets")


def image(rank=0, ranks=1, entries=()):
    # Independent binary construction, not the C++ writer reused as an oracle.
    def word(value):
        return int(value).to_bytes(8, "little")

    return (
        b"POPSDIA1"
        + b"".join(map(word, (64, rank, ranks, len(entries))))
        + b"".join(word(len(name)) + name + word(bits) for name, bits in entries)
    )


def owner(rank=0, ranks=1, capacity=512, *, kind="uniform"):
    base = CheckpointResourceBudget(kind, 2, 100, 100, 1000, 5000, "base-" + kind)
    native = NS(
        _checkpoint_program_diagnostics=lambda: image(rank, ranks),
        _validate_checkpoint_program_diagnostics=lambda data: None,
    )
    result = NS(
        _s=native,
        _checkpoint_program_diagnostic_base_budget=base,
        _checkpoint_program_diagnostic_member_names=NAMES,
        _checkpoint_program_diagnostic_capacity_per_rank=capacity,
        _checkpoint_program_diagnostic_ranks=ranks,
        _checkpoint_program_diagnostic_byte_capacity=None if capacity is None else capacity * ranks,
    )
    result._checkpoint_program_diagnostic_artifact = NS(
        program=NS(_generated_cpp=None), verify=lambda: None
    )
    result._checkpoint_program_diagnostic_inventory = None
    result._checkpoint_resource_budget = (
        base
        if capacity is None
        else budget._diagnostic_capacity_budget(base, NAMES, capacity, ranks)[0]
    )
    return result


def runtime(engine):
    # Actual public method on the real RuntimeInstance type; no native bind claimed.
    result = object.__new__(RuntimeInstance)
    result._executor = engine
    result._checkpoint_resource_budget = engine._checkpoint_resource_budget
    result._consumer_finalize_pending = ()
    result._consumer_recoveries = {}
    return result


def seam(monkeypatch, *, rank=0, ranks=1, rows=None, remote_error=False):
    topology = collective.CheckpointTopology(rank, ranks, object() if ranks > 1 else None)
    calls = []
    monkeypatch.setattr(collective, "checkpoint_topology", lambda _: topology)

    def vote(topology, phase, *, error=None, value=None):
        calls.append((phase, error, value))
        if error is not None:
            raise error
        if remote_error:
            raise RuntimeError("remote preparation refused")
        return (
            rows
            if rows is not None
            else tuple({"rank": r, "value": value, "error": None} for r in range(ranks))
        )

    monkeypatch.setattr(collective, "consensus", vote)
    return calls


def test_literal_inventory_authenticates_real_component_evidence(monkeypatch):
    from tests.python.unit.runtime.test_runtime_planning import _artifact
    from pops.codegen._compiled_artifact import CompiledSimulationArtifact, CompiledLayoutProgram

    source = "\n".join(
        'ctx.record_scalar("field_residual_4.%s", 1);' % name
        for name in (
            "residual_norm",
            "reference_residual_norm",
            "rel_residual",
            "full_residual_evaluations",
            "finite_difference_jvps",
        )
    )
    from pops.codegen import _compiled_artifact
    from pops._platform_contracts import artifact_platform_manifest

    # Source-only Platform metadata is substituted; the real artifact/evidence verifier runs.
    monkeypatch.setattr(
        _compiled_artifact,
        "_common_platform_manifest",
        lambda **kw: artifact_platform_manifest(
            backend=kw["backend"], target=kw["target"], component=kw["blocks"][0].model
        ),
    )
    base = _artifact()
    base.program._generated_cpp = source
    rows = tuple(
        CompiledLayoutProgram(row.layout_id, row.target, row.block_names, row.program)
        for row in base.layout_programs
    )
    artifact = CompiledSimulationArtifact(base.plan, base.program, base.blocks, rows)
    inventory = budget._compiled_diagnostic_inventory(artifact)
    assert len(inventory) == 6  # Five emitted names plus native frontier duration.
    assert budget._diagnostic_inventory_capacity(inventory) == 40 + sum(
        16 + len(n.encode()) for n in inventory
    )
    # Local source provenance can include residence; it must not affect canonical bind content.
    other = _artifact()
    other.program._generated_cpp = '#include "/other/rank/cache.hpp"\n' + source
    other = CompiledSimulationArtifact(
        other.plan, other.program, other.blocks, other.layout_programs
    )
    assert artifact.artifact_identity == other.artifact_identity
    assert artifact._component_evidence == other._component_evidence
    assert artifact._retained_source_evidence != other._retained_source_evidence
    artifact.verify()
    other.verify()
    base.program._generated_cpp = source + '\nctx.record_scalar("forged", 1);'
    with pytest.raises(ValueError):
        artifact.verify()
    with pytest.raises(ValueError):
        budget._compiled_diagnostic_inventory(artifact)


def test_actual_original_field_lowering_inventory_covers_five_records(monkeypatch):
    from tests.review import test_sol61_original_captured_diffusion as physical

    case, layout, *_ = physical.authored(monkeypatch, uniform=True, width=1, order=(0,))
    cpp, _ = physical.physical.emit(case, layout)
    inventory = budget._compiled_diagnostic_inventory(
        NS(program=NS(_generated_cpp=cpp), verify=lambda: None)
    )
    for suffix in (
        "residual_norm",
        "reference_residual_norm",
        "rel_residual",
        "full_residual_evaluations",
        "finite_difference_jvps",
    ):
        assert any(
            name.startswith("field_residual_") and name.endswith("." + suffix) for name in inventory
        )


def test_capacity_and_retained_source_drift_refuse_before_capture(monkeypatch):
    seam(monkeypatch)
    engine = owner()
    engine._checkpoint_program_diagnostic_byte_capacity += 1
    with pytest.raises(RuntimeError, match="byte capacity changed"):
        codec.capture_checkpoint_program_diagnostics(engine, {})
    engine = owner()
    engine._checkpoint_program_diagnostic_artifact.verify = lambda: (_ for _ in ()).throw(
        ValueError("retained component evidence changed")
    )
    engine._s._checkpoint_program_diagnostics = lambda: pytest.fail("drift reached Native")
    with pytest.raises(ValueError, match="component evidence changed"):
        codec.capture_checkpoint_program_diagnostics(engine, {})


def test_explicit_configuration_enables_absent_source_capacity(monkeypatch):
    seam(monkeypatch)
    engine = owner(capacity=None)
    with pytest.raises(ValueError, match="configure_checkpoint_diagnostics"):
        codec.capture_checkpoint_program_diagnostics(engine, {})
    live = runtime(engine)
    live.configure_checkpoint_diagnostics(capacity_per_rank=1024)
    payload = {}
    codec.capture_checkpoint_program_diagnostics(engine, payload)
    assert codec.prepare_checkpoint_program_diagnostics(engine, payload) == image()


@pytest.mark.parametrize("text", (None, "ctx.record_scalar(dynamic_name, 1);"))
def test_absent_or_dynamic_source_requires_explicit_capacity(text):
    artifact = NS(program=NS(_generated_cpp=text), verify=lambda: None)
    assert budget._compiled_diagnostic_inventory(artifact) is None
    assert budget._diagnostic_inventory_capacity(None) is None


@pytest.mark.parametrize("bad", (True, False, 0, -1, 1.0, "512", sys.maxsize + 1))
def test_public_configuration_votes_invalid_before_native_access(monkeypatch, bad):
    calls = seam(monkeypatch, ranks=2)
    engine = owner(ranks=2)
    engine._s._checkpoint_program_diagnostics = lambda: pytest.fail(
        "native called before invalid proposal vote"
    )
    live = runtime(engine)
    previous = live._checkpoint_resource_budget
    with pytest.raises((TypeError, ValueError, OverflowError)):
        live.configure_checkpoint_diagnostics(capacity_per_rank=bad)
    assert len(calls) == 1 and calls[0][1] is not None
    assert live._checkpoint_resource_budget == previous


def test_public_configuration_overflow_remote_mismatch_attempt_and_replace(monkeypatch):
    engine = owner(ranks=2)
    live = runtime(engine)
    previous = live._checkpoint_resource_budget
    calls = seam(monkeypatch, ranks=2)
    with pytest.raises(OverflowError):
        live.configure_checkpoint_diagnostics(capacity_per_rank=sys.maxsize)
    assert len(calls) == 1
    seam(monkeypatch, ranks=2, remote_error=True)
    with pytest.raises(RuntimeError, match="remote preparation"):
        live.configure_checkpoint_diagnostics(capacity_per_rank=1024)
    assert live._checkpoint_resource_budget == previous
    seam(
        monkeypatch,
        ranks=2,
        rows=({"value": {"capacity_per_rank": 1024}}, {"value": {"capacity_per_rank": 2048}}),
    )
    with pytest.raises(ValueError, match="differs across ranks"):
        live.configure_checkpoint_diagnostics(capacity_per_rank=1024)
    assert live._checkpoint_resource_budget == previous
    seam(monkeypatch, ranks=2)
    engine._s._checkpoint_program_diagnostics = lambda: (_ for _ in ()).throw(
        RuntimeError("pending attempt")
    )
    with pytest.raises(RuntimeError, match="pending attempt"):
        live.configure_checkpoint_diagnostics(capacity_per_rank=1024)
    assert live._checkpoint_resource_budget == previous
    engine._s._checkpoint_program_diagnostics = lambda: image(0, 2)
    live.configure_checkpoint_diagnostics(capacity_per_rank=1024)
    assert live._checkpoint_resource_budget is engine._checkpoint_resource_budget
    assert engine._checkpoint_program_diagnostic_capacity_per_rank == 1024
    assert engine._checkpoint_program_diagnostic_byte_capacity == 2048
    assert live._checkpoint_resource_budget.authority != previous.authority
    live.configure_checkpoint_diagnostics(capacity_per_rank=2048)
    assert engine._checkpoint_program_diagnostic_byte_capacity == 4096


@pytest.mark.parametrize("kind", ("uniform", "amr"))
def test_multilayout_configuration_is_one_joined_authority(monkeypatch, kind):
    seam(monkeypatch)
    executor = object.__new__(_MultiLayoutUniformExecutor)
    executor._engines = {"first": owner(kind=kind), "second": owner(kind=kind)}
    executor._active_transfer_generation = None
    executor._mapping_evaluations = {"map": 0}
    executor._checkpoint_resource_budget = executor._engines["first"]._checkpoint_resource_budget
    live = runtime(executor)
    live._install_plan = NS(
        artifact=NS(artifact_identity=NS(token="artifact")), bind_identity=NS(token="bind")
    )
    aggregates = []

    def aggregate(rows, **kw):
        aggregates.append((rows, kw))
        return CheckpointResourceBudget(
            "multi_layout_" + kind, 2, 100, 1000, 5000, 10000, kw["authority"]
        )

    monkeypatch.setattr(budget, "aggregate_checkpoint_resource_budgets", aggregate)
    live.configure_checkpoint_diagnostics(capacity_per_rank=2048)
    assert len(aggregates) == 1
    assert aggregates[0][1]["layout_ids"] == ("first", "second")
    assert all(
        child._checkpoint_program_diagnostic_capacity_per_rank == 2048
        for child in executor._engines.values()
    )
    assert live._checkpoint_resource_budget is executor._checkpoint_resource_budget
    previous = live._checkpoint_resource_budget
    executor._active_transfer_generation = 3
    with pytest.raises(RuntimeError, match="mapping attempt"):
        live.configure_checkpoint_diagnostics(capacity_per_rank=4096)
    assert live._checkpoint_resource_budget is previous


def test_capture_and_preflight_rank_local_bits_legacy_and_perrank_bounds(monkeypatch):
    from pops import _native_collectives

    calls = seam(monkeypatch, rank=1, ranks=2)
    records = (
        image(0, 2, ((b"opaque\0key", 0x8000000000000000),)),
        image(1, 2, ((b"different", 0x7FF8000000001234),)),
    )
    monkeypatch.setattr(_native_collectives, "allgather_bytes", lambda _, local: records)
    engine = owner(rank=1, ranks=2)
    payload = {}
    codec.capture_checkpoint_program_diagnostics(engine, payload)
    assert len(calls) == 2
    checked = []
    engine._s._validate_checkpoint_program_diagnostics = checked.append
    assert codec.prepare_checkpoint_program_diagnostics(engine, payload) == records[1]
    assert codec.prepare_checkpoint_program_diagnostics(engine, {}) == b""
    assert checked == [records[1], b""]
    oversized = (image(0, 2, ((b"x" * 30, 1),)), image(1, 2))
    assert len(b"".join(oversized)) < 70 * 2 and len(oversized[0]) > 70
    engine = owner(rank=1, ranks=2, capacity=70)
    payload = {
        NAMES[0]: np.frombuffer(b"".join(oversized), dtype=np.uint8),
        NAMES[1]: np.array([0, len(oversized[0]), sum(map(len, oversized))], dtype=np.int64),
    }
    monkeypatch.setattr(_native_collectives, "allgather_bytes", lambda _, local: oversized)
    with pytest.raises(ValueError, match="configure_checkpoint_diagnostics"):
        codec.prepare_checkpoint_program_diagnostics(engine, payload)
    with pytest.raises(ValueError, match="configure_checkpoint_diagnostics"):
        codec.capture_checkpoint_program_diagnostics(engine, {})


@pytest.mark.parametrize("refusal", ("oversized", "unconfigured", "peer"))
def test_capture_capacity_votes_before_transport(monkeypatch, refusal):
    from pops import _native_collectives

    calls = seam(monkeypatch, ranks=2, remote_error=refusal == "peer")
    engine = owner(ranks=2, capacity=None if refusal == "unconfigured" else 60)
    encoded = []

    def capture():
        encoded.append(True)
        return image(0, 2, ((b"large" * 20, 1),)) if refusal == "oversized" else image(0, 2)

    engine._s._checkpoint_program_diagnostics = capture
    monkeypatch.setattr(
        _native_collectives, "allgather_bytes", lambda *_: pytest.fail("transport before admission")
    )
    payload = {"preserved": object()}
    original = dict(payload)
    with pytest.raises((ValueError, RuntimeError)):
        codec.capture_checkpoint_program_diagnostics(engine, payload)
    assert payload == original
    assert len(calls) == 1 and calls[0][0] == "accepted Program diagnostic preparation"
    assert encoded == ([] if refusal == "unconfigured" else [True])


def test_array_preflight_copies_only_headers_then_admitted_owner_body(monkeypatch):
    seam(monkeypatch, ranks=2)
    copies = []

    class TracedArray(np.ndarray):
        def tobytes(self, *args, **kwargs):
            copies.append(len(self))
            return super().tobytes(*args, **kwargs)

    asarray = np.asarray
    monkeypatch.setattr(
        codec.np, "asarray", lambda data, *a, **kw: asarray(data, *a, **kw).view(TracedArray)
    )
    records = (image(0, 2, ((b"x" * 30, 1),)), image(1, 2))
    payload = {
        NAMES[0]: np.frombuffer(b"".join(records), dtype=np.uint8).view(TracedArray),
        NAMES[1]: np.array([0, len(records[0]), sum(map(len, records))], dtype=np.int64),
    }
    # General archive preflight may inspect headers before owner admission, never full bodies.
    assert codec.validate_checkpoint_program_diagnostic_arrays(payload)
    assert copies == [40, 40]
    copies.clear()
    engine = owner(ranks=2, capacity=70)
    engine._s._validate_checkpoint_program_diagnostics = lambda _: pytest.fail("oversized native input")
    with pytest.raises(ValueError, match="configure_checkpoint_diagnostics"):
        codec.prepare_checkpoint_program_diagnostics(engine, payload)
    assert copies == []
    checked = []
    engine = owner(ranks=2, capacity=100)
    engine._s._validate_checkpoint_program_diagnostics = checked.append
    assert codec.prepare_checkpoint_program_diagnostics(engine, payload) == records[0]
    assert copies == [40, 40, len(records[0])] and checked == [records[0]]


@pytest.mark.parametrize("attack", ("partial", "dtype", "offset", "empty", "rankdup", "rankcount"))
def test_arrays_refuse_before_native_or_publication(monkeypatch, attack):
    seam(monkeypatch, ranks=2)
    records = (image(0, 2), image(1, 2))
    payload = {
        NAMES[0]: np.frombuffer(b"".join(records), dtype=np.uint8),
        NAMES[1]: np.array([0, 40, 80], dtype=np.int64),
    }
    if attack == "partial":
        del payload[NAMES[1]]
    if attack == "dtype":
        payload[NAMES[0]] = payload[NAMES[0]].astype(np.int8)
    if attack == "offset":
        payload[NAMES[1]][1] = 41
    if attack == "empty":
        payload[NAMES[1]][1] = 0
    if attack == "rankdup":
        payload[NAMES[0]] = np.frombuffer(records[0] * 2, dtype=np.uint8)
    if attack == "rankcount":
        payload = {
            NAMES[0]: np.frombuffer(image(), dtype=np.uint8),
            NAMES[1]: np.array([0, 40], dtype=np.int64),
        }
    engine = owner(ranks=2)
    engine._s._validate_checkpoint_program_diagnostics = lambda _: pytest.fail(
        "invalid shard reached native"
    )
    with pytest.raises(ValueError):
        codec.prepare_checkpoint_program_diagnostics(engine, payload)


def test_restore_order_and_archive_extension_do_not_replace_amr_flux_contract():
    uniform = (ROOT / "python/pops/runtime/_system_io.py").read_text()
    amr = (ROOT / "python/pops/runtime/_amr_checkpoint_v3.py").read_text()
    for source, target in ((uniform, "self"), (amr, "owner")):
        assert source.index(
            "capture_checkpoint_program_diagnostics(" + target + ","
        ) < source.index(
            "seal_checkpoint_payload(" + target + ",",
            source.index("capture_checkpoint_program_diagnostics(" + target + ","),
        )
        last = source.index(
            "._restore_checkpoint_program_diagnostics(prepared.program_diagnostic_checkpoint)"
        )
        assert source.rfind("._restore_checkpoint_program_exchanges(", 0, last) >= 0
        assert "prepare_checkpoint_program_diagnostics(" + target + "," in source
    assert amr.index("sim.restore_checkpoint_accepted_state(program_state)") < amr.index(
        "sim._restore_checkpoint_program_diagnostics("
    )
