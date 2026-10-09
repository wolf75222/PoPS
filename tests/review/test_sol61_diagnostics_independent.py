"""Independent source/host reception of e86d07cc; no native runtime or MPI claim."""

from concurrent.futures import ThreadPoolExecutor
import ast
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace as NS
import subprocess
import sys

import numpy as np
import pytest

from pops.output import _checkpoint_collective as collective
from pops.output._checkpoint_contract import CheckpointResourceBudget
from pops.runtime import _checkpoint_program_diagnostics as diagnostics
from pops.runtime import _checkpoint_resource_budget as resources
from pops.runtime._runtime_instance import RuntimeInstance

ROOT = Path(__file__).resolve().parents[2]
FROZEN = "e86d07cc77c9e4b43868dae46feb1cbe5bc364b3"
KEYS = ("program_diagnostics_state", "program_diagnostics_offsets")


def image(rank, ranks=2, name=b"rank-owned", bits=0x8000000000000000):
    return (
        b"POPSDIA1"
        + b"".join(x.to_bytes(8, "little") for x in (64, rank, ranks, 1, len(name)))
        + name
        + bits.to_bytes(8, "little")
    )


class NativeAdapter:
    def __init__(self, rank):
        self.rank = rank
        self.calls = 0
        self.validated = []
        self.pending = False

    def _checkpoint_program_diagnostics(self):
        self.calls += 1
        if self.pending:
            raise RuntimeError("host accepted-boundary refusal")
        return image(self.rank)

    def _validate_checkpoint_program_diagnostics(self, data):
        self.validated.append(data)


def owner(rank, capacity=96):
    base = CheckpointResourceBudget("uniform", 2, 64, 256, 1024, 4096, "independent-base")
    result = NS(
        rank=rank,
        _s=NativeAdapter(rank),
        _checkpoint_program_diagnostic_base_budget=base,
        _checkpoint_program_diagnostic_member_names=KEYS,
        _checkpoint_program_diagnostic_capacity_per_rank=capacity,
        _checkpoint_program_diagnostic_ranks=2,
        _checkpoint_program_diagnostic_byte_capacity=2 * capacity,
        _checkpoint_program_diagnostic_artifact=NS(program=None, verify=lambda: None),
        _checkpoint_program_diagnostic_inventory=None,
    )
    result._checkpoint_resource_budget = resources._diagnostic_capacity_budget(
        base, KEYS, capacity, 2
    )[0]
    return result


def runtime(rank):
    value = object.__new__(RuntimeInstance)
    value._executor = owner(rank)
    value._checkpoint_resource_budget = value._executor._checkpoint_resource_budget
    value._consumer_finalize_pending = ()
    value._consumer_recoveries = {}
    return value


class TwoRanks:
    """Transport substitute only: real consensus validates/votes these envelopes."""

    def __init__(self):
        self.barrier = Barrier(2, timeout=10)
        self.rows = [None, None]
        self.history = []

    def gather(self, rank, row):
        self.rows[rank] = row
        self.barrier.wait()
        result = tuple(self.rows)
        if rank == 0:
            self.history.append(result)
        self.barrier.wait()
        return result

    def install(self, monkeypatch):
        def topology(x):
            rank = x._executor.rank if isinstance(x, RuntimeInstance) else x.rank
            return collective.CheckpointTopology(rank, 2, rank)

        monkeypatch.setattr(collective, "checkpoint_topology", topology)
        monkeypatch.setattr(collective, "allgather_value", self.gather)


def configure_pair(values, capacities):
    def call(index):
        try:
            values[index].configure_checkpoint_diagnostics(capacity_per_rank=capacities[index])
        except BaseException as error:
            return error
        return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(call, (0, 1)))


@pytest.mark.parametrize("bad", [True, 0, -1, 2**63, 2**64 - 1])
def test_invalid_rank_votes_before_any_capacity_publication(monkeypatch, bad):
    transport = TwoRanks()
    transport.install(monkeypatch)
    pair = [runtime(0), runtime(1)]
    previous = [v._checkpoint_resource_budget for v in pair]
    errors = configure_pair(pair, (bad, 512))
    expected_family = TypeError if type(bad) is not int or bad <= 0 else RuntimeError
    assert all(type(e) is expected_family for e in errors)
    assert str(errors[0]) == str(errors[1])
    assert len(transport.history) == 1
    assert transport.history[0][0]["error"] is not None
    if type(bad) is int and bad > sys.maxsize:
        assert "OverflowError" in str(errors[0])
    assert transport.history[0][1]["error"] is None
    assert pair[0]._executor._s.calls == 0
    assert [v._checkpoint_resource_budget for v in pair] == previous
    assert all(v._executor._checkpoint_program_diagnostic_capacity_per_rank == 96 for v in pair)


@pytest.mark.parametrize("attack", ["proposal", "pending", "consumer", "authority"])
def test_collective_preparation_refuses_atomically(monkeypatch, attack):
    transport = TwoRanks()
    transport.install(monkeypatch)
    pair = [runtime(0), runtime(1)]
    capacities = [512, 512]
    if attack == "proposal":
        capacities[1] = 513
    elif attack == "pending":
        pair[1]._executor._s.pending = True
    elif attack == "consumer":
        pair[1]._consumer_recoveries = {"unfinished": object()}
    else:
        pair[1]._executor._checkpoint_program_diagnostic_byte_capacity += 1
    previous = [v._checkpoint_resource_budget for v in pair]
    errors = configure_pair(pair, capacities)
    assert all(isinstance(e, (RuntimeError, ValueError)) for e in errors)
    assert str(errors[0]) == str(errors[1])
    assert [v._checkpoint_resource_budget for v in pair] == previous
    assert all(v._executor._checkpoint_program_diagnostic_capacity_per_rank == 96 for v in pair)
    assert len(transport.history) == 1


def test_actual_consensus_positive_and_exact_capacity_arithmetic(monkeypatch):
    transport = TwoRanks()
    transport.install(monkeypatch)
    pair = [runtime(0), runtime(1)]
    assert configure_pair(pair, (512, 512)) == [None, None]
    assert pair[0]._checkpoint_resource_budget == pair[1]._checkpoint_resource_budget
    for v in pair:
        assert v._checkpoint_resource_budget is v._executor._checkpoint_resource_budget
        assert v._executor._checkpoint_program_diagnostic_byte_capacity == 1024
        assert v._checkpoint_resource_budget.max_uncompressed_bytes == 1024 + 1024 + 24
    assert all(row["error"] is None for row in transport.history[0])


@pytest.mark.parametrize("refuse_second_child", [False, True])
def test_multilayout_uses_real_joined_budget_and_atomic_child_preparation(
    monkeypatch, refuse_second_child
):
    from pops.runtime._multi_layout_executor import _MultiLayoutUniformExecutor

    transport = TwoRanks()
    transport.install(monkeypatch)
    pair = [runtime(0), runtime(1)]
    for rank, live in enumerate(pair):
        engine = object.__new__(_MultiLayoutUniformExecutor)
        engine.rank = rank
        engine._engines = {"layout-a": owner(rank), "layout-b": owner(rank)}
        engine._active_transfer_generation = None
        engine._mapping_evaluations = {"transfer": 0}
        live._executor = engine
        live._install_plan = NS(
            artifact=NS(artifact_identity=NS(token="artifact"), plan=NS(consumer_graph=None)),
            bind_identity=NS(token="bind"),
        )
        engine._checkpoint_resource_budget = resources.aggregate_checkpoint_resource_budgets(
            tuple(x._checkpoint_resource_budget for x in engine._engines.values()),
            authority="initial-joined",
            install_plan=live._install_plan,
            layout_ids=tuple(engine._engines),
            mapping_ids=tuple(engine._mapping_evaluations),
        )
        live._checkpoint_resource_budget = engine._checkpoint_resource_budget
    old = [v._checkpoint_resource_budget for v in pair]
    if refuse_second_child:
        pair[1]._executor._engines["layout-b"]._s.pending = True
    errors = configure_pair(pair, (512, 512))
    if refuse_second_child:
        assert all(type(error) is RuntimeError for error in errors)
        assert [v._checkpoint_resource_budget for v in pair] == old
        assert all(
            x._checkpoint_program_diagnostic_capacity_per_rank == 96
            for v in pair
            for x in v._executor._engines.values()
        )
    else:
        assert errors == [None, None]
        assert pair[0]._checkpoint_resource_budget == pair[1]._checkpoint_resource_budget
        for v in pair:
            children = tuple(v._executor._engines.values())
            assert all(x._checkpoint_program_diagnostic_byte_capacity == 1024 for x in children)
            assert v._checkpoint_resource_budget.runtime_kind == "multi_layout_uniform"
            assert v._checkpoint_resource_budget.max_uncompressed_bytes > sum(
                x._checkpoint_resource_budget.max_archive_bytes for x in children
            )
            assert v._checkpoint_resource_budget is v._executor._checkpoint_resource_budget


@pytest.mark.parametrize("attack", ["rank", "cardinality", "offset", "partial", "capacity"])
def test_archive_offsets_and_rank_capacity_refuse_before_native_validate(monkeypatch, attack):
    monkeypatch.setattr(
        collective,
        "checkpoint_topology",
        lambda x: collective.CheckpointTopology(x.rank, 2, x.rank),
    )
    live = owner(1)
    first, second = image(0), image(1)
    if attack == "rank":
        second = image(0)
    data = {
        KEYS[0]: np.frombuffer(first + second, dtype=np.uint8).copy(),
        KEYS[1]: np.array([0, len(first), len(first) + len(second)], dtype=np.int64),
    }
    if attack == "cardinality":
        only = image(0, ranks=1)
        data = {
            KEYS[0]: np.frombuffer(only, dtype=np.uint8).copy(),
            KEYS[1]: np.array([0, len(only)], dtype=np.int64),
        }
    elif attack == "offset":
        data[KEYS[1]][1] = 0
    elif attack == "partial":
        del data[KEYS[1]]
    elif attack == "capacity":
        # Authenticated chosen capacity is smaller than one real opaque record.
        live = owner(1, capacity=40)
    with pytest.raises(ValueError):
        diagnostics.prepare_checkpoint_program_diagnostics(live, data)
    assert live._s.validated == []


def test_rank_local_bits_selected_and_legacy_absence_clear(monkeypatch):
    monkeypatch.setattr(
        collective,
        "checkpoint_topology",
        lambda x: collective.CheckpointTopology(x.rank, 2, x.rank),
    )
    first, second = image(0, bits=1), image(1, name=b"\x00\xff", bits=0x7FF8000000000042)
    data = {
        KEYS[0]: np.frombuffer(first + second, dtype=np.uint8),
        KEYS[1]: np.array([0, len(first), len(first) + len(second)], dtype=np.int64),
    }
    for rank, expected in enumerate((first, second)):
        live = owner(rank)
        assert diagnostics.prepare_checkpoint_program_diagnostics(live, data) == expected
        assert diagnostics.prepare_checkpoint_program_diagnostics(live, {}) == b""
        assert live._s.validated == [expected, b""]


def test_e86_historical_oversize_is_refused_only_after_byte_transport(monkeypatch):
    """Records a resource-order defect, not a positive capacity qualification."""
    from pops import _native_collectives

    transport = TwoRanks()
    transport.install(monkeypatch)
    byte_transport = TwoRanks()
    monkeypatch.setattr(_native_collectives, "allgather_bytes", byte_transport.gather)
    owners = [owner(0, capacity=40), owner(1, capacity=40)]
    payloads = [{}, {}]
    historical = _historical_helper()

    def capture(rank):
        try:
            historical["capture_checkpoint_program_diagnostics"](owners[rank], payloads[rank])
        except ValueError as error:
            return str(error)
        raise AssertionError("oversized diagnostics were accepted")

    with ThreadPoolExecutor(max_workers=2) as pool:
        errors = list(pool.map(capture, (0, 1)))
    assert errors[0] == errors[1]
    assert "exceeds its chosen resource capacity" in errors[0]
    assert payloads == [{}, {}]
    assert len(byte_transport.history) == 1
    assert all(len(part) > 40 for part in byte_transport.history[0])
    assert len(transport.history) == 2
    assert all(row["error"] is None for row in transport.history[0])
    assert all(row["error"] is not None for row in transport.history[1])


def _historical_helper():
    source = subprocess.check_output(
        ["git", "show", FROZEN + ":python/pops/runtime/_checkpoint_program_diagnostics.py"],
        cwd=ROOT,
        text=True,
    )
    namespace = {}
    exec(compile(source, "frozen-e86-checkpoint-program-diagnostics.py", "exec"), namespace)
    return namespace


def test_e86_historical_preflight_copies_full_rank_slices_before_capacity_guard(monkeypatch):
    monkeypatch.setattr(
        collective,
        "checkpoint_topology",
        lambda x: collective.CheckpointTopology(x.rank, 2, x.rank),
    )
    historical = _historical_helper()
    seen = []
    exact = historical["_exact_native_image"]

    def inspect(data, **authority):
        seen.append((type(data), len(data)))
        return exact(data, **authority)

    historical["_exact_native_image"] = inspect
    first, second = image(0), image(1)
    data = {
        KEYS[0]: np.frombuffer(first + second, dtype=np.uint8),
        KEYS[1]: np.array([0, len(first), len(first) + len(second)], dtype=np.int64),
    }
    live = owner(1, capacity=40)
    with pytest.raises(ValueError, match="exceeds its chosen resource capacity"):
        historical["prepare_checkpoint_program_diagnostics"](live, data)
    assert seen == [(bytes, len(first)), (bytes, len(second))]
    assert all(size > 40 for _, size in seen)
    assert live._s.validated == []


@pytest.mark.parametrize("oversized_ranks", [(0,), (0, 1)])
def test_521_current_oversize_votes_before_byte_transport(monkeypatch, oversized_ranks):
    from pops import _native_collectives

    transport = TwoRanks()
    transport.install(monkeypatch)
    owners = [owner(rank, capacity=40 if rank in oversized_ranks else 96) for rank in (0, 1)]
    payloads = [{}, {}]

    def forbidden_transport(*unused):
        raise AssertionError("capacity refusal must precede byte transport")

    monkeypatch.setattr(_native_collectives, "allgather_bytes", forbidden_transport)

    def capture(rank):
        try:
            diagnostics.capture_checkpoint_program_diagnostics(owners[rank], payloads[rank])
        except ValueError as error:
            return str(error)
        raise AssertionError("oversized diagnostics were accepted")

    with ThreadPoolExecutor(max_workers=2) as pool:
        errors = list(pool.map(capture, (0, 1)))
    assert errors[0] == errors[1]
    assert "exceeds its chosen resource capacity" in errors[0]
    assert payloads == [{}, {}]
    assert len(transport.history) == 1
    assert tuple(row["rank"] for row in transport.history[0] if row["error"] is not None) == (
        oversized_ranks
    )
    assert all(row["value"] is None for row in transport.history[0])


def test_521_current_positive_capture_preserves_exact_rank_images(monkeypatch):
    from pops import _native_collectives

    transport, byte_transport = TwoRanks(), TwoRanks()
    transport.install(monkeypatch)
    monkeypatch.setattr(_native_collectives, "allgather_bytes", byte_transport.gather)
    owners, payloads = [owner(0), owner(1)], [{}, {}]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(
            pool.map(
                lambda rank: diagnostics.capture_checkpoint_program_diagnostics(
                    owners[rank], payloads[rank]
                ),
                (0, 1),
            )
        )
    first, second = image(0), image(1)
    for payload in payloads:
        assert payload[KEYS[0]].tobytes() == first + second
        assert payload[KEYS[1]].dtype == np.dtype(np.int64)
        assert payload[KEYS[1]].tolist() == [0, len(first), len(first) + len(second)]
    assert len(transport.history) == 2 and len(byte_transport.history) == 1
    assert all(row["error"] is None for vote in transport.history for row in vote)


@pytest.mark.parametrize("attack", ["perrank", "aggregate"])
def test_521_current_capacity_refuses_before_header_or_body_copy(monkeypatch, attack):
    monkeypatch.setattr(
        collective,
        "checkpoint_topology",
        lambda x: collective.CheckpointTopology(x.rank, 2, x.rank),
    )
    first, second = image(0), image(1)
    data = {
        KEYS[0]: np.frombuffer(first + second, dtype=np.uint8),
        KEYS[1]: np.array([0, len(first), len(first) + len(second)], dtype=np.int64),
    }
    live = owner(1, capacity=40)
    copied_images = []

    def forbidden_header(data, **authority):
        copied_images.append(data)
        raise AssertionError("capacity refusal must precede slice conversion")

    monkeypatch.setattr(diagnostics, "_exact_native_image", forbidden_header)
    if attack == "perrank":
        with pytest.raises(ValueError, match="exceeds its chosen resource capacity"):
            diagnostics.prepare_checkpoint_program_diagnostics(live, data)
        assert live._s.validated == []
    else:
        with pytest.raises(ValueError, match="exceeds its chosen resource capacity"):
            diagnostics.validate_checkpoint_program_diagnostic_arrays(data, capacity=96, bound=40)
    assert copied_images == []


def test_521_current_only_fixed_headers_copied_before_bounded_selection(monkeypatch):
    monkeypatch.setattr(
        collective,
        "checkpoint_topology",
        lambda x: collective.CheckpointTopology(x.rank, 2, x.rank),
    )
    first, second = image(0), image(1, name=b"\x00\xff", bits=0x7FF8000000000042)
    data = {
        KEYS[0]: np.frombuffer(first + second, dtype=np.uint8),
        KEYS[1]: np.array([0, len(first), len(first) + len(second)], dtype=np.int64),
    }
    exact = diagnostics._exact_native_image
    copied_headers = []

    def inspect(header, **authority):
        copied_headers.append((type(header), len(header), authority))
        return exact(header, **authority)

    monkeypatch.setattr(diagnostics, "_exact_native_image", inspect)
    assert diagnostics.validate_checkpoint_program_diagnostic_arrays(data)
    assert [length for _, length, _ in copied_headers] == [40, 40]
    copied_headers.clear()
    live = owner(1)
    assert diagnostics.prepare_checkpoint_program_diagnostics(live, data) == second
    assert live._s.validated == [second]
    assert copied_headers == [
        (bytes, 40, {"rank": 0, "ranks": 2}),
        (bytes, 40, {"rank": 1, "ranks": 2}),
    ]


def test_521_helper_exact_authored_blob():
    relative = "python/pops/runtime/_checkpoint_program_diagnostics.py"
    authored = subprocess.check_output(
        ["git", "show", "521cca58a9719dc0843065ac0af0bfaf9aa242fe:" + relative], cwd=ROOT
    )
    assert (ROOT / relative).read_bytes() == authored


def _restore_definition(source, family):
    marker = "template <int Dim>\nvoid " + family + "<Dim>::restore_checkpoint_program_diagnostics("
    start = source.index(marker)
    body = source.index("{", start)
    depth = 1
    end = body + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.mark.parametrize("real_type", ["double", "float"])
@pytest.mark.parametrize("ndebug", [False, True])
def test_real_codec_and_frozen_restore_bodies_host(tmp_path, real_type, ndebug):
    bodies = []
    for path, family in (
        ("src/runtime/system/system_program.cpp", "System"),
        ("src/runtime/amr/amr_system.cpp", "AmrSystem"),
    ):
        current = _restore_definition((ROOT / path).read_text(), family)
        frozen = _restore_definition(
            subprocess.check_output(["git", "show", FROZEN + ":" + path], cwd=ROOT, text=True),
            family,
        )
        assert current == frozen  # Pin reviewed body, permit unrelated additive changes.
        bodies.append(current)
    host = (ROOT / "tests/review/sol61_diagnostics_independent_host.cpp").read_text()
    source = tmp_path / "host.cpp"
    source.write_text(host.replace("// SOURCE_EXTRACTED_BODIES", "\n".join(bodies)))
    command = [
        "c++",
        "-std=c++20",
        "-O0",
        "-I" + str(ROOT / "include"),
        "-DPOPS_REAL_TYPE=" + real_type,
    ]
    if ndebug:
        command.append("-DNDEBUG")
    command += [str(source), "-o", str(tmp_path / "host")]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    run = subprocess.run(
        [str(tmp_path / "host")], check=True, capture_output=True, text=True, timeout=10
    )
    assert "independent assertions=132 " in run.stdout


def test_retained_source_authority_is_local_and_verified_outside_bind(monkeypatch):
    # Existing source-only artifact constructor, real verifier; no author review helper.
    from tests.python.unit.runtime.test_runtime_planning import _artifact
    from pops.codegen import _compiled_artifact as compiled
    from pops._platform_contracts import artifact_platform_manifest

    monkeypatch.setattr(
        compiled,
        "_common_platform_manifest",
        lambda **kw: artifact_platform_manifest(
            backend=kw["backend"], target=kw["target"], component=kw["blocks"][0].model
        ),
    )
    base = _artifact(names=("left", "right"))
    base.program._generated_cpp = 'ctx.record_scalar("before", 0);'
    first = compiled.CompiledSimulationArtifact(
        base.plan, base.program, base.blocks, base.layout_programs
    )
    frozen_evidence = first._retained_source_evidence
    first.verify()
    base.program._generated_cpp += '\n#include "/rank1/residence.hpp"'
    with pytest.raises(ValueError, match="retained"):
        first.verify()
    second = compiled.CompiledSimulationArtifact(
        base.plan, base.program, base.blocks, base.layout_programs
    )
    second.verify()
    assert first.artifact_identity == second.artifact_identity
    assert first._component_evidence == second._component_evidence
    assert frozen_evidence != second._retained_source_evidence
    # Removing a different component's retained text also invalidates this frozen evidence.
    base.blocks[1].model._generated_cpp = "fresh unsealed text"
    with pytest.raises(ValueError, match="retained"):
        second.verify()
    with pytest.raises(ValueError):
        resources._compiled_diagnostic_inventory(second)
    base.blocks[1].model._generated_cpp = None
    base.program._generated_cpp = None
    with pytest.raises(ValueError, match="retained"):
        second.verify()


def test_source_restart_order_keeps_one_diagnostic_publication_and_snapshot_join():
    for relative, function in (
        ("python/pops/runtime/_system_io.py", "_apply_checkpoint_restart"),
        ("python/pops/runtime/_amr_checkpoint_v3.py", "apply_v3"),
    ):
        tree = ast.parse((ROOT / relative).read_text())
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == function
        )
        calls = sorted(
            (node.lineno, node.func.attr)
            for node in ast.walk(method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        )
        diagnostic = [
            line for line, name in calls if name == "_restore_checkpoint_program_diagnostics"
        ]
        assert len(diagnostic) == 1
        if function == "_apply_checkpoint_restart":
            exchange = next(
                line for line, name in calls if name == "_restore_checkpoint_program_exchanges"
            )
            assert diagnostic[0] > exchange
        else:
            history = next(
                node.lineno
                for node in ast.walk(method)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "stage_history_flux_snapshots"
            )
            assert diagnostic[0] > history
    runtime_state = (ROOT / "include/pops/runtime/program/program_runtime_state.hpp").read_text()
    assert "diagnostics_(accepted.diagnostics_)" in runtime_state
    assert "diagnostics_.swap(prepared.diagnostics_);" in runtime_state
    for relative, prepare, publish in (
        (
            "src/runtime/system/system_impl.hpp",
            "owner.program_.prepare_accepted_restore(program)",
            "owner.program_.publish_prepared_accepted_restore(std::move(prepared_program_restore))",
        ),
        (
            "src/runtime/amr/amr_system.cpp",
            "owner.program.prepare_accepted_restore(snapshot.program)",
            "owner.program.publish_prepared_accepted_restore(std::move(*prepared.program_restore))",
        ),
    ):
        source = (ROOT / relative).read_text()
        assert prepare in source and publish in source
