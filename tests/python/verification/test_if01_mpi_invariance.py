"""IF-01 MPI decomposition invariance (in-memory exact fields; no live MPI)."""
from __future__ import annotations

import ast
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.campaign import CampaignJob, CampaignRequest
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.provenance import RUN_FIELDS
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "infrastructure" / "mpi_invariance"
TR01_EXACT = (
    REPO_ROOT / "verification" / "cases" / "transport" / "advection_sine" / "exact.py"
)
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
PLACEMENTS = ("one_block", "two_block", "1x4", "4x1")
ONE_BLOCK_EDGES = (0.0, 1.0)
TWO_BLOCK_EDGES = (0.0, 0.5, 1.0)
FOUR_BLOCK_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)
FOUR_BLOCK_SPLITS = (0.25, 0.5, 0.75)
N_CELLS = 32
T = 0.25
ORDERS_REASON = "exact-field identity / no live MPI"



def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _pops_run_call_owners(source: str) -> list[str]:
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    owners: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "run"
            and isinstance(func.value, ast.Name)
            and func.value.id == "pops"
        ):
            continue
        current: ast.AST | None = node
        owner = "<module>"
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                owner = current.name
                break
            current = parents.get(current)
        owners.append(owner)
    return owners


def test_exact_loads_tr01_via_load_sibling_module():
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "advection_sine" in text
    assert "from exact import" not in text
    exact = _load_case_module("exact")
    tr01 = load_sibling_module(TR01_EXACT)
    centers = exact.cell_centers(N_CELLS)
    np.testing.assert_array_equal(
        exact.exact_sine(centers, T),
        tr01.exact_sine(centers, T),
    )


def test_one_and_two_block_placements_cover_the_unit_interval():
    exact = _load_case_module("exact")
    assert tuple(exact.PLACEMENTS) == PLACEMENTS
    assert tuple(exact.placement_edges("one_block")) == ONE_BLOCK_EDGES
    assert tuple(exact.placement_edges("two_block")) == TWO_BLOCK_EDGES
    centers = exact.cell_centers(N_CELLS)
    one = exact.exact_on_placement(N_CELLS, "one_block", T)
    two = exact.exact_on_placement(N_CELLS, "two_block", T)
    assert one.shape == (N_CELLS,)
    assert two.shape == (N_CELLS,)
    left = centers[centers < 0.5]
    right = centers[centers > 0.5]
    assert left.size == N_CELLS // 2
    assert right.size == N_CELLS // 2
    assert left.max() < 0.5 < right.min()
    np.testing.assert_array_equal(one, two)


def test_one_by_four_and_four_by_one_use_splits_at_quarter_points():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    assert tuple(exact.FOUR_BLOCK_SPLITS) == FOUR_BLOCK_SPLITS
    assert tuple(exact.placement_edges("1x4")) == FOUR_BLOCK_EDGES
    assert tuple(exact.placement_edges("4x1")) == FOUR_BLOCK_EDGES
    assert tuple(run.four_block_splits()) == FOUR_BLOCK_SPLITS
    width = 1.0 / float(N_CELLS)
    for interface in FOUR_BLOCK_SPLITS:
        n_left = exact.n_left_cells(N_CELLS, interface)
        assert n_left * width == interface
        assert 0 < n_left < N_CELLS
    one_by_four = exact.exact_on_placement(N_CELLS, "1x4", T)
    four_by_one = exact.exact_on_placement(N_CELLS, "4x1", T)
    np.testing.assert_array_equal(one_by_four, four_by_one)


def test_all_placements_agree_exactly():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    volumes = exact.cell_volumes(N_CELLS)
    fields = run.exact_fields_for_placements(N_CELLS, t=T)
    assert tuple(fields) == PLACEMENTS
    for left, right in combinations(fields.values(), 2):
        errors = reference_errors(left, right, volumes)
        assert errors.l1 == 0.0
        assert errors.l2 == 0.0
        assert errors.linf == 0.0
        np.testing.assert_array_equal(left, right)
    assert analyze.placements_agree(N_CELLS, t=T) == 0.0
    assert run.max_decomposition_difference(N_CELLS, t=T) == 0.0


def test_write_if01_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_if01_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"] == ORDERS_REASON


def test_run_native_is_public_pipeline_without_rank_launcher():
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "advection_sine" in text
    assert "from exact import" not in text
    assert "mpirun" not in text
    assert "if01_mpi_worker" not in text
    assert "_mpi_launcher" not in text
    assert "run_native_ranks" not in text
    assert "ExecutionContext.mpi_world" in (
        Path(__file__).resolve().parents[3]
        / "verification"
        / "pops_verify"
        / "case_authoring.py"
    ).read_text(encoding="utf-8")
    assert "bind_public" in text
    assert "attach_case_diagnostics" in text
    try:
        field = np.asarray(run.run_native(16, t_end=0.05), dtype=np.float64)
    except run.NativeUnavailable as exc:
        assert str(exc)
        return
    assert field.shape == (16,)
    assert np.isfinite(field).all()


def test_if01_request_returns_run_fields(monkeypatch):
    """A campaign request must return provenance fields, including honest MPI ranks."""
    import inspect

    run = _load_case_module("run")
    assert "request" in inspect.signature(run.run_native).parameters
    serial = CampaignRequest.from_job(
        CampaignJob(case_id="IF-01", pops_native_dim=1, mpi_mode="off", min_resolution=16)
    )
    serial_fields = run.campaign_run_fields(16, 0.25, serial)
    missing = [key for key in RUN_FIELDS if key not in serial_fields]
    assert missing == []
    assert serial_fields["mpi_enabled"] is False
    assert serial_fields["mpi_ranks"] == 1
    assert serial_fields["mpi_library"] == "none"
    assert serial_fields["resolution"] == [16]
    assert serial_fields["cfl"] == run.CFL
    assert serial_fields["time_program"] == "SSPRK2"

    mpi = CampaignRequest.from_job(
        CampaignJob(case_id="IF-01", pops_native_dim=1, mpi_mode="on", min_resolution=16)
    )
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
    monkeypatch.setenv("PMI_SIZE", "2")
    with pytest.raises(run.NativeUnavailable, match="native|communicator|unavailable"):
        run.campaign_run_fields(16, 0.25, mpi)


def _patch_native_world(monkeypatch, *, size: int | None, rank: int = 0, has_mpi: bool = True):
    import sys
    import types

    selector = types.ModuleType("pops._native_selector")
    if size is None:
        selector.selected_native_module = lambda *, required=False: None
    else:

        class _Module:
            __has_mpi__ = has_mpi

            def n_ranks(self):
                return size

            def my_rank(self):
                return rank

        selector.selected_native_module = lambda *, required=False: _Module()
    monkeypatch.setitem(sys.modules, "pops._native_selector", selector)


def test_if01_mpi_on_refuses_launcher_env_without_native_world(monkeypatch):
    """Rejected job 695285: SLURM_NTASKS=2 without a native communicator."""
    run = _load_case_module("run")
    _patch_native_world(monkeypatch, size=None)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
    monkeypatch.setenv("PMI_SIZE", "2")
    mpi = CampaignRequest.from_job(
        CampaignJob(case_id="IF-01", pops_native_dim=1, mpi_mode="on", min_resolution=16)
    )
    with pytest.raises(run.NativeUnavailable, match="native|communicator|unavailable"):
        run.discovered_mpi_ranks()
    with pytest.raises(run.NativeUnavailable, match="native|communicator|unavailable"):
        run.campaign_run_fields(16, 0.25, mpi)


def test_if01_mpi_on_refuses_native_singleton_despite_slurm(monkeypatch):
    """Rejected job 695285: two singleton MPI worlds under SLURM_NTASKS=2."""
    run = _load_case_module("run")
    _patch_native_world(monkeypatch, size=1, rank=0)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
    mpi = CampaignRequest.from_job(
        CampaignJob(case_id="IF-01", pops_native_dim=1, mpi_mode="on", min_resolution=16)
    )
    assert run.discovered_mpi_ranks() == 1
    with pytest.raises(run.NativeUnavailable, match="serial fallback|1 rank"):
        run.campaign_run_fields(16, 0.25, mpi)


def test_if01_mpi_on_records_native_world_size_two(monkeypatch):
    run = _load_case_module("run")
    _patch_native_world(monkeypatch, size=2, rank=0)
    monkeypatch.setenv("SLURM_NTASKS", "1")
    mpi = CampaignRequest.from_job(
        CampaignJob(case_id="IF-01", pops_native_dim=1, mpi_mode="on", min_resolution=16)
    )
    fields = run.campaign_run_fields(16, 0.25, mpi)
    assert fields["mpi_enabled"] is True
    assert fields["mpi_ranks"] == 2


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text
