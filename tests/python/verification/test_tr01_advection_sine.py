"""TR-01 3-d oblique periodic advection sine (Annexe A.1 / §35.1)."""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from tests.python.support.requirements import missing_compiler_requirement
from verification.pops_verify.campaign import CampaignRequest, CampaignResources
from verification.pops_verify.cell_averages import analytic_cell_averages
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.provenance import RUN_FIELDS
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "transport" / "advection_sine"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
RESOLUTIONS = (16, 32, 64, 128)


def _load_case_module(name: str):
    path = CASE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tr01_advection_sine_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_canonical_3d_data_matches_annexe_a():
    exact = _load_case_module("exact")
    assert exact.REQUIRED_NATIVE_DIM == 3
    assert tuple(exact.A) == (1.0, 1.0, 1.0)
    assert tuple(exact.K) == (1.0, 2.0, 3.0)
    assert float(exact.T_END) == 1.0
    assert tuple(exact.RESOLUTIONS) == RESOLUTIONS
    xx, yy, zz, volumes = exact.uniform_cell_mesh(8)
    q0 = exact.exact_sine_3d(xx, yy, zz, 0.0)
    q1 = exact.exact_sine_3d(xx, yy, zz, 1.0)
    np.testing.assert_allclose(q0, q1, atol=1.0e-12)
    assert volumes.shape == (8, 8, 8)
    assert xx.shape == (8, 8, 8)


def test_shared_1d_exact_sine_accepts_centers_and_time():
    exact = _load_case_module("exact")
    centers, volumes = exact.uniform_cell_centers(16)
    q0 = exact.exact_sine(centers, 0.0)
    q1 = exact.exact_sine(centers, 1.0, a=1.0, k=1.0)
    np.testing.assert_allclose(q0, q1, atol=1.0e-12)
    assert q0.shape == (16,)
    assert volumes.shape == (16,)


def test_tr01_runtime_stays_dedicated_1d():
    text = (REPO_ROOT / "verification" / "pops_verify" / "tr01_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "_require_native_dim3" not in text
    assert "(count, count, count)" not in text
    assert "exact_sine(" in text
    assert "(authored.n_cells,)" in text or "(count,)" in text


def test_reference_errors_of_exact_vs_exact_are_zero():
    exact = _load_case_module("exact")
    xx, yy, zz, volumes = exact.uniform_cell_mesh(8)
    field = exact.exact_sine_3d(xx, yy, zz, 0.0)
    errors = reference_errors(field, field, volumes)
    assert errors.l1 == 0.0
    assert errors.l2 == 0.0
    assert errors.linf == 0.0


def test_cell_average_oracle_is_finite_on_the_cube():
    exact = _load_case_module("exact")
    lo, hi = exact.cell_bounds(8)

    def _u(x, y, z, time):
        return exact.exact_sine_3d(x, y, z, time)

    averages = analytic_cell_averages(_u, lo, hi, 0.0)
    assert averages.shape == (8, 8, 8)
    assert np.isfinite(averages).all()


def test_analyze_refuses_injected_h2_as_order_pass(tmp_path: Path):
    n = np.asarray(RESOLUTIONS, dtype=np.float64)
    spacings = 1.0 / n
    errors = spacings**2
    helper_orders = observed_order(errors, spacings)
    np.testing.assert_allclose(helper_orders, np.full(helper_orders.shape, 2.0))

    analyze = _load_case_module("analyze")
    with pytest.raises(analyze.NativeSeriesError, match="synthetic|injected|native"):
        analyze.evaluate_order_claim(
            {
                "source": "synthetic",
                "resolutions": tuple(int(v) for v in RESOLUTIONS),
                "spacings": tuple(float(v) for v in spacings),
                "linf": tuple(float(v) for v in errors),
            }
        )
    with pytest.raises(analyze.NativeSeriesError, match="synthetic|injected|native"):
        analyze.analyze_series(errors, spacings, tmp_path)


def test_build_case_and_resolve_plan_default_to_canonical_3d():
    run = _load_case_module("run")
    case = run.build_case(8)
    plan = run.resolve_plan(8)
    assert case is not None
    assert getattr(plan, "resolved_dimension", None) == 3
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "Cartesian3D" in text
    assert "REQUIRED_NATIVE_DIM" in text or "canonical" in text.lower()


def test_run_native_refuses_missing_or_non_three_dim(monkeypatch):
    run = _load_case_module("run")
    monkeypatch.delenv("POPS_NATIVE_DIM", raising=False)
    with pytest.raises(run.NativeUnavailable, match="POPS_NATIVE_DIM=3"):
        run._require_native_dim3()
    monkeypatch.setenv("POPS_NATIVE_DIM", "1")
    with pytest.raises(run.NativeUnavailable, match="no 1-d/2-d fallback"):
        run._require_native_dim3()
    monkeypatch.setenv("POPS_NATIVE_DIM", "2")
    with pytest.raises(run.NativeUnavailable, match="no 1-d/2-d fallback"):
        run._require_native_dim3()


def test_write_tr01_report_without_native_series_is_not_an_order_pass(
    tmp_path: Path,
):
    analyze = _load_case_module("analyze")
    written = analyze.write_tr01_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert "orders" in loaded["not_applicable_reason"]
    assert loaded["coverage"]["cases_passed"] == 0


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert owners
            assert set(owners) <= {"run_native", "run_order_campaign", "_execute"}
        else:
            assert owners == []
            assert "pops.run(" not in text


@pytest.mark.compiler
def test_run_native_dim3_returns_cube_or_skips(tmp_path: Path):
    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        field = np.asarray(
            run.run_native(8, t_end=0.05, output_dir=tmp_path), dtype=np.float64
        )
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.shape == (8, 8, 8)
    assert np.isfinite(field).all()
    assert (tmp_path / "provenance.json").is_file()
    document = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert document["schema"] == "pops.verification.provenance.v1"
    assert document["pops_native_dim"] == 3
    assert document["dimension"] == 3
    assert document["resolution"] == [8, 8, 8]


@pytest.mark.compiler
def test_order_campaign_requires_four_resolutions():
    run = _load_case_module("run")
    with pytest.raises(ValueError, match="four resolutions"):
        run.run_order_campaign((16, 32, 64))


def _request(
    *,
    dim: int = 3,
    space: str = "KokkosSerial",
    mpi: str = "off",
    resolutions: tuple[int, ...] = RESOLUTIONS,
    n: int = 16,
    output_dir: Path | None = None,
) -> CampaignRequest:
    return CampaignRequest(
        case_id="TR-01",
        pops_native_dim=dim,
        suite="pr",
        execution_space=space,
        mpi_mode=mpi,
        min_resolution=n,
        resources=CampaignResources(
            nodes=1,
            mpi_ranks=2 if mpi == "on" else 1,
            omp_threads=4 if space == "KokkosOpenMP" else 1,
            resolutions=resolutions,
        ),
        evidence_status="required",
        output_dir=output_dir,
    )


def test_run_native_accepts_campaign_request():
    run = _load_case_module("run")
    signature = inspect.signature(run.run_native)
    assert "request" in signature.parameters


def test_dispatch_selects_1d_2d_and_canonical_3d():
    run = _load_case_module("run")
    one = run.resolve_config(_request(dim=1))
    two = run.resolve_config(_request(dim=2))
    three = run.resolve_config(_request(dim=3))
    assert one.dim == 1
    assert one.velocity == (1.0,)
    assert one.wave == (1.0,)
    assert one.t_end == 1.0
    assert one.label != "canonical"
    assert two.dim == 2
    assert two.velocity == (1.0, 1.0)
    assert two.wave == (1.0, 2.0)
    assert two.label != "canonical"
    assert three.dim == 3
    assert three.velocity == (1.0, 1.0, 1.0)
    assert three.wave == (1.0, 2.0, 3.0)
    assert three.t_end == 1.0
    assert three.label == "canonical"


def test_dispatch_refuses_config_dim_mismatch():
    run = _load_case_module("run")
    with pytest.raises(run.NativeUnavailable, match="dim|mismatch|canonical"):
        run.resolve_config(_request(dim=1), config_id="canonical")
    with pytest.raises(run.NativeUnavailable, match="POPS_NATIVE_DIM"):
        run.require_exact_rank(_request(dim=2), launched_dim=3)


def test_truthful_run_fields_follow_request_space_and_mpi():
    run = _load_case_module("run")
    serial = run.campaign_run_fields(
        _request(dim=1, space="KokkosSerial", mpi="off"),
        run.resolve_config(_request(dim=1)),
        n_cells=16,
        t_end=1.0,
    )
    openmp = run.campaign_run_fields(
        _request(dim=2, space="KokkosOpenMP", mpi="off"),
        run.resolve_config(_request(dim=2)),
        n_cells=16,
        t_end=1.0,
    )
    mpi = run.campaign_run_fields(
        _request(dim=1, space="KokkosSerial", mpi="on"),
        run.resolve_config(_request(dim=1)),
        n_cells=16,
        t_end=1.0,
    )
    for fields in (serial, openmp, mpi):
        assert set(RUN_FIELDS) <= set(fields)
    assert serial["kokkos_execution_space"] == "KokkosSerial"
    assert serial["mpi_enabled"] is False
    assert serial["mpi_library"] == "none"
    assert openmp["kokkos_execution_space"] == "KokkosOpenMP"
    assert openmp["mpi_enabled"] is False
    assert mpi["mpi_enabled"] is True
    assert mpi["mpi_ranks"] == 2
    assert serial["resolution"] == [16]
    assert openmp["resolution"] == [16, 16]


def test_analyze_refuses_absent_and_short_native_series():
    analyze = _load_case_module("analyze")
    with pytest.raises(analyze.NativeSeriesError, match="absent|native"):
        analyze.evaluate_order_claim({})
    short = analyze.evaluate_order_claim(
        {
            "source": "native",
            "dimension": 3,
            "label": "canonical",
            "resolutions": (16, 32),
            "spacings": (1.0 / 16.0, 1.0 / 32.0),
            "l1": (1.0e-2, 3.0e-3),
            "l2": (1.0e-2, 3.0e-3),
            "linf": (2.0e-2, 6.0e-3),
            "fields": {16: np.ones((16, 16, 16)), 32: np.ones((32, 32, 32))},
            "oracles": {16: np.ones((16, 16, 16)), 32: np.ones((32, 32, 32))},
            "volumes": {
                16: np.full((16, 16, 16), 1.0 / 16.0**3),
                32: np.full((32, 32, 32), 1.0 / 32.0**3),
            },
        }
    )
    assert short["verdict"] in {"smoke", "not-run"}
    assert short["order_pass"] is False
    assert short["orders"] == []


def test_analyze_truthful_orders_from_native_shaped_fields(tmp_path: Path):
    exact = _load_case_module("exact")
    analyze = _load_case_module("analyze")
    fields = {}
    oracles = {}
    volumes = {}
    l1 = []
    l2 = []
    linf = []
    for n_cells in RESOLUTIONS:
        lo, hi = exact.cell_bounds(n_cells)
        oracle = analytic_cell_averages(
            lambda x, y, z, time: exact.exact_sine_3d(x, y, z, time), lo, hi, 1.0
        )
        width = 1.0 / float(n_cells)
        field = oracle + (width**2) * np.sin(2.0 * np.pi * exact.uniform_cell_mesh(n_cells)[0])
        _, _, _, cell_volumes = exact.uniform_cell_mesh(n_cells)
        errors = reference_errors(field, oracle, cell_volumes)
        fields[n_cells] = field
        oracles[n_cells] = oracle
        volumes[n_cells] = cell_volumes
        l1.append(float(errors.l1))
        l2.append(float(errors.l2))
        linf.append(float(errors.linf))
    campaign = {
        "source": "native",
        "case_id": "TR-01",
        "dimension": 3,
        "label": "canonical",
        "velocity": (1.0, 1.0, 1.0),
        "wave": (1.0, 2.0, 3.0),
        "t_end": 1.0,
        "resolutions": RESOLUTIONS,
        "spacings": tuple(1.0 / float(n) for n in RESOLUTIONS),
        "l1": tuple(l1),
        "l2": tuple(l2),
        "linf": tuple(linf),
        "fields": fields,
        "oracles": oracles,
        "volumes": volumes,
        "diagnostics": {
            n: {"phase_error": 0.0, "amplitude_loss": 0.0, "mass_error": 0.0}
            for n in RESOLUTIONS
        },
    }
    claim = analyze.evaluate_order_claim(campaign)
    assert claim["order_pass"] is True
    assert len(claim["orders"]) == 3
    np.testing.assert_allclose(claim["orders"], np.full(3, 2.0), atol=0.15)
    written = analyze.write_native_campaign_report(tmp_path, campaign)
    assert written == ARTIFACTS
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(summary)
    assert summary["orders"]
    assert (tmp_path / "metrics.json").is_file()
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["schema"] == "pops.verification.metrics.v1"
    assert metrics["errors"]["q"]["linf"] == pytest.approx(linf[-1])
    visual = json.loads((tmp_path / "visual_manifest.json").read_text(encoding="utf-8"))
    assert visual["case_id"] == "TR-01"
    assert (tmp_path / "visual_data" / "spatial_convergence.json").is_file()
    convergence = json.loads(
        (tmp_path / "visual_data" / "spatial_convergence.json").read_text(encoding="utf-8")
    )
    assert convergence["linf"] == list(linf)


def test_variant_catalog_covers_v15_directions_blocks_and_periods():
    run = _load_case_module("run")
    one = run.variant_catalog(dim=1)
    two = run.variant_catalog(dim=2)
    three = run.variant_catalog(dim=3)
    assert {(row.velocity[0],) for row in one} >= {(1.0,), (-1.0,)}
    velocities_2d = {tuple(row.velocity) for row in two}
    assert {(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.37)} <= velocities_2d
    velocities_3d = {tuple(row.velocity) for row in three}
    assert {
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.37, 0.61),
    } <= velocities_3d
    assert {row.layout for row in one} >= {
        "U-C",
        "U-F",
        "A-S0",
        "A-S2",
        "A-DP",
        "A-DT",
    }
    assert {row.periods for row in one} >= {1, 2, 4}
    assert {8, 16, 32, 64} <= {row.block_size for row in one} - {None}
    assert not any(row.n_cells >= 256 and row.dim == 3 for row in three)
    assert any(row.label == "canonical" and row.velocity == (1.0, 1.0, 1.0) for row in three)


def test_resolve_plan_matches_requested_dimension():
    run = _load_case_module("run")
    one = run.resolve_plan_for(run.resolve_config_id("restriction_1d", n_cells=8))
    two = run.resolve_plan_for(run.resolve_config_id("restriction_2d", n_cells=8))
    three = run.resolve_plan_for(run.resolve_config_id("canonical", n_cells=8))
    assert getattr(one, "resolved_dimension", None) == 1
    assert getattr(two, "resolved_dimension", None) == 2
    assert getattr(three, "resolved_dimension", None) == 3


def test_no_helper_contamination():
    runtime = (
        Path(__file__).resolve().parents[3]
        / "verification"
        / "pops_verify"
        / "tr01_runtime.py"
    ).read_text(encoding="utf-8")
    assert "Cartesian3D" not in runtime
    assert "exact_sine_3d" not in runtime
    assert "complement" not in runtime
    assert "_require_native_dim3" not in runtime
    analyze_text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    run_text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    for text in (analyze_text, run_text):
        assert "verification.pops_verify.visuals" not in text
        assert "render_verification_visuals" not in text
        assert "from exact import" not in text
    if_exact = (
        Path(__file__).resolve().parents[3]
        / "verification"
        / "cases"
        / "infrastructure"
        / "native_dim_guard"
        / "exact.py"
    ).read_text(encoding="utf-8")
    assert 'native_dimensions": [1]' in if_exact or "native_dimensions\": [MATCHING_DIM]" in if_exact


def test_tr01_romeo_script_keeps_lexical_variants_root():
    """ROMEO /scratch_p is a symlink to /gpfs/scratch. Resolving the leaf
    publishes a second spelling and the selector refuses two manifests.
    """
    complement = (
        REPO_ROOT / "verification" / "machines" / "run_tr01_complement.py"
    ).read_text(encoding="utf-8")
    sbatch = (
        REPO_ROOT / "verification" / "machines" / "romeo_676_tr01_run.sbatch"
    ).read_text(encoding="utf-8")
    assert "Path(item).resolve()" not in complement
    assert "POPS_NATIVE_VARIANTS_ROOT" in sbatch
    assert '"$BUILD/python/pops/_native"' in sbatch
    assert "readlink" not in sbatch
    assert "realpath" not in sbatch
