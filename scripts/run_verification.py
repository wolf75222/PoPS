#!/usr/bin/env python3
"""Plan and optionally execute a verification campaign.

Validates the scientific campaign manifest, refuses more than two nodes, expands
selected cases into parameterized single-dimension jobs, and writes output/plan.json.
Planning does not require a native artifact. ``--execute`` requires and
authenticates an exact-rank leaf, then calls each job's public ``run_native``
with the campaign request. Multi-rank MPI is the native PoPS communicator:
launch this script under ``srun``/``mpiexec``. This script does not spawn ranks.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import re
from pathlib import Path
import sys
import traceback

SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from check_verification_manifest import (  # noqa: E402
    DEFAULT_MANIFEST,
    VerificationManifestError,
    check_verification_manifest,
)
from verification.pops_verify.campaign import (  # noqa: E402
    ALLOWED_DIMENSIONS,
    ALLOWED_EXECUTION_SPACES,
    ALLOWED_MPI_MODES,
    ALLOWED_STATUSES,
    MAX_NODES_LIMIT,
    CampaignError,
    CampaignJob,
    CampaignRequest,
    expand_jobs,
    job_to_dict,
    resolve_artifact_dim,
)
from verification.pops_verify.capabilities import (  # noqa: E402
    AuthenticatedArtifact,
    CapabilityError,
    authenticate_installed_artifact,
    missing_requirements,
)
from verification.pops_verify.case_authoring import load_sibling_module  # noqa: E402
from verification.pops_verify.evidence_bundle import EvidenceBundle, EvidenceError  # noqa: E402
from verification.pops_verify.evidence_contract import (  # noqa: E402
    EvidenceContractError,
    emit_job_directory,
    write_series_json,
)
from verification.pops_verify.metrics import collect_metrics, write_metrics  # noqa: E402
from verification.pops_verify.native_evidence import (  # noqa: E402
    emission_from_payload,
    report_from_native_series,
    run_fields_from_payload,
)
from verification.pops_verify.visualization.live import write_live_run_visuals  # noqa: E402
from verification.pops_verify.provenance import (  # noqa: E402
    ProvenanceError,
    RUN_FIELDS,
    collect_provenance,
    write_provenance,
)
from verification.pops_verify.report import write_verification_report  # noqa: E402

_SAFE_SERIES_JOB = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

ALLOWED_SUITES = ("pr", "nightly", "weekly", "release", "two_node")
_CASE_ID_RE = re.compile(r"[A-Z0-9-]+")


class VerificationRunnerError(RuntimeError):
    pass


def parse_dimensions(raw: str) -> list[int]:
    if not raw or not raw.strip():
        raise VerificationRunnerError(
            "invalid --dimensions (expected comma-separated 1, 2, and/or 3)"
        )
    parts = [part.strip() for part in raw.split(",")]
    if not parts or any(part == "" for part in parts):
        raise VerificationRunnerError(
            f"invalid --dimensions {raw!r} (expected comma-separated 1, 2, and/or 3)"
        )
    dimensions: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise VerificationRunnerError(
                f"invalid --dimensions {raw!r} (expected comma-separated 1, 2, and/or 3)"
            ) from exc
        if value not in ALLOWED_DIMENSIONS:
            raise VerificationRunnerError(
                f"invalid --dimensions {raw!r} (expected comma-separated 1, 2, and/or 3)"
            )
        if value not in dimensions:
            dimensions.append(value)
    return dimensions


def parse_cases(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    if not raw.strip():
        raise VerificationRunnerError("invalid --cases (expected comma-separated case ids)")
    cases = [part.strip() for part in raw.split(",") if part.strip()]
    if not cases:
        raise VerificationRunnerError("invalid --cases (expected comma-separated case ids)")
    return cases


def select_cases(
    manifest: dict,
    suite: str,
    dimensions: list[int],
    *,
    case_ids: list[str] | None = None,
    mpi_mode: str | None = None,
    execution_space: str | None = None,
) -> list[dict]:
    requested = set(dimensions)
    selected = []
    wanted = set(case_ids) if case_ids is not None else None
    for case in manifest.get("case", []):
        if wanted is not None and case.get("id") not in wanted:
            continue
        if suite not in case.get("suites", []):
            continue
        native = set(case.get("native_dimensions", []))
        if not native.intersection(requested):
            continue
        modes = case.get("mpi_modes") or ["off"]
        if mpi_mode is not None and mpi_mode not in modes:
            continue
        spaces = case.get("execution_spaces") or ["KokkosSerial"]
        if execution_space is not None and execution_space not in spaces:
            continue
        selected.append(case)
    return selected


def write_plan(output: Path, plan: dict) -> None:
    if not is_campaign_writer_rank():
        return
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(
        json.dumps(plan, indent=2) + "\n",
        encoding="utf-8",
    )


def invoke_run_native(runner, request: CampaignRequest):
    """Pass the full campaign request; refuse signatures that drop it."""
    signature = inspect.signature(runner)
    params = signature.parameters
    accepts_var_kw = any(
        item.kind is inspect.Parameter.VAR_KEYWORD for item in params.values()
    )
    if "request" not in params and not accepts_var_kw:
        raise VerificationRunnerError(
            "run_native must accept request=CampaignRequest; "
            "n_cells-only signatures drop campaign fields"
        )
    kwargs: dict[str, object] = {"request": request}
    if request.min_resolution is not None and ("n_cells" in params or accepts_var_kw):
        kwargs["n_cells"] = request.min_resolution
    return runner(**kwargs)


def campaign_writer_rank() -> int:
    """Native communicator rank only. Launcher env is never a writer identity."""
    from verification.pops_verify.mpi_world import native_world_rank

    rank = native_world_rank(required=False)
    return 0 if rank is None else int(rank)


def is_campaign_writer_rank() -> bool:
    """Serial rank 0 may write. An MPI singleton world must not write a ledger."""
    from verification.pops_verify.mpi_world import is_native_writer_rank

    return is_native_writer_rank()


def provenance_nodes(job: CampaignJob) -> int:
    """Prefer the scheduled Slurm node count when it is 1 or 2."""
    raw = os.environ.get("SLURM_JOB_NUM_NODES")
    if raw is not None and str(raw).strip() != "":
        try:
            value = int(raw)
        except ValueError:
            return job.resources.nodes
        if value in (1, 2):
            return value
    return job.resources.nodes


def validate_case_id(case_id: str) -> str:
    if not isinstance(case_id, str) or not _CASE_ID_RE.fullmatch(case_id):
        raise VerificationRunnerError(f"unsafe case id {case_id!r}")
    return case_id


def resolve_case_module_path(raw: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    resolved = (repo_root / path).resolve()
    root = repo_root.resolve()
    if os.path.commonpath([str(root), str(resolved)]) != str(root):
        raise VerificationRunnerError(f"case path escapes repository: {raw}")
    return resolved


def _job_dir(output: Path, job: CampaignJob) -> Path:
    validate_case_id(job.case_id)
    job_dir = (
        output / job.case_id / f"dim{job.pops_native_dim}-{job.execution_space}-{job.mpi_mode}"
    ).resolve()
    output_root = output.resolve()
    if os.path.commonpath([str(output_root), str(job_dir)]) != str(output_root):
        raise VerificationRunnerError(f"job path escapes output for case id {job.case_id!r}")
    return job_dir


def _series_job_name(n_cells: int) -> str:
    name = f"n{int(n_cells):03d}"
    if not _SAFE_SERIES_JOB.fullmatch(name) or ".." in name:
        raise VerificationRunnerError(f"unsafe series job name {name!r}")
    return name


def _job_resolutions(job: CampaignJob) -> list[int | None]:
    values = [int(item) for item in (job.resources.resolutions or ())]
    if values:
        return values
    if job.min_resolution is not None:
        return [int(job.min_resolution)]
    return [None]


def _leaf_identity(artifact: AuthenticatedArtifact) -> dict[str, object]:
    return {
        "path": str(artifact.path),
        "sha256": artifact.sha256,
        "dimension": int(artifact.dimension),
    }


def _split_run_payload(run_result: object) -> tuple[dict[str, object], dict[str, object] | None]:
    """Separate RUN_FIELDS from EvidenceBundle emission keys."""
    if not isinstance(run_result, dict):
        return {}, None
    fields = run_fields_from_payload(run_result)
    if "result" not in run_result or "program_bytes" not in run_result:
        return fields, None
    try:
        return fields, emission_from_payload(run_result)
    except (KeyError, TypeError, ValueError):
        return fields, None


def _request_for_resolution(
    job: CampaignJob,
    output_dir: Path,
    n_cells: int | None,
) -> CampaignRequest:
    return CampaignRequest(
        case_id=job.case_id,
        pops_native_dim=job.pops_native_dim,
        suite=job.suite,
        execution_space=job.execution_space,
        mpi_mode=job.mpi_mode,
        min_resolution=n_cells,
        resources=job.resources,
        evidence_status=job.evidence_status,
        output_dir=output_dir,
    )


def _emit_resolution_directory(
    job: CampaignJob,
    case: dict | None,
    request: CampaignRequest,
    record: dict,
    artifact: AuthenticatedArtifact,
    *,
    run_fields: dict[str, object],
    emission: dict[str, object],
    job_dir: Path,
) -> None:
    if not is_campaign_writer_rank():
        return
    import pops

    resolved = {
        "case": case,
        "job": job_to_dict(job),
        "status": record.get("status"),
        "reason": record.get("reason"),
    }
    resolved["job"]["min_resolution"] = request.min_resolution
    if emission.get("dt") is not None:
        resolved["job"]["dt"] = float(emission["dt"])
    provenance = collect_provenance(
        job.case_id,
        pops_native_dim=job.pops_native_dim,
        dimension=job.pops_native_dim,
        nodes=provenance_nodes(job),
        pops_version=str(pops.__version__),
        doctor_ok=artifact.doctor_ok,
        component_catalog_digest=artifact.component_catalog_digest,
        native_header_signature=artifact.native_header_signature,
        native_variant_manifest_digest=artifact.native_variant_manifest_digest,
        **run_fields,
    )
    reason = str(record.get("reason") or "runner foundation records no scientific field yet")
    emit_job_directory(
        job_dir,
        resolved_case=resolved,
        provenance=provenance,
        metrics=collect_metrics(job.case_id, reason=reason),
        result=emission["result"],
        program_bytes=emission["program_bytes"],
        native_artifact=_leaf_identity(artifact),
        pair_result=emission.get("pair_result"),
        pair_program_bytes=emission.get("pair_program_bytes"),
        coupling=emission.get("coupling"),
    )


def _honest_run_fields(
    job: CampaignJob,
    artifact: AuthenticatedArtifact,
    run_result: object,
) -> dict[str, object]:
    """Request and discovered facts only. Missing schema-required fields fail."""
    fields: dict[str, object] = {
        "kokkos_execution_space": job.execution_space,
        "mpi_enabled": job.mpi_mode == "on",
        "hdf5_collective_enabled": artifact.hdf5_collective,
        "mpi_ranks": job.resources.mpi_ranks,
        "omp_threads_per_rank": job.resources.omp_threads,
    }
    if job.mpi_mode == "off":
        fields["mpi_library"] = "none"
        fields["mpi_thread_level_requested"] = "none"
        fields["mpi_thread_level_provided"] = "none"
    if job.min_resolution is not None:
        fields["resolution"] = [job.min_resolution] * job.pops_native_dim
    if isinstance(run_result, dict):
        for key in RUN_FIELDS:
            if key in run_result:
                fields[key] = run_result[key]
    missing = [key for key in RUN_FIELDS if key not in fields]
    if missing:
        raise VerificationRunnerError(f"unknown provenance field: {missing[0]}")
    return fields


def _write_job_artifacts(
    job: CampaignJob,
    case: dict | None,
    request: CampaignRequest,
    record: dict,
    artifact: AuthenticatedArtifact,
    *,
    run_fields: dict[str, object] | None = None,
) -> None:
    if not is_campaign_writer_rank():
        return
    output_dir = request.output_dir
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "case": case,
        "job": job_to_dict(job),
        "status": record.get("status"),
        "reason": record.get("reason"),
    }
    (output_dir / "resolved_case.json").write_text(
        json.dumps(resolved, indent=2) + "\n",
        encoding="utf-8",
    )
    reason = str(record.get("reason") or "runner foundation records no scientific field yet")
    metrics = collect_metrics(job.case_id, reason=reason)
    write_metrics(output_dir / "metrics.json", metrics)
    record["metrics_ref"] = f"{job.case_id}/{output_dir.name}/metrics.json"
    if run_fields is None:
        return
    try:
        import pops

        pops_version = str(pops.__version__)
        provenance = collect_provenance(
            job.case_id,
            pops_native_dim=job.pops_native_dim,
            dimension=job.pops_native_dim,
            nodes=provenance_nodes(job),
            pops_version=pops_version,
            doctor_ok=artifact.doctor_ok,
            component_catalog_digest=artifact.component_catalog_digest,
            native_header_signature=artifact.native_header_signature,
            native_variant_manifest_digest=artifact.native_variant_manifest_digest,
            **run_fields,
        )
        write_provenance(output_dir / "provenance.json", provenance)
        record["provenance_ref"] = f"{job.case_id}/{output_dir.name}/provenance.json"
    except (ProvenanceError, VerificationRunnerError) as exc:
        if record.get("status") == "pass":
            record["status"] = "fail"
            record["reason"] = str(exc)


def _write_campaign_report(
    output: Path,
    *,
    suite: str,
    jobs: list[CampaignJob],
    results: list[dict],
) -> None:
    import subprocess

    sha = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not sha:
        raise VerificationRunnerError("cannot read repository SHA")
    passed = sum(1 for row in results if row.get("scientific_pass") is True)
    failed = sum(1 for row in results if row.get("status") == "fail")
    not_supported = sum(1 for row in results if row.get("status") == "not-supported")
    ran = sum(
        1 for row in results if row.get("status") in {"pass", "fail", "not-supported"}
    )
    spaces = sorted({job.execution_space for job in jobs}) or ["KokkosSerial"]
    dims = sorted({job.pops_native_dim for job in jobs}) or [1]
    failures = [
        {
            "case_id": row["case_id"],
            "reason": row.get("reason") or "fail",
            "metrics_ref": row.get("metrics_ref") or "",
            "provenance_ref": row.get("provenance_ref") or "",
        }
        for row in results
        if row.get("status") == "fail"
    ]
    foundation = "runner foundation does not aggregate this diagnostic"
    summary = {
        "schema": "pops.verification.report.v1",
        "repository": "wolf75222/PoPS",
        "repository_sha": sha,
        "suite": suite,
        "max_nodes": 2,
        "native_dimensions": dims,
        "execution_spaces": spaces,
        "coverage": {
            "components": ["verification"],
            "cases_planned": len(jobs),
            "cases_run": ran,
            "cases_passed": passed,
            "cases_failed": failed,
            "cases_not_supported": not_supported,
            "not_tested": [],
        },
        "failures": failures,
        "orders": [],
        "amr": {
            "order_retained": None,
            "invariants_ok": None,
            "interface_error": None,
            "bulk_error": None,
        },
        "poisson": {
            "potential_error": None,
            "field_error": None,
            "residual_l2": None,
        },
        "coupling": {
            "phase_error": None,
            "sign_ok": None,
            "energy_drift": None,
        },
        "parallel_invariance": {
            "ranks_ok": None,
            "threads_ok": None,
            "gpu_ok": None,
        },
        "performance": {"one_node": None, "two_node": None},
        "not_applicable_reason": {
            "orders": foundation,
            "amr.*": foundation,
            "poisson.*": foundation,
            "coupling.*": foundation,
            "parallel_invariance.*": foundation,
            "performance.one_node": foundation,
            "performance.two_node": foundation,
        },
        "artifacts": {
            "report_md": "REPORT.md",
            "summary_json": "summary.json",
            "coverage_csv": "coverage.csv",
            "failures_csv": "failures.csv",
        },
    }
    write_verification_report(summary, output)


def execute_jobs(
    jobs: list[CampaignJob],
    cases: list[dict],
    output: Path,
    *,
    artifact: AuthenticatedArtifact,
    manifest: dict,
) -> list[dict]:
    """Run each planned job's public ``run_native`` with campaign parameters."""
    by_id = {case["id"]: case for case in cases}
    current = manifest.get("current_capabilities") or {}
    results: list[dict] = []
    for job in jobs:
        validate_case_id(job.case_id)
        case = by_id.get(job.case_id)
        job_dir = _job_dir(output, job)
        request = CampaignRequest.from_job(job, output_dir=job_dir)
        record: dict = {
            "case_id": job.case_id,
            "pops_native_dim": job.pops_native_dim,
            "suite": job.suite,
            "execution_space": job.execution_space,
            "mpi_mode": job.mpi_mode,
            "status": "not-run",
        }
        run_result: object = None
        if case is None:
            record["status"] = "fail"
            record["reason"] = "missing case"
            _write_job_artifacts(job, None, request, record, artifact)
            results.append(record)
            continue
        record["path"] = str(resolve_case_module_path(case["path"]))
        missing = missing_requirements(case, artifact, current, job=job)
        if missing:
            reason = "missing installed capabilities: " + ", ".join(missing)
            record["reason"] = reason
            record["status"] = (
                "not-supported" if job.evidence_status == "capability-gated" else "fail"
            )
            _write_job_artifacts(job, case, request, record, artifact)
            results.append(record)
            continue
        env_dim = os.environ.get("POPS_NATIVE_DIM")
        os.environ["POPS_NATIVE_DIM"] = str(job.pops_native_dim)
        series_jobs: list[str] = []
        first_fields: dict[str, object] | None = None
        try:
            module = load_sibling_module(resolve_case_module_path(case["path"]))
            runner = getattr(module, "run_native", None)
            if not callable(runner):
                record["status"] = "fail"
                record["reason"] = "missing run_native"
            else:
                record["status"] = "pass"
                run_result = invoke_run_native(runner, request)
                payload_fields, emission = _split_run_payload(run_result)
                first_fields = _honest_run_fields(job, artifact, payload_fields)
                if emission is not None:
                    for n_cells in _job_resolutions(job):
                        name = (
                            _series_job_name(n_cells) if n_cells is not None else "n000"
                        )
                        resolution_dir = job_dir / name
                        resolution_request = _request_for_resolution(
                            job, resolution_dir, n_cells
                        )
                        if n_cells != request.min_resolution:
                            run_result = invoke_run_native(runner, resolution_request)
                            payload_fields, emission = _split_run_payload(run_result)
                            if emission is None:
                                continue
                            run_fields = _honest_run_fields(
                                job, artifact, payload_fields
                            )
                        else:
                            run_fields = first_fields
                        _emit_resolution_directory(
                            job,
                            case,
                            resolution_request,
                            record,
                            artifact,
                            run_fields=run_fields,
                            emission=emission,
                            job_dir=resolution_dir,
                        )
                        series_jobs.append(name)
        except VerificationRunnerError as exc:
            record["status"] = "fail"
            record["reason"] = str(exc)
        except Exception as exc:
            name = exc.__class__.__name__
            if name == "NativeUnavailable":
                if job.evidence_status == "capability-gated":
                    record["status"] = "not-supported"
                else:
                    record["status"] = "fail"
                record["reason"] = str(exc)
            else:
                record["status"] = "fail"
                record["reason"] = f"{name}: {exc}"
                record["traceback"] = traceback.format_exc()
        finally:
            if env_dim is None:
                os.environ.pop("POPS_NATIVE_DIM", None)
            else:
                os.environ["POPS_NATIVE_DIM"] = env_dim
        if record["status"] not in ALLOWED_STATUSES:
            record["status"] = "fail"
            record["reason"] = f"illegal status {record['status']!r}"
        _write_job_artifacts(
            job, case, request, record, artifact, run_fields=first_fields
        )
        if record["status"] == "pass" and series_jobs and is_campaign_writer_rank():
            analysis = None
            bundle = None
            try:
                write_series_json(job_dir, job.case_id, series_jobs)
                bundle = EvidenceBundle(job_dir)
                try:
                    analysis = report_from_native_series(
                        job.case_id,
                        job_dir,
                        native_dimensions=[job.pops_native_dim],
                        components=["verification"],
                    )
                    record["scientific_pass"] = (
                        int((analysis.get("coverage") or {}).get("cases_passed") or 0)
                        >= 1
                    )
                except (EvidenceError, EvidenceContractError, VerificationRunnerError) as exc:
                    record["scientific_pass"] = False
                    record["reason"] = record.get("reason") or str(exc)
                scientific_pass = record.get("scientific_pass") is True
                write_live_run_visuals(
                    job_dir,
                    case_id=job.case_id,
                    run_id=job_dir.name,
                    scientific_pass=scientific_pass,
                    verdict="pass" if scientific_pass else "fail",
                    analysis=analysis,
                    bundle=bundle,
                )
            except (EvidenceError, EvidenceContractError, VerificationRunnerError) as exc:
                record["scientific_pass"] = False
                record["reason"] = record.get("reason") or str(exc)
                write_live_run_visuals(
                    job_dir,
                    case_id=job.case_id,
                    run_id=job_dir.name,
                    scientific_pass=False,
                    verdict="fail",
                    bundle=bundle,
                )
        elif is_campaign_writer_rank():
            write_live_run_visuals(
                job_dir,
                case_id=job.case_id,
                run_id=job_dir.name,
                scientific_pass=False,
                verdict="not-run" if record["status"] == "pass" else "fail",
            )
        results.append(record)
    if is_campaign_writer_rank():
        output.mkdir(parents=True, exist_ok=True)
        (output / "results.json").write_text(
            json.dumps(results, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_campaign_report(
            output, suite=jobs[0].suite if jobs else "pr", jobs=jobs, results=results
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--dimensions", required=True)
    parser.add_argument("--max-nodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pops-native-dim", type=int, choices=ALLOWED_DIMENSIONS)
    parser.add_argument("--cases", help="comma-separated case ids")
    parser.add_argument("--mpi-mode", choices=ALLOWED_MPI_MODES)
    parser.add_argument("--execution-space", choices=ALLOWED_EXECUTION_SPACES)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run each planned job's run_native in-process (no rank launcher)",
    )
    args = parser.parse_args(argv)

    try:
        instance = check_verification_manifest(args.manifest)

        if args.max_nodes > MAX_NODES_LIMIT:
            raise VerificationRunnerError(
                f"--max-nodes {args.max_nodes} exceeds the two-node limit"
            )
        if args.suite not in ALLOWED_SUITES:
            raise VerificationRunnerError(
                f"unknown --suite {args.suite!r} (expected {', '.join(ALLOWED_SUITES)})"
            )
        dimensions = parse_dimensions(args.dimensions)
        if args.max_nodes < 1:
            raise VerificationRunnerError("--max-nodes must be >= 1")
        case_ids = parse_cases(args.cases)

        artifact_dim = resolve_artifact_dim(args.pops_native_dim)
        cases = select_cases(
            instance,
            args.suite,
            dimensions,
            case_ids=case_ids,
            mpi_mode=args.mpi_mode,
            execution_space=args.execution_space,
        )
        jobs = expand_jobs(
            cases,
            dimensions,
            artifact_dim=artifact_dim,
            suite=args.suite,
            execution_space=args.execution_space,
            mpi_mode=args.mpi_mode,
        )
        for job in jobs:
            validate_case_id(job.case_id)
        write_plan(
            args.output,
            {
                "suite": args.suite,
                "dimensions": dimensions,
                "max_nodes": args.max_nodes,
                "manifest": str(args.manifest.resolve()),
                "cases": cases,
                "jobs": [job_to_dict(job) for job in jobs],
            },
        )
        if args.execute:
            if artifact_dim is None:
                raise VerificationRunnerError(
                    "--execute requires an authenticated exact-rank artifact "
                    "(set --pops-native-dim or POPS_NATIVE_DIM)"
                )
            try:
                artifact = authenticate_installed_artifact(dimension=artifact_dim)
            except CapabilityError as exc:
                raise VerificationRunnerError(
                    f"unauthenticated exact-rank artifact: {exc}"
                ) from exc
            results = execute_jobs(
                jobs,
                cases,
                args.output,
                artifact=artifact,
                manifest=instance,
            )
            failed = sum(
                1
                for row in results
                if row.get("status") == "fail" or row.get("scientific_pass") is False
            )
            print(f"planned {len(cases)} cases")
            print(f"executed {len(results)} jobs ({failed} failed)")
            return 1 if failed else 0
    except (
        VerificationRunnerError,
        VerificationManifestError,
        CampaignError,
        CapabilityError,
    ) as exc:
        print(f"VERIFICATION: FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"planned {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
