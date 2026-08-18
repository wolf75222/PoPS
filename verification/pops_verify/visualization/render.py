"""Render a verification run into the Phase 8 analysis tree."""
from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Iterable

from jsonschema import Draft202012Validator

from verification.pops_verify.visualization.animation import (
    assemble_ffmpeg,
    write_animation_frames,
)
from verification.pops_verify.visualization.catalog import catalog_entry, visual_contract_for
from verification.pops_verify.visualization.data import (
    VisualsError,
    load_run_bundle,
    load_visual_series,
)
from verification.pops_verify.visualization.fixtures import FIXTURE_LABEL, FIXTURE_SHA
from verification.pops_verify.visualization.plots import (
    PUBLICATION_FORMATS,
    derive_figure_verdict,
    file_sha256,
    prepare_amr_boxes,
    prepare_backend_parity,
    prepare_convergence,
    prepare_field,
    prepare_performance_breakdown,
    prepare_profile,
    prepare_signed_error_profile,
    qualify_fixture_caption,
    render_prepared,
)
from verification.pops_verify.visualization.storyboard import render_storyboard
from verification.pops_verify.visualization.style import RENDERER_SCRIPT, RENDERER_VERSION

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_visuals.v1.json"
PR_KINDS = frozenset(
    {
        "spatial_convergence",
        "temporal_convergence",
        "reference_profile",
        "signed_error_profile",
        "reference_comparison",
        "report_figure",
        "backend_parity",
        "performance_breakdown",
    }
)


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _axis(dimension: int) -> str:
    return {1: "1d", 2: "2d", 3: "3d"}[dimension]


def _selected_kinds(case_id: str, dimension: int, suite: str) -> tuple[str, ...]:
    entry = catalog_entry(case_id)
    kinds = entry.artifacts[_axis(dimension)]
    if suite == "pr":
        return tuple(kind for kind in kinds if kind in PR_KINDS)
    return kinds


def _prepare(payload: dict[str, Any]) -> dict[str, Any]:
    kind = payload.get("kind")
    if kind in {"spatial_convergence", "temporal_convergence", "coarse_fine_error"}:
        return prepare_convergence(payload)
    if kind == "signed_error_profile":
        return prepare_signed_error_profile(payload)
    if kind == "backend_parity":
        return prepare_backend_parity(payload)
    if kind == "performance_breakdown":
        return prepare_performance_breakdown(payload)
    if kind == "amr_boxes":
        return prepare_amr_boxes(payload)
    if kind == "report_figure":
        return {
            "kind": "report_figure",
            "figure_id": "report_figure",
            "panels": payload.get("panels") or [],
            "title": payload.get("title") or "report figure",
            "verdict": derive_figure_verdict(payload) if payload.get("series") else "pass",
        }
    if kind in {
        "reference_profile",
        "reference_comparison",
        "linecuts",
        "linecut",
        "phase_amplitude",
        "invariants_vs_time",
        "frequency_spectrum",
        "symmetry_metric",
    }:
        prepared = prepare_profile(payload)
        prepared["kind"] = kind
        prepared["figure_id"] = payload.get("figure_id") or kind
        return prepared
    if "field" in payload or kind in {
        "exact_field",
        "numerical_field",
        "signed_error_field",
        "absolute_error_field",
        "field_snapshot",
        "amr_patch_map",
        "slice_xy",
        "slice_xz",
        "slice_yz",
        "hero_figure",
    }:
        prepared = prepare_field(payload)
        prepared["figure_id"] = payload.get("figure_id") or kind
        return prepared
    raise VisualsError(f"unsupported figure kind: {kind}")


def _role(kind: str) -> str:
    if kind == "hero_figure":
        return "hero"
    if kind == "storyboard":
        return "storyboard"
    if kind == "animation":
        return "animation"
    if kind in PR_KINDS:
        return "publication"
    return "diagnostic"


def _companion(kind: str) -> str | None:
    if kind in {
        "spatial_convergence",
        "temporal_convergence",
        "report_figure",
        "backend_parity",
        "performance_breakdown",
    }:
        return kind
    if kind == "hero_figure":
        return "report_figure"
    return "report_figure"


def _transform(kind: str, prepared: dict[str, Any] | None) -> str:
    if kind in {"spatial_convergence", "temporal_convergence", "coarse_fine_error"}:
        return "loglog"
    if kind == "signed_error_field" or (prepared or {}).get("center") == 0.0:
        return "diverging"
    return "none"


def _dimension_block(
    entry,
    executed_axis: str,
    rendered: list[str],
    omitted: list[str],
    verdict: str,
) -> dict[str, Any]:
    blocks = {}
    for axis, code in zip(("1d", "2d", "3d"), entry.dimension_codes, strict=True):
        status = {"R": "required", "E": "extended", "N/A": "not_applicable"}[code]
        if code == "N/A":
            blocks[axis] = {
                "status": status,
                "justification": entry.na_reasons[axis],
                "artifacts": [],
            }
        elif axis == executed_axis:
            reason = None
            if verdict in {"not-run", "not-supported"}:
                reason = f"run verdict is {verdict}; no scientific curve was invented"
            elif omitted:
                reason = "omitted required visuals marked not-run: " + ", ".join(omitted)
            blocks[axis] = {
                "status": status,
                "justification": reason,
                "artifacts": rendered,
            }
        else:
            blocks[axis] = {
                "status": status,
                "justification": f"{axis} was not the executed dimension of this run.",
                "artifacts": [],
            }
    return blocks


def _caption_and_sha(bundle) -> tuple[str, str]:
    if bundle.data_kind == "deterministic_fixture":
        sha = (
            bundle.provenance["repository_sha"]
            if str(bundle.provenance["repository_sha"]).startswith("fixture:")
            else FIXTURE_SHA
        )
        return qualify_fixture_caption(bundle.data_kind_label or FIXTURE_LABEL), sha
    return bundle.data_kind_label, bundle.provenance["repository_sha"]


def _figure_entry(
    *,
    bundle,
    figure_id: str,
    kind: str,
    source_files: list[str],
    units: dict[str, str],
    outputs: dict[str, str],
    run_dir: Path,
    verdict: str,
    times: list[float],
    steps: list[int],
    transform: str,
) -> dict[str, Any]:
    rel_outputs = {
        fmt: str(Path(path).resolve().relative_to(run_dir.resolve()))
        for fmt, path in outputs.items()
    }
    hashes = {fmt: file_sha256(path) for fmt, path in outputs.items()}
    sha = (
        bundle.provenance["repository_sha"]
        if bundle.data_kind != "deterministic_fixture"
        or str(bundle.provenance["repository_sha"]).startswith("fixture:")
        else FIXTURE_SHA
    )
    return {
        "case_id": bundle.case_id,
        "run_id": bundle.run_id,
        "figure_id": figure_id,
        "kind": kind,
        "role": _role(kind),
        "source_files": source_files,
        "variables": ["scalar"],
        "units": units,
        "transform": transform,
        "color_range": None,
        "resolutions": [16, 32, 64, 128],
        "amr_levels": [0],
        "times": times,
        "step_numbers": steps,
        "repository_sha": sha,
        "resolved_case_digest": bundle.provenance["component_catalog_digest"],
        "program_digest": bundle.provenance["native_header_signature"],
        "native_artifact_digest": bundle.provenance["native_variant_manifest_digest"],
        "renderer": {"script": RENDERER_SCRIPT, "version": RENDERER_VERSION},
        "output_hashes": hashes,
        "outputs": rel_outputs,
        "proves": f"{kind} reconstructed from versioned visual_data.",
        "does_not_prove": (
            "This fixture is not a live PoPS campaign result."
            if bundle.data_kind == "deterministic_fixture"
            else "The figure does not replace the quantitative assertion in metrics.json."
        ),
        "quantitative_companion": _companion(kind),
        "pr": kind in PR_KINDS,
        "verdict": verdict,
    }


def _combine_verdict(figure_verdicts: list[str], omitted: list[str], requested: str) -> str:
    if requested in {"not-run", "not-supported"}:
        return requested
    if omitted:
        return "not-run"
    if any(item == "fail" for item in figure_verdicts):
        return "fail"
    if figure_verdicts and all(item in {"pass", "not-supported"} for item in figure_verdicts):
        if any(item == "pass" for item in figure_verdicts) and "fail" not in figure_verdicts:
            if all(item == "not-supported" for item in figure_verdicts):
                return "not-supported"
            return "pass"
    if not figure_verdicts:
        return "not-run"
    return "not-run"


def render_run(
    run_dir: str | Path,
    *,
    suite: str = "pr",
    formats: Iterable[str] = ("svg", "png", "pdf"),
    strict: bool = True,
) -> dict[str, str]:
    bundle = load_run_bundle(run_dir)
    root = Path(run_dir)
    dimension = int(bundle.provenance["dimension"])
    entry = catalog_entry(bundle.case_id)
    axis = _axis(dimension)
    static_formats = tuple(fmt for fmt in formats if fmt in {"svg", "png", "pdf"})
    if not static_formats:
        static_formats = ("svg",)
    figures: list[dict[str, Any]] = []
    rendered: list[str] = []
    omitted: list[str] = []
    caption, sha = _caption_and_sha(bundle)
    if bundle.verdict in {"pass", "fail"}:
        for kind in _selected_kinds(bundle.case_id, dimension, suite):
            try:
                payload = load_visual_series(root, kind)
            except VisualsError:
                omitted.append(kind)
                continue
            role = _role(kind)
            write_formats = (
                PUBLICATION_FORMATS if role == "publication" else static_formats
            )
            if kind == "storyboard":
                out_dir = root / "analysis" / "storyboards"
                outputs = {}
                for fmt in write_formats if role == "publication" else static_formats:
                    path = out_dir / f"{kind}.{fmt}"
                    render_storyboard(payload, path, caption=caption, provenance_sha=sha)
                    outputs[fmt] = str(path)
                times = [float(frame["time"]) for frame in payload["frames"]]
                steps = [int(frame.get("step") or 0) for frame in payload["frames"]]
                figure_verdict = "pass"
                sources = [f"analysis/visual_data/{kind}.json"]
                units = payload.get("units") or {"x": "x / L", "y": "scalar"}
                transform = "none"
            elif kind == "animation":
                anim_id = visual_contract_for(bundle.case_id)["animation"]
                name = anim_id["id"] if isinstance(anim_id, dict) else "canonical"
                frames_dir = root / "analysis" / "animations" / "frames" / name
                frames = write_animation_frames(
                    payload, frames_dir, caption=caption, provenance_sha=sha
                )
                sources = [f"analysis/visual_data/{kind}.json"] + [
                    str(path.resolve().relative_to(root.resolve())) for path in frames
                ]
                outputs = {"png": str(frames[-1])}
                figure_verdict = "not-supported"
                want_movie = any(fmt in {"mp4", "gif"} for fmt in formats)
                if want_movie:
                    try:
                        movies = assemble_ffmpeg(
                            frames_dir,
                            root / "analysis" / "animations" / "mp4" / f"{name}.mp4",
                            root / "analysis" / "animations" / "gif" / f"{name}.gif",
                            periodic=bool(payload.get("periodic")),
                        )
                        outputs.update(movies)
                        figure_verdict = "pass"
                    except VisualsError:
                        if strict and suite == "release":
                            figure_verdict = "not-supported"
                times = [float(frame["time"]) for frame in payload["frames"]]
                steps = [int(frame.get("step") or 0) for frame in payload["frames"]]
                units = payload.get("units") or {"x": "x / L", "y": "y / L"}
                transform = "none"
            else:
                prepared = _prepare(payload)
                role_dir = (
                    "hero"
                    if kind == "hero_figure"
                    else "publication"
                    if kind in PR_KINDS
                    else "diagnostic"
                )
                out_dir = root / "analysis" / "figures" / role_dir
                outputs = render_prepared(
                    prepared,
                    out_dir,
                    formats=write_formats,
                    caption=caption,
                    provenance_sha=sha,
                )
                figure_verdict = prepared.get("verdict") or derive_figure_verdict(payload)
                times = list(payload.get("times") or [1.0])
                steps = list(payload.get("step_numbers") or [0])
                sources = [f"analysis/visual_data/{kind}.json"]
                units = payload.get("units") or {"x": "1", "y": "1"}
                transform = _transform(kind, prepared)
            figures.append(
                _figure_entry(
                    bundle=bundle,
                    figure_id=kind,
                    kind=kind,
                    source_files=sources,
                    units=units,
                    outputs={
                        key: value
                        for key, value in outputs.items()
                        if key in {"svg", "png", "pdf", "mp4", "gif"}
                    },
                    run_dir=root,
                    verdict=figure_verdict,
                    times=times,
                    steps=steps,
                    transform=transform,
                )
            )
            rendered.append(kind)
    run_verdict = _combine_verdict(
        [item["verdict"] for item in figures], omitted, bundle.verdict
    )
    proves = (
        "Figures reconstruct from versioned fixture visual_data."
        if bundle.data_kind == "deterministic_fixture"
        else "Figures reconstruct from schema-valid metrics, provenance, and visual_data."
    )
    does_not = (
        "This fixture is not a live PoPS campaign result."
        if bundle.data_kind == "deterministic_fixture"
        else "Plots do not invent values missing from the run artefacts."
    )
    if run_verdict in {"not-run", "not-supported"}:
        proves = f"Run verdict {run_verdict} is preserved; no curve was invented."
    manifest = {
        "schema": "pops.verification.visual_manifest.v1",
        "case_id": bundle.case_id,
        "run_id": bundle.run_id,
        "data_kind": bundle.data_kind,
        "suite": suite,
        "verdict": run_verdict,
        "repository_sha": sha
        if bundle.data_kind == "deterministic_fixture"
        else bundle.provenance["repository_sha"],
        "resolved_case_digest": bundle.provenance["component_catalog_digest"],
        "program_digest": bundle.provenance["native_header_signature"],
        "native_artifact_digest": bundle.provenance["native_variant_manifest_digest"],
        "renderer": {"script": RENDERER_SCRIPT, "version": RENDERER_VERSION},
        "figures": figures,
        "dimensions": _dimension_block(entry, axis, rendered, omitted, run_verdict),
        "proves": proves,
        "does_not_prove": does_not,
    }
    _validator().validate(manifest)
    dest = root / "analysis" / "visual_manifest.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"visual_manifest": str(dest)}
