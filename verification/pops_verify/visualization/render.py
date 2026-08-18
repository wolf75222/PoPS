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
from verification.pops_verify.visualization.plots import (
    file_sha256,
    prepare_convergence,
    prepare_field,
    prepare_profile,
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
CONVERGENCE_KINDS = frozenset(
    {
        "spatial_convergence",
        "temporal_convergence",
        "report_figure",
        "backend_parity",
        "performance_breakdown",
        "coarse_fine_error",
    }
)
PROFILE_KINDS = frozenset(
    {
        "reference_profile",
        "signed_error_profile",
        "reference_comparison",
        "linecuts",
        "linecut",
        "phase_amplitude",
        "invariants_vs_time",
        "frequency_spectrum",
        "symmetry_metric",
    }
)
FIELD_KINDS = frozenset(
    {
        "exact_field",
        "numerical_field",
        "signed_error_field",
        "absolute_error_field",
        "field_snapshot",
        "amr_patch_map",
        "slice_xy",
        "slice_xz",
        "slice_yz",
        "isosurface",
        "amr_boxes",
        "hero_figure",
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
    if kind in CONVERGENCE_KINDS:
        prepared = prepare_convergence({**payload, "kind": "spatial_convergence"})
        prepared["kind"] = "spatial_convergence" if kind != "temporal_convergence" else "temporal_convergence"
        prepared["figure_id"] = payload.get("figure_id") or kind
        if kind == "temporal_convergence":
            prepared["xlabel"] = payload.get("units", {}).get("x") or "dt"
        return prepared
    if kind in PROFILE_KINDS:
        prepared = prepare_profile(payload)
        prepared["figure_id"] = payload.get("figure_id") or kind
        prepared["kind"] = "reference_profile"
        return prepared
    if kind in FIELD_KINDS or "field" in payload:
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


def _dimension_block(entry, executed_axis: str, rendered: list[str], verdict: str) -> dict[str, Any]:
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


def _figure_entry(
    *,
    bundle,
    figure_id: str,
    kind: str,
    source: str,
    units: dict[str, str],
    outputs: dict[str, str],
    run_dir: Path,
) -> dict[str, Any]:
    rel_outputs = {
        fmt: str(Path(path).resolve().relative_to(run_dir.resolve()))
        for fmt, path in outputs.items()
    }
    hashes = {fmt: file_sha256(path) for fmt, path in outputs.items()}
    return {
        "case_id": bundle.case_id,
        "run_id": bundle.run_id,
        "figure_id": figure_id,
        "kind": kind,
        "role": _role(kind),
        "source_files": [source],
        "variables": ["scalar"],
        "units": units,
        "transform": "loglog" if "convergence" in kind else "none",
        "color_range": None,
        "resolutions": [16, 32, 64, 128],
        "amr_levels": [0],
        "times": [1.0],
        "step_numbers": [0],
        "repository_sha": bundle.provenance["repository_sha"],
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
    }


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
    caption = bundle.data_kind_label
    sha = bundle.provenance["repository_sha"]
    if bundle.verdict in {"pass", "fail"}:
        for kind in _selected_kinds(bundle.case_id, dimension, suite):
            payload = load_visual_series(root, kind)
            if kind == "storyboard":
                out_dir = root / "analysis" / "storyboards"
                outputs = {}
                for fmt in static_formats:
                    path = out_dir / f"{kind}.{fmt}"
                    render_storyboard(payload, path, caption=caption, provenance_sha=sha)
                    outputs[fmt] = str(path)
            elif kind == "animation":
                anim_id = visual_contract_for(bundle.case_id)["animation"]
                name = anim_id["id"] if isinstance(anim_id, dict) else "canonical"
                frames_dir = root / "analysis" / "animations" / "frames" / name
                frames = write_animation_frames(
                    payload, frames_dir, caption=caption, provenance_sha=sha
                )
                outputs = {"png": str(frames[0])}
                storyboard_path = root / "analysis" / "storyboards" / f"{name}.png"
                if "storyboard" not in rendered:
                    story_payload = {
                        "units": payload.get("units") or {"x": "x", "y": "y"},
                        "frames": [
                            {
                                "event": f"frame-{index}",
                                "time": frame.get("time"),
                                "step": frame.get("step"),
                                "series": [
                                    {
                                        "name": "midline",
                                        "x": frame["x"],
                                        "y": frame["field"][len(frame["field"]) // 2],
                                    }
                                ],
                            }
                            for index, frame in enumerate(payload["frames"][:4])
                        ],
                    }
                    render_storyboard(
                        story_payload,
                        storyboard_path,
                        caption=caption,
                        provenance_sha=sha,
                    )
                    outputs["storyboard"] = str(storyboard_path)
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
                    except VisualsError:
                        if strict and suite == "release":
                            raise
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
                    formats=static_formats,
                    caption=caption,
                    provenance_sha=sha,
                )
            figures.append(
                _figure_entry(
                    bundle=bundle,
                    figure_id=kind,
                    kind=kind,
                    source=f"analysis/visual_data/{kind}.json",
                    units=payload.get("units") or {"x": "1", "y": "1"},
                    outputs={key: value for key, value in outputs.items() if key in {"svg", "png", "pdf", "mp4", "gif"}},
                    run_dir=root,
                )
            )
            rendered.append(kind)
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
    if bundle.verdict in {"not-run", "not-supported"}:
        proves = f"Run verdict {bundle.verdict} is preserved; no curve was invented."
    manifest = {
        "schema": "pops.verification.visual_manifest.v1",
        "case_id": bundle.case_id,
        "run_id": bundle.run_id,
        "data_kind": bundle.data_kind,
        "suite": suite,
        "verdict": bundle.verdict,
        "repository_sha": bundle.provenance["repository_sha"],
        "resolved_case_digest": bundle.provenance["component_catalog_digest"],
        "program_digest": bundle.provenance["native_header_signature"],
        "native_artifact_digest": bundle.provenance["native_variant_manifest_digest"],
        "renderer": {"script": RENDERER_SCRIPT, "version": RENDERER_VERSION},
        "figures": figures,
        "dimensions": _dimension_block(entry, axis, rendered, bundle.verdict),
        "proves": proves,
        "does_not_prove": does_not,
    }
    _validator().validate(manifest)
    dest = root / "analysis" / "visual_manifest.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"visual_manifest": str(dest)}
