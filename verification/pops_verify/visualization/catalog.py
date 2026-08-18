"""Exhaustive Phase 8 case × dimension × artefact catalog (plan §40.6.11)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DimCode = Literal["R", "E", "N/A"]
AnimCode = Literal["required", "optional", "none"]

SCIENTIFIC_CASE_IDS: tuple[str, ...] = (
    "TR-01",
    "TR-02",
    "TR-03",
    "TR-04",
    "TR-05",
    "TR-06",
    "TR-07",
    "EU-01",
    "EU-02",
    "EU-03",
    "EU-04",
    "EU-05",
    "EU-06",
    "PO-01",
    "PO-02",
    "PO-03",
    "PO-04",
    "PO-05",
    "PO-06",
    "PO-07",
    "CP-01",
    "CP-02",
    "CP-03",
    "CP-04",
    "CP-05",
    "CP-06",
    "CP-07",
    "CP-08",
    "CP-09",
    "CP-10",
    "CP-11",
    "CP-12",
    "TM-01",
    "TM-02",
    "TM-03",
    "TM-04",
    "TM-05",
    "TM-06",
    "TM-07",
    "TM-08",
    "AM-01",
    "AM-02",
    "AM-03",
    "AM-04",
    "AM-05",
    "AM-06",
    "AM-07",
    "AM-08",
    "AM-09",
    "AM-10",
    "AM-11",
    "AM-12",
    "RB-01",
    "RB-02",
    "RB-03",
    "RB-04",
    "RB-05",
    "RB-06",
    "RB-07",
    "RB-08",
    "RB-09",
    "GE-01",
    "GE-02",
    "GE-03",
    "GE-04",
    "GE-05",
    "GE-06",
    "IF-01",
    "IF-02",
    "IF-03",
    "IF-04",
    "IF-05",
    "IF-06",
    "IF-07",
    "IF-08",
    "IF-09",
    "IF-10",
    "PF-01",
    "PF-02",
    "PF-03",
    "PF-04",
    "PF-05",
    "PF-06",
    "PF-07",
    "PF-08",
    "PF-09",
    "PF-10",
    "PF-11",
    "PF-12",
)

P0_STATIC_CASES: tuple[str, ...] = (
    "TR-01",
    "EU-01",
    "EU-02",
    "PO-01",
    "CP-01",
    "CP-02",
    "TM-01",
    "TM-02",
    "AM-01",
    "AM-09",
    "RB-01",
    "RB-05",
    "IF-01",
    "IF-03",
    "PF-06",
)

HERO_CASES: tuple[str, ...] = (
    "TR-03",
    "EU-02",
    "CP-02",
    "CP-11",
    "GE-06",
    "AM-01",
    "AM-02",
    "AM-03",
    "AM-11",
    "RB-05",
    "RB-08",
    "GE-03",
    "PF-06",
    "PF-11",
)

_DIMS: dict[str, tuple[DimCode, DimCode, DimCode]] = {
    "TR-01": ("R", "R", "R"),
    "TR-02": ("R", "R", "E"),
    "TR-03": ("N/A", "R", "E"),
    "TR-04": ("N/A", "E", "R"),
    "TR-05": ("R", "R", "R"),
    "TR-06": ("E", "R", "R"),
    "TR-07": ("R", "R", "E"),
    "EU-01": ("R", "R", "R"),
    "EU-02": ("N/A", "R", "E"),
    "EU-03": ("R", "R", "R"),
    "EU-04": ("R", "R", "E"),
    "EU-05": ("N/A", "R", "E"),
    "EU-06": ("R", "R", "R"),
    "PO-01": ("R", "R", "R"),
    "PO-02": ("R", "R", "E"),
    "PO-03": ("R", "R", "E"),
    "PO-04": ("E", "R", "R"),
    "PO-05": ("R", "R", "R"),
    "PO-06": ("R", "R", "R"),
    "PO-07": ("R", "R", "R"),
    "CP-01": ("R", "R", "E"),
    "CP-02": ("R", "E", "E"),
    "CP-03": ("R", "E", "E"),
    "CP-04": ("N/A", "R", "R"),
    "CP-05": ("R", "E", "E"),
    "CP-06": ("R", "E", "N/A"),
    "CP-07": ("R", "R", "E"),
    "CP-08": ("R", "R", "R"),
    "CP-09": ("R", "R", "E"),
    "CP-10": ("R", "E", "N/A"),
    "CP-11": ("N/A", "R", "N/A"),
    "CP-12": ("R", "R", "R"),
    "TM-01": ("R", "R", "E"),
    "TM-02": ("R", "E", "N/A"),
    "TM-03": ("R", "R", "R"),
    "TM-04": ("R", "R", "R"),
    "TM-05": ("R", "E", "N/A"),
    "TM-06": ("R", "E", "E"),
    "TM-07": ("R", "R", "E"),
    "TM-08": ("R", "R", "E"),
    "AM-01": ("R", "R", "R"),
    "AM-02": ("R", "R", "E"),
    "AM-03": ("E", "R", "R"),
    "AM-04": ("R", "R", "R"),
    "AM-05": ("R", "R", "E"),
    "AM-06": ("R", "R", "R"),
    "AM-07": ("R", "R", "R"),
    "AM-08": ("R", "R", "R"),
    "AM-09": ("R", "R", "R"),
    "AM-10": ("R", "R", "R"),
    "AM-11": ("R", "R", "E"),
    "AM-12": ("E", "R", "R"),
    "RB-01": ("R", "R", "R"),
    "RB-02": ("R", "E", "E"),
    "RB-03": ("R", "E", "E"),
    "RB-04": ("R", "E", "N/A"),
    "RB-05": ("E", "R", "R"),
    "RB-06": ("R", "R", "R"),
    "RB-07": ("N/A", "R", "N/A"),
    "RB-08": ("N/A", "R", "N/A"),
    "RB-09": ("R", "N/A", "N/A"),
    "GE-01": ("E", "R", "N/A"),
    "GE-02": ("E", "R", "N/A"),
    "GE-03": ("R", "R", "R"),
    "GE-04": ("E", "R", "N/A"),
    "GE-05": ("E", "R", "N/A"),
    "GE-06": ("N/A", "R", "N/A"),
    "IF-01": ("R", "R", "R"),
    "IF-02": ("R", "R", "R"),
    "IF-03": ("R", "R", "R"),
    "IF-04": ("R", "R", "R"),
    "IF-05": ("R", "R", "R"),
    "IF-06": ("R", "R", "R"),
    "IF-07": ("R", "R", "E"),
    "IF-08": ("R", "R", "R"),
    "IF-09": ("R", "R", "R"),
    "IF-10": ("R", "R", "R"),
    "PF-01": ("R", "R", "R"),
    "PF-02": ("R", "R", "R"),
    "PF-03": ("R", "R", "R"),
    "PF-04": ("R", "R", "R"),
    "PF-05": ("R", "R", "R"),
    "PF-06": ("R", "R", "R"),
    "PF-07": ("E", "R", "R"),
    "PF-08": ("R", "R", "R"),
    "PF-09": ("E", "R", "R"),
    "PF-10": ("R", "R", "R"),
    "PF-11": ("E", "R", "R"),
    "PF-12": ("E", "R", "E"),
}

_NA_REASONS: dict[str, dict[str, str]] = {
    "TR-03": {"1d": "The reversible transport vortex is intrinsically multidimensional."},
    "TR-04": {"1d": "Face-edge-corner crossing is not defined in one dimension."},
    "EU-02": {"1d": "The isentropic vortex is intrinsically multidimensional."},
    "EU-05": {"1d": "The Gresho vortex is intrinsically multidimensional."},
    "CP-02": {
        "2d": (
            "v1.5 selected package is the 1-d cold Langmuir wave; "
            "2-d is extended and not-run."
        ),
        "3d": (
            "v1.5 selected package is the 1-d cold Langmuir wave; "
            "3-d is extended and not-run."
        ),
    },
    "CP-04": {"1d": "An oblique electrostatic wave requires at least two dimensions."},
    "CP-06": {"3d": "Three-dimensional ion-acoustic execution is N/A by default."},
    "CP-10": {"3d": "Three-dimensional Jeans execution is N/A by default."},
    "CP-11": {
        "1d": "The linear diocotron mode is intrinsically two-dimensional.",
        "3d": "Three-dimensional diocotron is N/A in this catalogue.",
    },
    "TM-02": {"3d": "Three-dimensional noncommuting Strang is N/A by default."},
    "TM-05": {"3d": "Three-dimensional AP-limit execution requires a dedicated model."},
    "RB-04": {"3d": "Three-dimensional Shu-Osher is N/A by default."},
    "RB-07": {
        "1d": "The Liska-Wendroff implosion is a two-dimensional configuration.",
        "3d": "Three-dimensional Liska-Wendroff is N/A.",
    },
    "RB-08": {
        "1d": "Double Mach Reflection is a two-dimensional configuration.",
        "3d": "Three-dimensional Double Mach Reflection is N/A.",
    },
    "RB-09": {
        "2d": "Woodward-Colella blast waves are specified in one dimension.",
        "3d": "Woodward-Colella blast waves are specified in one dimension.",
    },
    "GE-01": {"3d": "Polar manufactured Poisson has no 3-d polar runtime in this catalogue."},
    "GE-02": {"3d": "Solid-body polar rotation has no 3-d polar runtime in this catalogue."},
    "GE-04": {"3d": "Cartesian/polar oracle comparison is N/A in three dimensions."},
    "GE-05": {"3d": "Polar-axis regularity has no 3-d polar runtime in this catalogue."},
    "GE-06": {
        "1d": "Cartesian diocotron is intrinsically two-dimensional.",
        "3d": "Cartesian diocotron is N/A in three dimensions.",
    },
}

_TITLES: dict[str, str] = {
    "TR-01": "Periodic sinusoidal advection",
    "TR-02": "Transported Gaussian pulse",
    "TR-03": "Reversible transport vortex",
    "TR-04": "Face-edge-corner crossing",
    "TR-05": "Block-boundary translation",
    "TR-06": "Axis permutation and rotation",
    "TR-07": "Transported discontinuous slot or disk",
    "EU-01": "Linear Euler eigenmodes",
    "EU-02": "Advected isentropic vortex",
    "EU-03": "Complete Euler manufactured solution",
    "EU-04": "Standing acoustic wave with reflecting walls",
    "EU-05": "Stationary Gresho vortex",
    "EU-06": "Exact uniform-flow preservation",
    "PO-01": "Periodic trigonometric Poisson",
    "PO-02": "Poisson with inhomogeneous Dirichlet",
    "PO-03": "Poisson with Neumann nullspace",
    "PO-04": "Huang-Greengard on AMR",
    "PO-05": "FFT versus geometric multigrid",
    "PO-06": "Gradient order and coarse-fine placement",
    "PO-07": "Elliptic tolerance sensitivity",
    "CP-01": "Complete Euler-Poisson manufactured solution",
    "CP-02": "Cold Langmuir wave",
    "CP-03": "Warm Langmuir wave and dispersion",
    "CP-04": "Oblique electrostatic wave",
    "CP-05": "Generic multifluid eigenmodes",
    "CP-06": "Ion-acoustic wave",
    "CP-07": "Pressure-electric equilibrium",
    "CP-08": "Uniform electric-field acceleration",
    "CP-09": "Linearized Debye screening",
    "CP-10": "Stable and unstable Jeans modes",
    "CP-11": "Linear diocotron mode",
    "CP-12": "Charge cancellation and multi-species equilibrium",
    "TM-01": "Pure temporal convergence",
    "TM-02": "Strang splitting with noncommuting operators",
    "TM-03": "Exact two-species collisional relaxation",
    "TM-04": "Exact Larmor rotation",
    "TM-05": "Asymptotic-preserving limit",
    "TM-06": "Multirate species substeps",
    "TM-07": "Field update at every RK stage",
    "TM-08": "Ordering symmetry and reversibility",
    "AM-01": "Wave crossing a static coarse-fine interface",
    "AM-02": "Prescribed moving patch",
    "AM-03": "Tagging-driven dynamic refinement",
    "AM-04": "Temporal subcycling",
    "AM-05": "Regrid frequency",
    "AM-06": "Total level count and provider capacity",
    "AM-07": "AMR versus equivalent uniform fine",
    "AM-08": "Interface-placement sweep",
    "AM-09": "Integrated conservation and reflux",
    "AM-10": "Composite multilevel Poisson",
    "AM-11": "Euler-Poisson synchronization on AMR",
    "AM-12": "Patch shape and symmetry",
    "RB-01": "Sod with AMR interface crossing",
    "RB-02": "Double rarefaction and near-vacuum",
    "RB-03": "Very strong shock",
    "RB-04": "Shu-Osher",
    "RB-05": "Off-center Sedov",
    "RB-06": "Noh",
    "RB-07": "Liska-Wendroff implosion",
    "RB-08": "Double Mach Reflection",
    "RB-09": "Woodward-Colella blast waves",
    "GE-01": "Radial manufactured Poisson in polar coordinates",
    "GE-02": "Exact solid-body scalar rotation",
    "GE-03": "Radial acoustic wave in Cartesian coordinates",
    "GE-04": "Same radial oracle in Cartesian and polar",
    "GE-05": "Polar-axis regularity and volume conservation",
    "GE-06": "Cartesian diocotron: modes, symmetry, and AMR",
    "IF-01": "MPI layout invariance",
    "IF-02": "Kokkos OpenMP thread invariance",
    "IF-03": "Kokkos space and MPI parity",
    "IF-04": "Checkpoint and restart",
    "IF-05": "Output-cadence invariance",
    "IF-06": "Deterministic mode and reductions",
    "IF-07": "Native, DSL, and hybrid parity",
    "IF-08": "Compilers, build modes, and exact native specialization",
    "IF-09": "Floating-point precision",
    "IF-10": "HDF5 I/O and distributed reread",
    "PF-01": "MultiFab arithmetic and periodic halo",
    "PF-02": "Scalar multigrid",
    "PF-03": "Finite-volume advection RHS",
    "PF-04": "Complete Euler step",
    "PF-05": "Composite AMR Poisson",
    "PF-06": "Complete Euler-Poisson step",
    "PF-07": "Regrid and clustering",
    "PF-08": "Reflux and AMR synchronization",
    "PF-09": "Load balancing and migration",
    "PF-10": "Checkpoint and parallel HDF5",
    "PF-11": "End-to-end dynamic AMR",
    "PF-12": "High-moment-count state",
}

# Per-dimension animation requirement from §40.6.11 storyboard/animation lines.
_ANIM: dict[str, tuple[AnimCode, AnimCode, AnimCode]] = {
    "TR-01": ("required", "required", "required"),
    "TR-02": ("optional", "required", "optional"),
    "TR-03": ("none", "required", "optional"),
    "TR-04": ("none", "optional", "optional"),
    "TR-05": ("none", "optional", "optional"),
    "TR-06": ("optional", "optional", "optional"),
    "TR-07": ("required", "required", "optional"),
    "EU-01": ("optional", "required", "required"),
    "EU-02": ("none", "required", "optional"),
    "EU-03": ("none", "optional", "optional"),
    "EU-04": ("required", "required", "optional"),
    "EU-05": ("none", "required", "optional"),
    "EU-06": ("none", "none", "none"),
    "PO-01": ("none", "none", "none"),
    "PO-02": ("none", "none", "none"),
    "PO-03": ("none", "none", "none"),
    "PO-04": ("none", "optional", "optional"),
    "PO-05": ("none", "none", "none"),
    "PO-06": ("optional", "optional", "optional"),
    "PO-07": ("none", "none", "none"),
    "CP-01": ("optional", "optional", "optional"),
    "CP-02": ("required", "optional", "optional"),
    "CP-03": ("optional", "optional", "optional"),
    "CP-04": ("none", "optional", "optional"),
    "CP-05": ("optional", "optional", "optional"),
    "CP-06": ("optional", "optional", "none"),
    "CP-07": ("none", "none", "none"),
    "CP-08": ("none", "none", "none"),
    "CP-09": ("none", "none", "none"),
    "CP-10": ("optional", "optional", "none"),
    "CP-11": ("none", "required", "none"),
    "CP-12": ("none", "none", "none"),
    "TM-01": ("none", "none", "none"),
    "TM-02": ("none", "none", "none"),
    "TM-03": ("none", "none", "none"),
    "TM-04": ("optional", "optional", "optional"),
    "TM-05": ("none", "none", "none"),
    "TM-06": ("optional", "optional", "optional"),
    "TM-07": ("none", "none", "none"),
    "TM-08": ("optional", "optional", "optional"),
    "AM-01": ("required", "required", "required"),
    "AM-02": ("optional", "required", "optional"),
    "AM-03": ("optional", "required", "required"),
    "AM-04": ("optional", "optional", "optional"),
    "AM-05": ("optional", "optional", "optional"),
    "AM-06": ("none", "none", "none"),
    "AM-07": ("none", "none", "none"),
    "AM-08": ("optional", "optional", "optional"),
    "AM-09": ("none", "none", "none"),
    "AM-10": ("none", "none", "none"),
    "AM-11": ("optional", "optional", "optional"),
    "AM-12": ("optional", "optional", "optional"),
    "RB-01": ("required", "required", "none"),
    "RB-02": ("optional", "optional", "optional"),
    "RB-03": ("none", "none", "none"),
    "RB-04": ("optional", "optional", "none"),
    "RB-05": ("none", "required", "required"),
    "RB-06": ("optional", "optional", "optional"),
    "RB-07": ("none", "required", "none"),
    "RB-08": ("none", "optional", "none"),
    "RB-09": ("optional", "none", "none"),
    "GE-01": ("none", "none", "none"),
    "GE-02": ("none", "required", "none"),
    "GE-03": ("optional", "optional", "none"),
    "GE-04": ("none", "none", "none"),
    "GE-05": ("none", "none", "none"),
    "GE-06": ("none", "required", "none"),
    "IF-01": ("none", "none", "none"),
    "IF-02": ("none", "none", "none"),
    "IF-03": ("none", "none", "none"),
    "IF-04": ("none", "none", "none"),
    "IF-05": ("none", "none", "none"),
    "IF-06": ("none", "none", "none"),
    "IF-07": ("none", "none", "none"),
    "IF-08": ("none", "none", "none"),
    "IF-09": ("none", "none", "none"),
    "IF-10": ("none", "none", "none"),
    "PF-01": ("none", "none", "none"),
    "PF-02": ("none", "none", "none"),
    "PF-03": ("none", "none", "none"),
    "PF-04": ("none", "none", "none"),
    "PF-05": ("none", "none", "none"),
    "PF-06": ("none", "none", "none"),
    "PF-07": ("none", "optional", "optional"),
    "PF-08": ("none", "none", "none"),
    "PF-09": ("none", "required", "required"),
    "PF-10": ("none", "none", "none"),
    "PF-11": ("none", "required", "required"),
    "PF-12": ("none", "none", "none"),
}

_STORYBOARD_REQUIRED = frozenset(
    {
        "TR-01",
        "TR-02",
        "TR-03",
        "TR-04",
        "TR-05",
        "TR-06",
        "TR-07",
        "EU-01",
        "EU-02",
        "EU-03",
        "EU-04",
        "EU-05",
        "EU-06",
        "PO-04",
        "PO-06",
        "CP-01",
        "CP-02",
        "CP-04",
        "CP-07",
        "CP-11",
        "TM-07",
        "AM-01",
        "AM-02",
        "AM-03",
        "AM-04",
        "AM-05",
        "AM-06",
        "AM-07",
        "AM-08",
        "AM-09",
        "AM-11",
        "AM-12",
        "RB-01",
        "RB-05",
        "RB-07",
        "RB-08",
        "GE-02",
        "GE-03",
        "GE-06",
        "IF-04",
        "IF-10",
        "PF-09",
        "PF-11",
    }
)


def _unique(items: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def _artifacts_for(case_id: str, axis: str, code: DimCode) -> tuple[str, ...]:
    if code == "N/A":
        return ()
    family = case_id.split("-", 1)[0]
    if family == "PF":
        kinds = ["performance_breakdown", "report_figure"]
        if case_id in {"PF-07", "PF-09", "PF-11"} and axis != "1d":
            kinds.append("amr_patch_map")
        if _ANIM[case_id][{"1d": 0, "2d": 1, "3d": 2}[axis]] == "required":
            kinds.extend(["storyboard", "animation"])
        return _unique(kinds)
    if family == "IF":
        kinds = ["backend_parity", "report_figure"]
        if axis == "1d":
            kinds.insert(0, "reference_comparison")
        elif axis == "2d":
            kinds.insert(0, "signed_error_field")
        else:
            kinds[0:0] = ["slice_xy", "slice_xz", "slice_yz"]
            kinds.append("isosurface")
        if case_id in {"IF-04", "IF-10"}:
            kinds.append("storyboard")
        return _unique(kinds)
    if family == "TM":
        kinds = ["temporal_convergence", "invariants_vs_time", "report_figure"]
        if axis == "2d":
            kinds.insert(0, "field_snapshot")
        if axis == "3d":
            kinds[0:0] = ["slice_xy", "linecut"]
        if case_id == "TM-07":
            kinds.append("storyboard")
        return _unique(kinds)
    if axis == "1d":
        kinds = [
            "reference_profile",
            "signed_error_profile",
            "spatial_convergence",
            "report_figure",
        ]
    elif axis == "2d":
        kinds = [
            "exact_field",
            "numerical_field",
            "signed_error_field",
            "linecuts",
            "report_figure",
        ]
    else:
        kinds = [
            "slice_xy",
            "slice_xz",
            "slice_yz",
            "linecut",
            "report_figure",
        ]
    if family in {"TR", "EU", "CP"}:
        kinds.append("invariants_vs_time")
    if case_id in {"TR-01", "EU-01", "CP-02", "CP-03"}:
        kinds.append("phase_amplitude")
    if family in {"AM", "TR", "RB", "GE"} and axis != "1d":
        kinds.append("amr_patch_map")
    if family == "AM":
        kinds.append("coarse_fine_error")
    if family == "PO":
        kinds.append("spatial_convergence")
    if axis == "3d" and family in {"TR", "EU", "PO", "AM", "RB"}:
        kinds.append("amr_boxes")
    if axis == "3d" and family in {"TR", "EU", "PO", "AM", "CP", "RB", "GE"}:
        kinds.append("isosurface")
    anim = _ANIM[case_id][{"1d": 0, "2d": 1, "3d": 2}[axis]]
    if anim == "required" or case_id in _STORYBOARD_REQUIRED:
        kinds.append("storyboard")
    if anim == "required":
        kinds.append("animation")
    if case_id in HERO_CASES:
        kinds.append("hero_figure")
    return _unique(kinds)


def _status(code: DimCode) -> str:
    return {"R": "required", "E": "extended", "N/A": "not_applicable"}[code]


@dataclass(frozen=True, slots=True)
class VisualCatalogEntry:
    case_id: str
    title: str
    dimension_codes: tuple[DimCode, DimCode, DimCode]
    na_reasons: dict[str, str]
    artifacts: dict[str, tuple[str, ...]]
    animation: tuple[AnimCode, AnimCode, AnimCode]
    storyboard_required: bool
    p0: bool
    hero: bool


def catalog_entry(case_id: str) -> VisualCatalogEntry:
    if case_id not in _DIMS:
        raise KeyError(case_id)
    codes = _DIMS[case_id]
    reasons = dict(_NA_REASONS.get(case_id, {}))
    artifacts = {
        axis: _artifacts_for(case_id, axis, code)
        for axis, code in zip(("1d", "2d", "3d"), codes, strict=True)
    }
    return VisualCatalogEntry(
        case_id=case_id,
        title=_TITLES[case_id],
        dimension_codes=codes,
        na_reasons=reasons,
        artifacts=artifacts,
        animation=_ANIM[case_id],
        storyboard_required=case_id in _STORYBOARD_REQUIRED,
        p0=case_id in P0_STATIC_CASES,
        hero=case_id in HERO_CASES,
    )


def iter_catalog() -> tuple[VisualCatalogEntry, ...]:
    return tuple(catalog_entry(case_id) for case_id in SCIENTIFIC_CASE_IDS)


def dimension_code(case_id: str) -> tuple[DimCode, DimCode, DimCode]:
    return catalog_entry(case_id).dimension_codes


def visual_contract_for(case_id: str) -> dict:
    entry = catalog_entry(case_id)
    required: list[str] = []
    optional = ["hero_figure"] if entry.hero else []
    for axis, code in zip(("1d", "2d", "3d"), entry.dimension_codes, strict=True):
        if code == "R":
            required.extend(entry.artifacts[axis])
    required = list(dict.fromkeys(required))
    animation = None
    am01_events = [
        "before_entry",
        "entry",
        "inside_fine",
        "exit",
        "periodic_crossing",
        "final",
    ]
    default_events = ["initial", "mid", "final"]
    if any(flag == "required" for flag in entry.animation):
        animation = {
            "id": f"{case_id.lower().replace('-', '_')}_canonical",
            "master": "mp4",
            "preview": "gif",
            "frames": "accepted_states",
            "color_limits": "global",
            "overlays": ["time", "leaf_cells"],
            "key_events": list(am01_events if case_id == "AM-01" else default_events),
        }
        if case_id == "AM-01":
            animation = {
                "id": "wave_coarse_fine_crossing",
                "master": "mp4",
                "preview": "gif",
                "frames": "accepted_states",
                "color_limits": "global",
                "overlays": [
                    "amr_patches",
                    "coarse_fine_interfaces",
                    "time",
                    "leaf_cells",
                ],
                "key_events": list(am01_events),
            }
    storyboard = None
    if entry.storyboard_required:
        if case_id == "AM-01":
            events = list(am01_events)
        elif animation and animation.get("key_events"):
            events = list(animation["key_events"])
        else:
            events = list(default_events)
        storyboard = {"key_events": events}
    dimensions = {}
    for axis, code in zip(("1d", "2d", "3d"), entry.dimension_codes, strict=True):
        block: dict = {"status": _status(code)}
        if code == "N/A":
            block["justification"] = entry.na_reasons[axis]
        elif code == "R":
            block["required"] = list(entry.artifacts[axis])
        else:
            block["required_when_executed"] = list(entry.artifacts[axis])
            if axis in entry.na_reasons:
                block["justification"] = entry.na_reasons[axis]
        dimensions[axis] = block
    return {
        "schema": "pops.verification.visuals.v1",
        "required": required,
        "optional": optional,
        "animation": animation,
        "storyboard": storyboard,
        "acceptance": {
            "require_source_data": True,
            "require_manifest": True,
            "require_fixed_animation_scale": True,
            "require_storyboard_for_animation": True,
            "forbid_missing_units": True,
        },
        "dimensions": dimensions,
        "report": {
            "require_quantitative_companion": True,
            "require_source_data": True,
            "require_same_run_identity": True,
        },
    }
