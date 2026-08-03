"""ADC-749: legacy transport-boundary authorities may only disappear.

The prepared hyperbolic boundary path is already the compiled Uniform/AMR
route.  A few older native authorities still exist while their replacements
need metric and characteristic kernels.  Keep their remaining
lexical surface bounded so adjacent work cannot silently create another
transport-boundary engine before that cutover is complete.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PRODUCTION_ROOTS = (ROOT / "include/pops", ROOT / "src/runtime")

# Exact upper bounds on the consolidated ADC-749 branch.  Counts deliberately
# include comments: closure requires deleting the legacy vocabulary as well as
# its executable branches.  A deletion passes without editing this ledger;
# any new file or additional occurrence fails closed.
LEGACY_AUTHORITY_LIMITS = {
    "AmrBoundaryFillAuthority": {
        "include/pops/coupling/amr/amr_coupler_mp.hpp": 2,
        "include/pops/numerics/time/amr/levels/amr_subcycling.hpp": 6,
        "include/pops/runtime/amr/amr_runtime.hpp": 1,
        "include/pops/runtime/builders/compiled/amr_dsl_block.hpp": 1,
    },
    "make_amr_boundary_fill_authority": {
        "include/pops/numerics/time/amr/levels/amr_subcycling.hpp": 1,
        "include/pops/runtime/builders/compiled/amr_dsl_block.hpp": 1,
    },
    "wall_radial": {
        "include/pops/numerics/spatial/operators/polar_operator.hpp": 9,
        "include/pops/runtime/builders/block/block_builder_polar.hpp": 13,
        "src/runtime/system/system_polar.cpp": 2,
    },
    "fill_ghosts_polar": {
        "include/pops/runtime/builders/block/block_builder_polar.hpp": 3,
    },
    "transport_bc": {
        "include/pops/runtime/amr/amr_runtime.hpp": 3,
        "include/pops/runtime/builders/compiled/amr_dsl_block.hpp": 7,
        "include/pops/runtime/program/amr_program_context.hpp": 2,
    },
}


def _production_sources() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for root in PRODUCTION_ROOTS
            for path in root.rglob("*")
            if path.suffix in {".cpp", ".hpp"}
        )
    )


def _occurrences() -> dict[str, dict[str, int]]:
    patterns = {
        identifier: re.compile(r"\b%s\b" % re.escape(identifier))
        for identifier in LEGACY_AUTHORITY_LIMITS
    }
    counts = {identifier: {} for identifier in patterns}
    for path in _production_sources():
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT).as_posix()
        for identifier, pattern in patterns.items():
            count = len(pattern.findall(source))
            if count:
                counts[identifier][relative] = count
    return counts


def test_legacy_transport_boundary_authorities_can_only_shrink() -> None:
    occurrences = _occurrences()
    violations = []
    for identifier, limits in LEGACY_AUTHORITY_LIMITS.items():
        for path, count in occurrences[identifier].items():
            limit = limits.get(path, 0)
            if count > limit:
                violations.append(
                    "%s: %s has %d occurrence(s), allowed at most %d"
                    % (identifier, path, count, limit)
                )

    assert not violations, (
        "legacy transport-boundary authority expanded; lower the route to "
        "PreparedBoundaryPlan instead:\n  " + "\n  ".join(violations)
    )


def test_resolved_transport_authority_accepts_only_executable_descriptors() -> None:
    """Numerical resolution, rather than a later compile/bind phase, owns acceptance."""
    source = ROOT / "python/pops/boundary/transport.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    authority = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ResolvedTransportBoundarySet"
    )
    post_init = next(
        node for node in authority.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "__post_init__"
    )
    calls = {
        node.func.attr
        for node in ast.walk(post_init)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
    }
    assert "_native_contract" in calls, (
        "ResolvedTransportBoundarySet must authenticate the complete executable boundary "
        "contract during numerical resolution; do not defer unsupported descriptors to compile"
    )


def test_boundary_provider_identity_cannot_erase_its_selected_law() -> None:
    """Every provider identity must retain the law selected by its public factory."""
    source = ROOT / "python/pops/mesh/boundaries/providers.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    provider = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BoundaryProvider"
    )
    annotations = {
        node.target.id
        for node in provider.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "kind" in annotations, "BoundaryProvider must retain one immutable typed law"
    canonical = next(
        node for node in provider.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "canonical_identity"
    )
    keys = {
        node.value for node in ast.walk(canonical)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "provider_kind" in keys, (
        "BoundaryProvider canonical identity must authenticate its selected law"
    )


def test_post_riemann_flux_is_one_typed_outward_oriented_pipeline_stage() -> None:
    catalog = json.loads(
        (ROOT / "schemas/component_catalog.v2.json").read_text(encoding="utf-8")
    )
    interface = next(
        row for row in catalog["native_interface_abis"]
        if row["name"] == "boundary_flux"
    )
    assert interface["cpp_table"] == "PopsBoundaryFluxApiV1"
    assert interface["operations"] == ["transform_faces"]
    route = catalog["boundary_handle_native_routes"]["boundary_flux_provider"]
    assert route["interface"] == "boundary_flux"
    assert route["operation"] == "transform_faces"

    executor = (
        ROOT / "include/pops/mesh/boundary/boundary_component_executor.hpp"
    ).read_text(encoding="utf-8")
    assert re.search(
        r"const double outward\s*=\s*static_cast<double>\(workspace\.side\)\s*\*",
        executor,
    )
    assert re.search(
        r"static_cast<Real>\(static_cast<double>\(workspace\.side\)\s*\*\s*outward\)",
        executor,
    )

    uniform = (
        ROOT / "include/pops/runtime/builders/block/block_builder.hpp"
    ).read_text(encoding="utf-8")
    uniform_stage = uniform[
        uniform.index("assemble_rhs_without_prepared_interfaces"):
        uniform.index("struct BlockRhsEval")
    ]
    assert uniform_stage.index("compute_face_fluxes") < uniform_stage.index(
        "transform_grid_boundary_fluxes"
    ) < uniform_stage.index("mf_eval_rhs")
    unqualified = uniform[
        uniform.index("void operator()(MultiFab& U, MultiFab& R) const"):
        uniform.index(
            "void operator()(const runtime::multiblock::BoundaryEvaluationPoint& point",
        )
    ]
    assert "has_flux_transformations()" in unqualified
    assert "requires a BoundaryEvaluationPoint" in unqualified
    assert "has_omitted_faces()" in unqualified
    assert "shared-interface flux requires BoundaryEvaluationPoint group authority" in unqualified
    assert "eval_core_filled(U, R);" in unqualified

    amr = (
        ROOT / "include/pops/runtime/builders/compiled/amr_dsl_block.hpp"
    ).read_text(encoding="utf-8")
    first_flux = amr.index("detail::compute_amr_face_fluxes")
    first_transform = amr.index("transform_grid_boundary_fluxes", first_flux)
    first_divergence = amr.index("pops::mf_eval_rhs", first_transform)
    assert first_flux < first_transform < first_divergence


def test_no_flux_is_a_builtin_face_law_of_the_same_prepared_pipeline() -> None:
    transport = (ROOT / "python/pops/boundary/transport.py").read_text(encoding="utf-8")
    hyperbolic = (
        ROOT / "include/pops/mesh/boundary/prepared_hyperbolic_boundary.hpp"
    ).read_text(encoding="utf-8")
    plan = (
        ROOT / "include/pops/mesh/boundary/prepared_boundary_plan.hpp"
    ).read_text(encoding="utf-8")
    operator = (
        ROOT / "include/pops/runtime/builders/block/block_builder.hpp"
    ).read_text(encoding="utf-8")

    assert 'condition_type: ClassVar[str] = "no_flux"' in transport
    assert '"no_flux": LowLevelNoFlux' in transport
    assert 'token == "no_flux"' in hyperbolic
    assert "HyperbolicBoundaryLaw::NoFlux" in plan
    assert "zero_prepared_boundary_fluxes" in operator
    assert "has_zero_flux_faces()" in operator
    assert operator.count("prepared_boundary_face_omission(ctx)") >= 2
    assert operator.count("PreparedBoundaryFluxFilter{&ctx}") >= 2
