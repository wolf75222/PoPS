"""ADC-749: legacy transport-boundary authorities may only disappear.

The prepared hyperbolic boundary path is already the compiled Uniform/AMR
route.  A few older native authorities still exist while their replacements
need metric, characteristic, and post-Riemann kernels.  Keep their remaining
lexical surface bounded so adjacent work cannot silently create another
transport-boundary engine before that cutover is complete.
"""

from __future__ import annotations

import ast
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
