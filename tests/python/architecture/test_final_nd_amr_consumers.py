"""ADC-738: AMR consumers have one ranked semantic closure, not shallow umbrellas."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
INCLUDE = ROOT / "include"

ROOTS = {
    "flux": "pops/numerics/time/amr/reflux/amr_flux_helpers.hpp",
    "subcycling": "pops/numerics/time/amr/levels/amr_subcycling.hpp",
    "program": "pops/runtime/program/amr_program_context.hpp",
}
UNCHANGED_CONSUMERS = (
    "pops/coupling/amr/amr_coupler_mp.hpp",
    "pops/coupling/amr/amr_regrid_coupler.hpp",
    "pops/coupling/system/amr_system_coupler.hpp",
    "pops/numerics/time/amr/levels/amr_patch_range.hpp",
)
CONTEXT_FRAGMENT_PATHS = frozenset(
    {
        "pops/runtime/program/amr_program_context_spatial.inc",
        "pops/runtime/program/amr_program_context_field_runtime_public.inc",
        "pops/runtime/program/amr_program_context_flux_expression_public.inc",
        "pops/runtime/program/amr_program_context_spatial_operations.inc",
        "pops/runtime/program/amr_program_context_history_checkpoint_public.inc",
        "pops/runtime/program/amr_program_context_field_runtime_solver.inc",
        "pops/runtime/program/amr_program_context_field_runtime_private.inc",
        "pops/runtime/program/amr_program_context_flux_expression_polynomial.inc",
        "pops/runtime/program/amr_program_context_cell_temporal_configuration.inc",
        "pops/runtime/program/amr_program_context_history_checkpoint_definitions.inc",
        "pops/runtime/program/amr_program_context_flux_basis_definitions.inc",
        "pops/runtime/program/amr_program_context_flux_expression_definitions.inc",
        "pops/runtime/program/amr_program_context_cell_temporal_level_runtime.inc",
        "pops/runtime/program/amr_program_context_field_runtime_definitions.inc",
        "pops/runtime/program/amr_program_context_flux_expression_services.inc",
        "pops/runtime/program/amr_program_context_cell_temporal_runtime.inc",
        "pops/runtime/program/amr_program_context_subcycling_runtime.inc",
        "pops/runtime/program/amr_program_context_flux_basis.inc",
        "pops/runtime/program/amr_program_context_flux_expression_runtime.inc",
        "pops/runtime/program/amr_program_context_history_checkpoint_runtime.inc",
        "pops/runtime/program/amr_program_context_field_runtime_services.inc",
        "pops/runtime/program/amr_program_context_history_checkpoint_services.inc",
        "pops/runtime/program/amr_program_context_spatial_operations_services.inc",
    }
)
PROGRAM_RESPONSIBILITY_AUTHORITIES = {
    "spatial_context": frozenset(
        {"pops/runtime/program/amr_program_context_spatial.inc"}
    ),
    "spatial_operations": frozenset(
        {
            "pops/runtime/program/amr_program_context_spatial_operations.inc",
            "pops/runtime/program/amr_program_context_spatial_operations_services.inc",
        }
    ),
    "field_runtime": frozenset(
        {
            "pops/runtime/program/amr_program_context_field_runtime_public.inc",
            "pops/runtime/program/amr_program_context_field_runtime_solver.inc",
            "pops/runtime/program/amr_program_context_field_runtime_private.inc",
            "pops/runtime/program/amr_program_context_field_runtime_definitions.inc",
            "pops/runtime/program/amr_program_context_field_runtime_services.inc",
        }
    ),
    "history_checkpoint": frozenset(
        {
            "pops/runtime/program/amr_program_context_history_checkpoint_public.inc",
            "pops/runtime/program/amr_program_context_history_checkpoint_definitions.inc",
            "pops/runtime/program/amr_program_context_history_checkpoint_runtime.inc",
            "pops/runtime/program/amr_program_context_history_checkpoint_services.inc",
        }
    ),
    "flux_expression": frozenset(
        {
            "pops/runtime/program/amr_program_context_flux_expression_public.inc",
            "pops/runtime/program/amr_program_context_flux_expression_polynomial.inc",
            "pops/runtime/program/amr_program_context_flux_expression_definitions.inc",
            "pops/runtime/program/amr_program_context_flux_expression_services.inc",
            "pops/runtime/program/amr_program_context_flux_expression_runtime.inc",
        }
    ),
    "flux_basis": frozenset(
        {
            "pops/runtime/program/amr_program_context_flux_basis.inc",
            "pops/runtime/program/amr_program_context_flux_basis_definitions.inc",
        }
    ),
    "subcycling_runtime": frozenset(
        {"pops/runtime/program/amr_program_context_subcycling_runtime.inc"}
    ),
    "cell_temporal_runtime": frozenset(
        {
            "pops/runtime/program/amr_program_context_cell_temporal_configuration.inc",
            "pops/runtime/program/amr_program_context_cell_temporal_level_runtime.inc",
            "pops/runtime/program/amr_program_context_cell_temporal_runtime.inc",
        }
    ),
}
PROGRAM_RESPONSIBILITY_BUDGETS = {
    "spatial_context": 350,
    "spatial_operations": 900,
    "field_runtime": 1_800,
    "history_checkpoint": 1_800,
    "flux_expression": 1_200,
    "flux_basis": 500,
    "subcycling_runtime": 800,
    "cell_temporal_runtime": 800,
}
# Intentional Phase 0 policy envelopes: fragment and scaffolding growth remain
# separately bounded, and their aggregate remains independently enforced.
PROGRAM_FRAGMENT_BUDGET = 7_450
PROGRAM_SCAFFOLDING_BUDGET = 1_850
PROGRAM_SEMANTIC_CLOSURE_BUDGET = 9_300
SEMANTIC_AUTHORITIES = frozenset(
    {
        "pops/numerics/time/amr/reflux/amr_flux_execution.hpp",
        "pops/numerics/time/amr/reflux/amr_flux_preparation.hpp",
        "pops/numerics/time/amr/levels/amr_subcycling_plan.hpp",
        "pops/numerics/time/amr/levels/amr_subcycling_engine.hpp",
        *CONTEXT_FRAGMENT_PATHS,
    }
)
COMPATIBILITY_UMBRELLAS = frozenset(ROOTS.values())
PERMITTED_UPSTREAM_BOUNDARIES = frozenset(
    {
        "pops/amr/transfer/temporal_interpolation_provider.hpp",
        "pops/amr/transfer/transfer_provider.hpp",
        "pops/core/foundation/types.hpp",
        "pops/core/identity/prepared_provider.hpp",
        "pops/core/identity/sha256.hpp",
        "pops/mesh/execution/for_each.hpp",
        "pops/mesh/layout/refinement.hpp",
        "pops/mesh/parallel/region_transfer.hpp",
        "pops/mesh/storage/mf_arith.hpp",
        "pops/numerics/elliptic/interface/field_nullspace.hpp",
        "pops/numerics/elliptic/linear/generic_krylov.hpp",
        "pops/numerics/elliptic/linear/solve_outcome.hpp",
        "pops/numerics/elliptic/nd/cartesian_tensor_operator.hpp",
        "pops/numerics/time/amr/levels/amr_patch_range.hpp",
        "pops/parallel/execution_lane.hpp",
        "pops/runtime/amr/amr_runtime.hpp",
        "pops/runtime/amr/prepared_multiblock_hierarchy.hpp",
        "pops/runtime/amr_system.hpp",
        "pops/runtime/builders/compiled/generated_amr_system_block.hpp",
        "pops/runtime/multiblock/evaluation_point.hpp",
        "pops/runtime/program/amr_program_checkpoint.hpp",
        "pops/runtime/program/clock_schedule.hpp",
        "pops/runtime/program/prepared_scalar_boundary_session.hpp",
        "pops/runtime/program/prepared_tensor_boundary_session.hpp",
        "pops/runtime/program/program_runtime_state.hpp",
        "pops/runtime/program/same_level_cell_temporal_provider.hpp",
        "pops/runtime/program/step_transaction.hpp",
        "pops/runtime/system/provider_storage_binding.hpp",
    }
)
INCLUDE_RE = re.compile(
    r'^\s*#include\s*(?:<(pops/[^>]+)>|"(pops/[^"]+)")', re.MULTILINE
)
FIXED_RANK_PATTERNS = (
    re.compile(r"\bMultiFab\s*<\s*2(?:\s*,[^>]*)?\s*>"),
    re.compile(r"\bBox\s*<\s*2\s*>"),
    re.compile(r"\bIndex\s*<\s*2\s*>"),
    re.compile(r"\bFieldView\s*<[^,>]+,\s*2\s*>"),
)


def _local_includes(source: str) -> tuple[str, ...]:
    return tuple(next(part for part in match if part) for match in INCLUDE_RE.findall(source))


def _direct_local_includes(path: str) -> tuple[str, ...]:
    return _local_includes((INCLUDE / path).read_text(encoding="utf-8"))


def _require_classified_local_include(
    path: str, include: str, known: frozenset[str], permitted: frozenset[str]
) -> None:
    if include not in known:
        assert include in permitted, f"unclassified direct local include from {path}: {include}"


def _semantic_closure(root: str) -> tuple[str, ...]:
    known = SEMANTIC_AUTHORITIES | COMPATIBILITY_UMBRELLAS
    visiting: set[str] = set()
    visited: set[str] = set()
    ordered: list[str] = []

    def visit(path: str) -> None:
        assert path not in visiting, f"semantic-authority include cycle: {path}"
        if path in visited:
            return
        visiting.add(path)
        visited.add(path)
        ordered.append(path)
        for include in _direct_local_includes(path):
            _require_classified_local_include(
                path, include, known, PERMITTED_UPSTREAM_BOUNDARIES
            )
            if include in known:
                visit(include)
        visiting.remove(path)

    visit(root)
    return tuple(ordered)


def _source(paths: tuple[str, ...]) -> str:
    return "\n".join((INCLUDE / path).read_text(encoding="utf-8") for path in paths)


def _unique_paths(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(path for group in groups for path in group))


def _without_tuple_value_indices(source: str) -> str:
    return re.sub(r"std::get\s*<\s*\d+\s*>", "", source)


def _fixed_rank_authorities(source: str) -> tuple[str, ...]:
    return tuple(
        match.group(0)
        for pattern in FIXED_RANK_PATTERNS
        for match in pattern.finditer(source)
    )


def test_amr_consumer_closures_are_explicit_bounded_and_acyclic() -> None:
    closures = {name: _semantic_closure(root) for name, root in ROOTS.items()}

    assert {
        "pops/numerics/time/amr/reflux/amr_flux_execution.hpp",
        "pops/numerics/time/amr/reflux/amr_flux_preparation.hpp",
    } <= set(closures["flux"])
    assert {
        "pops/numerics/time/amr/levels/amr_subcycling_plan.hpp",
        "pops/numerics/time/amr/levels/amr_subcycling_engine.hpp",
    } <= set(closures["subcycling"])
    assert {
        path for path in SEMANTIC_AUTHORITIES if path.startswith("pops/runtime/program/")
    } <= set(closures["program"])

    for path in SEMANTIC_AUTHORITIES:
        lines = (INCLUDE / path).read_text(encoding="utf-8").count("\n") + 1
        assert lines <= 1_200, (path, lines)

    responsibility_groups = tuple(PROGRAM_RESPONSIBILITY_AUTHORITIES.values())
    responsibility_union = set().union(*responsibility_groups)
    assert responsibility_union == CONTEXT_FRAGMENT_PATHS
    assert sum(map(len, responsibility_groups)) == len(responsibility_union)
    for responsibility, paths in PROGRAM_RESPONSIBILITY_AUTHORITIES.items():
        lines = sum(
            (INCLUDE / path).read_text(encoding="utf-8").count("\n") + 1 for path in paths
        )
        assert lines <= PROGRAM_RESPONSIBILITY_BUDGETS[responsibility], (
            responsibility,
            lines,
            PROGRAM_RESPONSIBILITY_BUDGETS[responsibility],
        )

    assert len(_source(closures["flux"]).splitlines()) <= 700
    assert len(_source(closures["subcycling"]).splitlines()) <= 1_600
    program_fragments = tuple(
        path for path in closures["program"] if path in CONTEXT_FRAGMENT_PATHS
    )
    program_scaffolding = tuple(
        path for path in closures["program"] if path not in CONTEXT_FRAGMENT_PATHS
    )
    assert len(_source(program_fragments).splitlines()) <= PROGRAM_FRAGMENT_BUDGET
    assert len(_source(program_scaffolding).splitlines()) <= PROGRAM_SCAFFOLDING_BUDGET
    assert (
        len(_source(closures["program"]).splitlines())
        <= PROGRAM_SEMANTIC_CLOSURE_BUDGET
    )
    shallow_roots = (*UNCHANGED_CONSUMERS, *ROOTS.values())
    assert len(_source(shallow_roots).splitlines()) < 1_000


def test_local_include_parser_authenticates_both_delimiters_and_hidden_fragments() -> None:
    assert _local_includes('#include <pops/a.hpp>\n#include "pops/b.hpp"') == (
        "pops/a.hpp",
        "pops/b.hpp",
    )
    hidden = _local_includes('#include "pops/runtime/program/amr_program_context_hidden.inc"')
    try:
        _require_classified_local_include(
            ROOTS["program"],
            hidden[0],
            CONTEXT_FRAGMENT_PATHS,
            PERMITTED_UPSTREAM_BOUNDARIES,
        )
    except AssertionError as error:
        assert "unclassified direct local include" in str(error)
    else:
        raise AssertionError("quoted hidden fragment bypassed semantic closure classification")


def test_amr_consumer_semantic_closures_retain_one_rank_without_a_2d_authority() -> None:
    semantic_paths = _unique_paths(*(_semantic_closure(root) for root in ROOTS.values()))
    full_scan_paths = _unique_paths(semantic_paths, UNCHANGED_CONSUMERS)
    assert len(full_scan_paths) == len(set(full_scan_paths))
    joined = _source(full_scan_paths)

    for token in (
        "CoarseHierarchyRequest",
        "class AmrCouplerMP",
        "class AmrRegridCoupler",
        "class AmrSystemCoupler",
        "class PatchRange",
        "class CoarseFineInterface",
        "class PreparedAmrSubcycleTransition",
        "class PreparedAmrSubcyclePlan",
        "class PreparedMultiBlockAmrSubcyclingEngine",
        "struct PreparedTransferKernel",
        "class AmrProgramContext",
    ):
        assert token in joined, token

    for forbidden in (
        "Box2D",
        "Fab2D",
        "DistributionMapping",
        "TagBox",
        "kAmrRefRatio",
        "Array4",
        "FluxRegister",
        "ProgramExecutionServices<AmrProgramContext>",
        "pops/amr/hierarchy/refinement_ratio.hpp",
        "pops/amr/tagging/cluster.hpp",
        "pops/amr/tagging/tag_box.hpp",
    ):
        assert forbidden not in joined, forbidden

    without_tuple_value_indices = _without_tuple_value_indices(joined)
    assert not re.search(r"\bif\s+(?:constexpr\s*)?\(\s*Dim\b", without_tuple_value_indices)
    assert not _fixed_rank_authorities(without_tuple_value_indices)


def test_fixed_rank_scan_authenticates_first_and_second_template_parameters() -> None:
    sample = (
        "std::get<2>(pair); MultiFab<2>; MultiFab<2, MemorySpace>; Box<2>; Index<2>; "
        "FieldView<const Real, 2>;"
    )
    assert _fixed_rank_authorities(_without_tuple_value_indices(sample)) == (
        "MultiFab<2>",
        "MultiFab<2, MemorySpace>",
        "Box<2>",
        "Index<2>",
        "FieldView<const Real, 2>",
    )
    assert _fixed_rank_authorities(_without_tuple_value_indices("std::get<2>(pair)")) == ()


def test_consumers_delegate_to_canonical_prepared_authorities() -> None:
    transfer = _source(_semantic_closure(ROOTS["flux"]))
    assert "runtime.template prepare_transfer" in transfer
    assert "PreparedTransfer<Dim>" in transfer
    assert "for_each_cell" in transfer

    subcycling = _source(_semantic_closure(ROOTS["subcycling"]))
    assert "LevelStateSpatialContract<Dim>" in subcycling
    assert "runtime.spatial_contract()" in subcycling
    assert "runtime.reconcile_reflux" in subcycling

    regrid = _source(("pops/coupling/amr/amr_regrid_coupler.hpp",))
    assert "ClusterProvider<Dim>" in regrid
    assert "runtime_->prepare_regrid" in regrid
    assert "runtime_->publish_regrid" in regrid

    coupler = _source(("pops/coupling/amr/amr_coupler_mp.hpp",))
    assert "load_balance.prepare(" in coupler
    assert "ownership.plan().distribution()" in coupler
    assert "AmrRuntime<Dim, MemorySpace>" in coupler

    program = _source(_semantic_closure(ROOTS["program"]))
    for operation in (
        "prepare_subcycling(",
        "prepare_regrid(",
        "publish_regrid(",
        "prepare_rebalance(",
        "apply_rebalance(",
        "reconcile_reflux(",
    ):
        assert operation in program


def test_direct_native_proof_exercises_one_and_three_dimensional_consumers() -> None:
    proof = ROOT / "tests/cpp/unit/amr/test_nd_amr_consumers.cpp"
    assert proof.is_file()
    source = proof.read_text(encoding="utf-8")
    assert "PatchRange<1>" in source
    assert "PatchRange<3>" in source
    assert "RefinementRatio<3>{2, 3, 1}" in source
