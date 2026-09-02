"""Source-only parity between explicit AMR Program deferrals and their Python mirror.

``ProgramExecutionServices`` marks an unsupported capability only by calling
``deferred_op("<unambiguous-id>", ...)``. Ordinary runtime, validation, history-integrity and
error-policy exceptions are not capability declarations. This gate locks the explicit identifiers
against ``DEFERRED_GROUPS`` without importing ``pops`` or the compiled extension.
"""

import ast
import importlib.util
import pathlib
import re
import sys
from types import MappingProxyType

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SUPPORT_PY = REPO_ROOT / "python" / "pops" / "runtime" / "amr_program_support.py"
CONTEXT_HPP = REPO_ROOT / "include" / "pops" / "runtime" / "program" / "program_execution_services.hpp"
PRODUCTION_CODEGEN = (
    REPO_ROOT / "python" / "pops" / "codegen" / "program_codegen.py",
    REPO_ROOT / "python" / "pops" / "codegen" / "program_emit_ops.py",
    REPO_ROOT / "python" / "pops" / "codegen" / "program_emit_amr.py",
)
CONTEXT_ROOT = "pops/runtime/program/program_execution_services.hpp"
CONTEXT_FRAGMENT_PATHS = frozenset(
    {
        "pops/runtime/program/detail/program_execution_services_amr_backend.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_backend_preparation.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_backend_state.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_spatial.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_field_runtime_public.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_flux_expression_public.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_spatial_operations.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_public.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_field_runtime_solver.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_field_runtime_private.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_flux_expression_polynomial.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_cell_temporal_configuration.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_definitions.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_forward.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_history_reseed.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_lifecycle.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_capacity.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_flux_basis_definitions.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_flux_expression_definitions.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_cell_temporal_level_runtime.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_field_runtime_definitions.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_flux_expression_services.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_cell_temporal_runtime.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_subcycling_runtime.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_flux_basis.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_flux_expression_runtime.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_runtime.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_snapshot.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_field_runtime_services.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_services.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_spatial_operations_services.hpp",
        "pops/runtime/program/detail/program_execution_services_amr_subcycling_interface_payload.hpp",
    }
)
CONTEXT_DETAIL_PATHS = frozenset(
    path for path in CONTEXT_FRAGMENT_PATHS
    if path != "pops/runtime/program/detail/program_execution_services_amr_backend.hpp"
)
RETIRED_CONTEXT_ROOT = (
    REPO_ROOT / "include" / "pops" / "runtime" / "program" / "amr_program_context.hpp"
)
_LOCAL_INCLUDE_RE = re.compile(
    r'^\s*#include\s*(?:<(pops/[^>]+)>|"(pops/[^"]+)")', re.MULTILINE
)


def _local_includes(source: str) -> tuple[str, ...]:
    return tuple(next(part for part in match if part) for match in _LOCAL_INCLUDE_RE.findall(source))


def _require_classified_context_include(include: str, known: frozenset[str]) -> None:
    if include not in known and include.startswith(
        "pops/runtime/program/detail/program_execution_services_amr_"
    ):
        raise AssertionError(f"unclassified ProgramExecutionServices definition authority: {include}")


def _context_semantic_source() -> str:
    """Read precisely the context root and its classified definition authorities."""
    known = CONTEXT_FRAGMENT_PATHS | {CONTEXT_ROOT}
    visiting: set[str] = set()
    visited: set[str] = set()
    sources: list[str] = []

    def visit(relative: str) -> None:
        assert relative not in visiting, f"ProgramExecutionServices semantic include cycle: {relative}"
        if relative in visited:
            return
        visiting.add(relative)
        visited.add(relative)
        path = REPO_ROOT / "include" / relative
        source = path.read_text(encoding="utf-8")
        sources.append(source)
        for include in _local_includes(source):
            if include in known:
                visit(include)
            else:
                _require_classified_context_include(include, known)
        visiting.remove(relative)

    visit(CONTEXT_ROOT)
    assert visited == known, (
        "ProgramExecutionServices semantic closure differs from its classified authorities: "
        f"missing={sorted(known - visited)} extra={sorted(visited - known)}"
    )
    detail_dir = REPO_ROOT / "include" / "pops" / "runtime" / "program" / "detail"
    physical_details = frozenset(
        path.relative_to(REPO_ROOT / "include").as_posix()
        for path in detail_dir.glob("program_execution_services_amr_*.hpp")
    )
    assert len(CONTEXT_DETAIL_PATHS) == 31
    assert physical_details == CONTEXT_FRAGMENT_PATHS, (
        "ProgramExecutionServices AMR detail inventory differs from its authenticated closure: "
        f"missing={sorted(CONTEXT_FRAGMENT_PATHS - physical_details)} "
        f"extra={sorted(physical_details - CONTEXT_FRAGMENT_PATHS)}"
    )
    assert not RETIRED_CONTEXT_ROOT.exists(), (
        "the retired public AMR Program context header must not be recreated"
    )
    return "\n".join(sources)


def test_context_include_parser_authenticates_both_delimiters_and_hidden_fragments():
    assert _local_includes('#include <pops/a.hpp>\n#include "pops/b.hpp"') == (
        "pops/a.hpp",
        "pops/b.hpp",
    )
    hidden = _local_includes(
        '#include "pops/runtime/program/detail/program_execution_services_amr_hidden.hpp"'
    )
    try:
        _require_classified_context_include(hidden[0], CONTEXT_FRAGMENT_PATHS)
    except AssertionError as error:
        assert "unclassified ProgramExecutionServices definition authority" in str(error)
    else:
        raise AssertionError("quoted hidden fragment bypassed AMR Program classification")


def _load_support_module():
    """Load the import-free support query directly from its source path."""
    spec = importlib.util.spec_from_file_location("_amr_program_support_parity", SUPPORT_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _strip_comments(text):
    """Drop C++ comments while preserving string literals verbatim."""
    out = []
    i = 0
    in_string = False
    while i < len(text):
        char = text[i]
        if in_string:
            out.append(char)
            if char == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 2
                continue
            if char == '"':
                in_string = False
            i += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            i += 1
            continue
        if char == "/" and i + 1 < len(text) and text[i + 1] == "/":
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        if char == "/" and i + 1 < len(text) and text[i + 1] == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(char)
        i += 1
    return "".join(out)


_DEFERRED_OP_RE = re.compile(r'\bdeferred_op\(\s*"([A-Za-z_]\w*)"')


def _parse_header_deferred_set(raw):
    """Return only explicit capability-deferral identifiers from the live C++ source."""
    return set(_DEFERRED_OP_RE.findall(_strip_comments(raw)))


def _extract_method_body(header, signature):
    """Return one method body, matched with balanced braces from its signature."""
    start = header.index(signature)
    open_brace = header.index("{", start)
    depth = 0
    for index in range(open_brace, len(header)):
        if header[index] == "{":
            depth += 1
        elif header[index] == "}":
            depth -= 1
            if depth == 0:
                return header[open_brace : index + 1]
    raise AssertionError("unterminated method body: %s" % signature)


def _module_scope_pops_imports(source):
    """Return package imports which would break standalone source-only execution."""
    tree = ast.parse(source, filename=str(SUPPORT_PY))
    offenders = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            offenders.extend(
                alias.name
                for alias in node.names
                if alias.name == "pops" or alias.name.startswith("pops.")
            )
        elif isinstance(node, ast.ImportFrom) and node.module and (
            node.module == "pops" or node.module.startswith("pops.")
        ):
            offenders.append("from " + node.module)
    return offenders


def test_support_module_loads_standalone_and_stays_import_free():
    source = SUPPORT_PY.read_text(encoding="utf-8")
    offenders = _module_scope_pops_imports(source)
    assert offenders == [], (
        "amr_program_support.py must load source-only before _pops exists; found %r"
        % offenders
    )

    groups = _load_support_module().deferred_groups()
    assert groups
    assert set(groups.values()) <= {"green"} | {
        value for value in groups.values() if value.startswith("pending")
    }


def test_header_deferred_set_matches_the_python_mirror():
    module = _load_support_module()
    mirror = set(module.header_deferred_methods())
    header = _parse_header_deferred_set(_context_semantic_source())
    assert header == mirror, (
        "AMR Program explicit-deferral drift:\n"
        "  only in header: %s\n"
        "  only in mirror: %s" % (sorted(header - mirror), sorted(mirror - header))
    )


def test_parser_finds_only_explicit_known_deferrals():
    module = _load_support_module()
    header = _parse_header_deferred_set(_context_semantic_source())
    assert header == set()
    assert module.header_deferred_methods() == frozenset()
    assert "unqualified_coupled_solve" not in module.DEFERRED_GROUPS
    assert module.deferred_groups()["schedule_cache"] == "green"
    cache_manager = (
        REPO_ROOT / "include" / "pops" / "runtime" / "program" / "cache_manager.hpp"
    ).read_text(encoding="utf-8")
    assert "using ProgramCacheSlot = std::size_t;" in cache_manager
    assert "std::vector<CacheSlot<Dim>> slots_;" in cache_manager
    assert "checkpoint_slot_indices()" in cache_manager
    context_source = _context_semantic_source()
    for provider_method in (
        "cache_should_update",
        "cache_store_scratch",
        "cache_restore_scratch",
        "cache_accumulate_dt",
        "cache_effective_dt",
    ):
        assert provider_method in context_source
    cache_provider_source = (
        REPO_ROOT
        / "include"
        / "pops"
        / "runtime"
        / "program"
        / "detail"
        / "program_execution_services_amr_history_checkpoint_public.hpp"
    ).read_text(encoding="utf-8")
    for cache_operation in (
        "runtime_state().cache_.is_due(slot, macro_step(), every_n)",
        "runtime_state().cache_.store(slot, scratch, macro_step())",
        "runtime_state().cache_.restore_into(slot, scratch)",
        "runtime_state().cache_.accumulate_dt(slot, dt)",
        "runtime_state().cache_.effective_dt(slot, dt)",
    ):
        assert cache_operation in cache_provider_source
    assert 'unavailable_("checkpointed AMR scheduler cache provider")' not in cache_provider_source
    assert "neg_div_flux_into" not in header
    assert "solve_fields_from_state_at_fine_level" not in header
    assert "solve_fields_from_state_default" not in header
    assert "SolveOutcome solve_fields_from_state(const std::string&" not in (
        _context_semantic_source()
    )
    assert "SolveOutcome solve_fields_from_blocks(const std::string&" not in (
        _context_semantic_source()
    )
    context_header = context_source
    assert context_header.count("SolveOutcome solve_fields_from_blocks_at(") == 3
    multi_block_route = context_header.rsplit("SolveOutcome solve_fields_from_blocks_at(", 1)[
        1
    ].split("SolveOutcome solve_default_field_on_coarse_level() const", 1)[0]
    assert 'require_boundary_point_(point, "AMR Program simultaneous field solve")' in (
        multi_block_route
    )
    assert "overrides.size() != route.program_blocks.size()" in multi_block_route
    assert "std::fill(route.runtime_stages.begin(), route.runtime_stages.end(), nullptr)" in (
        multi_block_route
    )
    assert "std::find(route.unique_stages.begin(), route.unique_stages.end(), override_value.state)" in (
        multi_block_route
    )
    assert "require_same_field_contract_(" in multi_block_route
    multi_facade = multi_block_route.index("facade_->solve_program_field_from_blocks_at(")
    assert multi_block_route.index("overrides.size() != route.program_blocks.size()") < multi_facade
    assert multi_block_route.index("std::fill(route.runtime_stages.begin()") < multi_facade
    assert multi_block_route.index("std::find(route.unique_stages.begin()") < multi_facade
    assert multi_block_route.index("require_same_field_contract_(") < multi_facade
    scalar_candidate_route = context_header.rsplit("void evaluate_with_field_state_at(", 1)[1].split(
        "[[nodiscard]] SolveOutcome solve_fields_from_state_at(", 1
    )[0]
    assert 'request.text("pops.amr-program.scalar-field-candidate-route")' in (
        scalar_candidate_route
    )
    assert "field_layout_contract(perturbed)" in scalar_candidate_route
    assert "field_layout_contract(accepted)" in scalar_candidate_route
    assert (
        scalar_candidate_route.index("all_reduce_max(local_error ? 1L : 0L, lane)")
        < scalar_candidate_route.index("all_ranks_agree_exact_ordered_byte_pairs")
        < scalar_candidate_route.index("facade_->with_program_field_candidate_at(")
    )

    single_state_route = context_header.rsplit("SolveOutcome solve_fields_from_state_at(", 1)[
        1
    ].split("SolveOutcome solve_fields_from_blocks_at(", 1)[0]
    assert 'request.text("pops.amr-program.single-field-route")' in single_state_route
    assert ".scalar(std::int32_t{runtime_block})" in single_state_route
    assert "field_layout_contract(stage)" in single_state_route
    assert (
        single_state_route.index("all_reduce_max(local_error ? 1L : 0L, lane)")
        < single_state_route.index("all_ranks_agree_exact_ordered_byte_pairs")
        < single_state_route.index("facade_->solve_program_field_from_blocks_at(")
    )
    assert "solve_fields_from_blocks_at" in CONTEXT_HPP.read_text(encoding="utf-8")
    assert "program_execution_solve_generated_field_from_blocks_outcome_" not in context_header
    assert "facade_->solve_program_field_from_blocks_at(" in context_header
    assert "named_solve_reports_" not in context_header
    assert "fine_level_field_perturbation" not in module.DEFERRED_GROUPS
    assert "refined_shared_block_interfaces" not in module.DEFERRED_GROUPS
    assert "apply_projection" not in header
    assert not any(identifier.startswith("history") for identifier in header)


def test_projection_is_green_after_the_real_amr_implementation_landed():
    module = _load_support_module()
    assert module.DEFERRED_GROUPS["projection"]["header_methods"] == frozenset()
    assert module.deferred_groups()["projection"] == "green"


def test_named_flux_support_matches_the_resolved_interface_envelope():
    module = _load_support_module()
    context_header = _context_semantic_source()
    named_route = context_header.rsplit("void neg_div_named_flux_into(", 1)[1].split(
        "void apply_projection(", 1
    )[0]
    named_envelope = context_header.split("void require_named_flux_execution_envelope_(", 1)[
        1
    ].split("const field_type* staged_parent_for_block_", 1)[0]
    assert module.DEFERRED_GROUPS["named_flux"]["header_methods"] == frozenset()
    assert module.deferred_groups()["named_flux"] == "green"
    assert module.amr_program_op_support(
        _Program([{"op": "rhs", "attrs": {"fluxes": ["transport"]}}]),
        context=_context(module, refined=True, interfaces=False),
    ) == {"named_flux": "green"}
    assert module.amr_program_op_support(
        _Program([{"op": "rhs", "attrs": {"fluxes": ["transport"]}}]),
        context=_context(module, refined=True, interfaces=True),
    ) == {"named_flux": "pending:shared_block_interfaces"}
    assert "active AMR named-flux divergence has no authenticated" not in named_route
    assert "prepare_active_flux_basis_impl_(" in named_route
    assert "has_interface_flux_provider()" in named_envelope
    assert (
        '"AMR named flux currently refuses the complete prepared carrier pack when shared "'
        in named_envelope
    )
    assert '"topological interfaces are installed"' in named_envelope


def test_generated_programs_cannot_use_coarse_injection_as_a_fine_solve():
    header = _context_semantic_source()
    # The unified authority deliberately retains the Uniform no-argument route.  The AMR
    # generated path must nevertheless use the exact-ranked state-at seam below.
    assert header.count("SolveOutcome solve_fields() const") == 1
    assert header.count("SolveOutcome solve_default_field_on_coarse_level() const") == 1
    coarse_route = _extract_method_body(
        header, "SolveOutcome solve_default_field_on_coarse_level() const"
    )
    assert "if (active_level_ != 0)" in coarse_route
    assert "coarse-to-fine auxiliary injection is not a fine-level solve" in coarse_route
    assert coarse_route.index("active_level_ != 0") < coarse_route.index("refresh_resources_()")
    assert coarse_route.index("refresh_resources_()") < coarse_route.index(
        "facade_->solve_program_default_field(0)"
    )
    assert "default_solve_report_" not in header

    generated = "\n".join(path.read_text(encoding="utf-8") for path in PRODUCTION_CODEGEN)
    assert "ctx.solve_fields(" not in generated
    assert "solve_default_field_on_coarse_level" not in generated
    assert "ctx.solve_fields_from_state_at(" in generated


class _Program:
    def __init__(self, nodes, *, recursive_nodes=None):
        self._nodes = list(nodes)
        self._recursive_nodes = list(self._nodes if recursive_nodes is None else recursive_nodes)

    def ir_nodes(self, *, recursive=False):
        return list(self._recursive_nodes if recursive else self._nodes)


def _context(module, *, refined=False, interfaces=False, frozen=True):
    return module.AMRProgramSupportContext(
        hierarchy_level_count=2 if refined else 1,
        frozen_hierarchy=frozen,
        shared_block_interfaces=interfaces,
        field_routes_validated=True,
    )


def test_complete_query_requires_resolved_context():
    module = _load_support_module()
    with pytest.raises(TypeError, match="resolved AMRProgramSupportContext"):
        module.amr_program_op_support(_Program([]), context=None)


def test_immutable_ir_carriers_are_accepted_but_malformed_attrs_stay_fail_closed():
    module = _load_support_module()
    immutable_node = MappingProxyType(
        {
            "op": "rhs",
            "attrs": MappingProxyType({"fluxes": ["transport"]}),
        }
    )

    assert module.amr_program_op_support(
        _Program([immutable_node]), context=_context(module, refined=True)
    ) == {"named_flux": "green"}

    with pytest.raises(TypeError, match=r"\[0\]\.attrs must be a mapping"):
        module.amr_program_op_support(
            _Program([MappingProxyType({"op": "rhs", "attrs": ()})]),
            context=_context(module),
        )
    with pytest.raises(TypeError, match=r"\[0\]\.attrs must be a mapping"):
        module.amr_program_op_support(
            _Program([MappingProxyType({"op": "rhs"})]),
            context=_context(module),
        )


def test_context_sensitive_routes_report_green_or_pending_from_resolved_hierarchy():
    module = _load_support_module()
    matrix_free = {"op": "matrix_free_operator", "attrs": {"apply_block": ["#2"]}}
    field_jacobian = _Program(
        [matrix_free],
        recursive_nodes=[
            matrix_free,
            {"op": "rhs_jacvec", "attrs": {"field_coupled": True}},
        ],
    )
    assert (
        module.amr_program_op_support(field_jacobian, context=_context(module, refined=False)) == {}
    )
    assert (
        module.amr_program_op_support(field_jacobian, context=_context(module, refined=True)) == {}
    )
    assert (
        module.amr_program_op_support(
            _Program([]), context=_context(module, refined=True, interfaces=True, frozen=False)
        )
        == {}
    )
    assert (
        module.amr_program_op_support(
            _Program([]), context=_context(module, refined=False, interfaces=True, frozen=False)
        )
        == {}
    )
    assert (
        module.amr_program_op_support(
            _Program([]), context=_context(module, refined=True, interfaces=True, frozen=True)
        )
        == {}
    )

    frozen_three = module.AMRProgramSupportContext(
        hierarchy_level_count=3,
        frozen_hierarchy=True,
        shared_block_interfaces=True,
        field_routes_validated=True,
    )
    dynamic_three = module.AMRProgramSupportContext(
        hierarchy_level_count=3,
        frozen_hierarchy=False,
        shared_block_interfaces=True,
        field_routes_validated=True,
    )
    assert frozen_three.supports_shared_interface_fragments
    assert dynamic_three.supports_shared_interface_fragments


def test_ir_ops_mirror_the_codegen_op_group_sets():
    module = _load_support_module()
    kernels = (REPO_ROOT / "python" / "pops" / "codegen" / "program_emit_kernels.py").read_text(
        encoding="utf-8"
    )
    match = re.search(r"_CONDENSED_OPS\s*=\s*frozenset\(\{([^}]*)\}\)", kernels, re.S)
    assert match is not None
    codegen_condensed = set(re.findall(r'"([A-Za-z_]\w*)"', match.group(1)))
    assert set(module.DEFERRED_GROUPS["condensed"]["ir_ops"]) == codegen_condensed
    assert module.DEFERRED_GROUPS["named_field_solve"]["ir_ops"] == frozenset({"solve_fields"})
    assert module.amr_program_op_support(
        _Program([{"op": "solve_fields", "attrs": {"field": "potential"}}]),
        context=_context(module),
    ) == {"named_field_solve": "green"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
