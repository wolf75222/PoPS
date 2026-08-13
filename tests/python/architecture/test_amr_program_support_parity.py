"""Source-only parity between explicit AMR Program deferrals and their Python mirror.

``AmrProgramContext`` marks an unsupported capability only by calling
``deferred_op("<unambiguous-id>", ...)``. Ordinary runtime, validation, history-integrity and
error-policy exceptions are not capability declarations. This gate locks the explicit identifiers
against ``DEFERRED_GROUPS`` without importing ``pops`` or the compiled extension.
"""

import importlib.util
import pathlib
import re
import sys

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SUPPORT_PY = REPO_ROOT / "python" / "pops" / "runtime" / "amr_program_support.py"
CONTEXT_HPP = REPO_ROOT / "include" / "pops" / "runtime" / "program" / "amr_program_context.hpp"
UNIFORM_CONTEXT_HPP = REPO_ROOT / "include" / "pops" / "runtime" / "program" / "program_context.hpp"
PRODUCTION_CODEGEN = (
    REPO_ROOT / "python" / "pops" / "codegen" / "program_codegen.py",
    REPO_ROOT / "python" / "pops" / "codegen" / "program_emit_ops.py",
    REPO_ROOT / "python" / "pops" / "codegen" / "program_emit_amr.py",
)


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


def test_support_module_loads_standalone_and_stays_import_free():
    source = SUPPORT_PY.read_text(encoding="utf-8")
    offender = re.search(r"(?m)^\s*(?:import\s+pops|from\s+pops)\b", source)
    assert offender is None, (
        "amr_program_support.py must load source-only before _pops exists; found %r"
        % (offender.group(0) if offender else None)
    )

    groups = _load_support_module().deferred_groups()
    assert groups
    assert set(groups.values()) <= {"green"} | {
        value for value in groups.values() if value.startswith("pending")
    }


def test_header_deferred_set_matches_the_python_mirror():
    module = _load_support_module()
    mirror = set(module.header_deferred_methods())
    header = _parse_header_deferred_set(CONTEXT_HPP.read_text(encoding="utf-8"))
    assert header == mirror, (
        "AMR Program explicit-deferral drift:\n"
        "  only in header: %s\n"
        "  only in mirror: %s" % (sorted(header - mirror), sorted(mirror - header))
    )


def test_parser_finds_only_explicit_known_deferrals():
    module = _load_support_module()
    header = _parse_header_deferred_set(CONTEXT_HPP.read_text(encoding="utf-8"))
    assert header == set()
    assert module.header_deferred_methods() == frozenset()
    assert "unqualified_coupled_solve" not in module.DEFERRED_GROUPS
    assert module.deferred_groups()["schedule_cache"] == (
        "pending:checkpointed_hierarchy_cache"
    )
    context_source = CONTEXT_HPP.read_text(encoding="utf-8")
    for provider_method in (
        "cache_should_update",
        "cache_store_scratch",
        "cache_restore_scratch",
        "cache_accumulate_dt",
        "cache_effective_dt",
    ):
        assert provider_method in context_source
    assert 'unavailable_("checkpointed AMR scheduler cache provider")' in context_source
    assert "neg_div_flux_into" not in header
    assert "solve_fields_from_state_at_fine_level" not in header
    assert "solve_fields_from_state_default" not in header
    assert "SolveOutcome solve_fields_from_state(const std::string&" not in (
        CONTEXT_HPP.read_text(encoding="utf-8")
    )
    assert "SolveOutcome solve_fields_from_blocks(const std::string&" not in (
        CONTEXT_HPP.read_text(encoding="utf-8")
    )
    context_header = context_source
    assert context_header.count("SolveOutcome solve_fields_from_blocks_at(") == 1
    multi_block_route = context_header.split("SolveOutcome solve_fields_from_blocks_at(", 1)[
        1
    ].split("SolveOutcome solve_default_field_on_coarse_level() const", 1)[0]
    assert "all_reduce_max(local_error ? 1L : 0L, lane)" in multi_block_route
    assert "all_ranks_agree_exact_ordered_byte_pairs" in multi_block_route
    assert 'request.text("pops.amr-program.simultaneous-field-route")' in multi_block_route
    assert ".scalar(value_id)" in multi_block_route
    assert ".presence(override_value.state != nullptr)" in multi_block_route
    assert "field_layout_contract(*override_value.state)" in multi_block_route
    multi_local_consensus = multi_block_route.index("all_reduce_max(local_error ? 1L : 0L, lane)")
    multi_byte_consensus = multi_block_route.index("all_ranks_agree_exact_ordered_byte_pairs")
    multi_facade = multi_block_route.index("facade_->solve_program_field_from_blocks_at(")
    multi_cache = multi_block_route.index("generated_field_routes_.insert(")
    assert multi_local_consensus < multi_byte_consensus < multi_facade < multi_cache
    scalar_candidate_route = context_header.split("void evaluate_with_field_state_at(", 1)[1].split(
        "[[nodiscard]] SolveOutcome solve_fields() const", 1
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

    single_state_route = context_header.split("SolveOutcome solve_fields_from_state_at(", 1)[
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
    assert "solve_fields_from_blocks_at" in UNIFORM_CONTEXT_HPP.read_text(encoding="utf-8")
    assert "program_execution_solve_generated_field_from_blocks_outcome_" in (context_header)
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
    context_header = CONTEXT_HPP.read_text(encoding="utf-8")
    named_route = context_header.split("void neg_div_named_flux_into(", 1)[1].split(
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
    assert "shared topological interfaces are installed" in named_envelope


def test_generated_programs_cannot_use_coarse_injection_as_a_fine_solve():
    header = CONTEXT_HPP.read_text(encoding="utf-8")
    assert "SolveOutcome solve_fields() const" not in header
    assert header.count("SolveOutcome solve_default_field_on_coarse_level() const") == 1
    coarse_route = header.split("SolveOutcome solve_default_field_on_coarse_level() const", 1)[
        1
    ].split("SolveOutcome solve_fields_from_state_at(", 1)[0]
    assert "if (level_ != 0)" in coarse_route
    assert "coarse-to-fine auxiliary injection is not a " in coarse_route
    assert '"fine-level solve"' in coarse_route
    assert "return eng_->solve_default_field();" in coarse_route
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
