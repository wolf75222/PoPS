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
CONTEXT_HPP = (REPO_ROOT / "include" / "pops" / "runtime" / "program"
               / "amr_program_context.hpp")


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
        % (offender.group(0) if offender else None))

    groups = _load_support_module().deferred_groups()
    assert groups
    assert set(groups.values()) <= {"green"} | {
        value for value in groups.values() if value.startswith("pending")}


def test_header_deferred_set_matches_the_python_mirror():
    module = _load_support_module()
    mirror = set(module.header_deferred_methods())
    header = _parse_header_deferred_set(CONTEXT_HPP.read_text(encoding="utf-8"))
    assert header == mirror, (
        "AMR Program explicit-deferral drift:\n"
        "  only in header: %s\n"
        "  only in mirror: %s" % (sorted(header - mirror), sorted(mirror - header)))


def test_parser_finds_only_explicit_known_deferrals():
    header = _parse_header_deferred_set(CONTEXT_HPP.read_text(encoding="utf-8"))
    for identifier in (
        "cache_should_update",
        "cache_effective_dt",
        "neg_div_flux_into",
        "solve_fields_from_state_default",
        "solve_fields_from_blocks_default",
        "refined_shared_block_interfaces",
        "solve_fields_from_state_at_fine_level",
    ):
        assert identifier in header
    assert "apply_projection" not in header
    assert not any(identifier.startswith("history") for identifier in header)


def test_projection_is_green_after_the_real_amr_implementation_landed():
    module = _load_support_module()
    assert module.DEFERRED_GROUPS["projection"]["header_methods"] == frozenset()
    assert module.deferred_groups()["projection"] == "green"


class _Program:
    def __init__(self, nodes, *, recursive_nodes=None):
        self._nodes = list(nodes)
        self._recursive_nodes = list(
            self._nodes if recursive_nodes is None else recursive_nodes)

    def ir_nodes(self, *, recursive=False):
        return list(self._recursive_nodes if recursive else self._nodes)


def _context(module, *, refined=False, interfaces=False):
    return module.AMRProgramSupportContext(
        refined_hierarchy=refined,
        shared_block_interfaces=interfaces,
        field_routes_validated=True,
    )


def test_complete_query_requires_resolved_context():
    module = _load_support_module()
    with pytest.raises(TypeError, match="resolved AMRProgramSupportContext"):
        module.amr_program_op_support(_Program([]), context=None)


def test_context_sensitive_deferrals_are_reported_only_when_reachable():
    module = _load_support_module()
    matrix_free = {"op": "matrix_free_operator", "attrs": {"apply_block": ["#2"]}}
    field_jacobian = _Program(
        [matrix_free],
        recursive_nodes=[
            matrix_free,
            {"op": "rhs_jacvec", "attrs": {"field_coupled": True}},
        ],
    )
    assert module.amr_program_op_support(
        field_jacobian, context=_context(module, refined=False)) == {}
    assert module.amr_program_op_support(
        field_jacobian, context=_context(module, refined=True)) == {
            "fine_level_field_perturbation": "pending",
        }
    assert module.amr_program_op_support(
        _Program([]), context=_context(module, refined=True, interfaces=True)) == {
            "refined_shared_block_interfaces": "pending",
        }


def test_ir_ops_mirror_the_codegen_op_group_sets():
    module = _load_support_module()
    kernels = (REPO_ROOT / "python" / "pops" / "codegen"
               / "program_emit_kernels.py").read_text(encoding="utf-8")
    match = re.search(r"_CONDENSED_OPS\s*=\s*frozenset\(\{([^}]*)\}\)", kernels, re.S)
    assert match is not None
    codegen_condensed = set(re.findall(r'"([A-Za-z_]\w*)"', match.group(1)))
    assert set(module.DEFERRED_GROUPS["condensed"]["ir_ops"]) == codegen_condensed
    assert module.DEFERRED_GROUPS["named_field_solve"]["ir_ops"] == frozenset(
        {"solve_fields"})
    assert module.amr_program_op_support(
        _Program([{"op": "solve_fields", "attrs": {"field": "potential"}}]),
        context=_context(module),
    ) == {"named_field_solve": "green"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
