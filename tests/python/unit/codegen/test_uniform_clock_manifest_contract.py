"""Source-only tests of emitted Uniform ownership and installed capacity admission.

Execute the actual isolated pure helpers without importing an installed Native package.
The loader integration suite owns real exported-symbol rejection tests.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[4]


def _helper(relative: str, name: str, namespace: dict[str, Any] | None = None):
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    scope = {"Any": Any, "sys": sys, "json": json, **(namespace or {})}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def _emit(clocks, primary):
    literal = _helper("python/pops/codegen/cpp_strings.py", "cpp_string_literal")
    emitter = _helper("python/pops/codegen/program_codegen.py", "_emit_uniform_clock_manifest",
                      {"cpp_string_literal": literal})
    program = SimpleNamespace(temporal_manifest=lambda: {
        "clocks": [{"id": clock} for clock in clocks], "primary_clock": primary})
    return emitter(program)


def test_uniform_manifest_emits_exact_contract_and_preserves_clock_indices():
    clocks = ("macro", "nested/clock", "long-" + "q" * 4096)
    source = _emit(clocks, clocks[1])
    assert 'pops_program_checkpoint_clock_manifest_contract()' in source
    assert 'return "pops.program.owned-clock-manifest@1";' in source
    assert 'logical_clock_count() { return 3; }' in source
    for index, clock in enumerate(clocks):
        assert 'case %d: return %s;' % (index, json.dumps(clock)) in source
    assert 'primary_clock_identity() { return "nested/clock"; }' in source


@pytest.mark.parametrize("clocks,primary", [([], "macro"), (["macro", "macro"], "macro"),
                                             ([""], ""), (["macro"], "foreign")])
def test_uniform_manifest_rejects_invalid_owned_tables(clocks, primary):
    with pytest.raises(ValueError):
        _emit(clocks, primary)


def _capacity_reader():
    capacity = _helper("python/pops/output/_checkpoint_contract.py", "_capacity")
    return _helper("python/pops/runtime/_checkpoint_resource_budget.py",
                   "_installed_program_auxiliary_capacity", {"_capacity": capacity})


def test_installed_capacity_uses_actual_runtime_owner_and_never_authored_clocks():
    calls = []
    owner = SimpleNamespace(_s=SimpleNamespace(_checkpoint_program_auxiliary_capacity=
                                                lambda: calls.append("installed") or (8192, 7)))
    assert _capacity_reader()(owner) == (8192, 7)
    assert calls == ["installed"]


@pytest.mark.parametrize("capacity,error", [(None, TypeError), ([1, 2], TypeError),
                                               ((1,), TypeError), ((1, 2, 3), TypeError),
                                               ((True, 2), TypeError), ((-1, 2), ValueError),
                                               ((1, -1), ValueError),
                                               ((sys.maxsize + 1, 2), OverflowError)])
def test_installed_capacity_rejects_malformed_native_result(capacity, error):
    owner = SimpleNamespace(_s=SimpleNamespace(_checkpoint_program_auxiliary_capacity=lambda: capacity))
    with pytest.raises(error):
        _capacity_reader()(owner)


def test_handwritten_uniform_installer_exports_its_actual_primary_clock():
    source = _helper("tests/python/support/explicit_program.py", "_uniform_clock_exports")()
    assert 'return "pops.program.owned-clock-manifest@1";' in source
    assert 'logical_clock_count() { return 1; }' in source
    assert source.count('"pops.test.clock.macro"') == 2
