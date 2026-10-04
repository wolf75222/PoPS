"""Source-only execution of the AMR witness's real pre-bind release/origin guards."""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest


def _guards(root):
    source = root / "tests/python/integration/runtime/test_m19_amr_consumed_runtime.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "test_installed_amr_consumed_field_regrid_and_nonfinite_rollback")
    guard = next(n for n in function.body if isinstance(n, ast.With))
    assertions = [n for n in guard.body if isinstance(n, ast.Assert)]
    assert len(assertions) == 4
    return compile(ast.Module(body=assertions, type_ignores=[]), str(source), "exec")


@pytest.mark.parametrize("version,foreign,accepted", [(11, False, True), (10, False, False), (True, False, False), (11, True, False)])
def test_amr_witness_published_release_and_installed_origin(version, foreign, accepted):
    root = Path(__file__).resolve().parents[2]
    contract = ast.parse((root / "python/pops/_generated_release_contract.py").read_text())
    version_node = next(n.value for n in contract.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "NATIVE_ABI_VERSION" for t in n.targets))
    release = ast.literal_eval(version_node)
    assert type(release) is int and release == 11
    # Source doubles exercise assertions; no PoPS package or Native artifact is imported.
    package = Path("/outside-installed-prefix/pops/__init__.py") if foreign else Path(sys.prefix) / "lib/pops/__init__.py"
    namespace = {"Path": Path, "pops": SimpleNamespace(__file__=str(package)), "caps": {"abi_version": version, "mapped_consumed_field_output_amr": True}, "NATIVE_ABI_VERSION": release}
    if accepted:
        exec(_guards(root), namespace)
    else:
        with pytest.raises(AssertionError):
            exec(_guards(root), namespace)
