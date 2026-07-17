"""Pure-Python authoring-layer import and typed descriptor summary gates."""
import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PHYSICS = REPO_ROOT / "python" / "pops" / "physics"


def _module_scope_import_targets(tree):
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)) or node.col_offset != 0:
            continue
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif node.level == 0 and node.module:
            yield node.module


def test_physics_never_imports_codegen_at_module_scope():
    offenders = []
    for path in sorted(PHYSICS.rglob("*.py")):
        for target in _module_scope_import_targets(ast.parse(path.read_text(), str(path))):
            if target == "pops.codegen" or target.startswith("pops.codegen."):
                offenders.append("%s imports %s" % (path.relative_to(REPO_ROOT), target))
    assert not offenders


def test_typed_descriptor_repr_is_short_and_has_no_array_dump():
    from pops.diagnostics import Integral, MinMax, Norm
    from pops.linalg import L2
    from pops.numerics.reconstruction import FirstOrder, MUSCL, WENO5
    from pops.numerics.riemann import HLL, HLLC, Roe, Rusanov

    descriptors = [HLL(), Rusanov(), HLLC(), Roe(), FirstOrder(), MUSCL(), WENO5(),
                   Norm(L2()), Integral(), MinMax()]
    for descriptor in descriptors:
        for text in (repr(descriptor), str(descriptor)):
            assert len(text) < 800
            assert "array(" not in text
            assert text.strip()
