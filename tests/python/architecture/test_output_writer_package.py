"""Scientific-output backends form a private DAG behind one public facade."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
POPS = ROOT / "python" / "pops"
OUTPUT = POPS / "output"
WRITERS = OUTPUT / "_writers"


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), str(path))
    result = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
        elif isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
    return result


def test_writer_backends_have_one_unidirectional_private_package():
    assert not (OUTPUT / "writers.py").exists()
    assert {path.name for path in WRITERS.glob("*.py")} == {
        "__init__.py", "common.py", "hdf5.py", "npz.py", "paraview.py",
    }

    assert _module_imports(WRITERS / "__init__.py") == set()
    common_imports = _module_imports(WRITERS / "common.py")
    assert not any(name.startswith("pops.output._writers.") for name in common_imports)

    optional = {"numpy", "h5py"}
    for backend in ("hdf5", "npz", "paraview"):
        imports = _module_imports(WRITERS / (backend + ".py"))
        assert "pops.output._writers.common" in imports
        assert not any(
            name.startswith("pops.output._writers.")
            and name != "pops.output._writers.common"
            for name in imports
        )
        assert not imports & optional


def test_pops_output_is_the_exact_writer_facade():
    import pops.output as output
    from pops.output._writers.common import ScientificWriter, WriterSession
    from pops.output._writers.hdf5 import HDF5Writer
    from pops.output._writers.npz import NPZWriter
    from pops.output._writers.paraview import ParaViewWriter

    assert output.ScientificWriter is ScientificWriter
    assert output.WriterSession is WriterSession
    assert output.HDF5Writer is HDF5Writer
    assert output.NPZWriter is NPZWriter
    assert output.ParaViewWriter is ParaViewWriter
    assert {"HDF5Writer", "NPZWriter", "ParaViewWriter", "ScientificWriter", "WriterSession"} \
        <= set(output.__all__)

    public_session_methods = {
        name for name, member in WriterSession.__dict__.items()
        if not name.startswith("_") and callable(member)
    }
    assert public_session_methods == {
        "stage", "abort_prepare", "publish", "rollback", "finalize",
    }


def test_paraview_geometry_assembly_has_no_python_cell_loops():
    tree = ast.parse((WRITERS / "paraview.py").read_text(), "paraview.py")
    prepare = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_stage_file"
    )
    cell_indices = {"i", "j", "row", "column", "cell_index"}
    offenders = []
    for node in ast.walk(prepare):
        if not isinstance(node, ast.For):
            continue
        targets = {
            name.id for name in ast.walk(node.target) if isinstance(name, ast.Name)
        }
        if targets & cell_indices:
            offenders.append(node.lineno)
    assert not offenders, "ParaView geometry must stay NumPy-vectorized: %r" % offenders


def test_production_sources_do_not_import_the_retired_writer_module():
    offenders = []
    for path in POPS.rglob("*.py"):
        parts = path.relative_to(POPS).with_suffix("").parts
        if not all(part.isidentifier() for part in parts):
            continue
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = (node.module,)
            elif isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            else:
                names = ()
            if "pops.output.writers" in names:
                offenders.append("%s:%d" % (path.relative_to(ROOT), node.lineno))
    assert not offenders
