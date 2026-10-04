"""Exercise the actual Python opaque-wire admission guards; Native decodes the image."""
import ast
from pathlib import Path
import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[2] / "python/pops/runtime/_system_io.py"

@pytest.mark.parametrize("version", [b"POPSAUX2", b"POPSAUX3", b"POPSAUX1", b"POPSAUX4", b"", b"POPSAUX"])
@pytest.mark.parametrize("restore", [False, True])
def test_uniform_opaque_auxiliary_version_guard(version, restore):
    tree = ast.parse(SOURCE.read_text())
    guards = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and any(isinstance(value, ast.Constant) and value.value == b"POPSAUX3"
                      for value in ast.walk(node.test))]
    assert len(guards) == 2
    node = next(node for node in guards if (".size" in ast.unparse(node.test)) == restore)
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    value = np.frombuffer(version, dtype=np.uint8) if restore else version
    if version in (b"POPSAUX2", b"POPSAUX3"):
        exec(compile(module, str(SOURCE), "exec"), {"auxiliary_checkpoint": value})
    else:
        with pytest.raises((ValueError, RuntimeError)):
            exec(compile(module, str(SOURCE), "exec"), {"auxiliary_checkpoint": value})
