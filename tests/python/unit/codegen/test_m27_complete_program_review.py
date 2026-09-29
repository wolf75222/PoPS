"""Independent complete mixed-field Program syntax, not installed execution."""
from pathlib import Path

import pytest

from tests.python.unit.codegen import test_m23_complete_program_syntax as compiler


WORKER = r'''
import pathlib, sys
root, output = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2])
sys.path.insert(0, str(root / "python"))
sys.path.insert(0, str(root / "examples/migration/scientific"))
import pops
assert pathlib.Path(pops.__file__).resolve() == root / "python/pops/__init__.py"
from api040_m27_mixed_linear import build_case
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
case, layout, subject = build_case(16, permuted=bool(int(sys.argv[4])),
                                  solver_iterations=1 if int(sys.argv[5]) else 400)
resolved = pops.resolve(pops.validate(case), layout=layout)
graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
source = emit_cpp_program(resolved.time, model_graph=graph)
assert "prepare_general_field_coefficients<pops::kNativeDimension, 2, 4, false>" in source
assert "apply_general_field<pops::kNativeDimension, 2, 4>" in source
assert "ctx.commit_many(" in source
output.write_text(source)
print("source_import="+pops.__file__)
'''


@pytest.mark.parametrize("permuted,insufficient", [(False, False), (True, True)])
def test_m27_full_program_syntax(tmp_path, monkeypatch, permuted, insufficient):
    monkeypatch.setattr(compiler, "EMIT_WORKER", WORKER)
    monkeypatch.setenv("POPS_M23_SOURCE_ROOT", str(Path(__file__).resolve().parents[4]))
    compiler.test_m23_complete_ssprk2_program_syntax(tmp_path, 2, permuted, insufficient)
