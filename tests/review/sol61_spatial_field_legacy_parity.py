"""Emit genuine legacy IR/C++ digests from an explicitly selected source tree."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "python"))
sys.path.insert(0, str(root / "examples/migration/scientific"))
import pops  # noqa: E402
from api040_m27_mixed_linear import build_case  # noqa: E402
from pops.codegen.program_codegen import emit_cpp_program  # noqa: E402
from pops.codegen.program_models import ProgramModelGraph  # noqa: E402

assert Path(pops.__file__).resolve() == root / "python/pops/__init__.py"
implicit = runpy.run_path(str(root / "tests/python/unit/time/test_implicit_stage_request.py"))
results = {}
cases = [("mixed-linear", build_case(16)),
         ("mixed-linear-permuted", build_case(16, permuted=True)),
         ("implicit-stage", implicit["make_source_stage"]()),
         ("implicit-stage-nonlinear-map", implicit["make_source_stage"](nonlinear=True))]
for name, (case, layout, *_) in cases:
    resolved = pops.resolve(pops.validate(case), layout=layout)
    code = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    results[name] = {"ir_sha256": resolved.time._ir_hash(),
                     "cpp_sha256": hashlib.sha256(code.encode()).hexdigest()}
print(json.dumps(results, indent=2, sort_keys=True))
