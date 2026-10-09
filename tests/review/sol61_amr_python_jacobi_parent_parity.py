"""Explicit external checkout parity; same fixture/callsite, fresh interpreter.

Run with a source tree argument. Compare the full JSON profiles excluding source
and package paths; source location is evidence, never scientific normalization.
No fetch, build, installed native import, or author receipt is used.
"""
import hashlib
import json
from pathlib import Path
import runpy
import sys

source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source / "python"))
import pops  # noqa: E402
from pops.time._program.serialization import _json_ready  # noqa: E402

fixture = runpy.run_path(str(Path(__file__).with_name("test_sol61_amr_public_original.py")))
profiles = []
for width, order, uniform in ((2, (1, 0), False), (3, (2, 0, 1), True), (5, (4, 2, 0, 3, 1), False)):
    case, layout, program, token, _ = fixture["authored"](width, order, seed=True, uniform=uniform)
    cpp, resolved = fixture["emit"](case, layout)
    modules = [block.model.module for block in resolved.blocks]
    profiles.append({"width": width, "uniform": uniform,
        "authored_ir": program._ir_hash(), "resolved_ir": resolved.time._ir_hash(),
        "ir_version": resolved.time._serialize()["version"],
        "cpp": hashlib.sha256(cpp.encode()).hexdigest(),
        "module_hashes": [module.module_hash() for module in modules],
        "module_manifests": [module.manifest().to_dict() for module in modules],
        "equation": token.attrs["solve_request"]["equation_identity"],
        "solver": token.attrs["solver_identity"],
        "request": _json_ready(token.attrs["solve_request"]),
    })
print(json.dumps({"source":str(source), "package":pops.__file__, "profiles":profiles},sort_keys=True))
