"""Small independent parent/candidate serialization receipt; no native imports."""

import hashlib
import json
from pathlib import Path
import runpy
import sys

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "python"))
namespace = runpy.run_path(str(Path(__file__).with_name("test_sol61_detached_global_authority.py")))
emit_cpp_program = namespace["emit_cpp_program"]
ProgramModelGraph = namespace["ProgramModelGraph"]


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


receipt = {}
for physical, primitive in ((False, False), (True, False), (True, True)):
    plan, values = namespace["prepared"](global_port=physical, primitive=primitive)
    program = values[2]
    graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    cpp = emit_cpp_program(program, model_graph=graph)
    model = graph.models_by_block["material"]
    assert model.module.manifest().schema_version == (11 if physical else 10)
    key = f"physical={physical},primitive={primitive}"
    receipt[key] = {
        "program_ir_hash": program._ir_hash(),
        "resolved_plan_identity": str(plan.plan_identity),
        "module_hash": model.module.module_hash(),
        "manifest_sha256": sha(
            json.dumps(model.module.manifest().to_dict(), sort_keys=True, separators=(",", ":"))
        ),
        "cpp_sha256": sha(cpp),
    }
print(json.dumps(receipt, sort_keys=True, indent=2))
