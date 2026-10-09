"""Fresh source tree, same authoring callsite; no installed provider or JIT."""
import json
from pathlib import Path
import runpy
import sys

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "python"))
from pops.time._program.serialization import _json_ready  # noqa: E402
from pops.codegen.program_emit_spatial_interaction import emit_spatial_interaction  # noqa: E402

folder = Path(__file__).parent
state_fixture = runpy.run_path(str(folder / "test_sol61_spatial_interaction_reception.py"))
history_fixture = runpy.run_path(str(folder / "test_sol61_spatial_history_independent.py"))
rows = []
for name, build in (("issued",lambda:state_fixture["setup"](scope="issued",candidate=True)),
                    ("accepted",lambda:state_fixture["setup"](scope="accepted")),
                    ("history-two",history_fixture["setup"])):
    program, _owner, result = build()
    cpp = {}
    for target in ("system", "amr_system"):
        lines = []
        emit_spatial_interaction(result, {result.inputs[0].id:"same_original_source"}, lines,
                                block_indices=program._block_indices(), target=target)
        cpp[target] = "\n".join(lines)
    rows.append({"profile":name,"IR":_json_ready(program._serialize()),"hash":program._ir_hash(),"CPP":cpp})
print(json.dumps(rows,sort_keys=True,separators=(",", ":")))
