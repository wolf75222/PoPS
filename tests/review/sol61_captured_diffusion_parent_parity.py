"""Fresh exact legacy comparison at a fixed public fixture/callsite.

Pass a source checkout explicitly. No fetching, Native/JIT or production mutation.
Complete canonical request/manifests are hashed without provenance normalization.
"""
from contextlib import nullcontext
from functools import partial
import hashlib
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

source=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(source/"python"))
import pops  # noqa: E402
from pops.solvers import Newton  # noqa: E402
from pops.time._program.serialization import _json_ready  # noqa: E402

fixture=runpy.run_path(str(Path(__file__).with_name("test_sol61_amr_public_original.py")))
def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":")).encode()).hexdigest()
profiles=[]
for width,order,uniform,jacobi in ((2,(1,0),False,False),(3,(2,0,1),True,False),(5,(4,2,0,3,1),False,True)):
    context=(patch.dict(fixture["authored"].__globals__,Newton=partial(Newton,right_preconditioner="SpatialBasisJacobi@1"))
             if jacobi else nullcontext())
    with context:
        case,layout,program,token,_=fixture["authored"](width,order,seed=True,uniform=uniform)
    cpp,resolved=fixture["emit"](case,layout)
    modules=[block.model.module for block in resolved.blocks]
    profiles.append({"width":width,"uniform":uniform,"jacobi":jacobi,
        "authored_ir":program._ir_hash(),"resolved_ir":resolved.time._ir_hash(),
        "ir_version":resolved.time._serialize()["version"],
        "cpp":hashlib.sha256(cpp.encode()).hexdigest(),
        "module_hashes":[module.module_hash() for module in modules],
        "module_full_manifest_sha256":[digest(module.manifest().to_dict()) for module in modules],
        "full_request_sha256":digest(_json_ready(token.attrs["solve_request"])),
        "physical_equation":token.attrs["solve_request"]["equation_identity"],
        "solver_identity":token.attrs["solver_identity"],
        "request_version":token.attrs["solve_request"]["schema_version"],
    })
print(json.dumps({"source":str(source),"package":pops.__file__,"profiles":profiles},sort_keys=True))
