"""SOURCE_ONLY exact phase orchestration, independently supplied by Banach.

No checkpoint decoding, native execution, owner seals or scientific approval.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def test_exact_receive_checkpoint_phase_loop_retains_codec_module():
    source=Path(__file__).with_name("sol61_evolved_stage_amr_spatial_reception_v2.py").read_text()
    receive=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="receive")
    loop=next(n for n in receive.body if isinstance(n,ast.For) and ast.unparse(n.target)=="(phase, cp)")
    calls=[]
    def checkpoint(cp,phase,*args,**kwargs):
        calls.append(phase)
        return {},[np.ones((1,1),dtype=bool),np.zeros((2,2),dtype=bool)],[]
    carrier_calls=[]
    codec=SimpleNamespace(checkpoint=checkpoint,receive_carriers=lambda *args:carrier_calls.append(args),
        complete_carrier_geometry=lambda *args:"SOURCE_ONLY geometry",run_identity=lambda *args:"SOURCE_ONLY run")
    # Full receive() lexical scope must resolve the imported codec globally.
    # A loop-target assignment anywhere in the function otherwise makes it local
    # even at its earlier ABI preflight call.
    compiled=compile(ast.Module(body=[receive],type_ignores=[]),"whole-receive-lexical-scope","exec")
    function=next(value for value in compiled.co_consts if hasattr(value,"co_varnames") and value.co_name=="receive")
    assert "c" not in function.co_varnames
    def need(test,message):
        assert test,message
    scope=dict(c=codec,b=SimpleNamespace(need=need),need=need,same=lambda a,b,message:np.testing.assert_array_equal(a,b),
        arrays={p:{"program_hash":np.array("SOURCE_ONLY")} for p in ("accepted","continuous","replay")},
        hashes=["SOURCE_ONLY"],registry={"phases":{p:{"rows_by_rank":[[]]} for p in ("accepted","continuous","replay","reloaded")}},
        masks=None,manifests={},diagnostics={},images={},pins={"ranks":1,"abi_key":"SOURCE_ONLY"},
        case={k:"SOURCE_ONLY" for k in ("artifact","bind","semantic")},transfer_subjects=frozenset({"SOURCE_ONLY"}),history_registry={})
    code=compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),"exact-source-phase-loop","exec")
    exec(code,scope)
    assert calls==["accepted","continuous","replay"]
    assert scope["c"] is codec

    assert len(carrier_calls)==4
    assert scope["c"].complete_carrier_geometry(None)=="SOURCE_ONLY geometry"
    assert scope["c"].run_identity(None)=="SOURCE_ONLY run"
