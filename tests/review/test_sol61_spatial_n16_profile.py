"""SOURCE/offline profile routing only; no Native data, seals or approval created."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import numpy as np
P=Path(__file__).with_name("sol61_evolved_stage_amr_spatial_reception_v4.py")
spec=importlib.util.spec_from_file_location("n16_reader",P);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
oldspec=importlib.util.spec_from_file_location("n16_math",P.with_name("test_sol61_evolved_stage_amr_spatial_reception.py"));old=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(old)
receive=next(n for n in ast.parse(P.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="receive")


def selector(case):
    start=next(i for i,n in enumerate(receive.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=="cells")
    scope=dict(case=case,need=r.need)
    exec(compile(ast.fix_missing_locations(ast.Module(body=receive.body[start:start+2],type_ignores=[])),"actual-reader-case-profile","exec"),scope)
    return scope["cells"]


@pytest.mark.parametrize("cells", (8,16))
def test_exact_actual_profile_routes_grid_to_checkpoints(cells):
    assert selector(dict(cells=cells,width=2)) == cells
    loop=next(n for n in receive.body if isinstance(n,ast.For) and ast.unparse(n.target)=="(phase, cp)")
    calls=[]
    def checkpoint(cp,phase,images,n,width,ranks,*args,**kwargs):
        calls.append((phase,n,width,ranks))
        return {},[np.ones((n,n),dtype=bool),np.zeros((2*n,2*n),dtype=bool)],[]
    codec=SimpleNamespace(checkpoint=checkpoint,receive_carriers=lambda *args:None)
    scope=dict(c=codec,need=r.need,same=r.same,cells=cells,
        arrays={p:{"program_hash":np.array("SOURCE_ONLY")} for p in ("accepted","continuous","replay")},
        hashes=["SOURCE_ONLY"],registry={"phases":{p:{"rows_by_rank":[[]]} for p in ("accepted","continuous","replay","reloaded")}},
        masks=None,manifests={},diagnostics={},images={},pins={"ranks":1,"abi_key":"SOURCE_ONLY"},
        case={k:"SOURCE_ONLY" for k in ("artifact","bind","semantic")},transfer_subjects=frozenset(),history_registry={})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),"actual-reader-checkpoint-orchestration","exec"),scope)
    assert calls == [(p,cells,2,1) for p in ("accepted","continuous","replay")]


@pytest.mark.parametrize("cells,width", ((True,2),(8.,2),(16.,2),(32,2),(16,True),(16,3)))
def test_profile_rejects_unknown_or_nonexact_selection(cells,width):
    with pytest.raises(ValueError): selector(dict(cells=cells,width=width))


@pytest.mark.parametrize("cells", (8,16))
def test_original_science_same_strict_threshold_on_synthetic_math(cells):
    images,masks=old.fixture(cells)
    for rows in images.values():
        for row in rows:
            row["carrier_patch_boxes"] = row["native_patch_boxes"].copy()
            row["native_patch_boxes"] = row["native_patch_boxes"][row["native_patch_boxes"][:,0]>0]
    result=r.science(images,masks,cells)
    assert all(v["original_F_weighted_l2"]<1e-10 for v in result.values())
    assert all(v["original_F_linf"]<1e-12 for v in result.values())
    r.nonlinear_restriction_attacks(images,masks,cells)
    # These are explicitly synthetic leaf-algebra witnesses, never Native evidence.


def test_historical_readers_untouched_and_profile_abi8():
    assert r.contract()["native_abi_version"] == 8
    assert r.contract()["schema"].endswith("@4")
    assert "cells16" in r.contract()["cases"]


def test_v3_reader_and_fixture_bytes_exact_frozen_base():
    import subprocess
    root = P.parents[2]
    base = "3d8481c9d5dc0f2aaa99764a2dd7662ef4e1d23c"
    for relative in ("tests/review/sol61_evolved_stage_amr_spatial_reception_v3.py", "tests/python/integration/runtime/test_public_evolved_stage_amr_spatial.py"):
        assert (root / relative).read_bytes() == subprocess.check_output(["git", "show", base+":"+relative], cwd=root)
    spec=importlib.util.spec_from_file_location("v3_immutable",P.with_name("sol61_evolved_stage_amr_spatial_reception_v3.py"));v3=importlib.util.module_from_spec(spec);spec.loader.exec_module(v3)
    assert v3.QUALIFICATION.endswith("@3") and r.QUALIFICATION.endswith("@4")
    assert "cells16" not in v3.contract()["cases"]
    old_tree=ast.parse(P.with_name("sol61_evolved_stage_amr_spatial_reception_v3.py").read_text())
    new_tree=ast.parse(P.read_text())
    for name in ("science", "independent_original_norm", "norm_arithmetic_ir", "nonlinear_restriction_attacks"):
        old_fn=next(n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        new_fn=next(n for n in new_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(old_fn,include_attributes=False)==ast.dump(new_fn,include_attributes=False)
