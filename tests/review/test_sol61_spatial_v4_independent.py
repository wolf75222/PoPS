"""Independent Source/offline admission checks; no Native archives or seals."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "tests/review/sol61_evolved_stage_amr_spatial_reception_v4.py"
spec = importlib.util.spec_from_file_location("hooke_v4", PATH)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
receive = next(n for n in ast.parse(PATH.read_text()).body
               if isinstance(n, ast.FunctionDef) and n.name == "receive")


def admission(message, **scope):
    node = next(n for n in ast.walk(receive)
                if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Name) and n.value.func.id == "need"
                and len(n.value.args) > 1 and isinstance(n.value.args[1], ast.Constant)
                and n.value.args[1].value == message)
    env = dict(need=r.need, typed=r.typed, STOP_RULE=r.STOP_RULE,
               QUALIFICATION=r.QUALIFICATION, **scope)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 "actual-v4-admission", "exec"), env)


@pytest.mark.parametrize("cells", (8, 16))
def test_pinned_cells_bind_receipt_metric_and_junit(cells):
    case = {"artifact": "SOURCE_ONLY"}
    pins = {"ranks": 1, "seed_selection": "SOURCE_ONLY"}
    receipt = dict(fixture_schema="pops.evolved-stage-amr-spatial-native-fixture@4",
                   stop_rule=r.STOP_RULE, independent_original_l2_threshold=1e-10,
                   seed_selection="SOURCE_ONLY", cells=cells, width=2,
                   dimension=2, rank=0, size=1, artifact="SOURCE_ONLY")
    admission("actual spatial receipt differs", receipt=receipt, case=case, pins=pins, cells=cells)
    bad = dict(receipt, cells=16 if cells == 8 else 8)
    with pytest.raises(ValueError):
        admission("actual spatial receipt differs", receipt=bad, case=case, pins=pins, cells=cells)
    assignment = next(n for n in receive.body if isinstance(n, ast.Assign)
                      and ast.unparse(n.targets[0]) == "expected_metric")
    env = dict(cells=cells, arrays={"accepted": {"patch_boxes": np.empty((0, 5), dtype=np.int64)}})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])),
                 "actual-v4-metric", "exec"), env)
    metric = env["expected_metric"]
    assert metric["native_shape"] == [cells, cells]
    admission("derived metric provenance differs", receipt={"metric_authority": metric},
              expected_metric=metric)
    mismatch = deepcopy(metric); mismatch["native_shape"] = [24 - cells] * 2
    with pytest.raises(ValueError):
        admission("derived metric provenance differs", receipt={"metric_authority": mismatch},
                  expected_metric=metric)
    def selected(n):
        return [ET.Element("testcase", name=
                f"test_public_evolved_stage_amr_nonconstant_Q_restriction_and_flux_v4[{n}]")]
    admission("selected native spatial case differs", selected=selected(cells), cells=cells)
    with pytest.raises(ValueError):
        admission("selected native spatial case differs", selected=selected(24 - cells), cells=cells)


def test_root_approval_versions_cannot_cross_profiles():
    # In-memory validation witnesses only. No approval file or seal is created.
    legacy = dict(schema="sol61.evolved-stage-amr-spatial.root-approval@3",
                  approved_by="ROOT", qualification=r.QUALIFICATION, pins_sha256="SOURCE_ONLY")
    with pytest.raises(ValueError):
        admission("spatial ROOT approval differs", approval=legacy, pins_sha="SOURCE_ONLY")
    foreign = dict(legacy, schema="sol61.evolved-stage-amr-spatial.root-approval@4",
                   qualification="nonconstant-original-composite-Q-tag-selection-periodic-strip@3")
    with pytest.raises(ValueError):
        admission("spatial ROOT approval differs", approval=foreign, pins_sha="SOURCE_ONLY")


def test_all_mathematical_definitions_match_immutable_v3():
    old = ast.parse(PATH.with_name("sol61_evolved_stage_amr_spatial_reception_v3.py").read_text())
    new = ast.parse(PATH.read_text())
    functions = lambda tree: {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                             if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    before, after = functions(old), functions(new)
    assert before.keys() == after.keys()
    assert {name for name in before if before[name] != after[name]} == {"contract", "receive"}
    assert r.b.DT == .01 and r.b.TOL == 3e-8
    assert r.b.STEPS == dict(initial=0, accepted=1, continuous=2, reloaded=1, replay=2)
    assert len(r.b.CONTROLS) == 7 and r.b.CONTROLS["tolerance"] == 1e-10


def test_geometry_mismatch_refused_by_actual_science():
    spec = importlib.util.spec_from_file_location("hooke_math_fixture",
        PATH.with_name("test_sol61_evolved_stage_amr_spatial_reception.py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    images, masks = module.fixture(8)
    for rows in images.values():
        for row in rows:
            row["carrier_patch_boxes"] = row["native_patch_boxes"].copy()
            row["native_patch_boxes"] = row["native_patch_boxes"][row["native_patch_boxes"][:, 0] > 0]
    with pytest.raises((ValueError, AssertionError)):
        r.science(images, masks, 16)
