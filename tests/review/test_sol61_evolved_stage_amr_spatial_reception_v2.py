"""Source/offline-only spatial @2 attacks; homogeneous archives never qualify it."""
import copy
import ast
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import numpy as np

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("spatial_v2", HERE / "sol61_evolved_stage_amr_spatial_reception_v2.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
old_spec = importlib.util.spec_from_file_location("spatial_synthetic_math", HERE / "test_sol61_evolved_stage_amr_spatial_reception.py")
old = importlib.util.module_from_spec(old_spec)
old_spec.loader.exec_module(old)
ROOT = HERE.parents[1]


def test_distinct_current_scope_and_original_math_unchanged():
    assert r.QUALIFICATION != old.r.QUALIFICATION != r.c.QUALIFICATION
    assert r.contract()["schema"].endswith("@2")
    assert r.contract()["native_abi_version"] == 6
    images, masks = old.fixture()
    expected = old.r.science(images, masks, 8)
    for phase in images:
        for row in images[phase]:
            row["carrier_patch_boxes"] = row["native_patch_boxes"].copy()
            row["native_patch_boxes"] = row["native_patch_boxes"][row["native_patch_boxes"][:,0]>0]
    assert r.science(images, masks, 8) == expected
    assert "pops" not in sys.modules


@pytest.mark.parametrize("attack", ("none", "sign", "FD", "tolerance", "Q", "z"))
def test_current_public_source_authentication_without_import(attack):
    files = {k:(ROOT / "tests/python/support" / v).read_bytes()
             for k,v in dict(spatial="evolved_stage_amr_spatial.py", amr="evolved_stage_amr.py",
                             equations="evolved_stage_mms.py", controls="captured_diffusion_mms.py").items()}
    if attack == "sign": files["equations"] = files["equations"].replace(b"[-.001, .014]", b"[.001, .014]")
    if attack == "FD": files["controls"] = files["controls"].replace(b"FD_STEP = 1e-6", b"FD_STEP = 1e-5")
    if attack == "tolerance": files["controls"] = files["controls"].replace(b"tolerance=1e-10", b"tolerance=1e-8")
    if attack == "Q": files["spatial"] = files["spatial"].replace(b".1*ValueExpr(b)", b".2*ValueExpr(b)")
    if attack == "z": files["spatial"] = files["spatial"].replace(b"-.25", b"-.35")
    if attack == "none": r.declared_source(**files)
    else:
        with pytest.raises(ValueError): r.declared_source(**files)
    assert "pops" not in sys.modules


@pytest.fixture
def homogeneous_archive():
    base = Path("/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/installed-sdkbb416-amr12-variants-serial-dim2/pytest-tmp")
    for index in range(4):
        folder = base/f"test_public_evolved_stage_amr_{index}"/"evolved-stage-amr"
        if not (folder/"receipt.json").is_file(): continue
        receipt = r.b.strict_json((folder/"receipt.json").read_bytes())
        if receipt["width"] != 2: continue
        arrays = r.b.wire.archive((folder/"accepted-checkpoint.npz").read_bytes())
        ir = r.b.strict_json(next(folder.glob("*.ir.json")).read_bytes())
        images = {phase:[r.b.wire.archive(Path(row["path"]).read_bytes()) for row in receipt["phases"][phase]["levels"]] for phase in r.b.PHASES}
        registry = r.b.strict_json((folder/"carrier-registry.json").read_bytes())
        return receipt, arrays, ir, images, registry
    pytest.skip("retained authentic homogeneous width2 archive unavailable")


def test_authentic_homogeneous_archive_cannot_qualify_nonconstant(homogeneous_archive):
    receipt, arrays, ir, images, registry = homogeneous_archive
    masks = r.b.topology(r.c.complete_carrier_geometry(arrays), receipt["cells"], 1,
                        [str(arrays[f"distribution_mode_{l}"].item()) for l in (0,1)],
                        [arrays[f"dmap_{l}"] for l in (0,1)])
    # Attach only the declared metrics to exercise the scientific guard itself.
    for phase in images:
        for level,row in enumerate(images[phase]):
            row.update(old.geometry(receipt["cells"])[0][level])
            row["carrier_patch_boxes"] = r.c.complete_carrier_geometry(arrays)
            row["native_patch_boxes"] = arrays["patch_boxes"]
            row["active"] = masks[level]
            row["valid"] = masks[level] if level else masks[level] | ~masks[level]
    with pytest.raises(ValueError): r.science(images, masks, receipt["cells"])


@pytest.mark.parametrize("attack", ("old_schema", "missing_tag", "derived_dilation", "different_buffer"))
def test_authentic_contract_attacks_fail_closed(homogeneous_archive, attack):
    _, arrays, ir, _, _ = homogeneous_archive
    contract = r.b.strict_json(str(arrays["amr_accepted_contract"].item()))
    subjects = r.c.program_transfer_subjects(ir)
    r.c.accepted_contract(contract, 1, subjects)
    contract = copy.deepcopy(contract)
    if attack == "old_schema": contract["schema_version"] = 7
    if attack == "missing_tag": del contract["tag_selection"]
    if attack == "derived_dilation": contract["tag_selection"][1][1:] = ["3","3"]
    if attack == "different_buffer": contract["tag_selection"][1][1:] = ["1","1"]
    with pytest.raises(ValueError): r.c.accepted_contract(contract, 1, subjects)
    assert "pops" not in sys.modules


def test_authentic_geometry_capture_uses_all_base_patches(homogeneous_archive):
    _, arrays, _, _, registry = homogeneous_archive
    # Compile only the pure capture helper, without importing PoPS or its model.
    tree = ast.parse((ROOT/"tests/python/support/evolved_stage_amr_spatial.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "carrier_patch_boxes")
    scope = {"np":np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "pure-live-geometry-helper", "exec"), scope)
    observed = registry["phases"]["accepted"]["rows_by_rank"]
    expected = r.c.complete_carrier_geometry(arrays)
    np.testing.assert_array_equal(scope["carrier_patch_boxes"](observed), expected)
    assert (expected[:,0] == 0).any() and not (arrays["patch_boxes"][:,0] == 0).any()
    altered = copy.deepcopy(observed)
    state = next(row for rank in altered for row in rank if row[1] == "state")
    state[7] = str(int(state[7])+1)
    with pytest.raises(ValueError): scope["carrier_patch_boxes"](altered)
    assert "pops" not in sys.modules


def test_H_of_average_is_rejected_by_existing_scientific_guard():
    images, masks = old.fixture()
    for phase in images:
        for row in images[phase]:
            row["carrier_patch_boxes"] = row["native_patch_boxes"].copy()
            row["native_patch_boxes"] = row["native_patch_boxes"][row["native_patch_boxes"][:,0]>0]
    metrics = r.science(images, masks, 8)
    attacks = r.nonlinear_restriction_attacks(images, masks, 8)
    assert all(row["countermodel_rejected"] and row["maximum_absolute"] > row["existing_guard"]
               for row in attacks.values())
    # This measured synthetic gap demonstrates power of the unchanged 3e-8 guard;
    # ROOT must repeat this attack on the future authentic nonconstant archives.
    assert min(row["restriction_gap"] for row in metrics.values()) > r.b.TOL
    row = images["continuous"][0]
    temperature = np.stack([row[k].reshape(8,8) for k in ("T0","T1")])
    wrong = r.b.q_of(temperature)
    for i in range(2): row[f"Q{i}"].reshape(8,8)[~masks[0]] = wrong[i][~masks[0]]
    with pytest.raises(ValueError, match="covered Q must restrict"):
        r.science(images, masks, 8)


def test_signed_second_Jensen_gap_may_be_negative():
    mean = np.array([[.15],[.25]])
    delta = np.array([[.02],[-.001]])
    gap = (r.b.q_of(mean+delta)+r.b.q_of(mean-delta))/2-r.b.q_of(mean)
    np.testing.assert_allclose(gap[:,0], [.02**2+.1*.001**2, .001**2+.2*.02*(-.001)], atol=1e-16)
    assert gap[0,0] > 0 > gap[1,0]


@pytest.mark.parametrize("attack", ("transpose", "sign", "cross"))
def test_signed_flux_matrix_mutants_discriminated(attack):
    _, masks = old.geometry()
    covered = ~masks[0][0]
    tower = [old.temperature(8), old.temperature(16)]
    expected = r.flux_action(tower, covered)[0]
    matrix = r.D.copy()
    if attack == "transpose": matrix = matrix.T
    if attack == "sign": matrix[1,0] *= -1
    if attack == "cross": matrix[0,1] = matrix[1,0] = 0
    actual = r.flux_action(tower, covered, matrix=matrix)[0]
    assert max(np.max(np.abs(a-b)) for a,b in zip(expected,actual,strict=True)) > r.b.TOL


@pytest.mark.parametrize("attack", ("blob_bit", "manifest_hash"))
def test_authentic_full_carrier_mutation_refused(homogeneous_archive, attack):
    _, arrays, _, _, registry = homogeneous_archive
    rows = registry["phases"]["accepted"]["rows_by_rank"]
    r.c.receive_carriers(arrays, rows, {"Q0":1,"Q1":1,"forcing":3})
    if attack == "blob_bit":
        arrays = dict(arrays)
        arrays["state_carriers_checkpoint"] = arrays["state_carriers_checkpoint"].copy()
        arrays["state_carriers_checkpoint"][-1] ^= 1
    else:
        rows = copy.deepcopy(rows)
        state = next(row for rank in rows for row in rank if row[1] == "state")
        state[-1] = state[-1][:-1]+("0" if state[-1][-1] != "0" else "1")
    with pytest.raises(ValueError): r.c.receive_carriers(arrays, rows, {"Q0":1,"Q1":1,"forcing":3})
    assert "pops" not in sys.modules
