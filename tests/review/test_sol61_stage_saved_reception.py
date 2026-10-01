"""Independent protocol/math tests; --campaign reads actual archives without approving them."""
from __future__ import annotations
import argparse
import copy
from fractions import Fraction
from functools import partial
import importlib.util
import io
import json
from pathlib import Path
import struct

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("stage_reader_under_review", Path(__file__).with_name("sol61_stage_saved_reception.py"))
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def test_nonlinear_Q_is_cellwise_not_Q_of_average():
    t = np.array([[[.1, .3]], [[.2, .4]]])
    result = stage.q_of(t)
    assert result[0, 0, 0] == .1+.1**2+.1*.2**2
    assert result[1, 0, 1] == .4+.4**2+.2*.3*.4
    assert not np.array_equal(result.mean(axis=2), stage.q_of(t.mean(axis=2, keepdims=True))[:, :, 0])


@pytest.mark.parametrize("width", [1, 2])
def test_face_action_independent_rational_matrix(width):
    # Synthetic signed values, not saved native states or an alleged native solve.
    t = np.array([[[.125, -.25, .5], [.75, .25, -.125]],
                  [[.375, .5, -.25], [.125, -.375, .625]]])[:width]
    got, faces = stage.spatial(t, True)
    d = [[Fraction(3, 250), Fraction(1, 500)], [Fraction(-1, 1000), Fraction(7, 500)]]
    expected = np.zeros_like(t)
    for i in range(width):
        for y in range(2):
            for x in range(3):
                value = Fraction(0)
                for j in range(width):
                    for axis, length in ((0, 2), (1, 3)):
                        for sign in (-1, 1):
                            yy, xx = ((y+sign) % 2, x) if axis == 0 else (y, (x+sign) % 3)
                            a, b = Fraction(float(t[j, y, x])), Fraction(float(t[j, yy, xx]))
                            value += d[i][j]*(1+Fraction(1, 5)*(a*a+b*b))*(b-a)*length**2
                expected[i, y, x] = float(value)
    np.testing.assert_allclose(got, expected, rtol=2e-15, atol=2e-17)
    assert all(face > 0 for face in faces)
    assert np.max(np.abs(got.sum(axis=(1, 2)))) < 2e-16
    if width == 2:
        assert np.max(np.abs(got-stage.spatial(t, True, transposed=True)[0])) > .005
        assert np.max(np.abs(got-stage.spatial(t, True, harmonic=True)[0])) > .00001


def test_fraction_polynomial_discriminates_vanishing_at_three_samples():
    x = stage.symbol("x")
    probe = stage.multiply(x, stage.multiply(stage.add(x, stage.constant(-1)), stage.add(x, stage.constant(-2))))
    assert probe != {}  # Exact polynomial, although zero at x=0,1,2.


@pytest.mark.parametrize("rank,size", [(0, 1), (0, 2), (1, 2)])
def test_junit_empty_or_foreign_record_refused(rank, size):
    with pytest.raises(ValueError, match="eight successful real cases"):
        stage.junit(b"<testsuites><testsuite tests='0'/></testsuites>", rank, size, {})


@pytest.mark.parametrize("which", ["pins", "approval"])
def test_external_seal_not_inferred(tmp_path, which):
    pins, approval = tmp_path/"pending.json", tmp_path/"approval.json"
    pins.write_bytes(b"{}")
    approval.write_bytes(b"{}")
    correct = stage.files.digest(b"{}")
    with pytest.raises(ValueError, match="external seal differs"):
        stage.receive(pins, "00"*32 if which == "pins" else correct,
                      approval, "00"*32 if which == "approval" else correct)


def test_canonical_path_alias_refused(tmp_path):
    original = tmp_path/"leaf"
    original.write_bytes(b"observed bytes only")
    alias = tmp_path/"alias"
    alias.symlink_to(original)
    with pytest.raises(ValueError):
        stage.files.leaf(alias)


def test_strict_json_duplicate_nonfinite_refused():
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}'):
        with pytest.raises(ValueError):
            stage.strict_json(raw)


def roundtrip(arrays):
    output = io.BytesIO()
    np.savez(output, **arrays)
    raw = output.getvalue()
    return stage.protocol.archive(raw), stage.files.digest(raw)


def refuses(call, fragment):
    try:
        call()
    except (ValueError, AssertionError) as error:
        assert fragment in str(error), (fragment, str(error))
        return str(error)
    raise AssertionError("altered copy was accepted: "+fragment)


def inspect_campaign(root, mode):
    """RO observations and freshly resealed countermodels, never ROOT approval."""
    roots = [str(root.parent), "/Users/romaindespoulain/miniforge3/envs/pops-api040"]
    tmp = "pytest-tmp" if mode == "serial" else "rank0-tmp"
    cases, reports, refusals = {}, {}, {}
    for idx in range(8):
        directory = root/tmp/f"test_public_evolved_original_s{idx}"/"evolved-original-stage"
        case = stage.case_inventory(directory)
        receipt = stage.strict_json((directory/"receipt.json").read_bytes())
        n, dt, width = receipt["cells"], receipt["dt"], receipt["evolved_states"]
        label = stage.key(n, dt, width)
        initial = stage.npz(case["initial"], roots)
        states = {phase: stage.npz(pin, roots) for phase, pin in case["phases"].items()}
        cps = {phase: stage.npz(pin, roots) for phase, pin in case["checkpoints"].items()}
        ir = stage.strict_json(stage.files.pinned(case["program_irs"][0], roots)[1])
        assert stage.program_hash(ir) == receipt["program_irs"][0]["program_hash"]
        stage.ir_physics(ir, width, dt)
        reports[label] = stage.science(initial, states, n, dt, width)
        stage.lineage.review(*(cps[phase] for phase in stage.CP))
        stage.receipt_provenance(receipt, cps, dt)
        for phase in stage.CP:
            stage.checkpoint(cps[phase], states[phase], states["accepted"], n, dt, width,
                             cps[phase]["abi_key"].item(), receipt["artifact"], receipt["program_irs"][0]["program_hash"])
        for binary in receipt["binaries"]:
            stage.component(binary, roots)
        cases[label] = case
        # One scalar and one coupled witness exercise mechanisms. Other cases
        # remain unmodified; no altered bytes are persisted into the donor.
        if idx not in (0, 4):
            continue
        changed = copy.deepcopy(states)
        changed["accepted"]["Q0"] += .01
        changed["accepted"], digest = roundtrip(changed["accepted"])
        refusals[label+":Q"] = dict(sha256=digest, reason=refuses(partial(stage.science, initial, changed, n, dt, width), "original F="))
        changed = copy.deepcopy(states)
        changed["accepted"]["forcing"] += .01
        changed["accepted"], digest = roundtrip(changed["accepted"])
        refusals[label+":forcing"] = dict(sha256=digest, reason=refuses(partial(stage.science, initial, changed, n, dt, width), "readonly forcing"))
        changed = copy.deepcopy(states)
        changed["accepted"]["T0-previous"] += .01
        changed["accepted"], digest = roundtrip(changed["accepted"])
        refusals[label+":history"] = dict(sha256=digest, reason=refuses(partial(stage.science, initial, changed, n, dt, width), "history"))
        altered = copy.deepcopy(ir)
        next(node for node in altered["nodes"] if node["op"] == "solve_spatial_field")["attrs"]["local_expressions"][0] = ["literal", {"kind":"integer","value":"0"}]
        refusals[label+":body"] = dict(sha256=stage.program_hash(altered), reason=refuses(partial(stage.ir_physics, altered, width, dt), "exact original local residual"))
        altered = copy.deepcopy(ir)
        next(node for node in altered["nodes"] if node["op"] == "field_problem_coefficients")["attrs"]["expressions"][0] = ["literal", {"kind":"integer","value":"0"}]
        refusals[label+":D"] = dict(sha256=stage.program_hash(altered), reason=refuses(partial(stage.ir_physics, altered, width, dt), "signed per-candidate diffusion"))
        altered = copy.deepcopy(ir)
        altered["commits"][0]["value"] = altered["commits"][-1]["value"]
        refusals[label+":Qcommit"] = dict(sha256=stage.program_hash(altered), reason=refuses(partial(stage.ir_physics, altered, width, dt), "Q publication"))
        for mutation in ("sample", "slot", "lineage"):
            altered = {key: value.copy() for key, value in cps["replay"].items()}
            if mutation == "sample":
                altered["history_sample_identity_T0"][-8:] = np.frombuffer(struct.pack("<Q", 2), dtype=np.uint8)
            elif mutation == "slot":
                altered["history_T0_0"] += .01
            else:
                manifest = stage.lineage.strict_json(altered[stage.lineage.MANIFEST].item())
                manifest["run_identity"] = stage.lineage.authenticate(cps["continuous"])["run_identity"]
                altered[stage.lineage.MANIFEST] = np.array(json.dumps(manifest))
            stage.lineage.reseal(altered, stage.lineage.strict_json(altered[stage.lineage.MANIFEST].item()))
            altered, digest = roundtrip(altered)
            if mutation == "lineage":
                reason = refuses(partial(stage.lineage.review, cps["accepted"], cps["continuous"], altered), "run lineage")
            else:
                reason = refuses(partial(stage.checkpoint, altered, states["replay"], states["accepted"], n, dt, width,
                                  altered["abi_key"].item(), receipt["artifact"], receipt["program_irs"][0]["program_hash"]), "history")
            refusals[label+":"+mutation] = dict(sha256=digest, reason=reason)
    ranks = 1 if mode == "serial" else 2
    for rank in range(ranks):
        raw = (root/("pytest.xml" if ranks == 1 else f"rank{rank}.xml")).read_bytes()
        stage.junit(raw, rank, ranks, cases)
        # Resealed records reach the semantic guard, not a raw digest failure.
        tree = stage.ET.fromstring(raw)
        tree.find(".//property[@name='rank']").set("value", "99")
        refuses(partial(stage.junit, stage.ET.tostring(tree), rank, ranks, cases), "dimension/rank")
    return dict(schema="sol61.stage-observed-unsealed@1", mode=mode, status="pending_external_ROOT_seals",
                actual_cases=8, scientific_reports=reports, copied_countermodel_refusals=refusals,
                donor_receipts={label:case["receipt"] for label,case in cases.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--mode", choices=("serial", "mpi2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(inspect_campaign(args.campaign, args.mode), indent=2, sort_keys=True)+"\n")
    print("8 genuine archives inspected; ROOT scientific seals remain pending")
