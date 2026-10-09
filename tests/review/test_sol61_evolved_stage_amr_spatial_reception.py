"""Independent SOURCE_ONLY math, source and protocol attacks; no Native claim."""

import ast
import copy
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "spatial_reader", Path(__file__).with_name("sol61_evolved_stage_amr_spatial_reception.py")
)
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)
ROOT = Path(__file__).resolve().parents[2]


def geometry(n=8, start=2, stop=6):
    boxes = np.array(
        [[0, 0, 0, n - 1, n - 1], [1, 2 * start, 0, 2 * stop - 1, 2 * n - 1]], dtype=np.int64
    )
    masks = r.b.topology(
        boxes,
        n,
        2,
        ["partitioned"] * 2,
        [np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)],
    )
    rows = []
    for level, active in enumerate(masks):
        m = n * 2**level
        rows.append(
            dict(
                active=active.copy(),
                valid=(np.ones_like(active) if level == 0 else active.copy()),
                cartesian_cell_volume=np.full((m, m), 1 / m**2),
                declared_no_EB_kappa=np.ones((m, m)),
                x_edges=np.arange(m + 1) / m,
                y_edges=np.arange(m + 1) / m,
                native_base_shape=np.array([n, n], dtype=np.int64),
                native_patch_boxes=boxes.copy(),
            )
        )
    return rows, masks


def temperature(n):
    x = (np.arange(n) + 0.5) / n
    return np.stack((0.15 + 0.02 * np.cos(2 * np.pi * x), 0.25 + 0.015 * np.sin(2 * np.pi * x)))


def fixture(n=8):
    """Synthetic implicit algebra on a leaf graph, never authentic saved states."""
    rows, masks = geometry(n, n // 4, 3 * n // 4)
    covered = ~masks[0][0]
    leaves = [
        (l, i)
        for i in range(n)
        for l, i in ([(1, 2 * i), (1, 2 * i + 1)] if covered[i] else [(0, i)])
    ]
    initial = copy.deepcopy(rows)
    for level, row in enumerate(initial):
        m = n * 2**level
        for i in range(2):
            row[f"Q{i}"] = np.tile(r.initial_q(m)[i], (m, 1)).ravel()
        marker = 1 + 0.04 * m * (
            np.sin(2 * np.pi * (np.arange(1, m + 1) / m - 0.5))
            - np.sin(2 * np.pi * (np.arange(m) / m - 0.5))
        ) / (2 * np.pi)
        row["forcing"] = np.stack(
            [np.full((m, m), f) for f in r.b.declared_load(2)[3]] + [np.tile(marker, (m, 1))]
        ).ravel()
    images = {"initial": initial}
    current = [temperature(n), temperature(2 * n)]
    for phase in ("accepted", "continuous"):
        old = images["initial" if phase == "accepted" else "accepted"]
        previous = np.array(
            [
                [old[l][f"Q{j}"].reshape(n * 2**l, n * 2**l)[0, i] for l, i in leaves]
                for j in range(2)
            ]
        )

        def tower(x):
            out = [np.zeros((2, n)), np.zeros((2, 2 * n))]
            for col, (l, i) in enumerate(leaves):
                out[l][:, i] = x[:, col]
            for i in np.flatnonzero(covered):
                out[0][:, i] = out[1][:, 2 * i : 2 * i + 2].mean(axis=1)
            return out

        def f(x, previous=previous):
            t = tower(x.reshape(2, -1))
            diff, _, _ = r.flux_action(t, covered)
            action = np.array([[diff[l][j, i] for l, i in leaves] for j in range(2)])
            return (
                r.b.q_of(x.reshape(2, -1))
                - previous
                - 0.01 * (action + r.b.declared_load(2)[3][:, None])
            ).ravel()

        x = np.array([[current[l][j, i] for l, i in leaves] for j in range(2)]).ravel()
        for _ in range(8):
            residual = f(x)
            if np.max(np.abs(residual)) < 1e-13:
                break
            jac = np.stack(
                [(f(x + 1e-6 * e) - f(x - 1e-6 * e)) / (2e-6) for e in np.eye(x.size)], axis=1
            )
            x -= np.linalg.solve(jac, residual)
        assert np.max(np.abs(f(x))) < 1e-13
        current = tower(x.reshape(2, -1))
        saved = copy.deepcopy(rows)
        for level, row in enumerate(saved):
            m = n * 2**level
            q = r.b.q_of(current[level])
            if level == 0:
                q[:, covered] = r.b.q_of(current[1]).reshape(2, n, 2).mean(axis=2)[:, covered]
            for i in range(2):
                row[f"Q{i}"] = np.tile(q[i], (m, 1)).ravel()
            for name, value in (
                ("T0", current[level][0]),
                ("T1", current[level][1]),
                ("z", 0.25 * current[level][0] + 0.5 * current[level][1]),
            ):
                row[name] = np.tile(value, (m, 1)).ravel()
                row[name + "-previous"] = (
                    row[name].copy()
                    if phase == "accepted"
                    else images["accepted"][level][name].copy()
                )
                row["history_sample_identity_" + name] = np.array(
                    [1], dtype=np.uint8
                )  # protocol deliberately NOT Native
            row["forcing"] = initial[level]["forcing"].copy()
        images[phase] = saved
    images["reloaded"] = copy.deepcopy(images["accepted"])
    images["replay"] = copy.deepcopy(images["continuous"])
    return images, masks


@pytest.mark.parametrize("n", (8, 16))
def test_original_math_synthetic_only(n):
    images, masks = fixture(n)
    result = r.science(images, masks, n)
    assert max(p["original_F_linf"] for p in result.values()) < 1e-12
    assert all(p["restriction_gap"] > 1e-10 for p in result.values())
    assert "pops" not in sys.modules


@pytest.mark.parametrize(
    "mutation",
    (
        "transpose_D",
        "sign_D",
        "omit_cross",
        "no_reflux",
        "constant_ghost",
        "no_restrict",
        "wrong_fine_order",
    ),
)
def test_leaf_operator_discriminates_wrong_physics_and_transfer(mutation):
    n = 8
    covered = np.array([False, False, True, True, True, True, False, False])
    tower = (temperature(n), temperature(2 * n))
    original, _, _ = r.flux_action(tower, covered)
    options = {}
    if mutation == "transpose_D":
        options["matrix"] = r.D.T
    if mutation == "sign_D":
        options["matrix"] = -r.D
    if mutation == "omit_cross":
        options["matrix"] = np.diag(np.diag(r.D))
    if mutation == "no_reflux":
        options["reflux"] = False
    if mutation == "constant_ghost":
        options["quadratic"] = False
    if mutation == "no_restrict":
        options["restrict"] = False
    if mutation == "wrong_fine_order":
        tower = (tower[0], tower[1][:, ::-1])
    wrong, _, _ = r.flux_action(tower, covered, **options)
    masks = (~covered, np.repeat(covered, 2))
    assert (
        max(np.max(np.abs((a - c)[:, m])) for a, c, m in zip(original, wrong, masks, strict=True))
        > 1e-5
    )
    if mutation == "no_reflux":
        assert (
            np.max(
                np.abs(
                    wrong[0][:, ~covered].sum(axis=1) / n
                    + wrong[1][:, masks[1]].sum(axis=1) / (2 * n)
                )
            )
            > 1e-6
        )


@pytest.mark.parametrize("start,stop", ((0, 3), (2, 6), (5, 8)))
def test_interface_periodic_signs_conserve_each_signed_row(start, stop):
    n = 8
    covered = np.array([start <= i < stop for i in range(n)])
    tower = (temperature(n), temperature(2 * n))
    for matrix in (r.D, np.array([[0.0, -3.0], [2.0, 0.0]]), np.array([[1.0, 2.0], [2.0, 4.0]])):
        action, _, _ = r.flux_action(tower, covered, matrix=matrix)
        balance = action[0][:, ~covered].sum(axis=1) / n + action[1][:, np.repeat(covered, 2)].sum(
            axis=1
        ) / (2 * n)
        np.testing.assert_allclose(balance, 0.0, atol=2e-15)


@pytest.mark.parametrize(
    "mutation",
    (
        "covered_H_of_avgT",
        "coarse_temperature",
        "stale_T",
        "wrong_z",
        "seed_in_rhs",
        "wrong_initial",
        "shift_initial",
        "load_change",
        "marker_change",
        "fine_index",
        "replay_bits",
        "history_swap",
        "negative_branch",
        "y_mode",
        "active",
        "metric",
    ),
)
def test_original_science_refuses_equilibrated_countermodels(mutation):
    images, masks = fixture()
    row = images["continuous"][0]
    if mutation == "covered_H_of_avgT":
        t = np.stack([row[k].reshape(8, 8) for k in ("T0", "T1")])
        q = r.b.q_of(t)
        for i in range(2):
            row[f"Q{i}"].reshape(8, 8)[~masks[0]] = q[i][~masks[0]]
    elif mutation == "coarse_temperature":
        row["T0"][~masks[0].ravel()] += 0.001
    elif mutation == "stale_T":
        row["T0"] = images["accepted"][0]["T0"].copy()
    elif mutation == "wrong_z":
        row["z"] += 0.001
    elif mutation == "seed_in_rhs":
        for r0 in images["initial"]:
            r0["Q0"] += 0.002
    elif mutation in ("wrong_initial", "shift_initial"):
        images["initial"][0]["Q0"] += 0.001
    elif mutation == "load_change":
        row["forcing"][0] += 0.001
    elif mutation == "marker_change":
        row["forcing"][-1] += 0.001
    elif mutation == "fine_index":
        images["continuous"][1]["T0"] = (
            images["continuous"][1]["T0"].reshape(16, 16)[:, ::-1].ravel()
        )
    elif mutation == "replay_bits":
        images["replay"][0]["Q0"][0] = np.nextafter(images["replay"][0]["Q0"][0], np.inf)
    elif mutation == "history_swap":
        row["T0"], row["T0-previous"] = row["T0-previous"], row["T0"]
    elif mutation == "negative_branch":
        row["T0"] *= -1
    elif mutation == "y_mode":
        row["T0"].reshape(8, 8)[1, 0] += 0.001
    elif mutation == "active":
        row["active"][0, 0] = False
    else:
        row["cartesian_cell_volume"][0, 0] *= 2
    with pytest.raises((ValueError, AssertionError)):
        r.science(images, masks, 8)


def test_declared_exact_cell_integrals_against_independent_Fourier_moments():
    tree = ast.parse((ROOT / "tests/python/support/evolved_stage_amr_spatial.py").read_text())
    primitive = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "primitive"
    )
    # Only the actual declared IC primitive, not any author oracle/builder, is evaluated.
    namespace = dict(np=np, sin=math.sin, cos=math.cos)
    exec(
        compile(ast.Module(body=[primitive], type_ignores=[]), "declared IC only", "exec"),
        namespace,
    )
    for n in (1, 8, 16, 23):
        values = np.stack(
            [
                (np.array(namespace["primitive"]((i + 1) / n)) - namespace["primitive"](i / n)) * n
                for i in range(n)
            ],
            axis=1,
        )
        np.testing.assert_allclose(values, r.initial_q(n), atol=3e-15, rtol=0)
    np.testing.assert_allclose(r.initial_q(1)[:, 0], [0.17896125, 0.3201125], atol=2e-16)


@pytest.mark.parametrize(
    "mutation",
    (
        "none",
        "signed_D",
        "cos_to_sin",
        "no_CellBounds",
        "old_seed",
        "duplicate_build",
        "rebind_D",
        "foreign_import",
    ),
)
def test_reviewed_public_source_closed_without_importing_author(mutation):
    files = {
        k: (ROOT / ("tests/python/support/" + v)).read_bytes()
        for k, v in dict(
            spatial="evolved_stage_amr_spatial.py",
            amr="evolved_stage_amr.py",
            equations="evolved_stage_mms.py",
            controls="captured_diffusion_mms.py",
        ).items()
    }
    if mutation == "signed_D":
        files["equations"] = files["equations"].replace(b"[-.001, .014]", b"[.001, .014]")
    if mutation == "cos_to_sin":
        files["spatial"] = files["spatial"].replace(b"a, b = .15+.02*cos", b"a, b = .15+.02*sin")
    if mutation == "no_CellBounds":
        files["spatial"] = files["spatial"].replace(
            b"cell_integrals=(exact,)", b"cell_integrals=None"
        )
    if mutation == "old_seed":
        files["spatial"] = files["spatial"].replace(
            b"previous=tuple(states[i][0]", b"previous=tuple(temperature[i]"
        )
    if mutation == "duplicate_build":
        files["spatial"] += b"\ndef build(cells=8):\n    return None\n"
    if mutation == "rebind_D":
        files["spatial"] += b"\nDIFFUSION = np.eye(2)\n"
    if mutation == "foreign_import":
        files["spatial"] = files["spatial"].replace(
            b"from tests.python.support.evolved_stage_mms import", b"from unrelated_model import"
        )
    if mutation == "none":
        r.declared_source(**files)
    else:
        with pytest.raises(ValueError):
            r.declared_source(**files)
    assert "pops" not in sys.modules


def test_no_synthetic_native_approval_and_homogeneous_contract_unchanged(tmp_path):
    assert r.contract()["qualification"] != r.b.contract()["qualification"]
    assert "status" not in r.contract()
    pins = tmp_path / "pins.json"
    approval = tmp_path / "approval.json"
    pins.write_bytes(b"{}")
    approval.write_bytes(b'{"schema":"SOURCE_ONLY-NOT-ROOT"}')
    with pytest.raises(ValueError, match="approval"):
        r.receive(pins, r.b.digest(pins.read_bytes()), approval, r.b.digest(approval.read_bytes()))


def test_actual_cpp_stencil_authorities_read_not_native_execution():
    quadratic = (ROOT / "include/pops/numerics/elliptic/mg/composite_fac_nlevel.hpp").read_text()
    assert "weights[axis][0] = (d - s) / Real(2)" in quadratic
    assert "weights[axis][1] = Real(1) - d" in quadratic
    assert "weights[axis][2] = (d + s) / Real(2)" in quadratic
    fac = (ROOT / "include/pops/numerics/elliptic/amr/composite_fac_poisson.hpp").read_text()
    body = fac[
        fac.index("void apply_linear_composite(") : fac.index("void install_embedded_boundary(")
    ]
    assert (
        body.index("synchronize_linear_solution")
        < body.index("image -=")
        < body.index("apply_flux_mismatch")
    )
    assert "arithmetic ? Real(0.5) * lo + Real(0.5) * center" in body
    projection = (ROOT / "python/pops/codegen/program_emit_evolved_field.py").read_text()
    assert "if (masked && active(index, 0) <= 0) return pops::Real(0)" in projection
    assert "PureFieldAlgebra::copy(evolved_output, evolved_previous)" in projection
