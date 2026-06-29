"""Moment-model algebra and runtime smoke tests over the clean install route.

The host-side checks exercise the public ``pops.moments`` model specification API and assert that
the built facade exposes a ``pops.model.Module`` view. The compiled smoke test keeps the current
block-compile seam for generated moment facades, then wires the block through ``sim.install(...)``
with typed ``pops.numerics`` descriptors and ``pops.runtime.bricks.Explicit``.
"""

import os

import numpy as np
import pytest

pops = pytest.importorskip("pops")
from pops import model as model_api
from pops.codegen import AOT
from pops.codegen.toolchain import _default_cxx
from pops.moments import (
    CartesianVelocityMoments,
    bgk_source,
    gaussian_closure,
    lorentz_sources,
    maxwellian_moments,
    moment_indices,
    moment_names,
)
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import HLL
from pops.numerics.spatial import spatial as spatial_catalog
from pops.runtime.bricks import Explicit


INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))
RHO, UU, VV, C20v, C11v, C02v = 1.3, 0.4, -0.25, 0.9, 0.15, 0.6


def moment_model(name, order, closure=None, *, robust=False, exact_speeds=True, roe=False):
    spec = (
        CartesianVelocityMoments(
            order,
            closure=closure or gaussian_closure(order),
            robust=robust,
            exact_speeds=exact_speeds,
            roe=roe,
        )
        .add_transport()
    )
    built = spec.build(name)
    assert isinstance(built.module, model_api.Module)
    return built


def gauss_raw(p, q, u, v, c20, c11, c02, memo=None):
    """Raw Gaussian moment E[x^p y^q] via Stein recurrence, independent of the DSL path."""
    if memo is None:
        memo = {}
    if p < 0 or q < 0:
        return 0.0
    if (p, q) == (0, 0):
        return 1.0
    if (p, q) not in memo:
        if p >= 1:
            memo[(p, q)] = (
                u * gauss_raw(p - 1, q, u, v, c20, c11, c02, memo)
                + (p - 1) * c20 * gauss_raw(p - 2, q, u, v, c20, c11, c02, memo)
                + q * c11 * gauss_raw(p - 1, q - 1, u, v, c20, c11, c02, memo)
            )
        else:
            memo[(p, q)] = (
                v * gauss_raw(p, q - 1, u, v, c20, c11, c02, memo)
                + (q - 1) * c02 * gauss_raw(p, q - 2, u, v, c20, c11, c02, memo)
            )
    return memo[(p, q)]


def gauss_state(order):
    return np.array(
        [
            RHO * gauss_raw(p, q, UU, VV, C20v, C11v, C02v)
            for (p, q) in moment_indices(order)
        ]
    )


def funky_closure(S):
    return {
        "S30": 0.7 * S["S11"] + 0.2,
        "S21": S["S11"] * S["S11"] - 0.1,
        "S12": -0.4 * S["S11"],
        "S03": 1.1,
    }


def copy_closure(S):
    return {
        "S40": S["S30"],
        "S31": S["S21"],
        "S22": S["S12"],
        "S13": S["S03"],
        "S04": S["S11"],
    }


def asymmetric_mixture():
    w1, w2 = 0.4, 0.6
    g1 = (0.9, -0.3, 0.7, 0.10, 0.5)
    g2 = (-0.5, 0.45, 1.2, -0.20, 0.8)
    return {
        pq: w1 * gauss_raw(pq[0], pq[1], *g1) + w2 * gauss_raw(pq[0], pq[1], *g2)
        for pq in moment_indices(4)
    }


def test_moment_ordering_contract():
    assert moment_names(4) == [
        "M00",
        "M10",
        "M20",
        "M30",
        "M40",
        "M01",
        "M11",
        "M21",
        "M31",
        "M02",
        "M12",
        "M22",
        "M03",
        "M13",
        "M04",
    ]
    assert len(moment_indices(2)) == 6
    assert len(moment_indices(3)) == 10


def test_gaussian_closure_flux_matches_shifted_raw_moments():
    for order in (2, 3, 4):
        mg = moment_model("g%d" % order, order)
        u = gauss_state(order)
        emax = 0.0
        for direction, shift in ((0, (1, 0)), (1, (0, 1))):
            flux = np.asarray(mg.eval_flux(u, {}, direction)).ravel()
            ref = np.array(
                [
                    RHO * gauss_raw(p + shift[0], q + shift[1], UU, VV, C20v, C11v, C02v)
                    for (p, q) in moment_indices(order)
                ]
            )
            emax = max(emax, (np.abs(flux - ref) / np.maximum(np.abs(ref), 1e-12)).max())
        assert emax < 1e-12


def test_custom_closure_destandardization_matches_numpy_mirror():
    mf = moment_model("funky", 2, funky_closure)
    u6 = gauss_state(2)
    sx, sy = np.sqrt(C20v), np.sqrt(C02v)
    s11 = C11v / (sx * sy)
    c30 = (0.7 * s11 + 0.2) * sx**3
    c21 = (s11 * s11 - 0.1) * sx**2 * sy
    c12 = (-0.4 * s11) * sx * sy**2
    c03 = 1.1 * sy**3
    m30 = UU**3 + 3 * UU * C20v + c30
    m21 = UU * UU * VV + VV * C20v + 2 * UU * C11v + c21
    m12 = UU * VV * VV + UU * C02v + 2 * VV * C11v + c12
    m03 = VV**3 + 3 * VV * C02v + c03

    fx = np.asarray(mf.eval_flux(u6, {}, 0)).ravel()
    fy = np.asarray(mf.eval_flux(u6, {}, 1)).ravel()
    fx_ref = np.array([u6[1], u6[2], RHO * m30, u6[4], RHO * m21, RHO * m12])
    fy_ref = np.array([u6[3], u6[4], RHO * m21, u6[5], RHO * m12, RHO * m03])
    assert max(np.abs(fx - fx_ref).max(), np.abs(fy - fy_ref).max()) < 1e-13


def test_standardized_order_three_input_on_asymmetric_state():
    mix = asymmetric_mixture()
    u10 = RHO * np.array([mix[pq] for pq in moment_indices(3)])
    mc = moment_model("probe_s", 3, copy_closure)

    um, vm = mix[(1, 0)], mix[(0, 1)]
    k20 = mix[(2, 0)] - um * um
    k11 = mix[(1, 1)] - um * vm
    k02 = mix[(0, 2)] - vm * vm
    k30 = mix[(3, 0)] - 3 * um * mix[(2, 0)] + 2 * um**3
    k21 = mix[(2, 1)] - 2 * um * mix[(1, 1)] - vm * mix[(2, 0)] + 2 * um * um * vm
    k12 = mix[(1, 2)] - 2 * vm * mix[(1, 1)] - um * mix[(0, 2)] + 2 * um * vm * vm
    k03 = mix[(0, 3)] - 3 * vm * mix[(0, 2)] + 2 * vm**3
    assert min(abs(k30), abs(k21), abs(k12), abs(k03)) > 1e-3

    sx, sy = np.sqrt(k20), np.sqrt(k02)
    s11 = k11 / (sx * sy)
    s30, s21 = k30 / sx**3, k21 / (sx**2 * sy)
    s12, s03 = k12 / (sx * sy**2), k03 / sy**3
    k40 = s30 * sx**4
    k31 = s21 * sx**3 * sy
    k22 = s12 * sx**2 * sy**2
    k13 = s03 * sx * sy**3
    k04 = s11 * sy**4
    top_ref = {
        (4, 0): um**4 + 6 * um * um * k20 + 4 * um * k30 + k40,
        (3, 1): um**3 * vm + 3 * um * um * k11 + 3 * um * vm * k20
        + 3 * um * k21 + vm * k30 + k31,
        (2, 2): um * um * vm * vm + vm * vm * k20 + um * um * k02
        + 4 * um * vm * k11 + 2 * vm * k21 + 2 * um * k12 + k22,
        (1, 3): um * vm**3 + 3 * vm * vm * k11 + 3 * um * vm * k02
        + 3 * vm * k12 + um * k03 + k13,
        (0, 4): vm**4 + 6 * vm * vm * k02 + 4 * vm * k03 + k04,
    }

    err = 0.0
    for direction, shift in ((0, (1, 0)), (1, (0, 1))):
        flux = np.asarray(mc.eval_flux(u10, {}, direction)).ravel()
        ref = np.array(
            [
                RHO
                * (
                    top_ref[(p + shift[0], q + shift[1])]
                    if p + q == 3
                    else mix[(p + shift[0], q + shift[1])]
                )
                for (p, q) in moment_indices(3)
            ]
        )
        err = max(err, (np.abs(flux - ref) / np.maximum(np.abs(ref), 1e-12)).max())
    assert err < 1e-12


def test_exact_wave_speeds_match_finite_difference_jacobian():
    mg2 = moment_model("g2ws", 2)
    u = gauss_state(2)
    for direction in (0, 1):
        smin, smax = np.asarray(mg2.eval_wave_speeds(u, {}, direction)).ravel()
        eps = 1e-7
        jac = np.zeros((6, 6))
        for col in range(6):
            up, um = u.copy(), u.copy()
            up[col] += eps
            um[col] -= eps
            jac[:, col] = (
                np.asarray(mg2.eval_flux(up, {}, direction)).ravel()
                - np.asarray(mg2.eval_flux(um, {}, direction)).ravel()
            ) / (2 * eps)
        lam = np.linalg.eigvals(jac).real
        assert abs(smin - lam.min()) < 1e-5
        assert abs(smax - lam.max()) < 1e-5


def test_robust_floor_is_finite_on_vacuum_and_identity_on_healthy_state():
    robust = moment_model("g2rob", 2, robust=True)
    raw = moment_model("g2raw", 2, robust=False)
    uvac = np.zeros(6)
    with np.errstate(all="ignore"):
        fvac_r = np.asarray(robust.eval_flux(uvac, {}, 0)).ravel()
        fvac_raw = np.asarray(raw.eval_flux(uvac, {}, 0)).ravel()
    assert np.isfinite(fvac_r).all()
    assert not np.isfinite(fvac_raw).all()

    u = gauss_state(2)
    fh_r = np.asarray(robust.eval_flux(u, {}, 0)).ravel()
    fh_raw = np.asarray(raw.eval_flux(u, {}, 0)).ravel()
    err = (np.abs(fh_r - fh_raw) / np.maximum(np.abs(fh_raw), 1e-12)).max()
    assert err < 1e-10


def test_lorentz_sources_match_manual_order_two_table():
    mf = {pq: float(k + 2) * (0.5 + 0.1 * k) for k, pq in enumerate(moment_indices(2))}
    qm, oc, ex, ey = 1.7, -0.6, 0.3, 0.9
    src = lorentz_sources(mf, ex, ey, qm, oc)
    expected = [
        0.0,
        qm * ex * mf[(0, 0)] + oc * mf[(0, 1)],
        qm * 2 * ex * mf[(1, 0)] + oc * 2 * mf[(1, 1)],
        qm * ey * mf[(0, 0)] - oc * mf[(1, 0)],
        qm * (ex * mf[(0, 1)] + ey * mf[(1, 0)]) + oc * (mf[(0, 2)] - mf[(2, 0)]),
        qm * 2 * ey * mf[(0, 1)] - oc * 2 * mf[(1, 1)],
    ]
    assert max(abs(a - b) for a, b in zip(src, expected)) < 1e-14
    assert len(lorentz_sources({pq: 1.0 for pq in moment_indices(4)}, ex, ey, qm, oc)) == 15


def test_maxwellian_moments_and_bgk_source():
    for order in (2, 3, 4):
        idx = moment_indices(order)
        mg = {pq: RHO * gauss_raw(pq[0], pq[1], UU, VV, C20v, C11v, C02v) for pq in idx}
        meq = maxwellian_moments(mg)
        assert max(abs(meq[k] - mg[pq]) for k, pq in enumerate(idx)) < 1e-12
        source = bgk_source(mg, 7.0)
        assert max(abs(float(x)) for x in source) < 1e-12
        invariants = [source[k] for k, pq in enumerate(idx) if pq in ((0, 0), (1, 0), (0, 1))]
        assert all(float(x) == 0.0 for x in invariants)

    mix = asymmetric_mixture()
    mne = {pq: RHO * mix[pq] for pq in moment_indices(4)}
    meq = maxwellian_moments(mne)
    idx4 = moment_indices(4)
    elow = max(abs(meq[k] - mne[pq]) for k, pq in enumerate(idx4) if pq[0] + pq[1] <= 2)
    dhi = max(abs(meq[k] - mne[pq]) for k, pq in enumerate(idx4) if pq[0] + pq[1] >= 3)
    assert elow < 1e-12
    assert dhi > 1e-3
    sne = bgk_source(mne, 3.0)
    assert all(
        float(sne[k]) == 0.0
        for k, pq in enumerate(idx4)
        if pq in ((0, 0), (1, 0), (0, 1))
    )


def test_moment_model_guards():
    with pytest.raises(ValueError, match="order >= 2"):
        CartesianVelocityMoments(1, closure=gaussian_closure(1))
    with pytest.raises(ValueError, match="S30"):
        moment_model("bad2", 2, lambda _s: {"S30": 0.0})


def test_compile_aot_hll_system_installs_with_public_route(tmp_path):
    if not _default_cxx(None):
        pytest.skip("no C++ compiler available")
    if not os.path.isdir(INCLUDE):
        pytest.skip("pops headers are not available")

    try:
        compiled = moment_model("g2sys", 2)._compile_for_runtime(
            str(tmp_path / "g2sys.so"), INCLUDE, backend=AOT()
        )
    except RuntimeError as exc:
        if "Kokkos" in str(exc) or "compile_aot" in str(exc):
            pytest.skip("AOT moment runtime requires Kokkos: %s" % str(exc)[:160])
        raise

    n = 16
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    pert = 1.0 + 0.1 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    u0 = gauss_state(2)[:, None, None] * pert[None, :, :]

    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.install(
        None,
        instances={
            "mom": {
                "model": compiled,
                "spatial": spatial_catalog.FiniteVolume(
                    reconstruction=FirstOrder(),
                    riemann=HLL(),
                ),
                "time": Explicit.ssprk2(),
                "initial": u0,
            }
        },
    )
    for _ in range(10):
        sim.step(5e-4)
    out = np.asarray(sim._get_state("mom"))
    assert np.isfinite(out).all()
    dm = abs(out[0].sum() - u0[0].sum()) / abs(u0[0].sum())
    assert dm < 1e-12
