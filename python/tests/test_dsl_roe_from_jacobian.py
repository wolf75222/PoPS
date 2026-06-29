"""Generic moment Roe-from-Jacobian tests over the clean install route."""

import os

import numpy as np
import pytest

pops = pytest.importorskip("pops")
from pops import model as model_api
from pops.codegen import AOT
from pops.codegen.toolchain import _default_cxx
from pops.moments import CartesianVelocityMoments, ExactSpeeds
from pops.moments.closures import gaussian_closure
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Roe
from pops.numerics.spatial import spatial as spatial_catalog
from pops.runtime.bricks import Explicit


INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def moment_model(name, *, roe=False):
    spec = (
        CartesianVelocityMoments(
            2,
            closure=gaussian_closure(2),
            robust=False,
            exact_speeds=True,
            roe=roe,
        )
        .add_transport()
    )
    built = spec.build(name)
    assert isinstance(built.module, model_api.Module)
    return spec, built


def gaussian(n, amp=0.4):
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="xy")
    return 1.0 + amp * np.exp(-60.0 * ((xx - 0.5) ** 2 + (yy - 0.5) ** 2))


def compile_or_skip(model, path):
    if not _default_cxx(None):
        pytest.skip("no C++ compiler available")
    if not os.path.isdir(INCLUDE):
        pytest.skip("pops headers are not available")
    try:
        return model._compile_for_runtime(str(path), INCLUDE, backend=AOT())
    except RuntimeError as exc:
        if "Kokkos" in str(exc) or "compile_aot" in str(exc):
            pytest.skip("AOT moment runtime requires Kokkos: %s" % str(exc)[:160])
        raise


def test_roe_from_jacobian_public_spec_and_exclusivity():
    spec, with_roe = moment_model("roe_src", roe=True)
    assert spec.hierarchy().speeds.options()["kind"] == ExactSpeeds.ROE_DISSIPATION
    assert callable(with_roe.roe_from_jacobian)

    with pytest.raises(ValueError, match="provider"):
        with_roe.enable_roe()

    _, no_roe = moment_model("noroe_src", roe=False)
    no_roe.enable_roe()
    with pytest.raises(ValueError, match="provider"):
        no_roe.roe_from_jacobian()


def test_compile_aot_roe_system_installs_with_public_route(tmp_path):
    _, model_with_roe = moment_model("g2roe", roe=True)
    compiled = compile_or_skip(model_with_roe, tmp_path / "g2roe.so")
    assert getattr(compiled, "has_roe", False)

    n = 24
    base = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    u0 = base[:, None, None] * gaussian(n)[None, :, :]

    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.install(
        None,
        instances={
            "mom": {
                "model": compiled,
                "spatial": spatial_catalog.FiniteVolume(
                    reconstruction=FirstOrder(),
                    riemann=Roe(),
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


def test_roe_flux_rejected_without_roe_capability(tmp_path):
    _, model_without_roe = moment_model("g2noroe", roe=False)
    compiled = compile_or_skip(model_without_roe, tmp_path / "g2noroe.so")
    assert not getattr(compiled, "has_roe", False)

    sim = pops.System(n=16, L=1.0, periodic=True)
    with pytest.raises(RuntimeError, match="Roe requires capability"):
        sim.install(
            None,
            instances={
                "mom": {
                    "model": compiled,
                    "spatial": spatial_catalog.FiniteVolume(
                        reconstruction=FirstOrder(),
                        riemann=Roe(),
                    ),
                    "time": Explicit.ssprk2(),
                }
            },
        )
