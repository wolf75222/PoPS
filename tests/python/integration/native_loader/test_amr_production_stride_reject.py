#!/usr/bin/env python3
"""Pre-dispatch guards for a detached compiled-AMR package.

This module is deliberately *not* an end-to-end native-loader test: the package metadata below is
synthetic and no shared library is loaded.  It verifies only the Python contract that rejects values
which the flat AMR package ABI cannot transport (multirate stride and a partial IMEX mask) before
native loading.  Actual compiled-package execution is covered by ``test_dsl_production_amr.py``.
"""

import sys

import pytest

import pops.runtime._engine_descriptors as engine
from pops.codegen.abi import module_header_signature
from pops.codegen.loader import CompiledModel
from pops.runtime._system import AmrSystem  # private engine seam exercised by this guard


def _compiled_amr_metadata(*, so_path: str = "/nonexistent/pops-amr-guard.so") -> CompiledModel:
    """Return exact detached metadata for the pre-loader branch, without claiming native execution."""
    return CompiledModel(
        so_path=so_path,
        backend="production",
        cons_names=["rho", "rho_u", "rho_v", "E"],
        cons_roles=["Density", "MomentumX", "MomentumY", "Energy"],
        prim_names=["rho", "u", "v", "p"],
        n_vars=4,
        gamma=1.4,
        n_aux=3,
        params={},
        caps={},
        abi_key=f"{module_header_signature()}|c++|c++23",
        model_hash="amr-preloader-guard",
        cxx="c++",
        std="c++23",
        target="amr_system",
    )


@pytest.mark.parametrize(
    "time",
    [engine.IMEX(stride=5), engine.Explicit(stride=5)],
    ids=["imex", "explicit"],
)
def test_compiled_amr_guard_rejects_untransported_stride(time):
    sim = AmrSystem(n=16, periodic=True)
    with pytest.raises(
        ValueError,
        match=r"stride=5 not transported by the production AMR path",
    ):
        sim.add_equation(
            "gas", _compiled_amr_metadata(), spatial=engine.Spatial(), time=time
        )


@pytest.mark.parametrize(
    ("time", "selector"),
    [
        (engine.IMEX(implicit_vars=["rho_u"]), "implicit_vars"),
        (engine.IMEX(implicit_roles=["momentum_x"]), "implicit_roles"),
    ],
)
def test_compiled_amr_guard_rejects_untransported_partial_imex_mask(time, selector):
    sim = AmrSystem(n=16, periodic=True)
    with pytest.raises(
        ValueError,
        match=r"implicit_vars / implicit_roles .* not transported",
    ) as excinfo:
        sim.add_equation(
            "gas", _compiled_amr_metadata(), spatial=engine.Spatial(), time=time
        )
    assert selector in str(excinfo.value)


@pytest.mark.parametrize("time", [engine.Explicit(), engine.IMEX()], ids=["explicit", "imex"])
def test_supported_defaults_cross_the_python_guard_and_reach_the_native_loader(time, tmp_path):
    missing = tmp_path / "missing-amr-package.so"
    sim = AmrSystem(n=16, periodic=True)

    with pytest.raises(RuntimeError) as excinfo:
        sim.add_equation(
            "gas",
            _compiled_amr_metadata(so_path=str(missing)),
            spatial=engine.Spatial(),
            time=time,
        )

    message = str(excinfo.value)
    assert str(missing) in message
    assert "stride" not in message and "implicit_vars" not in message


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
