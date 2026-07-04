#!/usr/bin/env python3
"""ADC-515 (Spec 6 sec.20): introspection + runtime CFL/profile on the AMR route, end-to-end.

The inspection column of the sec.20 matrix, joining the two ADC-515 introspection seams
(``arguments`` / ``estimate_memory`` on the AMR-route ``CompiledModel``) to the live AMR runtime
surface (``profile`` / ``step_cfl`` / ``amr.patch_table``):

  * INERT metadata on the AMR-route handle: ``arguments()`` reports ``layout='amr'`` with the block
    instance / named aux / typed params, ``estimate_memory(mesh)`` is a conservative patch-budget
    FORMULA, and ``inspect_amr()`` surfaces the carried refine / regrid tags. These run on a stub
    ``CompiledModel`` carrying the AMR ``_layout`` exactly as ``pops.compile(layout=AMR(...))``'s
    ``_orchestration_amr._compile_amr`` attaches it -- no ``.so`` dlopen, so the inert surface is
    validated locally without the Kokkos AOT compile the real per-block AMR loader needs.
  * LIVE runtime on a real ``AmrSystem``: a typed ``profile(Profile.Basic())`` context wraps two
    ``step_cfl`` runtime-CFL steps (the engine picks a CFL-bounded dt and advances the clock), the
    closing ``PerformanceSummary.by_amr_mpi()`` answers, and ``amr.patch_table()`` reads the built
    hierarchy. ``profile`` is the SAME seam ``bind``'s ``BoundSimulation`` whitelists.

Runtime: ``importorskip('pops')`` skips on a bare box; the live cells step a real Kokkos-Serial
engine on the CI runner. ``__main__`` runs pytest.
"""
import sys

import numpy as np
import pytest

pops = pytest.importorskip("pops", exc_type=ImportError)

from pops.codegen.loader import CompiledModel  # noqa: E402
from pops.mesh import CartesianMesh  # noqa: E402
from pops.mesh.amr import Refine, RegridEvery  # noqa: E402
from pops.mesh.layouts import AMR, Uniform  # noqa: E402
from pops.physics.model import Param  # noqa: E402
from pops.runtime.system import AmrSystem  # noqa: E402  (ADC-545 advanced runtime seam)


def _amr_route_handle():
    """A stub AMR-route ``CompiledModel`` (target='amr_system', no ``.so``) carrying the AMR layout.

    The shape ``pops.compile(problem, layout=AMR(...))`` returns: the first block's model with
    target='amr_system' and the Problem's AMR layout attached on ``_layout`` (what ``_compile_amr``
    attaches). No ``.so`` is dlopened -- the arguments / estimate_memory / inspect_amr surface is
    pure metadata + formula, so it is validated here without the Kokkos AOT per-block loader compile.
    """
    handle = CompiledModel(
        so_path="<stub-amr>", backend="production", adder="add_native_block",
        cons_names=["rho", "mx", "my"], cons_roles=["Density", "MomentumX", "MomentumY"],
        prim_names=["rho", "mx", "my"], n_vars=3, gamma=1.4, n_aux=1,
        params={"alpha": Param("alpha", 1.0, kind="runtime")},
        caps={"cpu": True, "amr": True, "mpi": True}, abi_key="k", model_hash="h", cxx="c++",
        std="c++23", target="amr_system", aux_extra_names=["B_z"])
    handle._layout = AMR(base=CartesianMesh(n=64, periodic=True), max_levels=2, ratio=2,
                         regrid=RegridEvery(4), refine=Refine.on("rho").above(0.1))
    return handle


# --- inert introspection on the AMR-route handle ---------------------------------
def test_arguments_on_the_amr_route_handle():
    args = _amr_route_handle().arguments()
    assert args.layout_runtime["layout"] == "amr"
    assert args.layout_runtime["supports_mpi"] is True
    inst = next(iter(args.instances.values()))
    assert inst["components"] == 3 and inst["conservative"] == ["rho", "mx", "my"]
    assert set(args.aux) == {"B_z"}
    assert args.params["alpha"]["kind"] == "runtime" and args.params["alpha"]["required"] is True


def test_estimate_memory_on_the_amr_route_handle_adds_a_patch_budget():
    handle = _amr_route_handle()
    mesh = CartesianMesh(n=64, L=1.0, periodic=True)
    amr_est = handle.estimate_memory(mesh)                       # auto AMR from _layout
    uni_est = handle.estimate_memory(mesh, layout=Uniform(mesh))
    assert amr_est.layout == "amr" and uni_est.layout == "uniform"
    assert amr_est.categories.get("amr_patch", 0) > 0
    # Conservative full-refinement worst case: the AMR estimate dominates the Uniform one.
    assert amr_est.total_bytes >= uni_est.total_bytes > 0
    # A pure formula: no MultiFab allocated, every assumption inspectable.
    assert amr_est.assumptions and any("CONSERVATIVE" in a for a in amr_est.assumptions)


def test_inspect_amr_surfaces_the_carried_refine_regrid_tags():
    rep = _amr_route_handle().inspect_amr().to_dict()
    assert rep["layout"] == "amr" and rep["max_levels"] == 2
    slots = {row["slot"] for row in rep["policies"]}
    assert "refine" in slots and "regrid" in slots


# --- live runtime profile + CFL on a real AmrSystem ------------------------------
def _built_amr(n=32):
    sim = AmrSystem(n=n, L=1.0, periodic=True, regrid_every=2, coarse_max_grid=16)
    sim.add_block("ne", pops.Model(pops.Scalar(), pops.ExB(B0=1.0), pops.NoSource(),
                                   pops.ChargeDensity(charge=1.0)),
                  spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    sim.set_poisson(bc="periodic")
    sim.set_refinement(threshold=1.05)
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    ne = 1.0 + 0.4 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)
    sim.set_density("ne", ne + (1.0 - ne.mean()))
    return sim


def test_profile_context_and_step_cfl_on_a_built_amr_system():
    """A typed ``profile()`` context wraps two runtime-CFL steps; the AMR/MPI summary answers.

    ``profile`` is the seam ``bind``'s ``BoundSimulation`` whitelists, exercised here on the engine
    it delegates to. ``step_cfl`` advances by a CFL-bounded dt; ``by_amr_mpi()`` must answer (counters
    may be zero on a host build, but the surface never raises).
    """
    n = 32
    sim = _built_amr(n)
    with sim.profile(pops.Profile.Basic()) as prof:
        sim.step_cfl(0.4)
        sim.step_cfl(0.4)
    assert sim.time() > 0.0 and np.isfinite(sim.time()), "step_cfl did not advance the clock"
    summary = prof.summary()
    assert summary.by_amr_mpi() is not None
    table = sim.amr.patch_table()
    assert table.built is True and table.base_n == n


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
