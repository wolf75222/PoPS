#!/usr/bin/env python3
"""AMR metadata introspection contract plus a real private-engine runtime integration.

The historical filename says ``endtoend``, but the two scopes are intentionally separate:

  * STATIC metadata contract on an explicitly non-executable detached artifact: ``arguments()``
    reports ``layout='amr'`` with the block instance / named aux / typed params,
    ``estimate_memory(mesh)`` is a conservative patch-budget formula, and
    ``pops.inspect(artifact.layout)`` surfaces the carried refine / regrid tags. This phase-record
    fixture is not a native package or a production execution test.
  * LIVE runtime on a real ``AmrSystem``: a typed ``profile(Profile.Basic())`` context wraps two
    ``step_cfl`` runtime-CFL steps (the engine picks a CFL-bounded dt and advances the clock), the
    closing ``PerformanceSummary.by_amr_mpi()`` answers, and ``amr.patch_table()`` reads the built
    hierarchy. ``profile`` is exercised on the internal AMR engine seam.

The live cells step a real Kokkos-Serial engine on the CI runner. ``__main__`` runs pytest.
"""
import sys

import numpy as np
import pytest

import pops
import pops.runtime._engine_descriptors as engine  # noqa: E402
from pops.runtime._engine_descriptors import Periodic  # noqa: E402

from pops.codegen.loader import CompiledModel  # noqa: E402
from pops.codegen._plans import (  # noqa: E402
    BindInputs, InstallPlan,
)
from pops.codegen._compiled_artifact import (  # noqa: E402
    CompiledBlockArtifact, CompiledSimulationArtifact,
)
from pops.codegen._compiled_model_identity import compiled_model_identity  # noqa: E402
from pops.layouts import Uniform  # noqa: E402
from pops.params import RuntimeParam  # noqa: E402
from pops.runtime._system import AmrSystem  # noqa: E402  (ADC-545 advanced runtime seam)
from tests.python.support.layout_plan import cartesian_grid  # noqa: E402
from tests.python.support.resolved_amr_plan import resolved_amr_plan  # noqa: E402


def _amr_metadata_fixture():
    """Build exact phase records for static reporting, without a loadable native component.

    ``backend='production'`` is the compiled-package phase tag required by the record schema; the
    sentinel path makes explicit that this fixture is never installed or executed.
    """
    alpha = RuntimeParam("alpha", default=1.0)
    handle = CompiledModel(
        so_path="<metadata-only-amr-component>", backend="production",
        cons_names=["rho", "mx", "my"], cons_roles=["Density", "MomentumX", "MomentumY"],
        prim_names=["rho", "mx", "my"], n_vars=3, gamma=1.4, n_aux=1,
        params={"alpha": alpha},
        caps={"cpu": True, "amr": True, "mpi": True}, abi_key="k", model_hash="h", cxx="c++",
        std="c++23", target="amr_system", aux_extra_names=["B_z"])
    handle.definition_identity = compiled_model_identity(model_hash="h")
    resolved = resolved_amr_plan(
        block_names=("ne",),
        parameters=(alpha,),
        tag_parameter="alpha",
        cells=64,
        name="amr-introspection-metadata",
    )
    schema = resolved.bind_schema
    artifact = CompiledSimulationArtifact(
        plan=resolved,
        program=None,
        blocks=(CompiledBlockArtifact(
            "ne", handle, resolved.blocks[0].spatial, ("U",)),),
    )
    inputs = BindInputs()
    install = InstallPlan(
        artifact=artifact,
        bind_inputs=inputs,
        instances={"ne": {"model": handle, "spatial": resolved.blocks[0].spatial}},
        params=schema.resolve_bind({}, compile_values=resolved.compile_values),
        aux={},
    )
    install.verify()
    return artifact


# --- inert introspection on the AMR-route handle ---------------------------------
def test_static_metadata_arguments_report_the_amr_route():
    args = _amr_metadata_fixture().arguments()
    assert args.layout_runtime["layout"] == "amr"
    assert args.layout_runtime["supports_mpi"] is True
    inst = next(iter(args.instances.values()))
    assert inst["components"] == 3 and inst["conservative"] == ["rho", "mx", "my"]
    assert set(args.aux) == {"B_z"}
    alpha_qid = next(iter(args.params))
    assert args.params[alpha_qid]["name"] == "alpha"
    assert args.params[alpha_qid]["kind"] == "runtime"
    assert args.params[alpha_qid]["required"] is False  # declaration carries a bind default


def test_static_metadata_estimate_adds_an_amr_patch_budget():
    handle = _amr_metadata_fixture()
    mesh = cartesian_grid(n=64, L=1.0, periodic=True)
    amr_est = handle.estimate_memory(mesh)                       # auto AMR from InstallPlan
    uni_est = handle.estimate_memory(mesh, layout=Uniform(mesh))
    assert amr_est.layout == "amr" and uni_est.layout == "uniform"
    assert amr_est.categories.get("amr_patch", 0) > 0
    # Conservative full-refinement worst case: the AMR estimate dominates the Uniform one.
    assert amr_est.total_bytes >= uni_est.total_bytes > 0
    # A pure formula: no MultiFab allocated, every assumption inspectable.
    assert amr_est.assumptions and any("CONSERVATIVE" in a for a in amr_est.assumptions)


def test_static_metadata_inspection_surfaces_the_carried_refine_regrid_tags():
    artifact = _amr_metadata_fixture()
    assert not hasattr(artifact, "inspect_amr")
    inspected = pops.inspect(artifact.layout)
    rep = inspected["amr_report"]
    assert rep["layout"] == "amr" and rep["max_levels"] == 2
    # The final AMR descriptor owns typed authorities, not the retired ``refine``/``patches``
    # policy slots.  Their complete payloads remain visible under the ordinary layout options.
    assert {"tagging", "regrid", "transfer", "execution"} <= set(inspected["options"])
    assert inspected["options"]["hierarchy"]["max_levels"] == 2


# --- live runtime profile + CFL on a real AmrSystem ------------------------------
def _built_amr(n=32):
    sim = AmrSystem(n=n, L=1.0, periodicity=(True, True), regrid_every=2, coarse_max_grid=16)
    sim.set_temporal_relations([2], [1], ["integral_only"])
    sim.add_equation("ne", engine.Model(engine.Scalar(), engine.ExB(B0=1.0), engine.NoSource(),
                                   engine.BackgroundDensity(alpha=1.0, n0=1.0)),
                  spatial=engine.Spatial(minmod=True), time=engine.Explicit())
    sim.set_poisson(bc=Periodic())
    sim.set_refinement(threshold=1.05)
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    ne = 1.0 + 0.4 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)
    sim.set_density("ne", ne + (1.0 - ne.mean()))
    return sim


def test_profile_context_and_step_cfl_on_a_built_amr_system():
    """A typed ``profile()`` context wraps two runtime-CFL steps; the AMR/MPI summary answers.

    ``profile`` is an internal engine seam, exercised here on the engine
    it delegates to. ``step_cfl`` advances by a CFL-bounded dt; ``by_amr_mpi()`` must answer (counters
    may be zero on a host build, but the surface never raises).
    """
    n = 32
    sim = _built_amr(n)
    from pops.runtime._profile import Profile

    with sim.profile(Profile.Basic()) as prof:
        sim.step_cfl(0.4)
        sim.step_cfl(0.4)
    assert sim.time() > 0.0 and np.isfinite(sim.time()), "step_cfl did not advance the clock"
    summary = prof.summary()
    assert summary.by_amr_mpi() is not None
    table = sim.amr.patch_table()
    assert table.built is True and table.base_n == n


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
