"""Spec 5 (sec.8.12 / sec.8.4, criterion #34): the AMR runtime inspection surface.

Exercises ``AmrSystem.amr`` -- the live, INERT inspection handle
(:class:`pops.runtime.amr.AmrRuntimeView`) -- and its reports: ``patch_table()`` /
``hierarchy_snapshot()`` / ``explain_regrid()`` / ``explain_ghosts()`` / ``explain_reflux()`` /
``explain_checkpoint()``. The host-runnable parts build a SMALL real ``AmrSystem`` (Kokkos-Serial
on this Mac), add one native block, refine on a density bump and take a few steps so a real fine
patch forms, then read the reports off the LIVE runtime. A full multi-step regrid campaign / MPI
distribution / GPU run is Kokkos/ROMEO-gated and not asserted here.

Honesty: a measure the native build cannot answer (per-level ghost depth, per-stage reflux timing)
is asserted to be reported as UNAVAILABLE, never a fabricated number. The reports are deterministic
and array-free (sec.12.1).
"""
import operator
import sys

import numpy as np
import pytest
from pops.runtime.system import AmrSystem, System  # ADC-545 advanced runtime seam
from tests.python.support.layout_plan import resolved_layout_contract

pops = pytest.importorskip("pops")

from pops.runtime.amr import (  # noqa: E402
    AmrRuntimeView, PatchReport, RegridReport, GhostReport, RefluxReport, CheckpointReport,
    HierarchySnapshot, RuntimeInspection)


def _model():
    """A minimal single-scalar ExB block model (no DSL compile; native bricks)."""
    return pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                      source=pops.NoSource(), elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))


def _built_amr(regrid_every=2, n=32):
    """A small built AmrSystem with one refined patch (density bump + a few steps)."""
    sim = AmrSystem(n=n, L=1.0, periodic=True, regrid_every=regrid_every, coarse_max_grid=16)
    sim.block("ne", model=_model(), spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    sim.set_refinement(threshold=0.5)
    ne = np.ones((n, n))
    ne[n // 3:2 * n // 3, n // 3:2 * n // 3] = 5.0
    sim.set_density("ne", ne)
    for _ in range(3):
        sim.step_cfl(0.4)
    return sim


# --- the handle ----------------------------------------------------------------
def test_amr_handle_is_an_inert_runtime_view():
    sim = AmrSystem(n=16, L=1.0, periodic=True)
    view = sim.amr
    assert isinstance(view, AmrRuntimeView)
    # A fresh view every access (handle, not cached state); both bound to the same system.
    assert isinstance(sim.amr, AmrRuntimeView)
    # The str is short and array-free (sec.12.1).
    text = str(view)
    assert "AmrRuntimeView" in text and "array(" not in text and len(text) < 200


def test_system_has_no_amr_handle_with_a_clear_error():
    sim = System(n=16, L=1.0, periodic=True)
    assert not hasattr(sim, "amr")
    # The remedy speaks the bind vocabulary (layout=AMR on the Case), not the native engine.
    with pytest.raises(AttributeError, match=r"layout=AMR\(.*inspect\(layout\)"):
        operator.attrgetter("amr")(sim)


# --- patch_table ---------------------------------------------------------------
def test_patch_table_before_build_reports_unbuilt():
    sim = AmrSystem(n=16, L=1.0, periodic=True)
    rep = sim.amr.patch_table()
    assert isinstance(rep, PatchReport)
    assert rep.built is False
    assert rep.n_patches == 0
    assert "not built" in str(rep)


def test_patch_table_on_built_hierarchy_reports_live_patches():
    sim = _built_amr()
    rep = sim.amr.patch_table()
    assert rep.built is True
    assert rep.n_levels == 2
    assert rep.base_n == 32
    # A real fine patch formed on the bump (live runtime, not config).
    assert rep.n_patches >= 1
    levels = {lvl["level"]: lvl for lvl in rep.per_level}
    assert 0 in levels and levels[0]["level"] == 0          # base box reported
    assert levels[1]["n_patches"] == rep.n_patches          # the fine patches live on level 1
    assert levels[1]["cells"] > 0
    # Coarse box distribution comes from the live MPI diagnostic accessors.
    assert rep.coarse_local_boxes == 1 and rep.coarse_total_boxes == 1
    assert rep.coarse_is_distributed is False
    # Printable, deterministic, array-free.
    text = str(rep)
    assert text.startswith("AMR patch table") and "array(" not in text
    assert str(sim.amr.patch_table()) == text
    # to_dict round-trips the same numbers.
    d = rep.to_dict()
    assert d["n_patches"] == rep.n_patches and d["n_levels"] == 2


# --- hierarchy_snapshot --------------------------------------------------------
def test_hierarchy_snapshot_composes_config_envelope_and_live_patches():
    sim = _built_amr(regrid_every=2)
    snap = sim.amr.hierarchy_snapshot()
    assert isinstance(snap, HierarchySnapshot)
    # Config envelope comes from the native descriptor-free capability facts.
    assert snap.max_levels == "resource_policy" and snap.ratio == 2
    assert snap.config_available == "yes"
    assert any("resource-policy" in note for note in snap.limitations)
    # Live parts: the block registry + the patch table.
    assert snap.blocks == ["ne"]
    assert snap.frozen is False and snap.regrid_every == 2
    assert snap.patch_table.built is True and snap.patch_table.n_patches >= 1
    text = str(snap)
    assert text.startswith("AMR hierarchy snapshot") and "array(" not in text
    assert str(sim.amr.hierarchy_snapshot()) == text


# --- explain_regrid ------------------------------------------------------------
def test_explain_regrid_dynamic_vs_frozen():
    dyn = AmrSystem(n=16, L=1.0, periodic=True, regrid_every=4).amr.explain_regrid()
    assert isinstance(dyn, RegridReport)
    assert dyn.frozen is False and dyn.regrid_every == 4
    # The union-of-tags criteria are named (config-sourced shape, not a fabricated threshold).
    blob = " ".join(dyn.criteria)
    # The criteria are described in the Case vocabulary (AMR(refine=Refine.on(...))).
    assert "AMR(refine=Refine.on" in blob and "grad phi" in blob

    frozen = AmrSystem(n=16, L=1.0, periodic=True, regrid_every=0).amr.explain_regrid()
    assert frozen.frozen is True and frozen.regrid_every == 0
    assert any("frozen" in n for n in frozen.notes)


# --- explain_ghosts (honest deferral) -----------------------------------------
def test_explain_ghosts_defers_per_level_depth_honestly():
    rep = AmrSystem(n=16, L=1.0, periodic=True).amr.explain_ghosts()
    assert isinstance(rep, GhostReport)
    # Per-level ghost depth is NOT fabricated: it is None and rendered as unavailable.
    assert rep.per_level_depth is None
    assert "unavailable" in str(rep)
    # The requirement shape (stencil -> ghost depth) is still explained.
    assert "weno5" in rep.requirement_note


# --- explain_reflux ------------------------------------------------------------
def test_explain_reflux_reports_route_requirement():
    rep = AmrSystem(n=16, L=1.0, periodic=True).amr.explain_reflux()
    assert isinstance(rep, RefluxReport)
    assert rep.enabled is True
    # The per-stage timing is honestly unavailable (route property, not a counter).
    assert rep.per_stage is None
    assert "unavailable" in str(rep)


# --- explain_checkpoint --------------------------------------------------------
def test_explain_checkpoint_restartable_for_frozen_single_block():
    sim = AmrSystem(n=16, L=1.0, periodic=True, regrid_every=0)
    sim.block("ne", model=_model())
    rep = sim.amr.explain_checkpoint()
    assert isinstance(rep, CheckpointReport)
    assert rep.restartable is True and rep.violations == []
    assert "bit-identical v3" in str(rep)


def test_explain_checkpoint_supports_dynamic_regrid():
    sim = AmrSystem(n=16, L=1.0, periodic=True, regrid_every=3)
    sim.block("ne", model=_model())
    rep = sim.amr.explain_checkpoint()
    assert rep.restartable is True and rep.violations == []
    assert any("active regridding is supported" in n for n in rep.notes)


# --- inspect() (ADC-589/555 criterion #34: the unified hierarchy/patch/regrid/limitations view) --
def test_inspect_returns_unified_runtime_inspection():
    sim = _built_amr(regrid_every=2)
    report = sim.amr.inspect()
    assert isinstance(report, RuntimeInspection)
    # The four parts 589 asks for, each the SAME report class the standalone methods return.
    assert isinstance(report.hierarchy, HierarchySnapshot)
    assert isinstance(report.patches, PatchReport)
    assert isinstance(report.regrid, RegridReport)
    assert isinstance(report.limitations, list)

    # Consistent with the standalone reports read off the same live system.
    assert report.hierarchy.blocks == sim.amr.hierarchy_snapshot().blocks
    assert report.patches.n_patches == sim.amr.patch_table().n_patches
    assert report.regrid.regrid_every == sim.amr.explain_regrid().regrid_every

    payload = report.to_dict()
    assert set(payload) == {"hierarchy", "patches", "regrid", "limitations"}
    assert payload["hierarchy"]["patch_table"]["n_patches"] == payload["patches"]["n_patches"]
    assert payload["regrid"]["regrid_every"] == 2
    # limitations rows carry a feature/status/reason shape; only non-available rows are listed.
    for row in payload["limitations"]:
        assert row["status"] != "available"
        assert "feature" in row and "reason" in row

    text = str(report)
    assert "AMR runtime inspection" in text and "array(" not in text


def test_inspect_before_build_reports_unbuilt_patches_honestly():
    sim = AmrSystem(n=16, L=1.0, periodic=True, regrid_every=0)
    report = sim.amr.inspect()
    assert report.patches.built is False
    assert report.hierarchy.patch_table.built is False
    assert report.regrid.frozen is True


# --- compiled static delegation ------------------------------------------------
def test_compiled_model_has_no_competing_layout_inspector():
    # A tiny stub CompiledModel (no .so dlopen needed) exposes no retired AMR-specific inspector.
    from pops.codegen.loader import CompiledModel
    cm = CompiledModel(
        so_path="<stub>", backend="aot", adder="add_native_block", cons_names=["rho"],
        cons_roles=["Density"], prim_names=["rho"], n_vars=1, gamma=None, n_aux=0, params={},
        caps={}, abi_key="k", model_hash="h", cxx="c++", std="23", target="amr_system")
    assert not hasattr(cm, "inspect_amr")


def test_compiled_artifact_exposes_its_layout_to_the_generic_inspector():
    # The artifact retains the exact resolved layout; the sole public inspector reports its tags.
    from pops.codegen.loader import CompiledModel
    from pops.codegen._plans import ResolvedBlock, ResolvedSimulationPlan
    from pops.codegen.compiled_artifact import CompiledBlockArtifact, CompiledSimulationArtifact
    from pops.mesh import CartesianMesh
    from pops.mesh.amr import Refine, RegridEvery
    from pops.mesh.layouts import AMR
    from pops.model import Handle, OwnerPath
    from pops.model.bind_schema import BindSchema
    from pops.problem._snapshot import AuthoringSnapshot

    cm = CompiledModel(
        so_path="<stub>", backend="aot", adder="add_native_block", cons_names=["rho"],
        cons_roles=["Density"], prim_names=["rho"], n_vars=1, gamma=None, n_aux=0, params={},
        caps={}, abi_key="k", model_hash="h", cxx="c++", std="23", target="amr_system")
    from pops.codegen._compiled_model_identity import compiled_model_identity
    cm.definition_identity = compiled_model_identity(model_hash="h")
    rho = Handle("rho", kind="state", owner=OwnerPath.shared("amr-runtime-inspect"))
    carried = AMR(base=CartesianMesh(n=64), regrid=RegridEvery(4),
                  refine=Refine.on(rho).above(0.1))
    snapshot = AuthoringSnapshot({"kind": "amr-runtime-inspect-stub"})
    schema = BindSchema()
    layout_plan, layout_coverage = resolved_layout_contract(
        carried, target="amr_system", block_names=("ne",))
    resolved = ResolvedSimulationPlan(
        snapshot=snapshot,
        target="amr_system",
        backend="aot",
        layout=carried,
        layout_plan=layout_plan,
        time=None,
        blocks=(ResolvedBlock("ne", {"kind": "amr-runtime-inspect-stub"}, None, "aot"),),
        bind_schema=schema,
        compile_values=schema.resolve_compile(),
        field_solvers={},
        outputs=(),
        diagnostics=(),
        libraries=(),
        requirements={"amr": True},
        capabilities={"amr": True},
        lowering_coverage=layout_coverage,
    )
    artifact = CompiledSimulationArtifact(
        plan=resolved,
        program=None,
        blocks=(CompiledBlockArtifact("ne", cm, None),),
    )

    assert not hasattr(artifact, "inspect_amr")
    payload = pops.inspect(artifact.layout)["amr_report"]
    assert payload["layout"] == "amr"
    slots = {row["slot"] for row in payload["policies"]}
    assert "refine" in slots and "regrid" in slots

    # A separately authored layout is inspected directly, with no artifact-specific override API.
    override = pops.inspect(AMR(base=CartesianMesh(n=32)))["amr_report"]
    assert override["max_levels"] == 2 and "policies" in override
    assert {row["slot"] for row in override["policies"]} == set()


# The CI python runner invokes each test file as `python3 <file>`; run pytest on this
# module so the assertions execute (a bare import would only define the test functions).
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
