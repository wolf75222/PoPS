#!/usr/bin/env python3
"""ADC-515: ``CompiledModel.arguments()`` / ``estimate_memory()`` -- the AMR-route seam.

``pops.compile(problem, layout=AMR(...))`` returns the first block's :class:`CompiledModel` (the AMR
route lowers per-block native ``add_native_block`` loaders; there is no whole-system
``CompiledProblem`` on AMR). Before ADC-515 that handle exposed ``inspect_amr()`` /
``capability_matrix()`` but NOT ``arguments()`` / ``estimate_memory()`` (those lived only on
``CompiledProblem``), so introspecting the AMR-route artifact raised ``AttributeError``. This suite
pins the two new seams, built by the SHARED ``inspect_compiled`` builders via the model-as-handle
path (the ``CompiledModel`` IS its own physical model):

  * ``arguments()`` reports the block instance / params / named aux and the runtime layout, with
    ``layout == "amr"`` driven by the handle's ``target='amr_system'``;
  * ``estimate_memory(mesh)`` is a pure FORMULA (state / aux / halo + AMR patch budget) that defaults
    the AMR layout from the handle's carried ``InstallPlan`` and NEVER touches Program-only scratch (the
    no-Program branch is guarded), so the seam matches the ``CompiledProblem`` counterpart on the
    Uniform route.

Pure metadata + formula: no compile, bind, dlopen or allocation, so it runs on a bare box (the
stub ``CompiledModel`` needs no ``.so``). ``importorskip("pops")`` guards the imports; ``__main__``
runs pytest.
"""
import sys

import pytest

pops = pytest.importorskip("pops", exc_type=ImportError)

from pops.codegen.loader import CompiledModel  # noqa: E402
from pops.codegen._plans import InstallBlock, InstallPlan  # noqa: E402
from pops.mesh import CartesianMesh  # noqa: E402
from pops.mesh.layouts import AMR, Uniform  # noqa: E402
from pops.params import ConstParam, RuntimeParam  # noqa: E402
from pops.problem._snapshot import AuthoringSnapshot  # noqa: E402


def _amr_handle(*, n_aux=2, mpi=True, runtime_param=True):
    """A stub AMR-route ``CompiledModel`` (target='amr_system', no ``.so``): the handle the AMR route
    returns. Carries three conservative components, named aux, a runtime + a const param, and the
    ``caps`` the ``pops.compile`` AMR route produces. The layout lives only in InstallPlan."""
    params = {}
    if runtime_param:
        params["alpha"] = RuntimeParam("alpha", default=1.0)
    params["gamma"] = ConstParam("gamma", 1.4)
    aux = ["B_z", "phi_bg"][:n_aux]
    handle = CompiledModel(
        so_path="<stub-amr>", backend="production", adder="add_native_block",
        cons_names=["rho", "mx", "my"], cons_roles=["Density", "MomentumX", "MomentumY"],
        prim_names=["rho", "mx", "my"], n_vars=3, gamma=1.4, n_aux=n_aux, params=params,
        caps={"cpu": True, "amr": True, "mpi": mpi}, abi_key="k", model_hash="h", cxx="c++",
        std="c++23", target="amr_system", aux_extra_names=aux)
    layout = AMR(base=CartesianMesh(n=64, periodic=True), max_levels=2, ratio=2)
    snapshot = AuthoringSnapshot({"kind": "amr-route-introspection-stub"})
    handle.install_plan = InstallPlan(
        snapshot_hash=snapshot.hash,
        target="amr_system",
        layout=layout,
        blocks=(InstallBlock("block", handle, None),),
        bind_schema=None,
        field_solvers={},
        outputs=(),
        diagnostics=(),
        has_program=False,
    )
    handle._problem_snapshot = snapshot
    return handle


# --- arguments() on the AMR route ------------------------------------------------
def test_amr_handle_exposes_arguments_and_estimate_memory():
    handle = _amr_handle()
    # The two seams the AMR route was missing before ADC-515.
    assert callable(getattr(handle, "arguments", None))
    assert callable(getattr(handle, "estimate_memory", None))


def test_arguments_reports_the_amr_layout_and_the_block_instance():
    args = _amr_handle().arguments()
    lr = args.layout_runtime
    # The runtime layout is AMR (driven by target='amr_system'), MPI advertised from the caps.
    assert lr["layout"] == "amr"
    assert lr["supports_mpi"] is True
    # The block instance carries the model's conservative components (not a degenerate 1).
    assert len(args.instances) == 1
    inst = next(iter(args.instances.values()))
    assert inst["components"] == 3
    assert inst["conservative"] == ["rho", "mx", "my"]
    assert inst["required"] is True


def test_arguments_lists_the_named_aux_and_typed_params():
    args = _amr_handle().arguments()
    # Named aux from aux_extra_names (each a required cell input).
    assert set(args.aux) == {"B_z", "phi_bg"}
    assert all(spec["required"] is True and spec["layout"] == "cell"
               for spec in args.aux.values())
    # The runtime param is required at bind; the const param is frozen (not required).
    assert set(args.params) == {"alpha", "gamma"}
    assert args.params["alpha"]["kind"] == "runtime" and args.params["alpha"]["required"] is True
    assert args.params["gamma"]["kind"] == "const" and args.params["gamma"]["required"] is False


def test_arguments_supports_mpi_reflects_the_caps():
    # A handle whose model has no MPI cap reports supports_mpi False (no fabricated capability).
    args = _amr_handle(mpi=False).arguments()
    assert args.layout_runtime["supports_mpi"] is False


# --- estimate_memory() on the AMR route ------------------------------------------
def test_estimate_memory_defaults_the_amr_layout_from_the_carried_layout():
    handle = _amr_handle()
    mesh = CartesianMesh(n=64, L=1.0, periodic=True)
    # A BARE estimate_memory(mesh) auto-reports the AMR hierarchy (the handle carries the AMR
    # InstallPlan); the caller need not re-pass layout=AMR(...).
    est = handle.estimate_memory(mesh)
    assert est.layout == "amr"
    cats = est.categories
    # The no-Program branch returns POSITIVE state / aux / halo / amr_patch categories.
    assert cats["state"] > 0 and cats["aux"] > 0 and cats["halo"] > 0
    assert cats.get("amr_patch", 0) > 0
    # state = n_cons(3) x cells(64*64) x 8 bytes; aux = n_aux(2) x cells x 8 bytes.
    assert cats["state"] == 3 * 64 * 64 * 8
    assert cats["aux"] == 2 * 64 * 64 * 8
    assert est.total_bytes > 0


def test_estimate_memory_no_program_branch_skips_program_only_scratch():
    # A bare CompiledModel carries no time Program, so the Program-only solver categories are ZERO
    # (never a reference to Program scratch on a handle that has none).
    est = _amr_handle().estimate_memory(CartesianMesh(n=32, L=1.0, periodic=True))
    for program_only in ("scalar_field", "krylov", "multigrid", "field_output"):
        assert est.categories.get(program_only, 0) == 0, (
            "no-Program branch must not populate %r" % program_only)


def test_estimate_memory_amr_budget_dominates_the_uniform_budget():
    handle = _amr_handle()
    mesh = CartesianMesh(n=64, L=1.0, periodic=True)
    amr_est = handle.estimate_memory(mesh)                       # auto AMR (from _layout)
    uni_est = handle.estimate_memory(mesh, layout=Uniform(mesh))  # explicit Uniform wins
    assert uni_est.layout == "uniform"
    # The AMR hierarchy estimate adds the refined-patch budget on top of the Uniform state budget.
    assert amr_est.total_bytes >= uni_est.total_bytes
    assert "amr_patch" not in uni_est.by_scratch() or uni_est.by_scratch().get("amr_patch", 0) == 0


def test_estimate_memory_explicit_platform_adds_the_mpi_buffer():
    handle = _amr_handle()
    mesh = CartesianMesh(n=32, L=1.0, periodic=True)
    plain = handle.estimate_memory(mesh)
    with_mpi = handle.estimate_memory(mesh, platform="mpi")
    assert plain.categories.get("mpi_buffer", 0) == 0
    assert with_mpi.categories.get("mpi_buffer", 0) > 0


# --- a system-target handle is unchanged (no AMR leakage) ------------------------
def test_system_target_handle_still_reports_system_layout():
    # A target='system' bare model (no _layout) reports the Uniform layout: the AMR path is opt-in
    # via target='amr_system', never a default.
    handle = CompiledModel(
        so_path="<stub-sys>", backend="aot", adder="add_native_block", cons_names=["rho"],
        cons_roles=["Density"], prim_names=["rho"], n_vars=1, gamma=None, n_aux=0, params={},
        caps={"cpu": True}, abi_key="k", model_hash="h", cxx="c++", std="23")  # target='system'
    assert handle.arguments().layout_runtime["layout"] == "system"
    assert handle.estimate_memory(16).layout == "system"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
