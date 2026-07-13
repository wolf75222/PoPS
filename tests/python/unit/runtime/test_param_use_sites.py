"""ADC-654: parameter kind/phase is validated before structural coercion."""
from __future__ import annotations

from types import SimpleNamespace

import pytest


pops = pytest.importorskip("pops")

from pops.codegen._backends import lower_backend  # noqa: E402
from pops.math import Bool, Integer, Real  # noqa: E402
from pops.mesh import CartesianMesh, PolarMesh  # noqa: E402
from pops.mesh.amr import Refine, RegridEvery  # noqa: E402
from pops.mesh.geometry import Disc, DiscDomain, HalfPlane  # noqa: E402
from pops.model import Handle, OwnerPath, ParamHandle  # noqa: E402
from pops.numerics.reconstruction import (  # noqa: E402
    required_ghost_depth,
    validate_ghost_depth,
)
from pops.params import (  # noqa: E402
    ConstParam,
    DerivedParam,
    InvalidParamUseSite,
    PARAM_USE_MATRIX,
    ParamInvalidation,
    ParamPhase,
    ParamStorage,
    ParamUse,
    RuntimeParam,
    resolve_param_use,
)
from pops.runtime_environment import validate_dimension, validate_precision  # noqa: E402
from pops.time import AcceptedStep, Clock, Every  # noqa: E402
from pops.time.history_persistence import (  # noqa: E402
    Dense,
    Interval as HistoryInterval,
    Revolve,
)


class _ResolvedDerived:
    """The small resolved-plan protocol consumed by use_sites.py."""

    param_kind = "derived"

    def __init__(self, name, value, phase):
        self.name = name
        self.resolved_value = value
        self.phase = phase


def _runtime(name="dynamic"):
    return RuntimeParam(name, dtype=Integer)


def test_matrix_is_closed_and_runtime_is_rejected_by_every_structural_use():
    assert set(PARAM_USE_MATRIX) == set(ParamUse)
    for use, row in PARAM_USE_MATRIX.items():
        assert set(row) == {"runtime", "const", "derived"}
        expected = "preserve" if use is ParamUse.RUNTIME_VALUE else "reject"
        assert row["runtime"] == expected


@pytest.mark.parametrize(
    ("build", "use"),
    [
        (lambda p: CartesianMesh(n=p), ParamUse.SHAPE),
        (lambda p: CartesianMesh(L=p), ParamUse.MESH_EXTENT),
        (lambda p: CartesianMesh(periodic=p), ParamUse.MESH_TOPOLOGY),
        (lambda p: RegridEvery(p), ParamUse.REGRID_SCHEDULE),
        (lambda p: Every(AcceptedStep(Clock("macro")), p), ParamUse.SCHEDULE),
        (lambda p: validate_ghost_depth("weno5", available=p), ParamUse.GHOST_DEPTH),
        (lambda p: validate_dimension(p), ParamUse.ABI),
        (lambda p: lower_backend(p), ParamUse.BACKEND),
        (lambda p: PolarMesh(p, 1.0, 8, 16), ParamUse.MESH_EXTENT),
        (lambda p: PolarMesh(0.1, p, 8, 16), ParamUse.MESH_EXTENT),
        (lambda p: PolarMesh(0.1, 1.0, p, 16), ParamUse.SHAPE),
        (lambda p: PolarMesh(0.1, 1.0, 8, p), ParamUse.SHAPE),
        (lambda p: PolarMesh(0.1, 1.0, 8, 16, theta_boxes=p),
         ParamUse.MESH_TOPOLOGY),
        (lambda p: Disc(radius=p), ParamUse.MESH_EXTENT),
        (lambda p: Disc(center=(p, 0.0)), ParamUse.MESH_EXTENT),
        (lambda p: HalfPlane(point=(p, 0.0)), ParamUse.MESH_EXTENT),
        (lambda p: HalfPlane(normal=(p, 0.0)), ParamUse.MESH_EXTENT),
        (lambda p: DiscDomain(radius=p), ParamUse.MESH_EXTENT),
        (lambda p: DiscDomain(center=(p, 0.0)), ParamUse.MESH_EXTENT),
        (lambda p: HistoryInterval(p), ParamUse.SCHEDULE),
        (lambda p: Revolve(p), ParamUse.SHAPE),
        (lambda p: Dense().stored_slots(p), ParamUse.SHAPE),
        (lambda p: Dense().recomputed_slots(p), ParamUse.SHAPE),
    ],
)
def test_runtime_param_is_rejected_before_structural_coercion(build, use):
    with pytest.raises(InvalidParamUseSite) as caught:
        build(_runtime())
    assert caught.value.param_kind == "runtime"
    assert caught.value.use is use
    assert "compile-structural" in str(caught.value)


def test_runtime_precision_choice_is_rejected_before_string_coercion():
    with pytest.raises(InvalidParamUseSite, match="compile-structural") as caught:
        validate_precision(RuntimeParam("precision", dtype=Real))
    assert caught.value.use is ParamUse.ABI


def test_const_params_are_explicitly_unwrapped_at_structural_sites():
    mesh = CartesianMesh(
        n=ConstParam("n", 32, dtype=Integer),
        L=ConstParam("L", 2.5, dtype=Real),
        periodic=ConstParam("periodic", False, dtype=Bool),
        dim=ConstParam("dim", 2, dtype=Integer),
    )
    assert (mesh.n, mesh.L, mesh.periodic, mesh.dim) == (32, 2.5, False, 2)

    trigger = Every(
        AcceptedStep(Clock("macro")), ConstParam("cadence", 7, dtype=Integer))
    assert trigger.n == 7
    assert validate_ghost_depth(
        "weno5", available=ConstParam("ghosts", 3, dtype=Integer)) is True

    polar = PolarMesh(
        r_min=ConstParam("r_min", 0.1, dtype=Real),
        r_max=ConstParam("r_max", 1.0, dtype=Real),
        nr=ConstParam("nr", 8, dtype=Integer),
        ntheta=ConstParam("ntheta", 16, dtype=Integer),
        theta_boxes=ConstParam("theta_boxes", 4, dtype=Integer),
    )
    assert (polar.r_min, polar.r_max, polar.nr, polar.ntheta, polar.theta_boxes) == (
        0.1, 1.0, 8, 16, 4)

    center = (
        ConstParam("center_x", 0.25, dtype=Real),
        ConstParam("center_y", 0.75, dtype=Real),
    )
    assert Disc(center=center, radius=ConstParam("disc_radius", 0.4)).options() == {
        "center": (0.25, 0.75), "radius": 0.4}
    assert DiscDomain(
        center=center, radius=ConstParam("domain_radius", 0.3)).lower() == (
            0.25, 0.75, 0.3, "none")
    assert HalfPlane(
        point=center,
        normal=(ConstParam("normal_x", 1.0), ConstParam("normal_y", 0.0)),
    ).options() == {"point": (0.25, 0.75), "normal": (1.0, 0.0)}

    assert HistoryInterval(ConstParam("interval", 2, dtype=Integer)).k == 2
    assert Revolve(ConstParam("snapshots", 3, dtype=Integer)).snapshots == 3
    depth = ConstParam("history_depth", 5, dtype=Integer)
    assert Dense().stored_slots(depth) == (0, 1, 2, 3, 4)
    assert Dense().recomputed_slots(depth) == ()


def test_runtime_stencil_width_is_rejected_instead_of_becoming_unknown():
    descriptor = SimpleNamespace(
        options={"ghost_depth": RuntimeParam("stencil_width", dtype=Integer)},
        scheme="external",
    )
    with pytest.raises(InvalidParamUseSite) as caught:
        required_ghost_depth(descriptor)
    assert caught.value.use is ParamUse.STENCIL


def test_derived_phase_must_be_compile_for_structural_use():
    late = _ResolvedDerived("n", 64, "runtime")
    with pytest.raises(InvalidParamUseSite, match="requires phase=compile") as caught:
        CartesianMesh(n=late)
    assert caught.value.param_kind == "derived"
    assert caught.value.phase == "runtime"

    early = _ResolvedDerived("n", 64, "compile")
    assert CartesianMesh(n=early).n == 64


def test_canonical_compile_derived_is_resolved_for_structural_use():
    module = pops.model.Module("derived-structural-use")
    base = module.param(ConstParam("base", 4, dtype=Integer))
    cells = DerivedParam(
        "cells",
        module.value(base) * 2,
        depends_on=(base,),
        phase=ParamPhase.Compile,
        storage=ParamStorage.Inline,
        invalidation=ParamInvalidation.Never,
        dtype=Integer,
    )
    module.param(cells)

    assert cells.resolved_value == 8
    assert CartesianMesh(n=cells).n == 8


def test_compile_derived_values_resolve_at_new_structural_boundaries():
    polar = PolarMesh(
        _ResolvedDerived("r_min", 0.1, "compile"),
        _ResolvedDerived("r_max", 1.0, "compile"),
        _ResolvedDerived("nr", 8, "compile"),
        _ResolvedDerived("ntheta", 16, "compile"),
        theta_boxes=_ResolvedDerived("theta_boxes", 4, "compile"),
    )
    assert polar.options() == {
        "r_min": 0.1, "r_max": 1.0, "nr": 8, "ntheta": 16, "theta_boxes": 4}

    disc = Disc(
        center=(_ResolvedDerived("cx", 0.25, "compile"), 0.75),
        radius=_ResolvedDerived("radius", 0.4, "compile"),
    )
    assert disc.options() == {"center": (0.25, 0.75), "radius": 0.4}

    interval = HistoryInterval(_ResolvedDerived("interval", 2, "compile"))
    assert interval.stored_slots(_ResolvedDerived("depth", 5, "compile")) == (0, 2, 4)
    assert Revolve(_ResolvedDerived("snapshots", 3, "compile")).stored_slots(5) == (
        0, 2, 4)


@pytest.mark.parametrize("param_kind", ["runtime", "const", "derived"])
def test_param_handle_never_falls_through_to_python_numeric_coercion(param_kind):
    handle = ParamHandle(
        "n", owner=OwnerPath.shared("param-use-sites"), param_kind=param_kind)
    with pytest.raises(InvalidParamUseSite) as caught:
        CartesianMesh(n=handle)
    assert caught.value.param_kind == param_kind
    assert caught.value.use is ParamUse.SHAPE
    if param_kind == "runtime":
        assert "compile-structural" in str(caught.value)
    else:
        assert "ParamRegistry or resolved plan" in str(caught.value)


def test_runtime_tag_threshold_keeps_its_storage_class():
    field = Handle("rho", kind="state", owner=OwnerPath.shared("param-use-sites"))
    threshold = RuntimeParam("tag_threshold", dtype=Real, default=0.1)
    criterion = Refine.on(field).above(threshold)
    assert criterion.threshold is threshold


def test_resolver_passes_non_params_and_reports_bad_use_tokens():
    marker = object()
    assert resolve_param_use(marker, ParamUse.SHAPE, where="test") is marker
    with pytest.raises(TypeError, match="ParamUse"):
        resolve_param_use(1, "not-a-use", where="test")
