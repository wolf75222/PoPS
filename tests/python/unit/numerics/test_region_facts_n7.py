"""N7 lot 1 (spec 0.3 IR14, IR17, IR20, IR21, IR25, IR-T1 first part): region facts of the plans.

The face kernels of the resolved plans are regions over typed maps: the finite-volume methods of
M01 and M03, whose native catalogue declares its stencil (IR24), and the stored face expressions
of M05 and M10, whose reads are derived from the rows of each axis, at the top level of the
program or in a nested block. Every region reaches the class its builder expects, face incidence,
and ``pops.compile`` refuses one that does not with ``ClassNotReached`` before any native work.
The path-conservative method of M24 and the coordinated method of M07 deliver their side terms
as single-target productions beside the shared flux. The exact halos come from box arithmetic;
the b6 widths, which IR-T1 compares, are the stored ``ghost_depth`` and ``ghost_extent``. Level
E of K6: resolution only, no compilation.
"""
import pytest

import pops
import examples.api_v02.models_M01_M14 as cases
from examples.api_v02.faceflux_pilot import navier_stokes_scheme
from examples.api_v02.models_M22_M25_M28 import build_m24
from pops._ir.index_analysis import Read, Region, Truth
from pops._ir.index_maps import DataMap
from pops._ir.refusals import ClassNotReached
from pops.codegen import region_facts
from pops.lib.numerics import realize_face_flux
from pops.numerics import DiscretizationPlan, StateStorage, reconstruction, riemann, variables
from pops.numerics.nonconservative import CoordinatedFiniteVolume, PathConservativeFiniteVolume
from pops.numerics.spatial import FiniteVolume
from pops.regions import Incidence, cell, face
from pops.time import FixedDt


def _plan(build):
    spec = build()
    return pops.resolve(pops.validate(spec.case), layout=spec.layout)


PLANS = {name: _plan(build) for name, build in (
    ("M01", cases.build_m01), ("M03", cases.build_m03), ("M05", cases.build_m05),
    ("M10", cases.build_m10), ("M07", cases.build_m07), ("M24", build_m24))}


def _over_axes(regions, where=None):
    low, high = [0, 0], [0, 0]
    for region in regions:
        fact = region.halo(where)
        assert fact.truth is Truth.PROVED
        for axis in range(2):
            low[axis] = max(low[axis], fact.value[0][axis])
            high[axis] = max(high[axis], fact.value[1][axis])
    return tuple(low), tuple(high)


@pytest.mark.parametrize("name,basis,kernels", (
    ("M01", "declared", ("rate conservation",)), ("M03", "declared", ("rate Euler_balance",)),
    ("M05", "derived", ("program NS_stage0", "program NS_stage1")),
    ("M10", "derived", ("program stage0_fitted_face", "program stage1_fitted_face"))))
def test_every_face_kernel_reaches_its_expected_class(name, basis, kernels):
    rows = region_facts.check_expected_classes(PLANS[name])
    # One region per axis and per kernel: the rate of M01 and M03, the two SSPRK2 stages of the
    # stored face expression of M05 and M10.
    assert [row[0] for row in rows] == ["block fluid/%s/axis %d" % (kernel, axis)
                                        for kernel in kernels for axis in range(2)]
    assert {(expected, reached, kind) for _, expected, reached, kind in rows} == {
        ("face_incidence", "face_incidence", basis)}


@pytest.mark.parametrize("name", ("M01", "M03", "M05", "M10", "M07", "M24"))
def test_face_deliveries_are_conservative(name):
    for region in region_facts.plan_regions(PLANS[name]):
        assert region.conservation().truth is Truth.PROVED
        (left, wl), (right, wr) = region.targets
        axis = region.domain.axis
        assert (left, wl, right, wr) == (Incidence(face(axis), cell, "left"), -1,
                                         Incidence(face(axis), cell, "right"), 1)


@pytest.mark.parametrize("name,block,method", (
    ("M24", "mixture", PathConservativeFiniteVolume), ("M07", "fluid", CoordinatedFiniteVolume)))
def test_path_conservative_side_terms_are_single_target_productions(name, block, method):
    plan = PLANS[name]
    (rate,) = [row for row in plan.blocks[0].numerics.rates if type(row.method) is method]
    regions = region_facts.plan_regions(plan)
    assert [region.name for region in regions] == [
        "block %s/rate %s/axis %d" % (block, rate.rate.local_id, axis) for axis in range(2)]
    for axis, region in enumerate(regions):
        # The left term goes to the left cell and the right term to the right cell, each with
        # weight +1 beside the shared flux (path_flux.hpp, MaterializePathResidual): conservation
        # stays proved on the flux alone, and the class and the halo are those of the flux.
        assert region.productions == ((Incidence(face(axis), cell, "left"), 1),
                                      (Incidence(face(axis), cell, "right"), 1))
        assert region.conservation().value == 0
        assert region.check_expected_class() == "face_incidence"
        assert region.halo().value == tuple(tuple(int(d == axis) for d in range(2))
                                            for _ in range(2))
    # A conservative finite volume has no side term.
    assert {region.productions for name in ("M01", "M03")
            for region in region_facts.plan_regions(PLANS[name])} == {()}


@pytest.mark.parametrize("name", ("M01", "M03"))
def test_finite_volume_halo_equals_the_b6_ghost_depth(name):
    plan = PLANS[name]
    method = plan.blocks[0].numerics.rates[0].method
    regions = region_facts.plan_regions(plan)
    # First order: the left and the right cell of each face, one cell beyond each side.
    assert _over_axes(regions) == ((1, 1), (1, 1))
    assert max(max(side) for side in _over_axes(regions)) == method.ghost_depth == 1


def test_m05_exact_halo_is_tighter_than_the_b6_width_it_stores():
    plan = PLANS["M05"]
    regions = region_facts.plan_regions(plan)
    flux = cases.navier_stokes_face()
    assert _over_axes(regions) == ((1, 1), (1, 1))
    assert flux.ghost_extent == (2, 2)
    # IR-T1 compares the stored value: the b6 face formula over the derived reads gives it back.
    offsets = {read.map.offset(2) for region in regions for read in region.reads}
    b6 = tuple(max(max(-o[d], o[d] + 1) for o in offsets) for d in range(2))
    assert b6 == flux.ghost_extent
    # The reads are those of each axis: the faces normal to x read no cell at y offset 2.
    by_axis = [{read.map.offset(2) for read in region.reads} for region in regions[:2]]
    assert by_axis[0] == {(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1)}
    assert by_axis[1] == {(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0)}
    assert by_axis[0] | by_axis[1] == {offset for offset, _, _ in flux.samples}


def test_m10_state_and_field_halos_equal_b6():
    plan = PLANS["M10"]
    regions = region_facts.plan_regions(plan)
    flux = cases.scharfetter_gummel_face()
    assert _over_axes(regions, lambda array: array[0] == "state") == ((1, 1), (1, 1))
    assert _over_axes(regions, lambda array: array[0] == "field") == ((1, 1), (1, 1))
    assert flux.ghost_extent == (1, 1) and flux.field_ghost_extents == ((1, 1),)


def test_pilot_builder_reports_the_core_facts():
    frame, _ = cases._domain("M05_NS", (16, 8))
    model = pops.Model("M05_pilot", frame=frame)
    U = model.state("conservative", components=("rho", "mx", "my", "E"))
    realized = realize_face_flux(navier_stokes_scheme(U), U, frame=model.frame)
    assert realized.halo() == ((1, 1), (2, 2))          # the b6 bound
    assert realized.exact_halo() == ((1, 1), (1, 1))    # box arithmetic
    assert [region.check_expected_class() for region in realized.regions()] == [
        "face_incidence", "face_incidence"]


def _brute_force_reconstruction_halo(offsets, n=7):
    # The left trace of face i reads the cells i - 1 + o, the right trace i - o (mirrored).
    cells = {i - 1 + o for i in range(n + 1) for o in offsets} | {
        i - o for i in range(n + 1) for o in offsets}
    return max(0, -min(cells)), max(0, max(cells) - (n - 1))


@pytest.mark.parametrize("body,offsets", (
    (lambda sample: sample(0) + .5*(sample(1) - sample(0)), (0, 1)),
    (lambda sample: sample(-3) + sample(2), (-3, 2)),
    (lambda sample: sample(0) + .25*(sample(1) - sample(-1)), (-1, 0, 1)),
))
def test_a_source_reconstruction_has_derived_reads(body, offsets):
    frame, _ = cases._domain("user_reconstruction", (8, 8))
    model = pops.Model("user_reconstruction", frame=frame)
    U = model.state("amount", components=("u",))
    flux = model.flux("transport", state=U, frame=frame,
                      components={axis: (a*U[0],)
                                  for axis, a in zip(frame.axes, (.7, .3), strict=True)},
                      waves={axis: (a,) for axis, a in zip(frame.axes, (.7, .3), strict=True)})
    method = FiniteVolume(flux=flux, variables=variables.Conservative(U),
                          reconstruction=reconstruction.User(body, formal_order=1),
                          riemann=riemann.Rusanov())
    regions = method.regions("probe", 2)
    assert {region.basis for region in regions} == {"derived"}
    assert method.reconstruction.options["sample_offsets"] == offsets
    low, high = _over_axes(regions)
    assert (low[0], high[0]) == _brute_force_reconstruction_halo(offsets)
    # b6 stores max(1, 1 - min, max + 1), which bounds the exact widths.
    assert method.ghost_depth == max(1, 1 - min(offsets), max(offsets) + 1) >= max(low + high)


def test_compile_refuses_a_class_not_reached_before_native_work(monkeypatch):
    plan = PLANS["M05"]
    graph = DataMap("graph", source=face(0), target=cell)
    blocked = Region("block fluid/face flux/axis 0", face(0), 2,
                     (Read(("state", 0), Incidence(face(0), cell, "left")),
                      Read(("state", 0), graph)),
                     ((Incidence(face(0), cell, "left"), -1), (Incidence(face(0), cell, "right"), 1)),
                     expected="face_incidence")
    monkeypatch.setattr(region_facts, "plan_regions", lambda plan: (blocked,))

    def native_work(*_):
        raise AssertionError("compile reached the native cut line before classifying")
    monkeypatch.setattr("pops._native_selector.select_native_dimension", native_work)
    with pytest.raises(ClassNotReached, match="block fluid/face flux/axis 0") as refusal:
        pops.compile(plan)
    assert (refusal.value.reached, refusal.value.blocking) == ("csr", graph)


def _nested_plan(loop):
    # A stored face expression evaluated only inside the body of a loop of the program.
    frame, layout = cases._domain("M05_nested_%s" % loop, (16, 8))
    model = pops.Model("M05_nested_%s" % loop, frame=frame)
    U = model.state("conservative", components=("rho", "mx", "my", "E"))
    case = pops.Case("M05_nested_%s_case" % loop)
    block = case.block("fluid", model)
    flux = cases.navier_stokes_face()
    numerics = DiscretizationPlan()
    numerics.states.add(U, StateStorage(ghost_depth=max(flux.ghost_extent)))
    case.numerics(numerics, block=block)
    program = pops.Program("nested_%s" % loop,
                           history_field_metadata_budget=(8*1024*1024, 16*1024*1024))
    temporal = program.state(block[U])

    def body(p, x):
        k = p.stencil_flux_divergence("NS_in_loop", x, flux=flux)
        return p.value("loop_update", x + p.dt*k, at=temporal.next.point)
    if loop == "range":
        after = program.range(temporal.n, 2, body)
    else:
        after = program.while_(temporal.n, lambda p, x: p.norm2(x) > 1e30, body)
    program.commit(temporal.next,
                   program.value("accepted_after_loop", 1.0*after, at=temporal.next.point))
    program.step_strategy(FixedDt(1e-4))
    case.program(program)
    return pops.resolve(pops.validate(case), layout=layout)


@pytest.mark.parametrize("loop", ("range", "while"))
def test_a_face_kernel_in_a_nested_block_is_classified(loop):
    plan = _nested_plan(loop)
    assert not any("stencil_flux" in value.attrs for value in plan.time._values)
    rows = region_facts.check_expected_classes(plan)
    assert [row[0] for row in rows] == ["block fluid/program NS_in_loop/axis %d" % axis
                                        for axis in range(2)]
    assert {row[1:] for row in rows} == {("face_incidence", "face_incidence", "derived")}
