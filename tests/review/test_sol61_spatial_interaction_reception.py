"""Independent source + actual consumer-body host reception; no native run."""

from pathlib import Path
import subprocess
import pytest
import pops
from pops.domain import CartesianDomain
from pops.frames import Cartesian2D
from pops.model.spaces import FieldSpace
from pops.fields import (
    SpatialInteractionKernel,
    CellVolumeMeasure,
    CellMidpoint,
    DirectSpatialInteraction,
)
from pops.codegen.program_emit_spatial_interaction import emit_spatial_interaction
from pops.time._program.spatial_interaction import interaction_contract

ROOT = Path(__file__).resolve().parents[2]


def setup(*, relabel=False, scope="issued", components=(1, 0), candidate=False):
    frame = CartesianDomain("matter", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    model = pops.Model("density", frame=frame)
    rho = model.state("rho", components=("r", "s"), sampling="cell_average")
    case = pops.Case("kernel_review")
    block = case.block("matter", model, states=(rho,))
    p = pops.Program("kernel_review_step")
    u = p.state(block[rho])
    source = p._replace_value(u.n, point=u.next.point) if relabel else u.n
    if candidate == "branch":
        leaf = source
        source = p.branch(
            p.requested_dt() > 0,
            lambda P: P.value("true_candidate", 2 * leaf, at=u.next.point),
            lambda P: P.value("false_candidate", 3 * leaf, at=u.next.point),
        )
    elif candidate:
        source = p.value("candidate", 2 * source, at=u.next.point)
    space = FieldSpace(
        "interaction",
        ("s", "r"),
        frame=source.space.frame,
        clock=source.space.clock,
        support=source.space.support,
        sampling="cell_center",
    )
    value = p.spatial_interaction(
        source,
        SpatialInteractionKernel(2, lambda x, y: 1 + x[0] - 2 * y[1]),
        output_space=space,
        measure=CellVolumeMeasure(),
        quadrature=CellMidpoint(),
        realization=DirectSpatialInteraction(100000),
        components=components,
        source_scope=scope,
    )
    return p, source, value


@pytest.mark.parametrize("scope", ["issued", "accepted"])
@pytest.mark.parametrize("target", ["system", "amr_system"])
def test_valid_literal_signed_kernel(scope, target):
    p, s, v = setup(scope=scope)
    assert interaction_contract(v)[4] == (1, 0)
    lines = []
    emit_spatial_interaction(v, {s.id: "state_n"}, lines, block_indices={v.block: 0}, target=target)
    assert "{1, 0}" in "\n".join(lines)
    assert p._serialize()["version"] == 17


@pytest.mark.parametrize("scope", ["issued", "accepted"])
def test_relabelled_n_read_is_refused(scope):
    with pytest.raises(ValueError):
        p, s, v = setup(relabel=True, scope=scope)
        interaction_contract(v)


@pytest.mark.parametrize("tree", [("x", 99), ("constant", "nan"), ("foreign",)])
def test_serialization_revalidates_kernel(tree):
    p, _, v = setup()
    p._replace_value(v, attrs={**v.attrs, "kernel": {**v.attrs["kernel"], "tree": tree}})
    with pytest.raises(ValueError):
        p._serialize()


@pytest.mark.parametrize("candidate", [True, "branch"])
@pytest.mark.parametrize("target", ["system", "amr_system"])
def test_genuine_computed_next_is_admitted(target, candidate):
    p, s, v = setup(candidate=candidate)
    assert s.op in ("linear_combine", "branch")
    assert interaction_contract(v)[0] is s
    lines = []
    emit_spatial_interaction(
        v, {s.id: "candidate"}, lines, block_indices={v.block: 0}, target=target
    )
    assert "ctx.set_stage_time(1, 1);" == lines[0]
    assert p._serialize()["version"] == 17


@pytest.mark.parametrize("candidate", [True, "branch"])
def test_wrapped_relabelled_n_read_is_refused(candidate):
    with pytest.raises(ValueError):
        p, s, v = setup(candidate=candidate, relabel=True)
        interaction_contract(v)


@pytest.mark.parametrize("candidate", [True, "branch"])
def test_changed_canonical_leaf_is_refused_at_serialization(candidate):
    p, s, v = setup(candidate=candidate)
    leaf = s.attrs["true_result"].inputs[0] if candidate == "branch" else s.inputs[0]
    p._replace_value(leaf, point=s.point)
    with pytest.raises(ValueError):
        p._serialize()


@pytest.mark.parametrize(
    "powers", [(("length", True, 1),), (("length", 0, 1),), (("length", 2, 4),)]
)
def test_serialized_units_require_exact_canonical_typed_image(powers):
    p, _, v = setup()
    w = p._replace_value(
        v,
        attrs={
            **v.attrs,
            "kernel": {
                **v.attrs["kernel"],
                "units": {"kind": "physical_dimension", "powers": powers},
            },
        },
    )
    with pytest.raises(ValueError):
        interaction_contract(w)
    with pytest.raises(ValueError):
        p._serialize()


def test_measure_cannot_import_unknown_unit_leaves():
    p, _, v = setup()
    p._replace_value(
        v, attrs={**v.attrs, "measure": {**v.attrs["measure"], "coordinate_units": (None, None)}}
    )
    with pytest.raises(ValueError):
        p._serialize()


def test_dimensioned_component_permutation_has_exact_product_units():
    from fractions import Fraction
    from pops.model import PhysicalDimension

    length = PhysicalDimension((("length", Fraction(1)),))
    density = PhysicalDimension((("mass", Fraction(1)), ("length", Fraction(-2))))
    frequency = PhysicalDimension((("time", Fraction(-1)),))
    inverse_length = PhysicalDimension((("length", Fraction(-1)),))
    expected = (
        PhysicalDimension((("time", Fraction(-1)), ("length", Fraction(1)))),
        PhysicalDimension((("mass", Fraction(1)), ("length", Fraction(-1)))),
    )
    model = pops.Model("dimensioned_density")
    rho = model.state("rho", components=("r", "s"), units=(density, frequency))
    case = pops.Case("dimensioned_review")
    block = case.block("matter", model, states=(rho,))
    p = pops.Program("dimensioned_kernel")
    source = p.state(block[rho]).n
    v = p.spatial_interaction(
        source,
        SpatialInteractionKernel(2, lambda x, y: 1, units=inverse_length),
        output_space=FieldSpace("interaction", ("s", "r"), sampling="cell_center", units=expected),
        measure=CellVolumeMeasure((length, length)),
        quadrature=CellMidpoint(),
        realization=DirectSpatialInteraction(100000),
        components=(1, 0),
    )
    assert interaction_contract(v)[4] == (1, 0)
    assert p._serialize()["version"] == 17


@pytest.mark.parametrize("real", ["double", "float"])
def test_actual_native_consumer_body_host(tmp_path, real):
    native = ROOT / "include/pops/runtime/program/spatial_direct_interaction.hpp"
    body = "\n".join(
        line
        for line in native.read_text().splitlines()
        if not line.startswith("#include") and line != "#pragma once"
    )
    reduction = (ROOT / "include/pops/mesh/execution/for_each.hpp").read_text()
    start = reduction.index("struct FiniteCompensatedSum {")
    stop = reduction.index("\n};", start) + 3
    cpp = (
        Path(__file__).with_name("sol61_spatial_interaction_host.cpp").read_text()
        + "\nnamespace pops {"
        + reduction[start:stop]
        + "}\n"
        + body
        + r"""
template<int D> void dimension_smoke() {
  using namespace pops;using namespace pops::runtime::program;
  using F=MultiFab<D,Kokkos::HostSpace>;
  Index<D> zero{};Box<D> box(zero,zero);RealVector<D> upper{};upper.fill(1);
  Geometry<D> geometry{box,{},upper};F rho({box},{},{},3,{});
  auto r=rho.fab(0).view();r(zero,0)=2;r(zero,1)=-3;r(zero,2)=.5;
  std::array<InteractionLevelView<D,Kokkos::HostSpace>,1> levels{{{&rho,nullptr,nullptr,nullptr,geometry}}};
  std::array<int,3> selected{2,0,1};ExecutionLane lane;
  auto output=direct_spatial_interaction<D,Kokkos::HostSpace>(levels,0,selected,100000,"issued",lane,[](auto,auto){return 2.;});
  auto v=output.fab(0).view();assert(v(zero,0)==1);assert(v(zero,1)==4);assert(v(zero,2)==-6);
}
void tower_smoke() {
  using namespace pops;using namespace pops::runtime::program;
  using F=MultiFab<1,Kokkos::HostSpace>;
  Box<1> coarse_box({0},{1}),fine_box({0},{1}),fine_domain({0},{3});
  Geometry<1> coarse_geometry{coarse_box,{0},{1}},fine_geometry{fine_domain,{0},{1}};
  F coarse({coarse_box},{},{},1,{}),fine({fine_box},{},{},1,{}),coverage({coarse_box},{},{},1,{}),kappa({fine_box},{},{},1,{});
  coarse.fab(0).view()({0},0)=NAN;coarse.fab(0).view()({1},0)=2;
  fine.fab(0).view()({0},0)=4;fine.fab(0).view()({1},0)=6;
  coverage.fab(0).view()({0},0)=0;coverage.fab(0).view()({1},0)=1;
  kappa.fab(0).view()({0},0)=1;kappa.fab(0).view()({1},0)=.5;
  std::array<InteractionLevelView<1,Kokkos::HostSpace>,2> levels{{{&coarse,nullptr,&coverage,nullptr,coarse_geometry},{&fine,nullptr,nullptr,&kappa,fine_geometry}}};
  std::array<int,1> selected{0};ExecutionLane lane;
  auto output=direct_spatial_interaction<1,Kokkos::HostSpace>(levels,0,selected,100000,"issued",lane,[](auto,auto){return 1.;});
  assert(output.fab(0).view()({0},0)==0);assert(output.fab(0).view()({1},0)==2.75);
}
int main() {
  tower_smoke();
  dimension_smoke<1>();dimension_smoke<3>();
  using namespace pops; using namespace pops::runtime::program;
  using Field=MultiFab<2,Kokkos::HostSpace>;
  Box<2> box({0,0},{1,1}); Geometry<2> geometry{box,{0,0},{1,3}};
  Field rho({box},{true},{},3,{}), active({box},{true},{},1,{}), covered({box},{true},{},1,{}), kappa({box},{true},{},1,{});
  auto r=rho.fab(0).view(); auto a=active.fab(0).view();auto c=covered.fab(0).view();auto k=kappa.fab(0).view();
  for(int y=0;y<2;++y)for(int x=0;x<2;++x){a({x,y},0)=1;c({x,y},0)=1;k({x,y},0)=1;}
  r({0,0},0)=2;r({0,0},1)=-0.;r({0,0},2)=4;k({0,0},0)=.25;
  for(int j=0;j<3;++j)r({1,0},j)=NAN;c({1,0},0)=0;
  r({0,1},0)=-3;r({0,1},1)=5;r({0,1},2)=0;k({0,1},0)=.5;
  for(int j=0;j<3;++j)r({1,1},j)=NAN;a({1,1},0)=0;
  std::array<InteractionLevelView<2,Kokkos::HostSpace>,1> levels{{{&rho,&active,&covered,&kappa,geometry}}};
  std::array<int,3> selected{2,0,1};ExecutionLane lane;
  auto kernel=[](auto x,auto y){return 1+x[0]+x[1]-2*y[1];};
  auto output=direct_spatial_interaction<2,Kokkos::HostSpace>(levels,0,selected,100000,"issued",lane,kernel);
  auto v=output.fab(0).view();
  assert(v({0,0},0)==.375);assert(v({0,0},1)==3.);assert(v({0,0},2)==-4.6875);
  assert(v({0,1},0)==1.5);assert(v({0,1},1)==1.875);assert(v({0,1},2)==-1.875);
  assert(v({1,0},0)==0);assert(v({1,1},1)==0);
  assert(std::signbit(interaction_cell_value<2>(r,Index<2>{0,0},1)));
  assert(std::signbit(interaction_owner_value(-0.,true,lane)));
  auto refusal=[&](auto callback){try{callback();}catch(const std::exception&){return;}throw std::runtime_error("expected refusal");};
  refusal([&]{direct_spatial_interaction<2,Kokkos::HostSpace>(levels,0,selected,8,"issued",lane,kernel);});
  a({0,0},0)=.5;refusal([&]{direct_spatial_interaction<2,Kokkos::HostSpace>(levels,0,selected,100000,"issued",lane,kernel);});a({0,0},0)=1;
  r({0,0},0)=INFINITY;refusal([&]{direct_spatial_interaction<2,Kokkos::HostSpace>(levels,0,selected,100000,"issued",lane,kernel);});r({0,0},0)=2;
  refusal([&]{direct_spatial_interaction<2,Kokkos::HostSpace>(levels,0,selected,100000,"issued",lane,[](auto,auto){return NAN;});});
  k({1,0},0)=NAN;refusal([&]{direct_spatial_interaction<2,Kokkos::HostSpace>(levels,0,selected,100000,"issued",lane,kernel);});k({1,0},0)=1;
  std::array<int,2> duplicate{0,0};refusal([&]{direct_spatial_interaction<2,Kokkos::HostSpace>(levels,0,duplicate,100000,"issued",lane,kernel);});
  std::array<int,1> foreign{3};refusal([&]{direct_spatial_interaction<2,Kokkos::HostSpace>(levels,0,foreign,100000,"issued",lane,kernel);});
  refusal([&]{interaction_product(SIZE_MAX,2);});refusal([&]{interaction_add(SIZE_MAX,1);});
  std::cout << "host actual body: 27 checks PASS Realbits=" << interaction_real_bits << "\n";
}
"""
    )
    source = tmp_path / "consumer.cpp"
    source.write_text(cpp)
    executable = tmp_path / "consumer"
    subprocess.run(
        [
            "/usr/bin/clang++",
            "-std=c++20",
            "-O0",
            "-DPOPS_REAL_TYPE=" + real,
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    assert "27 checks PASS Realbits=" + str(32 if real == "float" else 64) in result.stdout
