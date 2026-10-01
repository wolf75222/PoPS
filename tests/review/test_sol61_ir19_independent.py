"""IR19 independent SOURCE_ONLY probes. Host storage is explicitly substituted."""

from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


def body(text, signature):
    begin = text.index(signature)
    brace = text.index("{", begin)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[begin:end]


@pytest.mark.parametrize("fixed", (False, True))
def test_historical_counter_and_fixed_real_sealer_receipt_values(tmp_path, fixed):
    # Real sealer and receipt reader; only low-level lane/field/core authority
    # storage is substituted. This demonstrates source semantics, not Kokkos/MPI.
    def historical(path):
        return subprocess.check_output(
            ["rtk", "proxy", "git", "show", "ac68714233d9f6e7040198ad67b0bd626db42a30:" + path],
            cwd=ROOT,
        ).decode()

    seal = body(
        (
            ROOT / "include/pops/runtime/program/amr_program_context_spatial_interaction.inc"
        ).read_text()
        if fixed
        else historical("include/pops/runtime/program/amr_program_context_spatial_interaction.inc"),
        "void seal_original_field_source(",
    )
    stamp = body(
        historical("include/pops/runtime/amr/hierarchy_tensor_solver_provider.hpp"),
        "std::uint64_t accepted_original_candidate_generation(",
    )
    prefix = r"""
#include <cassert>
#include <bit>
#define POPS_HD
#include <cstdint>
using InteractionRealWord=std::uint64_t;
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
constexpr int Dim=1;
using Real=double;
struct ExecutionLane {};
namespace Kokkos { void fence() {} }
template<int> struct Box { std::size_t numPts() const {return 1;} };
template<int> struct Index {};
template<class Fn> Real for_each_cell_reduce_max(Box<1>,Fn fn) {return fn(Index<1>{});}
template<int> struct Geometry {};
struct Field {
 std::vector<double> values;
 struct View { const double* data; double operator()(Index<1>,int c) const {return data[c];} };
 struct Fab { const std::vector<double>* data=nullptr; Box<1> grown_box() const {return {};}
   View view() const {return {data->data()};} };
 using fab_type=Fab;
 std::vector<int> layout() const {return {1};}
 int distribution() const {return 1;} int local_rank() const {return 0;}
 int ncomp() const {return 1;} std::size_t local_size() const {return 1;}
 Fab fab(std::size_t) const {return {&values};}
 Box<1> box(std::size_t) const {return {};}
};
using field_type=Field;
struct AmrFieldResidualAuthority {};
template<int> struct PreparedAmrFieldResidual {
 std::vector<Field> image{{{3.}}};
 const auto& candidate(AmrFieldResidualAuthority,const ExecutionLane&) const {return image;}
};
struct Provider {
 const ExecutionLane* prepared_lane_;
 bool publication_active_=false;
 const std::vector<Field>* original_accepted_candidate_;
 std::uint64_t original_acceptance_generation_=1;
 std::vector<Field> publication{{{3.}}};
 Field mask{{1.}};
 Field& solution(int i) {return publication.at(i);}
 const Field& active_cell_mask(int) const {return mask;}
"""
    middle = r"""
};
using hierarchy_tensor_solver_type=Provider;
struct Facade { Geometry<1> prepared_amr_level_geometry(int) {return {};} };
struct Context {
 struct ClosedOriginalFieldSource {
  std::shared_ptr<Provider> provider;
  std::function<const std::vector<Field>&()> candidate;
  std::uint64_t accepted_generation=0;
  std::string identity;
  std::vector<Field> image,active_image;
  std::vector<Geometry<1>> geometries;
  std::size_t retained_bytes=0;
  std::map<std::int64_t,std::tuple<std::int64_t,int,int,std::string>> bindings;
 };
 struct Resource {std::shared_ptr<Provider> solver;};
 ExecutionLane lane;
 Facade facade; Facade* facade_=&facade;
 std::map<std::int64_t,Resource> hierarchy_field_resources_;
 mutable std::map<std::int64_t,std::shared_ptr<const ClosedOriginalFieldSource>> closed_original_sources_;
 const ExecutionLane& prepared_execution_lane() const {return lane;}
 int nlev() const {return 1;}
 template<class Fn> void closed_interaction_phase_(Fn fn) const {fn();}
 static std::size_t interaction_add(std::size_t a,std::size_t b) {return a+b;}
 static std::size_t interaction_product(std::size_t a,std::size_t b) {return a*b;}
"""
    suffix = r"""
};
int main() {
 Context ctx;
 auto core=std::make_shared<PreparedAmrFieldResidual<1>>();
 auto provider=std::make_shared<Provider>();
 provider->prepared_lane_=&ctx.lane;
 provider->original_accepted_candidate_=&core->image;
 ctx.hierarchy_field_resources_[1]={provider};
 const auto generation=provider->accepted_original_candidate_generation(&core->image,ctx.lane);
 // Layout, masks, candidate pointer, generation and simulated authority stay exact.
 // Only the public provider solution value changes after the Accept receipt.
 provider->solution(0).values[0]=99.;
 ctx.seal_original_field_source(1,provider,core,[]{return AmrFieldResidualAuthority{};},
   {{2,3,0,0,"binding"}},"original",1ULL<<24);
 const auto& image=ctx.closed_original_sources_.at(1)->image;
 assert(core->image[0].values[0]==3.);
 assert(image[0].values[0]==99.);
 assert(provider->accepted_original_candidate_generation(&core->image,ctx.lane)==generation);
}
"""
    if fixed:
        suffix = (
            suffix[: suffix.index(" // Layout,")]
            + r"""
 ctx.seal_original_field_source(1,provider,core,[]{return AmrFieldResidualAuthority{};},
   {{2,3,0,0,"binding"}},"original",1ULL<<24);
 const auto prior=ctx.closed_original_sources_.at(1);
 provider->solution(0).values[0]=99.;
 bool refused=false;
 try { ctx.seal_original_field_source(1,provider,core,[]{return AmrFieldResidualAuthority{};},
   {{2,3,0,0,"binding"}},"original",1ULL<<24); }
 catch(const std::invalid_argument& e) {
   refused=std::string(e.what())=="accepted original source values changed after Accept";
 }
 assert(refused && ctx.closed_original_sources_.at(1)==prior && prior->image[0].values[0]==3.);
 // Signed zero is a bit difference, not a tolerance comparison.
 core->image[0].values[0]=0.; provider->solution(0).values[0]=-0.;
 refused=false;
 try { ctx.seal_original_field_source(1,provider,core,[]{return AmrFieldResidualAuthority{};},
   {{2,3,0,0,"binding"}},"original",1ULL<<24); }
 catch(const std::invalid_argument&) {refused=true;}
 assert(refused && ctx.closed_original_sources_.at(1)==prior);
}
"""
        )
    source = tmp_path / "actual_ir19_seal_host.cpp"
    source.write_text(prefix + stamp + middle + seal + suffix)
    executable = tmp_path / "actual_ir19_seal_host"
    subprocess.run(
        ["rtk", "proxy", "c++", "-std=c++20", "-O0", str(source), "-o", str(executable)], check=True
    )
    subprocess.run(["rtk", "proxy", str(executable)], check=True)


def test_deep_Fab_and_value_Geometry_scope_EB_refusal_are_actual():
    fab = (ROOT / "include/pops/mesh/storage/fab.hpp").read_text()
    copy = body(fab, "Fab(const Fab& other)")
    assert "Kokkos::deep_copy(data_, other.data_)" in copy
    geometry = (ROOT / "include/pops/mesh/geometry/geometry.hpp").read_text()
    assert "Kokkos::View" not in geometry
    authority = body(
        (
            ROOT / "include/pops/runtime/program/amr_program_context_general_field_public.inc"
        ).read_text(),
        "AmrFieldResidualAuthority original_hierarchy_field_authority(",
    )
    assert "|| shared ||" in authority
    assert "prepared_amr_block_level_active_mask(runtime_block, level)" in authority
    assert "original AMR field provider has no embedded-boundary operator authority" in authority
    assert "prepare_spatial_collectively" in authority


def public():
    import importlib.util

    source = ROOT / "tests/review/test_sol61_amr_public_original.py"
    spec = importlib.util.spec_from_file_location("original_independent_ir19", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case, layout, program, _, _ = module.authored(width=3, order=(2, 0, 1), seed=False)
    from pops.fields import (
        CellMidpoint,
        CellVolumeMeasure,
        DirectSpatialInteraction,
        SpatialInteractionKernel,
    )
    from pops.model import FieldSpace

    observation = next(
        value
        for value in program._values
        if value.op == "field_component" and value.attrs["component"] == 2
    )
    keeper = next(value for value in program._values if value.op == "state")
    output = FieldSpace(
        "independent-closed",
        components=("I",),
        sampling="cell_center",
        frame=keeper.space.frame,
        support=keeper.space.support,
        clock=keeper.space.clock,
    )
    value = program.spatial_interaction(
        observation,
        SpatialInteractionKernel(2, lambda x, y: 1 + x[1] * y[0]),
        output_space=output,
        measure=CellVolumeMeasure(),
        quadrature=CellMidpoint(),
        realization=DirectSpatialInteraction(2**24),
        source_scope="completed_original",
        owner_block=keeper.block,
    )
    program.sum_component(value, 0)
    return module, case, layout, program, observation, value


@pytest.mark.parametrize("mutation", ("source_id", "tuple_component", "owner", "point", "scope"))
def test_public_source_authority_changes_refuse_at_serialization(mutation):
    _, _, _, program, source, value = public()
    attrs = dict(value.attrs)
    metadata = dict(attrs["closed_field_source"])
    if mutation == "source_id":
        object.__setattr__(source, "id", source.id + 1000)
    elif mutation == "tuple_component":
        metadata["tuple_component"] = True
    elif mutation == "owner":
        metadata["owner_block"] = next(
            node.block
            for node in program._values
            if node.op == "state" and node.block != metadata["owner_block"]
        )
    elif mutation == "point":
        object.__setattr__(value, "point", program.stage("foreign", c=0))
    else:
        attrs["source_scope"] = "accepted"
    attrs["closed_field_source"] = metadata
    object.__setattr__(value, "attrs", attrs)
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        program._serialize()


def test_actual_public_emission_Accept_then_seal_then_full_prepare():
    module, case, layout, program, source, value = public()
    assert source.block is source.space is source.state_ref is None
    inputs = tuple(source.inputs[0].inputs)
    code, plan = module.emit(case, layout)
    assert plan.time._serialize()["version"] == 19
    accept = code.index(".consume(pops::SolveConsumption::kAccept)")
    seal = code.index("ctx.seal_original_field_source(")
    prepare = code.index("ctx.prepare_closed_original_interaction(")
    read = code.index("ctx.closed_original_interaction(", prepare)
    assert accept < seal < prepare < read
    assert all(a is b for a, b in zip(inputs, source.inputs[0].inputs, strict=True))
