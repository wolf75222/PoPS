#include <gtest/gtest.h>
#include <pops/numerics/diffusion/prepared_diffusion.hpp>

namespace {
using namespace pops;
using namespace pops::runtime::program;
using Field=MultiFab<2>;
struct DiffusionContext {
  Geometry<2> geometry_=Geometry<2>::from_bounds(Box<2>{Index<2>{0,0},Index<2>{3,3}},RealVector<2>{0,0},RealVector<2>{1,1});
  BoundaryTopology<2> topology=BoundaryTopology<2>::axis_periodic({true,true});
  ExecutionLane lane=ExecutionLane::world("prepared-diffusion-test");
  AcceptedExchangeLedger ledger;
  const auto& geometry() const {return geometry_;}
  const auto& prepared_execution_lane() const {return lane;}
  auto prepare_mesh_boundary_session(Field& field,const ExecutionLane& execution) {
    return PreparedScalarBoundarySession<2>::prepare(geometry_,topology,field,execution,1);
  }
  void stage_exchange(ExchangeRecord record) {ledger.stage(std::move(record));}
};
Field field(Box<2> box=Box<2>{Index<2>{0,0},Index<2>{3,3}},Extent<2> ghosts=Extent<2>{1,1}) {
  auto layout=mesh::BoxArray<2>::from_domain(box,Extent<2>{4,4});
  auto distribution=mesh::Distribution<2>::replicated(layout,mesh::RankSpace<2>{Index<2>{0,0},Extent<2>{1,1}});
  return Field(layout,distribution,Index<2>{0,0},1,ghosts);
}
auto identity_law(Field& q) {
  return [&q](std::size_t local) {
    const auto values=std::as_const(q).fab(local).view();
    return [=] POPS_HD(const Index<2>& cell) {return std::array<Real,4>{values(cell,0),.1,.1,1};};
  };
}
}

TEST(PreparedDiffusion, RejectsSameSizedForeignLayoutAndGhostsBeforeWriting) {
  DiffusionContext context;
  auto q=field(), output=field(), foreign=field(Box<2>{Index<2>{4,0},Index<2>{7,3}}), ghost=field(Box<2>{Index<2>{0,0},Index<2>{3,3}},Extent<2>{2,1});
  PreparedDiffusion<2> prepared(context,q,{});
  output.set_val(17);
  const auto layout=q.layout();
  auto other_distribution=mesh::Distribution<2>::partitioned(layout,mesh::RankSpace<2>{Index<2>{0,0},Extent<2>{1,1}},{Index<2>{0,0}});
  Field partitioned(layout,other_distribution,Index<2>{0,0},1,Extent<2>{1,1});
  auto two_ranks=mesh::Distribution<2>::replicated(layout,mesh::RankSpace<2>{Index<2>{0,0},Extent<2>{2,1}});
  Field other_rank(layout,two_ranks,Index<2>{1,0},1,Extent<2>{1,1});
  EXPECT_THROW(prepared.apply(foreign,output,identity_law(foreign)),std::invalid_argument);
  EXPECT_THROW(prepared.apply(q,foreign,identity_law(q)),std::invalid_argument);
  EXPECT_THROW(prepared.apply(ghost,output,identity_law(ghost)),std::invalid_argument);
  EXPECT_THROW(prepared.apply(q,ghost,identity_law(q)),std::invalid_argument);
  EXPECT_THROW(prepared.apply(q,partitioned,identity_law(q)),std::invalid_argument);
  EXPECT_THROW(prepared.apply(other_rank,output,identity_law(other_rank)),std::invalid_argument);
  EXPECT_DOUBLE_EQ(reduce_max_local(output),17);
}

TEST(PreparedDiffusion, RejectsNegativeFaceSecantEvenWhenSampleDerivativesArePositive) {
  DiffusionContext context;
  auto q=field(),output=field();
  const auto values=q.fab(0).view();
  for_each_cell(q.box(0),[=] POPS_HD(const Index<2>& cell){values(cell,0)=cell[0]==0 ? -1 : 1;});
  PreparedDiffusion<2> prepared(context,q,{});
  auto law=[&](std::size_t local) {
    const auto values=std::as_const(q).fab(local).view();
    return [=] POPS_HD(const Index<2>& cell) {
      const Real u=values(cell,0);
      return std::array<Real,4>{u*u*u-2*u,.1,.1,3*u*u-2};
    };
  };
  EXPECT_THROW(prepared.apply(q,output,law),DiffusiveEvaluationError);
  EXPECT_THROW(prepared.explicit_frequency(),std::logic_error);
  EXPECT_TRUE(context.ledger.records().empty());
}

TEST(PreparedDiffusion, VariableDiagonalStaysInsideDivergenceWithPhysicalValueTrace) {
  DiffusionContext context;
  context.topology=BoundaryTopology<2>::physical();
  auto q=field(),output=field();
  const auto values=q.fab(0).view();
  const auto geometry=context.geometry();
  for_each_cell(q.box(0),[=] POPS_HD(const Index<2>& cell){
    values(cell,0)=1+geometry.cell_coordinate(0,cell[0])+geometry.cell_coordinate(1,cell[1]);
  });
  std::array<DiffusiveBoundary<2>,4> physical{};
  for(auto& row:physical) row={DiffusiveBoundaryKind::value,1,{1,1}};
  PreparedDiffusion<2> prepared(context,q,physical);
  prepared.apply(q,output,[&](std::size_t local) {
    const auto values=std::as_const(q).fab(local).view();
    return [=] POPS_HD(const Index<2>& cell) {
      return std::array<Real,4>{values(cell,0),1+.25*geometry.cell_coordinate(0,cell[0]),
                               2+.5*geometry.cell_coordinate(1,cell[1]),1};
    };
  });
  const auto result=std::as_const(output).fab(0).view();
  EXPECT_NEAR(for_each_cell_reduce_max(output.box(0),[=] POPS_HD(const Index<2>& cell){
    return Kokkos::abs(result(cell,0)-.75);}),0,2e-13);
  prepared.stage_accepted_exchanges(context,"operator","occurrence","stage0",.05);
  Real exchange=0;
  for(const auto& record:context.ledger.records()) exchange+=record.integrated_amount();
  EXPECT_EQ(context.ledger.records().size(),64);
  EXPECT_NEAR(exchange,.05*.75,2e-13);
}
