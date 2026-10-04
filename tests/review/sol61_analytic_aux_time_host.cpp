// Source/Host execution of an actual emitted analytic launcher; no PoPS Native module.
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/system/exact_aux_registry.hpp>
#include <pops/runtime/system/auxiliary_checkpoint.hpp>
#include <pops/runtime/system/auxiliary_ghost_fill.hpp>
#include <pops/runtime/system/exact_field_marshaling.hpp>
#include <Kokkos_Core.hpp>
#include <bit>
#include <iostream>
#include <limits>
#include <stdexcept>
namespace s=pops::runtime::system;
using Provider=s::PreparedAuxiliaryProvider<2>;
using Dependency=s::AuxiliaryDependency<2>;
#include "actual-analytic-launcher.inc"
void require(bool ok){if(!ok)throw std::runtime_error("actual analytic-time Host oracle failed");}

s::AuxiliaryEvaluationPoint point(double dt,double time,int evaluation,int tick=4) {
  s::AuxiliaryEvaluationPoint result;result.clock=expected_clock;
  result.accepted_step=tick;result.stage=evaluation;
  result.nonlinear_iteration=evaluation;result.layout_generation=7;
  result.event=s::AuxiliaryEvaluationEvent::before_residual;
  pops::runtime::multiblock::BoundaryEvaluationPoint source;
  source.clock=result.clock;source.tick=tick;source.stage=evaluation;
  source.stage_fraction=pops::amr::Rational(evaluation?1:0,1);
  source.dt=dt;source.physical_time=time;
  result.qualify_physical_evaluation(source);return result;
}
void check(pops::Index<2> upper) {
  using namespace pops;using namespace pops::mesh;
  using Registry=s::ExactAuxiliaryRegistry<2>;
  const Box<2> domain{Index<2>{},upper};BoxArray<2> layout(std::vector<Box<2>>{domain});
  Extent<2> one{};one[0]=one[1]=1;
  auto distribution=Distribution<2>::replicated(layout,RankSpace<2>(Index<2>{},one));
  auto geometry=Geometry<2>::from_bounds(domain,RealVector<2>{0,0},RealVector<2>{1,1});
  s::AuxiliaryStorageShape<2> shape;shape.spatial_rank=2;shape.value_components=1;
  s::AuxiliaryComponentContract contract{"scalar","cell",std::nullopt,"cell","scalar"};
  s::AuxiliaryComponentKey key{"model_definition:actual-temporal-coefficient","aux","actual-value","actual-value"};
  Registry registry;
  registry.add(Provider{"actual-emitted-time",s::AuxiliaryProviderKind::derived,
      {s::AuxiliaryEvaluationEvent::before_residual,s::AuxiliaryFreshness::evaluation},
      {{key,contract,shape}},{},actual_launcher()});
  registry.add_consumer_plan({"actual-read",{{{key,contract,shape},0}}});registry.seal();
  s::AuxiliaryStorageGroups<2> accepted;
  for(const auto& group:registry.storage_groups())accepted.groups.emplace(group.identity,
      MultiFab<2>(layout,distribution,Index<2>{},int(group.component_count),Extent<2>{}));
  const auto address=registry.address_of(key);
  const auto values=[&](const auto& storage){
    const auto& fab=storage.find(address.group)->fab(0);auto mirror=fab.create_host_mirror();fab.copy_to_host(mirror);
    std::vector<std::uint64_t> result;
    s::marshaling::for_each_host_index(domain,[&](const Index<2>& cell,std::size_t){
      result.push_back(std::bit_cast<std::uint64_t>(double(mirror(s::marshaling::storage_ordinal(fab,cell,int(address.component))))));});
    return result;
  };
  const auto oracle=[&](const auto& storage,double t){
    const auto& fab=storage.find(address.group)->fab(0);auto mirror=fab.create_host_mirror();fab.copy_to_host(mirror);
    double max_error=0;
    s::marshaling::for_each_host_index(domain,[&](const Index<2>& cell,std::size_t){
      const double x=(cell[0]+0.5)/double(upper[0]+1),y=(cell[1]+0.5)/double(upper[1]+1);
      const double expected=1+x+2*y+t;
      const double actual=mirror(s::marshaling::storage_ordinal(fab,cell,int(address.component)));
      require(std::isfinite(actual));max_error=std::max(max_error,std::abs(actual-expected));
      std::cout<<cell[0]<<" "<<cell[1]<<" "<<std::hexfloat<<t<<" "<<actual<<" "<<expected<<"\n";
    });require(max_error<1e-13);
  };
  const auto evaluate=[&](const auto& at,bool accept){
    auto candidate=accepted;auto tx=registry.begin_publication(at,{}, {"actual-read"});
    require(tx.requires_staging("actual-emitted-time"));
    tx.launch_ready_native({&accepted,&candidate,&geometry});
    Kokkos::fence();s::require_finite_auxiliary_groups(candidate,nullptr,"actual-time Host oracle");
    oracle(candidate,at.require_physical_time(expected_clock));
    if(accept){tx.accept();std::swap(accepted,candidate);}else tx.reject();
  };
  const double dt=1./64.,begin=2.;const auto start=point(dt,begin,0);
  evaluate(start,true);const auto initial=values(accepted);
  const auto points=registry.accepted_points();const auto generation=registry.accepted_generation();
  const auto end=point(dt,begin+dt,1);evaluate(end,false);
  require(values(accepted)==initial&&registry.accepted_points()==points&&registry.accepted_generation()==generation);
  auto registry_snapshot=registry;auto storage_snapshot=accepted;
  evaluate(end,true);require(values(accepted)!=initial);
  registry.swap_complete(registry_snapshot);std::swap(accepted,storage_snapshot);
  require(values(accepted)==initial&&registry.accepted_points()==points&&registry.accepted_generation()==generation);
  // Retry a refused attempt with a changed trial dt and physical stage, same accepted epoch.
  const auto retry=point(dt/2,begin+dt/2,1);evaluate(retry,true);
  require(values(accepted)!=initial&&registry.accepted_generation()==generation+1);
  {auto tx=registry.begin_publication(retry,{}, {"actual-read"});require(!tx.requires_staging("actual-emitted-time"));tx.reject();}
  for(auto changed:{point(dt,begin+dt/2,1),point(dt/2,begin+dt,1),point(dt/2,begin+dt/2,1,5)}) {
    auto tx=registry.begin_publication(changed,{}, {"actual-read"});require(tx.requires_staging("actual-emitted-time"));tx.reject();
  }
  {auto changed=retry;++changed.layout_generation;auto tx=registry.begin_publication(changed,{}, {"actual-read"});require(tx.requires_staging("actual-emitted-time"));tx.reject();}
  auto snapshot=values(accepted);const auto accepted_generation=registry.accepted_generation();
  for(int mutation=0;mutation<3;++mutation){
    auto wrong=retry;if(mutation==0)wrong.clock="foreign logical Clock";
    if(mutation==1)wrong.physical_evaluation.reset();
    if(mutation==2)wrong.physical_evaluation->physical_time=std::numeric_limits<double>::quiet_NaN();
    bool refused=false;try{auto candidate=accepted;auto tx=registry.begin_publication(wrong,{}, {"actual-read"});
      require(tx.requires_staging("actual-emitted-time"));
      tx.launch_ready_native({&accepted,&candidate,&geometry});tx.accept();}
    catch(const std::invalid_argument&){refused=true;}require(refused);
    require(values(accepted)==snapshot&&registry.accepted_generation()==accepted_generation);
  }
  {auto stale=retry;runtime::multiblock::BoundaryEvaluationPoint source;
   source.clock=stale.clock;source.tick=stale.accepted_step-1;source.stage=stale.stage;source.dt=dt;source.physical_time=begin;
   bool refused=false;try{stale.qualify_physical_evaluation(source);}catch(const std::invalid_argument&){refused=true;}require(refused);}
  // Live rollback holds the full temporal point; durable checkpoint bytes carry only discrete stamps.
  const auto durable=s::capture_auxiliary_checkpoint_state(registry);
  require(!durable.providers.at(0).accepted_point->physical_evaluation);
  const auto bytes=s::serialize_auxiliary_checkpoint_state(durable);
  const auto decoded=s::deserialize_auxiliary_checkpoint_state<2>(bytes);
  require(s::serialize_auxiliary_checkpoint_state(decoded)==bytes);
  Registry restored=registry;restored.restore_accepted_publication(decoded.accepted_generation,{decoded.providers.at(0).accepted_point});
  auto after_restart=restored.begin_publication(retry,{}, {"actual-read"});require(after_restart.requires_staging("actual-emitted-time"));after_restart.reject();
  // The physical key is exact IEEE, including signed zero, while unrelated static v2 points retain old serialization.
  auto plus=point(dt,0.,0),minus=point(dt,-0.,0);require(plus!=minus);
  ExactContractBuilder plus_bytes,minus_bytes;plus.serialize_exact(plus_bytes);minus.serialize_exact(minus_bytes);
  require(std::move(plus_bytes).release()!=std::move(minus_bytes).release());
  std::cout<<"shape "<<upper[0]+1<<"x"<<upper[1]+1<<" clock/dt/epoch/cache/rollback/retry/checkpoint PASS\n";
}
int main(int argc,char** argv){Kokkos::initialize(argc,argv);try{check(pops::Index<2>{2,6});check(pops::Index<2>{6,2});}catch(...){Kokkos::finalize();throw;}Kokkos::finalize();}
