// Source/Host proof of the existing real registry/carrier transaction, not PoPS Native science.
#include <pops/runtime/system/exact_aux_registry.hpp>
#include <pops/runtime/system/auxiliary_ghost_fill.hpp>
#include <pops/runtime/system/exact_field_marshaling.hpp>
#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace s = pops::runtime::system;
void require(bool value) { if (!value) throw std::runtime_error("instance carrier assertion failed"); }

void check(pops::Index<2> upper, bool reverse) {
  using namespace pops;
  using namespace pops::mesh;
  using Registry=s::ExactAuxiliaryRegistry<2>;
  using Provider=s::PreparedAuxiliaryProvider<2>;
  const Box<2> domain{Index<2>{},upper};
  const BoxArray<2> layout(std::vector<Box<2>>{domain});
  Extent<2> one{};one[0]=one[1]=1;
  const auto distribution=Distribution<2>::replicated(layout,RankSpace<2>(Index<2>{},one));
  s::AuxiliaryStorageShape<2> shape;shape.spatial_rank=2;shape.value_components=1;
  s::AuxiliaryComponentContract contract{"scalar","cell",std::nullopt,"cell","scalar"};
  std::array<s::AuxiliaryComponentKey,3> fields,gains;
  Registry registry;
  for (int slot=0;slot<3;++slot) {
    int i=reverse?2-slot:slot;
    const auto owner="case:independent/block:instance"+std::to_string(i)+"/model_definition:shared";
    fields[i]={owner,"field","repeated space","same_name"};
    gains[i]={owner,"aux","gain","gain"};
    const auto id="published-"+std::to_string(i);
    registry.add(Provider{id,s::AuxiliaryProviderKind::field_output,
        {s::AuxiliaryEvaluationEvent::before_residual,s::AuxiliaryFreshness::accepted_step},
        {{fields[i],contract,shape}}, {}});
    registry.add(Provider{"derived-"+std::to_string(i),s::AuxiliaryProviderKind::derived,
        {s::AuxiliaryEvaluationEvent::before_residual,s::AuxiliaryFreshness::evaluation},
        {{gains[i],contract,shape}},{{fields[i],contract,shape}},
        Provider::launcher_type::trusted_extension(PreparedProviderIdentity{"source-host.instance-derived",1},
            "derived-"+std::to_string(i),[](const s::AuxiliaryKernelLaunchContext<2>& context) {
          auto* destination=context.storage.candidate->find(context.outputs[0].address.group);
          const auto* input=context.storage.candidate->find(context.dependencies[0].address.group);
          const auto out=context.outputs[0].address.component;
          const auto in=context.dependencies[0].address.component;
          for (std::size_t local=0;local<destination->local_size();++local) {
            const auto source=input->fab(local).view();const auto target=destination->fab(local).view();
            for_each_cell(destination->box(local),[=] POPS_HD(const Index<2>& cell) {
              target(cell,out)=Real(2)+source(cell,in)*source(cell,in);
            });
          }
          Kokkos::fence();
        })});
    registry.add_consumer_plan({"read-"+std::to_string(i),{{{gains[i],contract,shape},0}}});
  }
  registry.seal();
  for(int i=0;i<3;++i)for(int j=0;j<i;++j) {
    const auto left=registry.address_of(fields[i]),right=registry.address_of(fields[j]);
    require(left.group!=right.group||left.component!=right.component);
  }
  s::AuxiliaryStorageGroups<2> accepted;
  for(const auto& group:registry.storage_groups())
    accepted.groups.emplace(group.identity,MultiFab<2>(layout,distribution,Index<2>{},int(group.component_count),Extent<2>{}));
  const auto write_fields=[&](auto& storage,bool invalid) {
    for(const auto& group:registry.storage_groups()) {
      auto& field=*storage.find(group.identity);const auto view=field.fab(0).view();const int width=field.ncomp();
      for_each_cell(domain,[=] POPS_HD(const Index<2>& cell){for(int k=0;k<width;++k)view(cell,k)=Real(0);});
    }
    for(int i=0;i<3;++i) {
      const auto address=registry.address_of(fields[i]);auto& field=*storage.find(address.group);
      const auto view=field.fab(0).view();const Real value=(invalid&&i==1)?std::numeric_limits<Real>::quiet_NaN():Real(1+2*i);
      for_each_cell(domain,[=] POPS_HD(const Index<2>& cell){view(cell,address.component)=value;});
    }
    Kokkos::fence();
  };
  const auto values=[&](const auto& storage) {
    std::vector<Real> result;
    for(int i=0;i<3;++i)for(const auto& key:{fields[i],gains[i]}) {
      const auto address=registry.address_of(key);const auto& fab=storage.find(address.group)->fab(0);
      auto host=fab.create_host_mirror();fab.copy_to_host(host);
      s::marshaling::for_each_host_index(domain,[&](const Index<2>& cell,std::size_t) {
        result.push_back(host(s::marshaling::storage_ordinal(fab,cell,int(address.component))));
      });
    }
    return result;
  };
  s::AuxiliaryEvaluationPoint point;point.clock="independent-clock";point.event=s::AuxiliaryEvaluationEvent::before_residual;
  const std::vector<std::string> roots{"published-0","published-1","published-2"};
  auto candidate=accepted;write_fields(candidate,false);
  {
    auto tx=registry.begin_external_publication(point,roots);
    for(const auto& id:roots)tx.stage_external(id);
    tx.accept();std::swap(accepted,candidate);
  }
  {
    candidate=accepted;auto tx=registry.begin_publication(point,{}, {"read-0","read-1","read-2"});
    tx.launch_ready_native({&accepted,&candidate,nullptr});
    s::require_finite_auxiliary_groups(candidate,nullptr,"instance Host refresh");
    tx.accept();std::swap(accepted,candidate);
  }
  const auto before=values(accepted);const auto generation=registry.accepted_generation();
  const auto cells=static_cast<std::size_t>(domain.numPts());
  for(int i=0;i<3;++i)for(std::size_t n=0;n<cells;++n) {
    require(before[(2*i)*cells+n]==Real(1+2*i));
    require(before[(2*i+1)*cells+n]==Real(2+(1+2*i)*(1+2*i)));
  }
  point.accepted_step=1;
  {
    candidate=accepted;auto tx=registry.begin_external_publication(point,roots);
    write_fields(candidate,true);for(const auto& id:roots)tx.stage_external(id);
    bool refused=false;try{s::require_finite_auxiliary_groups(candidate,nullptr,"instance Host nonfinite");}
    catch(const std::runtime_error&){refused=true;}
    require(refused);tx.reject();
  }
  require(values(accepted)==before&&registry.accepted_generation()==generation);
  {
    candidate=accepted;auto tx=registry.begin_external_publication(point,roots);
    write_fields(candidate,false);for(const auto& id:roots)tx.stage_external(id);
    s::require_finite_auxiliary_groups(candidate,nullptr,"instance Host retry");
    tx.accept();std::swap(accepted,candidate);
  }
  require(registry.accepted_generation()==generation+1);
  std::cout<<"shape "<<upper[0]+1<<"x"<<upper[1]+1<<" order "<<reverse<<" distinct-addresses/derived/rollback/retry PASS\n";
}
int main(int argc,char** argv) {
  Kokkos::initialize(argc,argv);int result=0;
  try { std::cout<<"actual DefaultExecutionSpace "<<Kokkos::DefaultExecutionSpace::name()<<"\n";
    check(pops::Index<2>{6,2},false);check(pops::Index<2>{1,10},true); }
  catch(const std::exception& error){std::cerr<<error.what()<<"\n";result=1;}
  Kokkos::finalize();return result;
}
