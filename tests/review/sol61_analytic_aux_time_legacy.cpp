// Identical source built against old/new real headers: static point and checkpoint byte control.
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/system/auxiliary_checkpoint.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <iomanip>
#include <iostream>
namespace s=pops::runtime::system;
void dump(const std::string& bytes) {
  for(unsigned char c:bytes)std::cout<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(c);
  std::cout<<'\n';
}
int main() {
  s::AuxiliaryEvaluationPoint point;
  point.clock="static exact clock";point.accepted_step=17;point.layout_generation=23;
  point.level=1;point.substep=2;point.stage=3;point.nonlinear_iteration=5;
  point.event=s::AuxiliaryEvaluationEvent::before_residual;
  pops::ExactContractBuilder exact;point.serialize_exact(exact);dump(std::move(exact).release());
  s::AuxiliaryCheckpointAcceptedState<2> checkpoint;
  checkpoint.registry_contract="actual legacy registry bytes";checkpoint.accepted_generation=19;
  checkpoint.providers.push_back({"actual static provider",s::AuxiliaryProviderKind::derived,point});
  const auto bytes=s::serialize_auxiliary_checkpoint_state(checkpoint);
  const auto restored=s::deserialize_auxiliary_checkpoint_state<2>(bytes);
  if(s::serialize_auxiliary_checkpoint_state(restored)!=bytes)return 2;
  dump(std::string(bytes.begin(),bytes.end()));
  std::cout<<std::dec<<sizeof(s::AuxiliaryEvaluationPoint)<<' '
           <<sizeof(s::AuxiliaryKernelLaunchContext<2>)<<'\n'<<POPS_ABI_KEY_LITERAL<<'\n';
}
