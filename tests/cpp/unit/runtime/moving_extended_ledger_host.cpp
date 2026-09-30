// Actual-header host receipt seam, not a PDE/System/MPI qualification.
#include <pops/runtime/program/accepted_exchange.hpp>
#include <iostream>

using pops::runtime::program::AcceptedExchangeLedger;
using pops::runtime::program::ExchangeRecord;

void require(bool value,const char* message) {
  if(!value) throw std::runtime_error(message);
}

int main(int argc,char** argv) {
  AcceptedExchangeLedger empty;
  const auto legacy=empty.checkpoint();
  require(legacy.size()==16 && legacy[7]=='1',"legacy empty bytes changed");
  const auto extended=empty.checkpoint(true);
  require(extended.size()==32 && extended[7]=='2',"explicit extended image invalid");
  if(argc>1 && std::string(argv[1])=="--zero-baseline") {
    bool rejected=false;
    try { (void)AcceptedExchangeLedger::from_checkpoint(extended); }
    catch(const std::invalid_argument& error) {
      rejected=std::string(error.what())=="accepted exchange checkpoint has invalid integral count";
    }
    require(rejected,"old positive-baseline refusal was not reproduced");
    std::cout<<"actual-header baseline writer/read inconsistency reproduced\n";
    return 0;
  }
  auto restored=AcceptedExchangeLedger::from_checkpoint(extended);
  require(restored.checkpoint(true)==extended,"empty extended roundtrip differs");
  require(restored.checkpoint()==legacy,"default serialization changed");
  ExchangeRecord record{"mesh","volume","geometry","endpoint",1,1.,.04,.2,1};
  record.source_evaluation_identity="mesh-original-evaluation";
  empty.stage(record);
  const auto metadata=empty.checkpoint(true);
  restored=AcceptedExchangeLedger::from_checkpoint(metadata);
  require(restored.records().size()==1,"record was lost");
  require(restored.records()[0].source_evaluation_identity==record.source_evaluation_identity,
          "qualified geometry metadata was lost");
  require(restored.checkpoint(true)==metadata,"geometry metadata roundtrip differs");
  // A metadata-only exterior record is a valid unconsumed positive baseline.
  // Invent its exact consumption key without inventing an IntegralState.
  AcceptedExchangeLedger pending;
  record.trace_axis=0; record.trace_side=1; record.trace_component=0; record.exterior_trace=true;
  pending.stage(record);
  const auto pending_image=pending.checkpoint(true);
  restored=AcceptedExchangeLedger::from_checkpoint(pending_image);
  require(restored.checkpoint(true)==pending_image,"pending trace positive baseline differs");
  auto forged_consumption=pending_image;
  forged_consumption[forged_consumption.size()-8]=1;
  const auto text=[&](const std::string& value) {
    for(unsigned byte=0;byte<8;++byte)
      forged_consumption.push_back(static_cast<std::uint8_t>(value.size()>>(8*byte)));
    forged_consumption.insert(forged_consumption.end(),value.begin(),value.end());
  };
  text(record.operation_identity); text(record.occurrence_identity);
  text(record.evaluation_context); text(record.quadrature_identity);
  if(argc>1 && std::string(argv[1])=="--consumption-baseline") {
    restored=AcceptedExchangeLedger::from_checkpoint(forged_consumption);
    require(restored.checkpoint(true)==forged_consumption,"old synthetic consumption was not accepted");
    std::cout<<"actual-header synthetic consumption without integral reproduced\n";
    return 0;
  }
  unsigned rejected=0;
  auto malformed=extended;
  for(unsigned byte=16;byte<24;++byte) malformed[byte]=255;
  try { (void)AcceptedExchangeLedger::from_checkpoint(malformed); }
  catch(const std::invalid_argument&) { ++rejected; }
  malformed=extended; malformed.resize(24);
  try { (void)AcceptedExchangeLedger::from_checkpoint(malformed); }
  catch(const std::invalid_argument&) { ++rejected; }
  malformed=extended; malformed.back()=1;
  try { (void)AcceptedExchangeLedger::from_checkpoint(malformed); }
  catch(const std::invalid_argument&) { ++rejected; }
  try { (void)AcceptedExchangeLedger::from_checkpoint(forged_consumption); }
  catch(const std::invalid_argument& error) {
    require(std::string(error.what())=="accepted exchange checkpoint consumes trace without an integral state",
            "synthetic consumption was refused for an unrelated reason");
    ++rejected;
  }
  require(rejected==4,"corrupt count/truncation/consumption was accepted");
  empty.declare_integral("q",2.);
  const auto original02=empty.checkpoint();
  restored=AcceptedExchangeLedger::from_checkpoint(original02);
  require(restored.integral("q")==2. && restored.checkpoint()==original02,
          "existing integral POPSEX02 changed");
  std::cout<<"actual-header extended metadata roundtrip and 4 corruptions PASS\n";
}
