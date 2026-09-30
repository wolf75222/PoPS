#include <pops/core/identity/sha256.hpp>
#if __has_include(<pops/core/identity/physical_dimension_json.hpp>)
#include <pops/core/identity/physical_dimension_json.hpp>
#endif
#include <pops/runtime/program/accepted_exchange.hpp>
#include <pops/runtime/program/prepared_resource_cache.hpp>
#define private public
#include <pops/runtime/program/prepared_integral_capture.hpp>
#undef private
#include <cassert>
#include <iostream>

using namespace pops;
using namespace pops::runtime::program;
std::string key_of(const std::string& units) {
  return "pops.integral.v2/independent-units/" + identity::sha256_hex(
      std::vector<std::uint8_t>(units.begin(),units.end())) + "/q";
}
int main() {
  const auto lane=ExecutionLane::world("sol61.units.preflight");
  PreparedResourceCache cache;
  const auto attempt=cache.begin_attempt();
  runtime::multiblock::BoundaryEvaluationPoint point{"clock",0,0,0,0,{0,1},.01,0.};
  int owner=0, callbacks=0, checks=0;
  const std::string units=R"({"kind":"physical_dimension","powers":[]})";
  const auto key=key_of(units);
  AcceptedExchangeLedger ledger;
  ledger.declare_integral(key,.7);
  const auto image=ledger.checkpoint(true);
  auto attempt_get=[&] { ++callbacks; return attempt; };
  auto point_get=[&] { ++callbacks; return point; };
  auto read=[&] { ++callbacks; return PreparedIntegralCapture::ReadImage{ledger.integral(key),ledger.checkpoint(true)}; };
  const auto capture=PreparedIntegralCapture::prepare_(&owner,attempt_get,point_get,key,units,read,lane);
  for (const auto& malformed : {std::string("not-json"),
      std::string(R"({"powers":[],"kind":"physical_dimension"})"),
      std::string(R"({"kind":"physical_dimension","powers":[["a",2,4]]})"),
      std::string(R"({"kind":"physical_dimension","powers":[["a",01,1]]})")}) {
    callbacks=0;
    bool refused=false;
    try { (void)PreparedIntegralCapture::prepare_(&owner,attempt_get,point_get,key_of(malformed),malformed,read,lane); }
    catch (const std::invalid_argument& e) { refused=std::string(e.what()).find("canonical typed JSON")!=std::string::npos; }
    assert(refused && callbacks==0 && ledger.checkpoint(true)==image); ++checks;
    refused=false;
    try { (void)capture.consume_(&owner,attempt_get,point_get,key_of(malformed),malformed,read,lane); }
    catch (const std::invalid_argument& e) { refused=std::string(e.what()).find("canonical typed JSON")!=std::string::npos; }
    assert(refused && callbacks==0 && ledger.checkpoint(true)==image); ++checks;
    assert(capture.consume_(&owner,attempt_get,point_get,key,units,read,lane)==.7); ++checks;
  }
  std::cout<<"real_capture_preflight_checks="<<checks<<'\n';
}
