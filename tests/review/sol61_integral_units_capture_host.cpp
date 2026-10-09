// Actual class/cache/ledger host guard reception. This access shim exercises
// private typed gates, not a System run or native Kokkos/MPI qualification.
#include <pops/core/identity/physical_dimension_json.hpp>
#include <pops/core/identity/sha256.hpp>
#include <pops/runtime/program/accepted_exchange.hpp>
#include <pops/runtime/program/prepared_resource_cache.hpp>
#define private public
#include <pops/runtime/program/prepared_integral_capture.hpp>
#undef private
#include <iostream>

using namespace pops;
using namespace pops::runtime::program;

int main() try {
  int checks = 0;
  auto lane = ExecutionLane::world("pops.test.canonical-integral-host");
  for (const std::string& units : {
           std::string("not-json"),
           std::string(R"({"powers":[],"kind":"physical_dimension"})"),
           std::string(R"({"kind":"physical_dimension","powers":[["mass",2,4]]})"),
           std::string(R"({"kind":"physical_dimension","powers":[["mass",0,1]]})"),
           std::string(R"({"kind":"physical_dimension","powers":[["mass",1,1],["mass",2,1]]})")}) {
    const auto key = "pops.integral.v2/actual-guard/" + identity::sha256_hex(
        std::vector<std::uint8_t>(units.begin(), units.end())) + "/q";
    PreparedResourceCache cache;
    auto attempt = cache.begin_attempt();
    runtime::multiblock::BoundaryEvaluationPoint point{"clock", 0, 0, 0, 0, {0,1}, .01, 0.,
        "host-graph", "host-rate", "host-application"};
    AcceptedExchangeLedger ledger;
    ledger.declare_integral(key, .7);
    const auto original = ledger.checkpoint(true);
    int owner = 0, attempts = 0, reads = 0;
    bool refused = false;
    try {
      (void)PreparedIntegralCapture::prepare_(&owner,
          [&] { ++attempts; return attempt; }, [&] { return point; }, key, units,
          [&] { ++reads; return PreparedIntegralCapture::ReadImage{
              ledger.integral(key), ledger.checkpoint(true)}; }, lane);
    } catch (const std::invalid_argument&) { refused = true; }
    if (!refused || attempts != 0 || reads != 0 || ledger.checkpoint(true) != original)
      throw std::runtime_error("redigested units reached live capture authority");
    checks += 4;
  }
  for (const std::string& units : {
           std::string(R"({"kind":"physical_dimension","powers":[]})"),
           std::string(R"({"kind":"physical_dimension","powers":[["mass",-7,3],["\u6642\u9593",1,2]]})"),
           std::string(R"({"kind":"physical_dimension","powers":[["huge",18446744073709551617,18446744073709551616],["\ue000",1,1],["\ud800\udc00",1,1]]})")}) {
    const auto key = "pops.integral.v2/actual-guard/" + identity::sha256_hex(
        std::vector<std::uint8_t>(units.begin(), units.end())) + "/q";
    PreparedResourceCache cache;
    auto attempt = cache.begin_attempt();
    runtime::multiblock::BoundaryEvaluationPoint point{"clock", 0, 0, 0, 0, {0,1}, .01, 0.,
        "host-graph", "host-rate", "host-application"};
    AcceptedExchangeLedger ledger;
    ledger.declare_integral(key, .7);
    const auto original = ledger.checkpoint(true);
    int owner = 0;
    const auto read = [&] { return PreparedIntegralCapture::ReadImage{
        ledger.integral(key), ledger.checkpoint(true)}; };
    const auto capture = PreparedIntegralCapture::prepare_(&owner, [&] { return attempt; },
        [&] { return point; }, key, units, read, lane);
    if (capture.consume_(&owner, [&] { return attempt; }, [&] { return point; },
                        key, units, read, lane) != .7 || ledger.checkpoint(true) != original)
      throw std::runtime_error("canonical units altered value or ledger");
    checks += 2;
    int attempts = 0, reads = 0;
    bool refused = false;
    try {
      (void)capture.consume_(&owner, [&] { ++attempts; return attempt; },
          [&] { return point; }, key, units + " ", [&] { ++reads; return read(); }, lane);
    } catch (const std::invalid_argument&) { refused = true; }
    if (!refused || attempts || reads || ledger.checkpoint(true) != original)
      throw std::runtime_error("consume did not validate typed units before state reads");
    checks += 4;
  }
  std::cout << checks << " actual class/cache/ledger host assertions PASS\n";
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
