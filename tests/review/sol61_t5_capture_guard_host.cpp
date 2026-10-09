// Independent host guard reception. Actual production class/cache/ledger; serial lane.
// Access shim exposes only PreparedIntegralCapture, after all dependencies are included.
// It does not instantiate System/ProgramContext or qualify MPI/Kokkos execution.
#include <pops/core/identity/sha256.hpp>
#include <pops/runtime/program/accepted_exchange.hpp>
#include <pops/runtime/program/prepared_resource_cache.hpp>
#define private public
#include <pops/runtime/program/prepared_integral_capture.hpp>
#undef private
#include <iostream>
#include <cassert>

using namespace pops;
using namespace pops::runtime::program;
int checks = 0;
template<class Fn> void refuses(Fn&& fn, std::string_view fragment) {
  try { fn(); } catch (const std::exception& error) {
    if (std::string(error.what()).find(fragment) == std::string::npos) throw;
    ++checks; return;
  }
  throw std::runtime_error("guard accepted mutation: " + std::string(fragment));
}
std::string typed_key(const std::string& units) {
  return "pops.integral.v2/host-program/" + identity::sha256_hex(
      std::vector<std::uint8_t>(units.begin(), units.end())) + "/q";
}
int main() {
  auto lane = ExecutionLane::world("sol61.t5.capture.host");
  const std::string units = R"({"kind":"physical_dimension","powers":[]})";
  const auto key = typed_key(units);
  PreparedResourceCache cache, foreign_cache;
  auto attempt = cache.begin_attempt();
  auto foreign_attempt = foreign_cache.begin_attempt();
  runtime::multiblock::BoundaryEvaluationPoint point{"clock", 0, 0, 0, 0, {0,1}, .01, 0.};
  AcceptedExchangeLedger ledger;
  ledger.declare_integral(key, .7);
  const auto initial_ledger = ledger;
  int owner = 0, foreign_owner = 0;
  auto read = [&] { return PreparedIntegralCapture::ReadImage{ledger.integral(key), ledger.checkpoint(true)}; };
  auto prepare = [&] {
    return PreparedIntegralCapture::prepare_(&owner, [&]{return attempt;}, [&]{return point;}, key, units, read, lane);
  };
  auto consume = [&](const PreparedIntegralCapture& capture) {
    return capture.consume_(&owner, [&]{return attempt;}, [&]{return point;}, key, units, read, lane);
  };
  auto capture = prepare();
  assert(consume(capture) == .7); ++checks;
  bool overflow = false, runtime_failure = false;
  try {
    (void)PreparedIntegralCapture::prepare_(&owner,[&]{return attempt;},[&]{return point;},key,units,
        [&]() -> PreparedIntegralCapture::ReadImage { throw std::overflow_error("injected-overflow"); },lane);
  } catch (const std::overflow_error& error) {
    overflow = std::string(error.what()) == "injected-overflow";
  }
  assert(overflow); ++checks;
  try {
    (void)PreparedIntegralCapture::prepare_(&owner,[&]{return attempt;},[&]{return point;},key,units,
        [&]() -> PreparedIntegralCapture::ReadImage { throw std::runtime_error("injected-runtime"); },lane);
  } catch (const std::overflow_error&) { throw; }
    catch (const std::runtime_error& error) {
    runtime_failure = std::string(error.what()) == "injected-runtime";
  }
  assert(runtime_failure); ++checks;
  refuses([&] { (void)capture.consume_(&foreign_owner,[&]{return attempt;},[&]{return point;},key,units,read,lane); }, "owner/attempt/point");
  refuses([&] { (void)capture.consume_(&owner,[&]{return foreign_attempt;},[&]{return point;},key,units,read,lane); }, "owner/attempt/point");
  point.physical_time = .001;
  refuses([&] { (void)consume(capture); }, "owner/attempt/point");
  point.physical_time = 0.; point.graph_identity = "foreign-parent-graph";
  refuses([&] { (void)consume(capture); }, "owner/attempt/point");
  point.graph_identity.clear();
  refuses([&] { (void)capture.consume_(&owner,[&]{return attempt;},[&]{return point;},key,units+" ",read,lane); }, "exact typed v2 units");
  refuses([&] { (void)capture.consume_(&owner,[&]{return attempt;},[&]{return point;},key+"foreign",units,read,lane); }, "owner/attempt/point");
  ledger.declare_integral("unrelated-ledger-mutation", 0.);
  refuses([&] { (void)consume(capture); }, "ledger provenance changed");
  ledger = initial_ledger;
  assert(consume(capture) == .7); ++checks;
  ledger = AcceptedExchangeLedger{}; ledger.declare_integral(key, .8);
  refuses([&] { (void)consume(capture); }, "value changed");
  ledger = initial_ledger;
  cache.reject_attempt();
  refuses([&] { (void)consume(capture); }, "owner/attempt/point");
  attempt = cache.begin_attempt();
  refuses([&] { (void)consume(capture); }, "owner/attempt/point");
  auto retry = prepare();
  assert(consume(retry) == .7); ++checks;
  cache.finish_attempt();
  assert(consume(retry) == .7); ++checks;
  auto lease = cache.acquire_lease<std::vector<int>>(0,0,0,lane,
      [](const std::vector<int>& v){return v.size()==1;}, 1, 4);
  cache.clear();
  assert(!lease.current() && lease.get()[0] == 4); ++checks;
  refuses([&] { (void)consume(retry); }, "owner/attempt/point");
  attempt = cache.begin_attempt();
  assert(consume(prepare()) == .7); ++checks;

  // Findings are observations, not expected-refusal tests disguised as green.
  for (const std::string bogus_units : {std::string("not-json"), std::string(R"({"powers":[],"kind":"physical_dimension"})")}) {
    const auto bogus_key = typed_key(bogus_units);
    AcceptedExchangeLedger declared;
    declared.declare_integral(bogus_key,.7);
    auto bogus_read = [&] { return PreparedIntegralCapture::ReadImage{declared.integral(bogus_key),declared.checkpoint(true)}; };
    const auto bogus = PreparedIntegralCapture::prepare_(&owner,[&]{return attempt;},[&]{return point;},bogus_key,bogus_units,bogus_read,lane);
    assert(bogus.consume_(&owner,[&]{return attempt;},[&]{return point;},bogus_key,bogus_units,bogus_read,lane)==.7);
    std::cout << "OBSERVATION accepted noncanonical units: " << bogus_units << '\n';
  }
  point.stage_fraction.denominator = 0;
  const auto malformed_point = prepare();
  assert(consume(malformed_point)==.7);
  std::cout << "OBSERVATION accepted zero stage denominator\n";
  point.stage_fraction = {0,1};
  auto zero = prepare(); point.physical_time = -0.;
  assert(consume(zero)==.7);
  std::cout << "OBSERVATION accepted signed-zero point change\n";
  std::cout << "real_guard_checks=" << checks << "\n";
}
