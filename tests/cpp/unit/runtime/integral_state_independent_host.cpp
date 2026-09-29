// Independent host seam review. This is not a System transaction or PDE test.
#include <pops/runtime/program/accepted_exchange.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using pops::runtime::program::AcceptedExchangeLedger;
using pops::runtime::program::ExchangeRecord;
using pops::runtime::program::consume_external_trace_collectively;

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

template <class Function>
void rejects(Function operation, const char* reason) {
  bool matched = false;
  try { operation(); }
  catch (const std::exception& error) {
    matched = std::string(error.what()).find(reason) != std::string::npos;
    if (!matched) throw;
  }
  require(matched, "expected refusal was absent");
}

ExchangeRecord face(std::string key, double dt, bool exterior = true,
                    int axis = 0, int side = 1, int component = 0,
                    std::string evaluation = "stage") {
  ExchangeRecord record{"operator", "occurrence", std::move(evaluation), std::move(key),
                        1, 1., 2., dt, 1};
  record.trace_axis = axis;
  record.trace_side = side;
  record.trace_component = component;
  record.exterior_trace = exterior;
  pops::runtime::multiblock::BoundaryEvaluationPoint point;
  point.clock = "main";
  point.dt = dt;
  point.physical_time = 0.;
  record.qualify_runtime_point(point);
  return record;
}

int main() {
  const auto lane = pops::ExecutionLane::world();
  AcceptedExchangeLedger ledger;
  ledger.declare_integral("q", .7);
  const AcceptedExchangeLedger::TraceSelection selection{"operator", "occurrence", 0, 1, 0, "stage"};
  ledger.stage(face("selected", .1));
  ledger.stage(face("interior", 100., false));
  ledger.stage(face("other-axis", 100., true, 1));
  ledger.stage(face("other-side", 100., true, 0, 0));
  ledger.stage(face("other-component", 100., true, 0, 1, 1));
  auto other_occurrence = face("other-occurrence", 100.);
  other_occurrence.occurrence_identity = "another-occurrence";
  ledger.stage(other_occurrence);
  require(std::abs(consume_external_trace_collectively(ledger, "q", selection, 1., lane) - .9)
              < 1.e-15, "wrong support entered the trace integral");
  const auto consumed_image = ledger.checkpoint();
  rejects([&] { consume_external_trace_collectively(ledger, "q", selection, 1., lane); },
          "no unconsumed face contribution");
  require(ledger.checkpoint() == consumed_image, "duplicate refusal changed state/consumption");

  auto restarted = AcceptedExchangeLedger::from_checkpoint(consumed_image);
  require(restarted.checkpoint() == consumed_image, "restart changed the exact ledger image");
  require(restarted.same_integral_declarations(ledger), "restart lost declaration authority");
  rejects([&] { consume_external_trace_collectively(restarted, "q", selection, 1., lane); },
          "no unconsumed face contribution");
  restarted.clear();
  require(std::abs(restarted.integral("q") - .9) < 1.e-15,
          "starting a fresh exchange window erased persistent q");
  restarted.stage(face("selected", .2));
  require(std::abs(consume_external_trace_collectively(restarted, "q", selection, 1., lane) - 1.3)
              < 1.e-15, "new-duration exchange reused an old consumption");

  // One physical occurrence evaluated at two SSPRK stages must not collapse into one delivery.
  AcceptedExchangeLedger stages;
  stages.declare_integral("q", .7);
  stages.stage(face("shared-face", .05, true, 0, 1, 0, "stage:0/evaluation:2"));
  auto last = face("shared-face", .05, true, 0, 1, 0, "stage:1/evaluation:5");
  last.numerical_flux = 4.;
  stages.stage(last);
  const AcceptedExchangeLedger::TraceSelection first_stage{
      "operator", "occurrence", 0, 1, 0, "stage:0/evaluation:2"};
  const AcceptedExchangeLedger::TraceSelection last_stage{
      "operator", "occurrence", 0, 1, 0, "stage:1/evaluation:5"};
  require(std::abs(consume_external_trace_collectively(stages, "q", first_stage, 1., lane) - .8)
              < 1.e-15, "first stage consumed another rate evaluation");
  auto between_stages = AcceptedExchangeLedger::from_checkpoint(stages.checkpoint());
  require(std::abs(consume_external_trace_collectively(
      between_stages, "q", last_stage, 1., lane) - 1.) < 1.e-15,
      "restart lost second stage ownership or quadrature");
  rejects([&] { consume_external_trace_collectively(
      between_stages, "q", first_stage, 1., lane); }, "no unconsumed face contribution");

  AcceptedExchangeLedger missing;
  missing.declare_integral("q", .7);
  missing.stage(face("periodic-or-interior", .1, false));
  const auto missing_image = missing.checkpoint();
  rejects([&] { consume_external_trace_collectively(missing, "q", selection, 1., lane); },
          "no unconsumed face contribution");
  require(missing.checkpoint() == missing_image, "missing trace refusal changed persistent state");

  AcceptedExchangeLedger overflow;
  overflow.declare_integral("q", .7);
  overflow.stage(face("selected", 1.));
  const auto overflow_image = overflow.checkpoint();
  rejects([&] { consume_external_trace_collectively(
      overflow, "q", selection, std::numeric_limits<double>::max(), lane); }, "non-finite");
  require(overflow.checkpoint() == overflow_image, "failed update consumed an exchange");
  require(std::abs(consume_external_trace_collectively(overflow, "q", selection, .1, lane) - .9)
              < 1.e-15, "retry could not consume the previously rejected update");

  AcceptedExchangeLedger authenticated;
  authenticated.declare_integral("q", .7);
  authenticated.stage(face("wire-face", .1, true, 0, 1, 0, "stage:0/evaluation:2"));
  auto forged = authenticated.checkpoint();
  const std::string author_evaluation = "stage:0/evaluation:2";
  const auto source_field = std::find_end(forged.begin(), forged.end(),
                                         author_evaluation.begin(), author_evaluation.end());
  require(source_field != forged.end(), "fixture did not locate the wire evaluation field");
  // The last copy is the standalone source identity, after the qualified runtime context.
  *(source_field + author_evaluation.size() - 1) = '9';
  rejects([&] { (void)AcceptedExchangeLedger::from_checkpoint(forged); }, "evaluation");

  AcceptedExchangeLedger zero;
  zero.declare_integral("zero", 0.);
  rejects([&] { zero.declare_integral("zero", -0.); }, "different initial");
  AcceptedExchangeLedger negative_zero;
  negative_zero.declare_integral("zero", -0.);
  require(!zero.same_integral_declarations(negative_zero), "signed-zero authority collapsed");
  AcceptedExchangeLedger legacy;
  legacy.stage(face("legacy", .1, false));
  const auto legacy_bytes = legacy.checkpoint();
  require(legacy_bytes[7] == '1' && consumed_image[7] == '2', "wrong local wire version");
  require(AcceptedExchangeLedger::from_checkpoint(legacy_bytes).checkpoint() == legacy_bytes,
          "legacy POPSEX01 roundtrip changed");
  std::cout << "integral ledger support/atomicity/restart host checks passed\n";
}
