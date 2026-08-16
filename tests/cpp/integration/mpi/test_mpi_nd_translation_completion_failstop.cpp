#include <pops/mesh/nd_proof/translation_exchange.hpp>
#include <pops/parallel/comm.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

namespace {

using namespace pops;
using namespace pops::mesh::nd_proof;
using pops::mesh::BoxHashBudget;
using pops::mesh::Distribution;
using pops::mesh::RankSpace;

template <int Dim>
using ProductionBoxArray = pops::mesh::BoxArray<Dim>;

constexpr char kCompletionFailstopToken[] = "POPS_ND_COMPLETION_FAILSTOP_OBSERVED";
TranslationExchange<1>* g_exchange = nullptr;

[[noreturn]] void completion_terminate_handler() noexcept {
  const bool verified =
      g_exchange != nullptr && g_exchange->sealed() &&
      g_exchange->diagnostic_stage() == TranslationExchangeDiagnosticStage::completion &&
      g_exchange->live_request_count() == 0;
  std::fputs(verified ? kCompletionFailstopToken : "POPS_ND_COMPLETION_FAILSTOP_INVALID", stderr);
  std::fputc('\n', stderr);
  std::fflush(stderr);
  std::_Exit(verified ? 0 : 2);
}

TranslationSchedule<1> completion_schedule() {
  const Box<1> domain{Index<1>{0}, Index<1>{1}};
  const ProductionBoxArray<1> layout(std::vector<Box<1>>{domain});
  const RankSpace<1> ranks{Index<1>{}, Extent<1>{1}};
  const Distribution<1> distribution =
      Distribution<1>::partitioned(layout, ranks, std::vector<Index<1>>{Index<1>{}});
  return TranslationSchedule<1>{
      layout,
      distribution,
      domain,
      PeriodicTopology<1>::axis_translations(std::array<bool, 1>{true}),
      Extent<1>{1},
      1,
      0,
      1,
      Index<1>{},
      Extent<1>{2},
      BoxHashBudget{64, 64, 64},
      TranslationScheduleBudget{64, 8, 256, 256, 256,
                                LocalNeighborWorkBudget{64, 64, {64, 4096}, {4096, 4096}}}};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    comm_init(&argc, &argv);
    Kokkos::ScopeGuard kokkos(argc, argv);
    auto lane = ExecutionLane::duplicate_world_collectively("nd-exchange-completion-failstop");
    auto schedule = completion_schedule();
    if (schedule.local_job_count() == 0)
      return 10;
    MultiFab<1> fields(schedule.layout(), schedule.distribution(), Index<1>{}, 1,
                       schedule.ghosts());
    TranslationExchangeContext context{131, 137};
    context.fail_completion_rank = 0;
    TranslationExchange<1> exchange(schedule, lane, context);
    g_exchange = &exchange;
    std::set_terminate(completion_terminate_handler);
    try {
      exchange.execute(fields, lane);
    } catch (...) {
      return 11;
    }
    return 12;
  } catch (...) {
    return 13;
  }
}
