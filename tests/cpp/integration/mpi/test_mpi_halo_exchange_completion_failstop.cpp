#include <pops/mesh/boundary/halo_exchange.hpp>
#include <pops/parallel/comm.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

namespace {

using namespace pops;
using namespace pops::mesh;

constexpr char kPublicationFailstopToken[] = "POPS_HALO_PUBLICATION_FAILSTOP_OBSERVED";
HaloExchange<1>* g_exchange = nullptr;

[[noreturn]] void publication_terminate_handler() noexcept {
  const bool verified =
      g_exchange != nullptr && g_exchange->sealed() &&
      g_exchange->diagnostic_stage() == HaloExchangeDiagnosticStage::publication &&
      g_exchange->live_request_count() == 0;
  std::fputs(verified ? kPublicationFailstopToken : "POPS_HALO_PUBLICATION_FAILSTOP_INVALID",
             stderr);
  std::fputc('\n', stderr);
  std::fflush(stderr);
  std::_Exit(verified ? 0 : 2);
}

HaloSchedule<1> completion_schedule(const MultiFab<1>& fields, const Box<1>& domain) {
  return prepare_halo_schedule(fields, domain,
                               BoundaryTopology<1>::axis_periodic(std::array<bool, 1>{true}),
                               HaloScheduleBudget{{1, 0}, 16, 16, 3, 1, 64, 64, 64});
}

}  // namespace

int main(int argc, char** argv) {
  try {
    comm_init(&argc, &argv);
    Kokkos::ScopeGuard kokkos(argc, argv);
    auto lane = ExecutionLane::duplicate_world_collectively("halo-publication-failstop");
    const Box<1> domain{Index<1>{0}, Index<1>{1}};
    const BoxArray<1> layout(std::vector<Box<1>>{domain});
    const RankSpace<1> ranks{Index<1>{}, Extent<1>{1}};
    const Distribution<1> distribution =
        Distribution<1>::partitioned(layout, ranks, std::vector<Index<1>>{Index<1>{}});
    MultiFab<1> fields(layout, distribution, Index<1>{}, 1, Extent<1>{1});
    fields.set_val(Real{3});
    const HaloSchedule<1> schedule = completion_schedule(fields, domain);
    if (schedule.local_jobs().empty())
      return 10;
    HaloExchangeContext context{};
    context.context_generation = 1009;
    context.schedule_generation = 1013;
    context.fail_publication_rank = 0;
    HaloExchange<1> exchange(schedule, lane, context);
    g_exchange = &exchange;
    std::set_terminate(publication_terminate_handler);
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
