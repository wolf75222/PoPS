/// @file
/// @brief Uniform System integration witnesses for the shared Program transaction authority.

#include <gtest/gtest.h>

#include "native_dso_compiler.hpp"
#include "program_v5_fixture.hpp"
#include <pops/core/foundation/allocator.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/program/program_execution_services.hpp>
#include <pops/runtime/system/prepared_coupling_workspace.hpp>
#include <pops/runtime/system/system_coupling_registry.hpp>
#include <pops/runtime/system.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <future>
#include <memory>
#include <new>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace pops {

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

namespace {

// This test binary intercepts every ordinary, aligned, and nothrow global allocation form, but
// only counts while a witness explicitly arms its window.
std::atomic<bool> g_heap_measurement_enabled{false};
std::atomic<std::uint64_t> g_measured_heap_allocations{0};

void note_measured_heap_allocation() noexcept {
  if (g_heap_measurement_enabled.load(std::memory_order_relaxed))
    g_measured_heap_allocations.fetch_add(1, std::memory_order_relaxed);
}

void* measured_allocate(std::size_t size) {
  void* pointer = std::malloc(size == 0 ? 1 : size);
  if (pointer == nullptr)
    throw std::bad_alloc();
  note_measured_heap_allocation();
  return pointer;
}

void* measured_allocate_nothrow(std::size_t size) noexcept {
  void* pointer = std::malloc(size == 0 ? 1 : size);
  if (pointer != nullptr)
    note_measured_heap_allocation();
  return pointer;
}

void* measured_aligned_allocate(std::size_t size, std::size_t alignment) {
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size == 0 ? 1 : size) != 0)
    pointer = nullptr;
  if (pointer == nullptr)
    throw std::bad_alloc();
  note_measured_heap_allocation();
  return pointer;
}

void* measured_aligned_allocate_nothrow(std::size_t size, std::size_t alignment) noexcept {
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size == 0 ? 1 : size) != 0)
    pointer = nullptr;
  if (pointer != nullptr)
    note_measured_heap_allocation();
  return pointer;
}

class HeapAllocationWindow {
 public:
  HeapAllocationWindow() : before_(g_measured_heap_allocations.load(std::memory_order_relaxed)) {
    g_heap_measurement_enabled.store(true, std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t close() noexcept {
    g_heap_measurement_enabled.store(false, std::memory_order_relaxed);
    return g_measured_heap_allocations.load(std::memory_order_relaxed) - before_;
  }

 private:
  std::uint64_t before_ = 0;
};

}  // namespace

void* operator new(std::size_t size) {
  return measured_allocate(size);
}
void* operator new[](std::size_t size) {
  return measured_allocate(size);
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  return measured_allocate_nothrow(size);
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
  return measured_allocate_nothrow(size);
}
void operator delete(void* pointer) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer) noexcept {
  std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept {
  std::free(pointer);
}
void operator delete(void* pointer, const std::nothrow_t&) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, const std::nothrow_t&) noexcept {
  std::free(pointer);
}
void* operator new(std::size_t size, std::align_val_t alignment) {
  return measured_aligned_allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
  return measured_aligned_allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  return measured_aligned_allocate_nothrow(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  return measured_aligned_allocate_nothrow(size, static_cast<std::size_t>(alignment));
}
void operator delete(void* pointer, std::align_val_t) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, std::align_val_t) noexcept {
  std::free(pointer);
}
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept {
  std::free(pointer);
}
void operator delete(void* pointer, std::align_val_t, const std::nothrow_t&) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, std::align_val_t, const std::nothrow_t&) noexcept {
  std::free(pointer);
}

namespace {

template <class View>
concept HasRvalueViewAccess = requires(View&& view) {
  std::forward<View>(view).get();
  std::forward<View>(view).operator->();
};

template <int Dim>
pops::SystemConfig<Dim> unit_config(int cells) {
  pops::SystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = cells;
    config.lower[axis] = pops::Real(0);
    config.upper[axis] = pops::Real(1);
    config.periodicity[axis] = true;
  }
  return config;
}

template <int Dim>
void add_authority_block(pops::System<Dim>& system) {
  system.install_block_state_route("gas", "test.system-transaction-authority.gas.state");
  system.seal_auxiliary_providers();
  pops::add_compiled_model(system, "gas", pops::nd::ScalarAdvection<Dim>{}, "none", "rusanov",
                           "conservative", "explicit");
}

template <int Dim>
void add_coupling_blocks(pops::System<Dim>& system) {
  system.install_block_state_route("donor", "test.system-coupling.donor.state");
  system.install_block_state_route("receiver", "test.system-coupling.receiver.state");
  system.seal_auxiliary_providers();
  pops::add_compiled_model(system, "donor", pops::nd::ScalarAdvection<Dim>{}, "none", "rusanov",
                           "conservative", "explicit");
  pops::add_compiled_model(system, "receiver", pops::nd::ScalarAdvection<Dim>{}, "none", "rusanov",
                           "conservative", "explicit");
}

template <int Dim>
void add_unit_to_first_coupling_candidate(const std::vector<pops::MultiFab<Dim>*>& candidates) {
  if (candidates.size() != 2 || candidates.front() == nullptr)
    throw std::logic_error("coupling test received an incomplete candidate pack");
  pops::MultiFab<Dim>& donor = *candidates.front();
  if (donor.local_size() != 0) {
    const auto values = donor.fab(0).view();
    if constexpr (std::is_same_v<typename pops::MultiFab<Dim>::memory_space, Kokkos::HostSpace>)
      values(donor.box(0).lo, 0) += pops::Real(1);
    else
      pops::for_each_cell(
          pops::Box<Dim>{donor.box(0).lo, donor.box(0).lo},
          [=] POPS_HD(const pops::Index<Dim>& index) { values(index, 0) += pops::Real(1); });
  }
  if constexpr (!std::is_same_v<typename pops::MultiFab<Dim>::memory_space, Kokkos::HostSpace>)
    Kokkos::fence();
}

template <int Dim>
void install_authority_program(pops::System<Dim>& system, std::string_view mode,
                               std::string_view identity, std::string_view marker = {},
                               std::string_view release = {}) {
#if !defined(POPS_TEST_TMPDIR)
  throw std::runtime_error("ABI-v5 authority fixture requires POPS_TEST_TMPDIR");
#else
  static std::size_t fixture_index = 0;
  const std::string prefix = std::string(POPS_TEST_TMPDIR) + "/system_transaction_authority_" +
                             std::to_string(++fixture_index);
  const std::string source_path = prefix + ".cpp";
  const std::string library_path = prefix + ".so";
  {
    std::ofstream source(source_path);
    if (!source)
      throw std::runtime_error("cannot create ABI-v5 authority fixture source");
    source << pops::test::program_v5::authority_program_source(mode, identity, {"gas"}, marker,
                                                               release);
  }
  const auto compiled = pops::test::native_dso::compile_shared(source_path, library_path);
  if (!compiled.ok) {
    pops::test::native_dso::report_compile_failure("test_system_transaction_authority", compiled);
    throw std::runtime_error("ABI-v5 authority fixture compilation failed");
  }
  system.install_program(library_path);
#endif
}

constexpr std::string_view kMailboxBalanceRoute =
    "pops.balance-ledger-route.v1:sha256:"
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
constexpr std::string_view kMailboxProjection =
    "pops.test.system-transaction-authority/mailbox-projection";
const std::string kMailboxBalanceRouteText{kMailboxBalanceRoute};
const std::string kMailboxProjectionText{kMailboxProjection};
std::uint64_t mailbox_callback_calls = 0;

template <int Dim>
void install_mailbox_authority_program(pops::System<Dim>& system) {
#if !defined(POPS_TEST_TMPDIR)
  throw std::runtime_error("ABI-v5 mailbox fixture requires POPS_TEST_TMPDIR");
#else
  static std::size_t fixture_index = 0;
  const std::string prefix = std::string(POPS_TEST_TMPDIR) + "/system_transaction_mailbox_" +
                             std::to_string(++fixture_index);
  const std::string source_path = prefix + ".cpp";
  const std::string library_path = prefix + ".so";
  const pops::test::program_v5::CallbackProgramTransactionAuthorities authorities{
      .balance_routes = {std::string(kMailboxBalanceRoute)},
      .step_projections = {std::string(kMailboxProjection)},
  };
  {
    std::ofstream source(source_path);
    if (!source)
      throw std::runtime_error("cannot create ABI-v5 mailbox fixture source");
    source << pops::test::program_v5::callback_program_source(
        0, "test.system-transaction-authority.mailbox.v5", "test.authority.clock", {"gas"}, {},
        "pops_test_v5_mailbox_callback", "uniform", {}, authorities);
  }
  const auto compiled = pops::test::native_dso::compile_shared(source_path, library_path);
  if (!compiled.ok) {
    pops::test::native_dso::report_compile_failure("test_system_transaction_authority", compiled);
    throw std::runtime_error("ABI-v5 mailbox fixture compilation failed");
  }
  system.install_program(library_path);
#endif
}

}  // namespace

template <int Dim>
void install_lane(pops::System<Dim>& system, const char* identity) {
  system.install_prepared_boundary_execution_lane(std::make_shared<pops::ExecutionLane>(
      pops::ExecutionLane::duplicate_world_collectively(identity)));
}

template <int Dim>
void add_view_test_block(pops::System<Dim>& system) {
#if defined(POPS_HAS_KOKKOS)
  static std::unique_ptr<Kokkos::ScopeGuard> guard =
      !Kokkos::is_initialized() && !Kokkos::is_finalized() ? std::make_unique<Kokkos::ScopeGuard>()
                                                           : nullptr;
#endif
  install_lane(system, "pops.test.system-transaction-authority.view");
  add_authority_block(system);
  system.set_poisson("charge_density", "cartesian_cg");
  install_authority_program(system, "noop", "test.system-transaction-authority.view.v5");
  system.mark_bound();
}

namespace {

using AuthoritySystem = pops::System<pops::kNativeDimension>;
AuthoritySystem* authority_probe_system = nullptr;
bool* authority_public_time_refused = nullptr;
bool* authority_public_macro_refused = nullptr;
bool* authority_provisional_public_refused = nullptr;
bool* authority_provisional_scope_read = nullptr;

}  // namespace

extern "C" void pops_test_v5_public_reader_probe() noexcept {
  if (authority_probe_system == nullptr)
    return;
  try {
    (void)authority_probe_system->time();
  } catch (const std::logic_error&) {
    if (authority_public_time_refused != nullptr)
      *authority_public_time_refused = true;
  } catch (...) {
  }
  try {
    (void)authority_probe_system->macro_step();
  } catch (const std::logic_error&) {
    if (authority_public_macro_refused != nullptr)
      *authority_public_macro_refused = true;
  } catch (...) {
  }
}

extern "C" void pops_test_v5_provisional_probe() noexcept {
  if (authority_probe_system == nullptr)
    return;
  try {
    (void)authority_probe_system->program_diagnostic("candidate");
  } catch (const std::logic_error&) {
    if (authority_provisional_public_refused != nullptr)
      *authority_provisional_public_refused = true;
  } catch (...) {
  }
  try {
    auto scope = authority_probe_system->_provisional_read_scope();
    if (authority_provisional_scope_read != nullptr)
      *authority_provisional_scope_read =
          scope.valid() && authority_probe_system->program_diagnostic("candidate") == pops::Real(7);
  } catch (...) {
  }
}

extern "C" void pops_test_v5_mailbox_callback(std::uint64_t, void* opaque, double) {
  using Services = pops::runtime::program::ProgramExecutionServices<pops::kNativeDimension>;
  auto* context = static_cast<Services*>(opaque);
  if (context == nullptr)
    throw std::logic_error("ABI-v5 mailbox callback received a null execution context");
  if (mailbox_callback_calls++ != 0)
    return;
  context->record_balance_term(kMailboxBalanceRouteText, "storage_change", pops::Real(1));
  context->record_balance_term(kMailboxBalanceRouteText, "outward_boundary_flux", pops::Real(2));
  context->record_balance_term(kMailboxBalanceRouteText, "sources", pops::Real(3));
  context->record_balance_term(kMailboxBalanceRouteText, "reflux", pops::Real(4));
  context->record_balance_term(kMailboxBalanceRouteText, "projection", pops::Real(5));
  context->note_step_projection(kMailboxProjectionText);
  throw std::runtime_error("injected mailbox candidate rejection");
}

TEST(SystemTransactionAuthority, RejectRestoresProgramImageAndRetryCommitsOnce) {
  constexpr int dim = pops::kNativeDimension;
  pops::System<dim> system(unit_config<dim>(4));
  install_lane(system, "pops.test.system-transaction-authority.retry");
  add_authority_block(system);
  install_authority_program(system, "reject_first_attempt",
                            "test.system-transaction-authority.retry.v5");
  system.mark_bound();

  EXPECT_EQ(system.accepted_transaction_generation_(), 0u);
  EXPECT_THROW(system.step(0.125), std::runtime_error);
  EXPECT_EQ(system.accepted_transaction_generation_(), 0u);
  EXPECT_DOUBLE_EQ(system.time(), 0.0);
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_TRUE(system.program_diagnostics().empty());
  EXPECT_THROW((void)system.program_diagnostic("test.program.v5.authority.calls"),
               std::out_of_range);

  EXPECT_NO_THROW(system.step(0.125));
  EXPECT_EQ(system.accepted_transaction_generation_(), 1u);
  EXPECT_DOUBLE_EQ(system.time(), 0.125);
  EXPECT_EQ(system.macro_step(), 1);
  ASSERT_EQ(system.program_diagnostics().size(), 2u);
  EXPECT_DOUBLE_EQ(system.program_diagnostic("test.program.v5.authority.calls"), 2.0);
  EXPECT_DOUBLE_EQ(system.program_diagnostic("test.program.v5.authority.last_dt"), 0.125);
}

TEST(SystemTransactionAuthority, ExternalAcceptedWindowRemainsComposableWithInternalStep) {
  constexpr int dim = pops::kNativeDimension;
  pops::System<dim> system(unit_config<dim>(4));
  install_lane(system, "pops.test.system-transaction-authority.external");
  add_authority_block(system);
  install_authority_program(system, "noop", "test.system-transaction-authority.external.v5");
  system.mark_bound();

  system.begin_step_transaction();
  EXPECT_EQ(system.accepted_transaction_generation_(), 0u);
  // The generation remains at its accepted value until the outer envelope seals it.
  EXPECT_NO_THROW(system.step(0.25));
  EXPECT_EQ(system.accepted_transaction_generation_(), 0u);
  EXPECT_THROW((void)system.step_change_l2(), std::logic_error);
  {
    auto scope = system._provisional_read_scope();
    EXPECT_TRUE(scope.valid());
    EXPECT_NO_THROW((void)system.step_change_l2_for_block("gas"));
    // Candidate diagnostics validate their input collectively before any rank reaches the
    // reduction.  A local unknown block is therefore deliberately surfaced as one lane-wide
    // refusal instead of leaking a rank-local out_of_range exception.
    EXPECT_THROW((void)system.step_change_l2_for_block("unknown"), std::runtime_error);
  }
  // The outer savepoint retains the visibility writer until rollback/finalize. Public reads from
  // this writer thread are refused rather than manufacturing a lease over provisional state.
  EXPECT_THROW((void)system.time(), std::logic_error);
  EXPECT_THROW((void)system.macro_step(), std::logic_error);
  std::promise<void> reader_finished;
  std::future<void> reader_done = reader_finished.get_future();
  std::thread reader([&] {
    (void)system.time();
    reader_finished.set_value();
  });
  EXPECT_EQ(reader_done.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
  system.rollback_step_transaction();
  EXPECT_EQ(system.accepted_transaction_generation_(), 0u);
  EXPECT_EQ(reader_done.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  reader.join();

  EXPECT_DOUBLE_EQ(system.time(), 0.0);
  EXPECT_EQ(system.macro_step(), 0);

  system.begin_step_transaction();
  EXPECT_NO_THROW(system.step(0.25));
  EXPECT_NO_THROW(system.commit_step_transaction());
  // Commit performs hidden publication, but the visible generation is sealed only by finalize.
  EXPECT_EQ(system.accepted_transaction_generation_(), 0u);
  EXPECT_THROW((void)system.time(), std::logic_error);
  EXPECT_NO_THROW(system.finalize_step_transaction());
  EXPECT_EQ(system.accepted_transaction_generation_(), 1u);
  EXPECT_DOUBLE_EQ(system.time(), 0.25);
  EXPECT_EQ(system.macro_step(), 1);
}

TEST(SystemTransactionAuthority, ProvisionalMailboxLeasePreventsStaleBalanceOrProjectionRetry) {
  constexpr int dim = pops::kNativeDimension;
  pops::System<dim> system(unit_config<dim>(4));
  install_lane(system, "pops.test.system-transaction-authority.mailbox");
  add_authority_block(system);
  mailbox_callback_calls = 0;
  install_mailbox_authority_program(system);
  system.mark_bound();

  system.begin_step_transaction();
  EXPECT_THROW(system.step(0.125), std::runtime_error);
  EXPECT_THROW((void)system.accepted_balance_terms(std::string(kMailboxBalanceRoute)),
               std::logic_error);
  EXPECT_THROW((void)system.selected_accepted_balance_terms(std::string(kMailboxBalanceRoute),
                                                            "gas", 0, {0}, {}),
               std::logic_error);
  {
    auto scope = system._provisional_read_scope();
    ASSERT_TRUE(scope.valid());
    const auto balance = system.accepted_balance_terms(std::string(kMailboxBalanceRoute));
    const auto selected = system.selected_accepted_balance_terms(std::string(kMailboxBalanceRoute),
                                                                 "gas", 0, {0}, {});
    EXPECT_DOUBLE_EQ(balance.at("storage_change"), 1.0);
    EXPECT_DOUBLE_EQ(balance.at("projection"), 5.0);
    EXPECT_EQ(selected, balance);
  }
  system.rollback_step_transaction();

  system.begin_step_transaction();
  EXPECT_NO_THROW(system.step(0.125));
  EXPECT_THROW((void)system.accepted_balance_terms(std::string(kMailboxBalanceRoute)),
               std::logic_error);
  {
    auto scope = system._provisional_read_scope();
    ASSERT_TRUE(scope.valid());
    EXPECT_THROW((void)system.accepted_balance_terms(std::string(kMailboxBalanceRoute)),
                 std::runtime_error);
    EXPECT_TRUE(system.consume_step_projections().empty());
  }
  EXPECT_NO_THROW(system.commit_step_transaction());
  EXPECT_NO_THROW(system.finalize_step_transaction());
  EXPECT_EQ(mailbox_callback_calls, 2u);
}

TEST(SystemTransactionAuthority, PreparedFullAcceptedTransactionsAllocateNothing) {
  constexpr int dim = pops::kNativeDimension;
  pops::System<dim> system(unit_config<dim>(4));
  install_lane(system, "pops.test.system-transaction-authority.allocations");
  add_authority_block(system);
  install_authority_program(system, "noop", "test.system-transaction-authority.allocations.v5");
  system.mark_bound();

  // The Program candidate is installed and mark_bound before this point.  The interval below is
  // exactly one external accepted transaction: begin -> step -> commit -> finalize.
  const pops::AllocationEventStats first_before = pops::allocation_event_stats();
  HeapAllocationWindow first_heap;
  bool first_accepted = true;
  try {
    system.begin_step_transaction();
    system.step(0.125);
    system.commit_step_transaction();
    system.finalize_step_transaction();
  } catch (...) {
    first_accepted = false;
  }
  const std::uint64_t first_heap_allocations = first_heap.close();
  const pops::AllocationEventStats first_after = pops::allocation_event_stats();
  ASSERT_TRUE(first_accepted);
  EXPECT_EQ(first_heap_allocations, 0u);
  EXPECT_EQ(first_after, first_before);
  EXPECT_EQ(system.accepted_transaction_generation_(), 1u);

  const pops::AllocationEventStats repeated_before = pops::allocation_event_stats();
  HeapAllocationWindow repeated_heap;
  bool repeated_accepted = true;
  try {
    system.begin_step_transaction();
    system.step(0.125);
    system.commit_step_transaction();
    system.finalize_step_transaction();
  } catch (...) {
    repeated_accepted = false;
  }
  const std::uint64_t repeated_heap_allocations = repeated_heap.close();
  const pops::AllocationEventStats repeated_after = pops::allocation_event_stats();
  ASSERT_TRUE(repeated_accepted);
  EXPECT_EQ(repeated_heap_allocations, 0u);
  EXPECT_EQ(repeated_after, repeated_before);
  EXPECT_EQ(system.accepted_transaction_generation_(), 2u);
}

TEST(SystemTransactionAuthority, PreparedCouplingWorkspaceReusesImagesAndRestoresEveryFailure) {
  constexpr int dim = pops::kNativeDimension;
  pops::System<dim> system(unit_config<dim>(4));
  install_lane(system, "pops.test.system-transaction-authority.coupling");
  add_coupling_blocks(system);

  enum class Fault { none, throw_after_mutation, nonfinite };
  auto fault = std::make_shared<Fault>(Fault::none);
  system.install_prepared_coupling_operator(
      "test.system-coupling.operator", "test.system-coupling/operator@1", {},
      [fault](pops::Real, const std::vector<pops::MultiFab<dim>*>& candidates) {
        add_unit_to_first_coupling_candidate(candidates);
        if (*fault == Fault::nonfinite) {
          pops::MultiFab<dim>& donor = *candidates.front();
          if (donor.local_size() != 0) {
            const auto values = donor.fab(0).view();
            const pops::Index<dim> cell = donor.box(0).lo;
            pops::for_each_cell(pops::Box<dim>{cell, cell}, [=] POPS_HD(const pops::Index<dim>& i) {
              values(i, 0) = std::numeric_limits<pops::Real>::quiet_NaN();
            });
            Kokkos::fence();
          }
        }
        if (*fault == Fault::throw_after_mutation) {
          // Deliberately leave a launched kernel unfinished when the operator reports its fault.
          // The resident rollback image must fence before copying it back to the HostSpace field.
          pops::MultiFab<dim>& donor = *candidates.front();
          if (donor.local_size() != 0) {
            const auto values = donor.fab(0).view();
            const pops::Index<dim> cell = donor.box(0).lo;
            pops::for_each_cell(pops::Box<dim>{cell, cell},
                                [=] POPS_HD(const pops::Index<dim>& i) { values(i, 0) += 5; });
          }
          throw std::runtime_error("injected prepared coupling failure");
        }
      });
  system.mark_bound();

  pops::MultiFab<dim> donor = [&] {
    const auto accepted = system.block_state(0);
    return pops::MultiFab<dim>(*accepted);
  }();
  pops::MultiFab<dim> receiver = [&] {
    const auto accepted = system.block_state(1);
    return pops::MultiFab<dim>(*accepted);
  }();
  const std::vector<pops::MultiFab<dim>*> candidates{&donor, &receiver};

  // The first post-bind application is included: the workspace is fully cold-primed by mark_bound,
  // so neither first use nor repeated use may allocate an image, mirror, exact witness, or vector.
  const pops::AllocationEventStats first_before = pops::allocation_event_stats();
  HeapAllocationWindow first_heap;
  ASSERT_EQ(system.apply_coupling_operators(pops::Real(1), candidates), 1u);
  const std::uint64_t first_heap_allocations = first_heap.close();
  const pops::AllocationEventStats first_after = pops::allocation_event_stats();
  EXPECT_EQ(first_heap_allocations, 0u);
  EXPECT_EQ(first_after, first_before);

  const pops::AllocationEventStats repeated_before = pops::allocation_event_stats();
  HeapAllocationWindow repeated_heap;
  ASSERT_EQ(system.apply_coupling_operators(pops::Real(1), candidates), 1u);
  ASSERT_EQ(system.apply_coupling_operators(pops::Real(1), candidates), 1u);
  const std::uint64_t repeated_heap_allocations = repeated_heap.close();
  const pops::AllocationEventStats repeated_after = pops::allocation_event_stats();
  EXPECT_EQ(repeated_heap_allocations, 0u);
  EXPECT_EQ(repeated_after, repeated_before);
  EXPECT_EQ(pops::norm_inf(donor, 0), pops::Real(3));

  *fault = Fault::throw_after_mutation;
  EXPECT_THROW(system.apply_coupling_operators(pops::Real(1), candidates), std::runtime_error);
  EXPECT_EQ(pops::norm_inf(donor, 0), pops::Real(3));
  *fault = Fault::nonfinite;
  EXPECT_THROW(system.apply_coupling_operators(pops::Real(1), candidates), std::runtime_error);
  EXPECT_EQ(pops::norm_inf(donor, 0), pops::Real(3));
  *fault = Fault::none;

  EXPECT_THROW(system.apply_coupling_operators(pops::Real(1), {&donor, &donor}),
               std::invalid_argument);
  pops::MultiFab<dim> wrong(donor.layout(), donor.distribution(), donor.local_rank(),
                            donor.ncomp() + 1, donor.ghosts());
  EXPECT_THROW(system.apply_coupling_operators(pops::Real(1), {&wrong, &receiver}),
               std::invalid_argument);
  EXPECT_EQ(pops::norm_inf(donor, 0), pops::Real(3));

  // The registry-level conservation ledger uses the same reusable workspace.  It is intentionally
  // constructed here because System's public typed installer currently has no conservation-group
  // argument; no alternative dynamic image or host mirror is allowed in the hot overload.
  using Registry = pops::runtime::system::SystemCouplingRegistry<dim>;
  using Operator = pops::runtime::system::PreparedCouplingOperator<dim>;
  using Group = pops::runtime::system::PreparedCouplingConservationGroup;
  Registry registry;
  registry.operator_contracts = {"test.system-coupling/conservation@1"};
  registry.coupled_operators = {pops::CouplingOperatorView{}};
  registry.operators.emplace_back(Operator(
      [](pops::Real, const std::vector<pops::MultiFab<dim>*>& states) {
        add_unit_to_first_coupling_candidate(states);
      },
      std::vector<Group>{
          {"donor-receiver", {{"donor", 0, 0, "scalar"}, {"receiver", 1, 0, "scalar"}}}}));
  pops::runtime::system::PreparedCouplingWorkspace<dim> conservation_workspace;
  const std::vector<const pops::MultiFab<dim>*> prototypes{&donor, &receiver};
  registry.bind_workspace(conservation_workspace, prototypes);
  conservation_workspace.bind_candidates(candidates);
  conservation_workspace.capture_rollback();
  EXPECT_EQ(registry.apply(pops::Real(1), conservation_workspace), 1u);
  EXPECT_TRUE(conservation_workspace.numeric_failure());
  conservation_workspace.restore_rollback();
  EXPECT_EQ(pops::norm_inf(donor, 0), pops::Real(3));
}

TEST(SystemTransactionAuthority, ConcurrentAcceptedReaderBlocksDuringCandidate) {
  constexpr int dim = pops::kNativeDimension;
  pops::System<dim> system(unit_config<dim>(4));
  install_lane(system, "pops.test.system-transaction-authority.reader");
  add_authority_block(system);
  const std::string prefix = std::string(POPS_TEST_TMPDIR) + "/authority-reader";
  const std::string marker = prefix + ".started";
  const std::string release = prefix + ".release";
  std::remove(marker.c_str());
  std::remove(release.c_str());
  install_authority_program(system, "wait_for_release",
                            "test.system-transaction-authority.reader.v5", marker, release);
  system.mark_bound();

  std::promise<void> step_finished;
  std::thread stepper([&] {
    try {
      system.step(0.125);
    } catch (...) {
      // The fixture has no injected failure; retaining this guard prevents a detached test thread
      // from surviving if a future implementation rejects the candidate.
    }
    step_finished.set_value();
  });
  for (int attempt = 0; attempt != 1000; ++attempt) {
    std::ifstream marker_stream(marker);
    if (marker_stream.good())
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (attempt == 999)
      FAIL() << "v5 authority candidate did not reach its coordination point";
  }

  std::promise<void> reader_finished;
  std::future<void> reader_done = reader_finished.get_future();
  std::thread reader([&] {
    (void)system.time();
    reader_finished.set_value();
  });
  EXPECT_EQ(reader_done.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
  std::ofstream release_stream(release);
  release_stream << "release\n";
  release_stream.close();
  EXPECT_EQ(step_finished.get_future().wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  stepper.join();
  reader.join();
}

TEST(SystemTransactionAuthority, PublicReaderCannotReenterUniformCandidate) {
  constexpr int dim = pops::kNativeDimension;
  pops::System<dim> system(unit_config<dim>(4));
  install_lane(system, "pops.test.system-transaction-authority.public-reader");
  add_authority_block(system);
  bool time_refused = false;
  bool macro_refused = false;
  authority_probe_system = &system;
  authority_public_time_refused = &time_refused;
  authority_public_macro_refused = &macro_refused;
  install_authority_program(system, "probe_public_readers",
                            "test.system-transaction-authority.public-reader.v5");
  system.mark_bound();

  EXPECT_NO_THROW(system.step(0.125));
  EXPECT_TRUE(time_refused);
  EXPECT_TRUE(macro_refused);
  EXPECT_DOUBLE_EQ(system.time(), 0.125);
  authority_probe_system = nullptr;
  authority_public_time_refused = nullptr;
  authority_public_macro_refused = nullptr;
}

TEST(SystemTransactionAuthority, ProvisionalScopeReadsCandidateAndRefusesAfterHiddenPublish) {
  constexpr int dim = pops::kNativeDimension;
  pops::System<dim> system(unit_config<dim>(4));
  install_lane(system, "pops.test.system-transaction-authority.provisional");
  add_authority_block(system);
  bool public_reader_refused = false;
  bool candidate_scope_read = false;
  authority_probe_system = &system;
  authority_provisional_public_refused = &public_reader_refused;
  authority_provisional_scope_read = &candidate_scope_read;
  install_authority_program(system, "probe_provisional",
                            "test.system-transaction-authority.provisional.v5");
  system.mark_bound();

  system.begin_step_transaction();
  ASSERT_NO_THROW(system.step(0.125));
  EXPECT_TRUE(public_reader_refused);
  EXPECT_TRUE(candidate_scope_read);

  ASSERT_NO_THROW(system.commit_step_transaction());
  EXPECT_THROW((void)system._provisional_read_scope(), std::logic_error);
  ASSERT_NO_THROW(system.finalize_step_transaction());
  authority_probe_system = nullptr;
  authority_provisional_public_refused = nullptr;
  authority_provisional_scope_read = nullptr;
}

TEST(SystemTransactionAuthority, AcceptedMultiFabReadViewOwnsReadLeaseUntilDestruction) {
  constexpr int dim = pops::kNativeDimension;
  using Field = pops::MultiFab<dim>;
  using View = pops::AcceptedMultiFabReadView<dim>;
  static_assert(!std::is_convertible_v<View, Field&>);
  static_assert(!std::is_convertible_v<View, const Field&>);
  static_assert(!std::is_copy_constructible_v<View>);
  static_assert(std::is_move_constructible_v<View>);
  static_assert(!HasRvalueViewAccess<View>);

  pops::System<dim> system(unit_config<dim>(4));
  add_view_test_block(system);

  {
    auto view = system.block_state(0);
    ASSERT_TRUE(view);
    ASSERT_NE(view.get(), nullptr);
    ASSERT_GT(view->ncomp(), 0);

    auto writer = std::async(std::launch::async, [&] {
      try {
        system.begin_step_transaction();
        system.rollback_step_transaction();
      } catch (const std::runtime_error& error) {
        return std::string_view(error.what()).find("candidate phase rejected collectively") !=
               std::string_view::npos;
      }
      return false;
    });
    EXPECT_EQ(writer.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_TRUE(writer.get())
        << "an accepted view must prevent a candidate writer from entering its visible state";
  }
  EXPECT_NO_THROW(system.begin_step_transaction());
  EXPECT_NO_THROW(system.rollback_step_transaction());
}

TEST(SystemTransactionAuthority, AcceptedMultiFabReadViewDoesNotBypassSameThreadWriter) {
  constexpr int dim = pops::kNativeDimension;
  pops::System<dim> system(unit_config<dim>(4));
  add_view_test_block(system);

  system.begin_step_transaction();
  EXPECT_THROW((void)system.block_state(0), std::logic_error);
  system.rollback_step_transaction();
  EXPECT_NO_THROW({
    auto view = system.block_state(0);
    EXPECT_NE(view.get(), nullptr);
  });
}
